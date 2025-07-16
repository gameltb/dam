import difflib
import importlib.util
import inspect
import logging
import os
import textwrap

import libcst as cst
import libcst.matchers as m
from libcst.metadata import MetadataWrapper, PositionProvider

# Import the decorator and custom exception from the new module
from domarkx.tools.tool_decorator import ToolError, tool_handler


# Helper for symbol resolution from modify_python_ast.py
def _resolve_symbol_path(full_symbol: str) -> tuple[str, str]:
    """
    Resolves a fully qualified symbol path to its corresponding file path and the
    internal symbol path within that file.
    e.g., 'my_package.my_module.MyClass.my_method' -> ('/path/to/my_module.py', 'MyClass.my_method')
    If the symbol only points to a module, the internal symbol path is an empty string.
    """
    parts = full_symbol.split(".")
    file_path = None
    internal_symbol_parts = []

    current_module_path_elements = []
    module_found = False

    for i, part in enumerate(parts):
        current_module_path_elements.append(part)
        temp_module_name = ".".join(current_module_path_elements)
        try:
            spec = importlib.util.find_spec(temp_module_name)
            if spec and spec.origin and spec.origin != "built-in":
                file_path = spec.origin
                module_found = True
                internal_symbol_parts = parts[i + 1 :]
                logging.debug(f"Resolved module '{temp_module_name}' to file: {file_path}")
                break
        except Exception as e:
            logging.debug(f"Could not import {temp_module_name}: {e}")

    if not module_found:
        # Changed from ValueError to ToolError
        raise ToolError(
            f"Could not resolve file path for symbol '{full_symbol}'. No Python module found or accessible."
        )

    # Optional: Check if the resolved file is within the workspace
    workspace_root = "/workspace/domarkx"  # Adjust this if workspace root is dynamic
    if not os.path.commonpath([os.path.abspath(file_path), os.path.abspath(workspace_root)]) == os.path.abspath(
        workspace_root
    ):
        logging.warning(
            f"Symbol '{full_symbol}' resolves to a file outside the workspace: {file_path}. Modifications might be restricted."
        )

    return file_path, ".".join(internal_symbol_parts)


# --- New auxiliary function for LibCST 1.0+ code extraction ---
def _get_node_full_code(node: cst.CSTNode) -> str:
    """
    Given a LibCST node, generates its full source code by wrapping it in a temporary module.
    This is the recommended way to get node source code in LibCST 1.0+ as node.code is removed.
    """
    temp_module = cst.Module(body=[node])
    return temp_module.code


class LibCstEditor:
    """
    A LibCST-based code editor for performing true lossless Python code modifications.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File '{file_path}' does not exist.")
        if not os.path.isfile(file_path):
            raise IsADirectoryError(f"Path '{file_path}' is a directory, not a file.")

        with open(file_path, "r", encoding="utf-8") as f:
            self.source_code = f.read()
        self.module = cst.parse_module(self.source_code)
        logging.info(f"Initialized LibCstEditor for {file_path}")

    def save(self, output_path: str = None):
        """Saves the modified code. LibCST handles formatting automatically."""
        if output_path is None:
            output_path = self.file_path
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(self.module.code)
            logging.info(f"Code successfully saved to '{output_path}'.")
            return f"Code successfully saved to '{output_path}'."
        except Exception as e:
            logging.error(f"Failed to save code to '{output_path}': {e}")
            raise ToolError(f"Failed to save code to '{output_path}': {e}", original_exception=e)

    def _apply_transformer(self, transformer: cst.CSTTransformer) -> None:
        """Applies a transformer and updates the module."""
        modified_tree = self.module.visit(transformer)

        # Check if any changes were actually applied
        if modified_tree.deep_equals(self.module):
            operation_performed = False
            # Check for specific flags from the transformer if it indicates action
            if hasattr(transformer, "updated") and transformer.updated:
                operation_performed = True
            elif hasattr(transformer, "created") and transformer.created:
                operation_performed = True
            elif hasattr(transformer, "deleted") and transformer.deleted:
                operation_performed = True

            if not operation_performed:
                if hasattr(transformer, "found_target") and not transformer.found_target:
                    raise ValueError("Operation failed: Target node not found.")
                else:
                    raise ValueError(
                        "Operation failed: Target node found but no changes were applied (content might be identical or no specific action flag set)."
                    )
        self.module = modified_tree
        logging.info("Transformer applied.")

    def _apply_script(self, script_code: str) -> None:
        """Executes a custom libcst transformation script."""
        exec_globals = {
            "cst": cst,
            "tree": self.module,
            "m": m,
            "logging": logging,
        }
        try:
            exec(script_code, exec_globals)
            if "tree" in exec_globals and isinstance(exec_globals["tree"], cst.Module):
                self.module = exec_globals["tree"]
                logging.info("Custom libcst script executed and module updated.")
            else:
                logging.warning("Custom script did not result in a valid cst.Module update in 'tree' variable.")
        except Exception as e:
            logging.error(f"Error executing custom libcst script: {e}")
            raise ToolError(f"Error executing custom libcst script: {e}", original_exception=e)

    def _get_node_matcher(self, internal_symbol_path: str, target_type: str) -> m.BaseMatcherNode:
        """
        Constructs a LibCST matcher based on the internal relative symbol path and target type within a file.
        """
        if not internal_symbol_path:
            if target_type == "module":
                return m.Module()
            else:
                raise ValueError(
                    f"Invalid target_type '{target_type}' for empty internal_symbol_path. Only 'module' is supported when internal_symbol_path is empty."
                )

        parts = internal_symbol_path.split(".")
        target_name = parts[-1]

        if target_type == "function" or target_type == "method":
            innermost_matcher = m.FunctionDef(name=m.Name(target_name))
        elif target_type == "class":
            innermost_matcher = m.ClassDef(name=m.Name(target_name))
        elif target_type == "assignment":
            # Match top-level assignments for now, more complex cases can be added
            innermost_matcher = m.Assign(targets=[m.AssignTarget(target=m.Name(target_name))])
        else:
            raise ValueError(f"Unsupported target_type for matcher: {target_type}")

        current_matcher = innermost_matcher
        for i in range(len(parts) - 2, -1, -1):
            parent_name = parts[i]
            # Assumes parent is a class for nested definitions
            current_matcher = m.ClassDef(
                name=m.Name(parent_name),
                body=m.IndentedBlock(
                    body=[
                        m.ZeroOrMore(),
                        current_matcher,
                        m.ZeroOrMore(),
                    ]
                ),
            )

        return current_matcher

    def create_code(self, parent_internal_symbol_path: str, target_type: str, code_content: str):
        """
        Creates a new class, function, or assignment statement within the specified parent symbol (module or class).
        """
        parent_matcher = self._get_node_matcher(
            parent_internal_symbol_path,
            "module" if not parent_internal_symbol_path else "class",
        )

        class CreatorTransformer(cst.CSTTransformer):
            def __init__(self, target_type, code_content, parent_matcher):
                self.target_type = target_type
                self.code_content = code_content
                self.parent_matcher = parent_matcher
                self.created = False
                self.found_target = False

            def _get_new_node(self):
                try:
                    # Parse as a statement, which can be a function, class, or assignment
                    new_node = cst.parse_statement(self.code_content)
                    return new_node
                except cst.ParserSyntaxError as e:
                    raise ToolError(
                        f"Invalid code content for {self.target_type}: {e}",
                        original_exception=e,
                    )

            def leave_Module(self, original_node, updated_node):
                if m.matches(original_node, self.parent_matcher):
                    self.found_target = True
                    if not self.created:
                        if self.target_type in ["function", "class", "assignment"]:
                            new_node = self._get_new_node()
                            new_body = list(updated_node.body)
                            # Ensure proper spacing for new code
                            if new_body and not isinstance(new_body[-1], cst.Newline):
                                new_body.append(cst.Newline())
                            new_body.append(new_node)
                            new_body.append(cst.Newline())  # Add newline after for separation
                            self.created = True
                            return updated_node.with_changes(body=tuple(new_body))
                return updated_node

            def leave_ClassDef(self, original_node, updated_node):
                if m.matches(original_node, self.parent_matcher):
                    self.found_target = True
                    if not self.created:
                        if self.target_type == "method":
                            new_node = self._get_new_node()
                            if not isinstance(new_node, cst.FunctionDef):
                                # Change from ValueError to ToolError
                                raise ToolError("Code content for 'method' must be a function definition.")

                            existing_body = list(updated_node.body.body)
                            if not existing_body:
                                # For empty class body, ensure proper indentation is handled by libcst
                                new_body_elements = [
                                    cst.Newline(),
                                    new_node,
                                    cst.Newline(),
                                ]
                            else:
                                new_body_elements = list(existing_body)
                                if not isinstance(new_body_elements[-1], cst.Newline):
                                    new_body_elements.append(cst.Newline())
                                new_body_elements.append(new_node)
                                new_body_elements.append(cst.Newline())  # Add newline after for separation

                            self.created = True
                            return updated_node.with_changes(
                                body=updated_node.body.with_changes(body=new_body_elements)
                            )
                return updated_node

        self._apply_transformer(CreatorTransformer(target_type, code_content, parent_matcher))
        return f"Attempted to create {target_type} under '{parent_internal_symbol_path}'. Operation status: {'Created' if getattr(self._apply_transformer, 'created', False) else 'Failed to create (check logs)'}"

    def update_code(self, internal_symbol_path: str, target_type: str, new_code_content: str):
        """
        Updates the code content of the specified symbol (class, function, method, assignment).
        """
        target_matcher = self._get_node_matcher(internal_symbol_path, target_type)

        class UpdaterTransformer(cst.CSTTransformer):
            def __init__(self, target_matcher, new_code_content, target_type):
                self.target_matcher = target_matcher
                self.new_code_content = new_code_content
                self.target_type = target_type
                self.updated = False
                self.found_target = False

            def on_leave(self, original_node, updated_node):
                if m.matches(original_node, self.target_matcher):
                    self.found_target = True
                    if not self.updated:
                        try:
                            if self.target_type in ["function", "method", "class"]:
                                new_node = cst.parse_statement(self.new_code_content)
                                if self.target_type == "function" or self.target_type == "method":
                                    if not isinstance(new_node, cst.FunctionDef):
                                        # Change from ValueError to ToolError
                                        raise ToolError("New code content must be a function definition.")
                                elif self.target_type == "class":
                                    if not isinstance(new_node, cst.ClassDef):
                                        # Change from ValueError to ToolError
                                        raise ToolError("New code content must be a class definition.")
                                self.updated = True
                                return new_node
                            elif self.target_type == "assignment":
                                parsed_statement = cst.parse_statement(self.new_code_content)
                                if (
                                    not isinstance(parsed_statement, cst.SimpleStatementLine)
                                    or not len(parsed_statement.body) == 1
                                ):
                                    raise ToolError("Expected a single statement.")
                                new_assignment_statement = parsed_statement.body[0]

                                if (
                                    not isinstance(new_assignment_statement, cst.Assign)
                                    or len(new_assignment_statement.targets) != 1
                                    or not isinstance(
                                        new_assignment_statement.targets[0].target,
                                        cst.Name,
                                    )
                                ):
                                    # Change from ValueError to ToolError
                                    raise ToolError(
                                        "New code content for 'assignment' must be a single assignment statement (e.g., 'VAR = 10')."
                                    )

                                if not (
                                    isinstance(original_node, cst.Assign)
                                    and isinstance(original_node.targets[0].target, cst.Name)
                                    and original_node.targets[0].target.value
                                    == new_assignment_statement.targets[0].target.value
                                ):
                                    # This check ensures we are updating the value of the same variable
                                    # Not trying to change the variable name itself via update
                                    # Change from ValueError to ToolError
                                    raise ToolError(
                                        "Cannot change the variable name during an assignment update. Provide the same variable name."
                                    )

                                new_value = new_assignment_statement.value

                                if not isinstance(updated_node, cst.Assign):
                                    # Change from TypeError to ToolError
                                    raise ToolError("Target node is not an assignment.")
                                self.updated = True
                                return updated_node.with_changes(value=new_value)
                        except cst.ParserSyntaxError as e:
                            # Change from ValueError to ToolError
                            raise ToolError(
                                f"Invalid new code content for {self.target_type}: {e}",
                                original_exception=e,
                            )
                        except Exception as e:
                            logging.error(f"Error updating node for {self.target_type}: {e}")
                            # Change generic Exception to ToolError
                            raise ToolError(
                                f"Error updating node for {self.target_type}: {e}",
                                original_exception=e,
                            )
                return updated_node

        self._apply_transformer(UpdaterTransformer(target_matcher, new_code_content, target_type))
        return f"Attempted to update {target_type} at '{internal_symbol_path}'. Operation status: {'Updated' if getattr(self._apply_transformer, 'updated', False) else 'Failed to update (check logs)'}"

    def delete_code(self, internal_symbol_path: str, target_type: str):
        """
        Deletes the code of the specified symbol (class, function, method, assignment).
        """
        target_matcher = self._get_node_matcher(internal_symbol_path, target_type)

        class DeleterTransformer(cst.CSTTransformer):
            def __init__(self, target_matcher, internal_symbol_path, target_type):
                self.target_matcher = target_matcher
                self.internal_symbol_path_parts = internal_symbol_path.split(".")
                self.target_type = target_type
                self.deleted = False
                self.found_target = False
                self.innermost_target_name = (
                    self.internal_symbol_path_parts[-1] if self.internal_symbol_path_parts else None
                )

            def on_leave(self, original_node, updated_node):
                if self.deleted:  # Once deleted, propagate the change up
                    return updated_node

                if self.target_type == "method":
                    # For methods, we need to modify the parent ClassDef's body
                    if isinstance(original_node, cst.ClassDef) and m.matches(original_node, self.target_matcher):
                        # This matcher should ideally be for the class containing the method
                        self.found_target = True
                        logging.debug(f"Handling method deletion in class '{original_node.name.value}'")
                        new_body_elements = []
                        method_found_and_removed = False
                        for element in updated_node.body.body:
                            # Check if the element is a function definition and matches the method name
                            if (
                                isinstance(element, cst.FunctionDef)
                                and element.name.value == self.innermost_target_name
                            ):
                                logging.info(f"Found and removing method '{self.innermost_target_name}'.")
                                method_found_and_removed = True
                                continue  # Skip adding this element to new_body_elements
                            new_body_elements.append(element)

                        if method_found_and_removed:
                            self.deleted = True
                            return updated_node.with_changes(
                                body=updated_node.body.with_changes(body=tuple(new_body_elements))
                            )
                    return updated_node

                # For function, class, assignment (top-level or nested non-method)
                if m.matches(original_node, self.target_matcher):
                    self.found_target = True
                    logging.info(f"Deleting node matching: {self.target_matcher} (type: {self.target_type})")
                    self.deleted = True
                    return cst.RemoveFromParent()  # This is the key for deletion
                return updated_node

        self._apply_transformer(DeleterTransformer(target_matcher, internal_symbol_path, target_type))
        return f"Attempted to delete {target_type} at '{internal_symbol_path}'. Operation status: {'Deleted' if getattr(self._apply_transformer, 'deleted', False) else 'Failed to delete (check logs)'}"


def _handle_list_mode_by_symbol(full_symbol: str, target_type: str, list_detail_level: str) -> str:
    """
    Handles the logic for list mode when the target argument is an importable symbol (uses inspect).
    """
    try:
        # Dynamically import the object
        target_obj = None
        if "." in full_symbol:
            module_name, obj_name = full_symbol.rsplit(".", 1)
            module = importlib.import_module(module_name)
            target_obj = getattr(module, obj_name)
        else:
            target_obj = importlib.import_module(full_symbol)

        # If target_type is specified, check against it
        if target_type:
            if target_type == "function" and not inspect.isfunction(target_obj) and not inspect.isbuiltin(target_obj):
                return f"Symbol '{full_symbol}' is not a function."
            elif target_type == "class" and not inspect.isclass(target_obj):
                return f"Symbol '{full_symbol}' is not a class."
            elif target_type == "module" and not inspect.ismodule(target_obj):
                return f"Symbol '{full_symbol}' is not a module."

        results = [f"--- Symbol: {full_symbol} ---"]

        if list_detail_level in ["with_docstring", "full_definition"]:
            doc = inspect.getdoc(target_obj)
            if doc:
                results.append("--- Docstring ---")
                results.append(doc)
                results.append("---")
            else:
                results.append("This symbol has no docstring.")

        if list_detail_level == "full_definition":
            try:
                source_lines, start_lineno = inspect.getsourcelines(target_obj)
                results.append("--- Source Code ---")
                for i, line in enumerate(source_lines):
                    results.append(f"{start_lineno + i} | {line.rstrip()}")
            except TypeError:
                results.append(
                    "Note: Source code is not available for this symbol (might be built-in or dynamically generated)."
                )
            except OSError as e:
                results.append(f"Error: Could not retrieve source code file for this symbol: {e}")

        return "\n".join(results)

    except ImportError as e:
        # Changed from ImportError to ToolError
        raise ToolError(
            f"Could not import symbol '{full_symbol}'. Ensure the module is installed and accessible: {e}",
            original_exception=e,
        )
    except AttributeError as e:
        # Changed from AttributeError to ToolError
        raise ToolError(f"Object not found in symbol '{full_symbol}': {e}", original_exception=e)
    except Exception as e:
        # Changed from generic Exception to ToolError
        raise ToolError(
            f"An error occurred while querying symbol '{full_symbol}': {e}",
            original_exception=e,
        )


def _handle_list_mode_by_path(
    file_path: str,
    internal_target_symbol_path: str,
    target_type: str,
    list_detail_level: str,
) -> str:
    """
    Handles the logic for list mode when the path argument is provided (uses libcst).
    """
    files_to_process = []
    if os.path.isfile(file_path):
        files_to_process = [file_path]
    elif os.path.isdir(file_path):
        for root, _, filenames in os.walk(file_path):
            for filename in filenames:
                if filename.lower().endswith(".py"):
                    files_to_process.append(os.path.join(root, filename))
    else:
        raise ValueError(f"Path '{file_path}' is neither a file nor a directory.")

    output = []
    for current_file_path in files_to_process:
        if not current_file_path.lower().endswith(".py"):
            output.append(f"File: {current_file_path} (Not a Python file, skipping)")
            continue

        # Add file header if listing all definitions in a file/directory
        if not internal_target_symbol_path:
            output.append(f"\n--- File: {current_file_path} ---")

        with open(current_file_path, "r", encoding="utf-8") as f:
            file_content = f.read()
        try:
            parsed_cst = cst.parse_module(file_content)
        except cst.ParserSyntaxError as e:
            logging.error(f"Syntax error in file {current_file_path}: {e}")
            output.append(f"  Error: Syntax error in file {current_file_path} - {e}")
            continue

        class DefinitionCollector(cst.CSTVisitor):
            METADATA_DEPENDENCIES = (PositionProvider,)

            def __init__(self, target_symbol_path, target_type_filter, detail_level):
                self.definitions = []
                self.target_symbol_path_parts = target_symbol_path.split(".") if target_symbol_path else []
                self.target_type_filter = target_type_filter
                self.detail_level = detail_level
                self.found_specific_target = False
                self.node_stack = []  # To track current scope (e.g., inside a class)

            def on_visit(self, node: cst.CSTNode) -> bool:
                super().on_visit(node)
                # Push the node onto the stack before visiting its children
                self.node_stack.append(node)
                return True  # Always return True to continue visiting children

            def on_leave(self, original_node: cst.CSTNode) -> None:
                super().on_leave(original_node)
                # Pop the node from the stack after visiting its children
                if self.node_stack and self.node_stack[-1] is original_node:
                    self.node_stack.pop()

            def _get_current_symbol_path(self, node_name: str) -> str:
                path_parts = []
                for n in self.node_stack[:-1]:
                    if isinstance(n, cst.ClassDef):
                        path_parts.append(n.name.value)

                if not path_parts:  # If no classes in stack, it's a module-level definition
                    logging.debug(f"_get_current_symbol_path: Module-level item '{node_name}'")
                    return node_name  # The symbol path is just its own name

                path_parts.append(node_name)
                full_path = ".".join(path_parts)
                logging.debug(f"_get_current_symbol_path: Nested item '{node_name}', full path: {full_path}")
                return full_path

            def _is_target_match(self, node_name: str, node_type: str) -> bool:
                current_full_path = self._get_current_symbol_path(node_name)
                logging.debug(
                    f"_is_target_match: Checking '{current_full_path}' (type: {node_type}) against target '{'.'.join(self.target_symbol_path_parts)}' (filter: {self.target_type_filter})"
                )

                if self.target_symbol_path_parts:
                    # If a specific symbol path is targeted, check for an exact match
                    return current_full_path == ".".join(self.target_symbol_path_parts)
                else:
                    # If no specific symbol path is targeted (i.e., listing all or by type)
                    if self.target_type_filter is None or self.target_type_filter == "":
                        # No specific target_type filter, so include all recognized top-level definition types
                        # Explicitly exclude 'method' unless it's specifically requested
                        if node_type == "method":
                            return False  # Do not list methods by default when listing all
                        return node_type in ["function", "class", "assignment"]
                    else:
                        # Specific target_type filter is provided, check for a match
                        return self.target_type_filter == node_type

            def _add_definition(self, node, type_name, name=None):
                def_name = name if name else node.name.value
                logging.debug(f"_add_definition: Adding definition: name={def_name}, type={type_name}")

                if self.target_symbol_path_parts and self._get_current_symbol_path(def_name) == ".".join(
                    self.target_symbol_path_parts
                ):
                    self.found_specific_target = True

                position_data = self.get_metadata(PositionProvider, node)
                start_line_num = position_data.start.line

                docstring = ""
                full_code = ""
                # --- Docstring Extraction Logic ---
                if self.detail_level in ["with_docstring", "full_definition"]:
                    if isinstance(node, (cst.FunctionDef, cst.ClassDef, cst.Module)):
                        docstring = f"\n    --- Docstring ---\n{node.get_docstring()}"

                # --- Full Source Code Extraction Logic ---
                if self.detail_level == "full_definition":
                    try:
                        # Use the new helper function for LibCST 1.0+ compatibility
                        full_code_content = _get_node_full_code(node)
                        full_code = "\n    --- Source Code ---\n" + full_code_content
                    except Exception as e:
                        logging.warning(f"Could not get full source code for {def_name} ({type_name}): {e}")
                        full_code = f"\n    --- Source Code Unavailable (Error: {e}) ---\n"

                self.definitions.append(f"  {type_name}: {def_name} (Line {start_line_num})\n{docstring}{full_code}")

            def visit_FunctionDef(self, node: cst.FunctionDef):
                is_method = False
                if self.node_stack:  # Check if there's a parent node
                    # The parent of a FunctionDef that is a method would be an IndentedBlock,
                    # and the parent of that IndentedBlock would be a ClassDef.
                    # We need to look up the stack to find a ClassDef.
                    for parent_node in reversed(self.node_stack[:-1]):  # Exclude current node itself
                        if isinstance(parent_node, cst.ClassDef):
                            is_method = True
                            break

                node_type = "method" if is_method else "function"
                logging.debug(f"visit_FunctionDef: Found {node_type} {node.name.value}")
                if self._is_target_match(node.name.value, node_type):
                    self._add_definition(node, node_type, name=node.name.value)
                return True

            def visit_ClassDef(self, node: cst.ClassDef):
                logging.debug(f"visit_ClassDef: Found class {node.name.value}")
                if self._is_target_match(node.name.value, "class"):
                    self._add_definition(node, "class")
                return True

            def visit_Assign(self, node: cst.Assign):
                logging.debug(f"visit_Assign: Node: {node}")
                logging.debug(f"visit_Assign: Targets: {[type(t.target).__name__ for t in node.targets]} ")
                for target_node in node.targets:
                    if isinstance(target_node.target, cst.Name):
                        var_name = target_node.target.value
                        logging.debug(f"visit_Assign: Processing variable name: {var_name}")
                        # Pass 'assignment' as the node_type for assignments to match the filter
                        if self._is_target_match(var_name, "assignment"):
                            logging.debug(f"visit_Assign: Match found for {var_name}. Calling _add_definition.")
                            self._add_definition(
                                node, "assignment", name=var_name
                            )  # Changed from "变量" to "assignment"
                        else:
                            logging.debug(f"visit_Assign: No match for {var_name} (type: assignment).")
                    else:
                        logging.debug(f"visit_Assign: Skipping non-Name target: {type(target_node.target).__name__}")
                return True

        collector = DefinitionCollector(internal_target_symbol_path, target_type, list_detail_level)
        wrapper = MetadataWrapper(parsed_cst)
        # Use visit instead of walk if you want to traverse the CST
        wrapper.visit(collector)

        if internal_target_symbol_path and not collector.found_specific_target:
            output.append(f"Symbol '{internal_target_symbol_path}' not found in file '{current_file_path}'.")
        elif (
            not collector.definitions and not internal_target_symbol_path
        ):  # Only report no definitions if not looking for a specific symbol
            output.append("  No code definitions found.")
        else:
            output.extend(collector.definitions)

    return "\n".join(output)


def _handle_modify_mode(
    file_to_process: str,
    internal_target_symbol_path: str,
    operation: str,
    target_type: str,
    code_content: str,
    modification_script: str,
) -> str:
    """
    Handles the logic for modify mode.
    """
    if not file_to_process.lower().endswith(".py"):
        raise ValueError(f"Modify operations only support Python files. File '{file_to_process}' is not a Python file.")

    editor = LibCstEditor(file_to_process)
    initial_module_code = editor.module.code  # Store initial state for diff

    if modification_script:
        logging.info(f"Applying custom modification script to {file_to_process}...")
        editor._apply_script(modification_script)
        result_message = "Custom libcst script executed."
    elif operation:
        valid_target_types = {"function", "class", "method", "assignment", "module"}
        if not target_type or target_type not in valid_target_types:
            raise ValueError(
                f"Invalid or missing 'target_type'. Must be one of {valid_target_types}. Got: {target_type}"
            )

        # For create, internal_target_symbol_path can be empty string for module root
        if operation != "create_code" and not internal_target_symbol_path:
            raise ValueError("'operation' requires 'target' (as an internal symbol path).")

        if operation == "create_code":
            if not code_content:
                raise ValueError("create_code operation requires 'code_content'.")
            result_message = editor.create_code(
                internal_target_symbol_path if internal_target_symbol_path else "",
                target_type,
                code_content,
            )
        elif operation == "update_code":
            if not code_content:
                raise ValueError("update_code operation requires 'code_content'.")
            result_message = editor.update_code(internal_target_symbol_path, target_type, code_content)
        elif operation == "delete_code":
            result_message = editor.delete_code(internal_target_symbol_path, target_type)
        else:
            raise ValueError(f"Unknown operation: '{operation}'.")
    else:
        raise ValueError("Either 'operation' or 'modification_script' must be provided for 'modify' mode.")

    final_module_code = editor.module.code
    if (
        not final_module_code == initial_module_code
    ):  # Use string comparison for simplicity, deep_equals is for CST nodes
        logging.info("Code changes detected. Showing diff:")
        diff = difflib.unified_diff(
            initial_module_code.splitlines(keepends=True),
            final_module_code.splitlines(keepends=True),
            fromfile=f"a/{os.path.basename(file_to_process)}\n",
            tofile=f"b/{os.path.basename(file_to_process)}\n",
        )
        for line in diff:
            logging.info(line.strip())

    editor.save()  # Save changes back to the file
    return result_message


# Main tool function
@tool_handler(log_level=logging.INFO)  # Apply the decorator
def python_code_handler(
    mode: str,
    target: str,  # This replaces both 'symbol' and 'target_symbol'
    path: str = None,  # Becomes optional, provides file context for non-importable targets
    # --- List Mode Specific Args ---
    list_detail_level: str = "names_only",
    # --- Modify Mode Specific Args ---
    operation: str = None,
    target_type: str = None,
    code_content: str = None,
    modification_script: str = None,
) -> str:
    """
    A comprehensive tool for handling Python code, supporting listing definitions,
    querying details, and performing code modifications.

    This tool leverages the `libcst` library for Python code parsing and AST manipulation,
    ensuring lossless modifications and precise node targeting. It can scan files or
    symbols to list code definitions such as functions, classes, methods, variables, etc.
    When provided with a `path` and a `target` (as a relative symbol path within the file),
    it finds and returns information about that specific symbol in the file.
    When only `path` is provided (and `target` is an empty string or None), it recursively
    scans the directory or file for all definitions.
    When only `target` is provided (as a fully importable symbol), it attempts to import
    and introspect that symbol.
    The `target_type` can filter which definition types are listed (applicable when scanning files/directories).
    The `list_detail_level` controls the verbosity of the output:
        - 'names_only' (default): Returns only the names of definitions.
        - 'with_docstring': Returns definition names along with their docstrings.
        - 'full_definition': Returns definition names, docstrings, and their complete source code blocks.
          (Note: Retrieving full definitions may rely on the `inspect` module's capabilities
          when `path` is not provided, or `libcst`'s source code extraction when `path` is provided.)

    *   'modify':
        Performs modifications on code within a specified file (`path`) or symbol (`target`).
        Requires the `operation` parameter ('create', 'update', 'delete'),
        the `target` parameter, and `target_type` to specify the exact target.
        - `operation='create'`: Creates new code at the location specified by `target`.
            Requires `code_content` for the code to be created.
            For module-level creation, `target` can be an empty string (representing the module root).
            For in-class creation, `target` should be 'ClassName'.
        - `operation='update'`: Updates the code at the location specified by `target`.
            Requires `code_content` for the new code.
            The target must be an existing definition (e.g., function, class, variable).
        - `operation='delete'`: Deletes the code definition at the location specified by `target`.
        Alternatively, a `modification_script` parameter can provide a custom `libcst`
        transformation script for more complex refactoring or modification logic.

    **Parameter (`parameter`) Details:**

    *   `mode` (str, required): Operation mode, must be 'list' or 'modify'.
    *   `target` (str, required): The Python code target to process.
                                - If `path` is not provided, `target` must be a fully importable symbol
                                  (e.g., 'os.path.join', 'my_package.my_module').
                                  The tool will attempt to import and operate on this symbol.
                                - If `path` is provided, `target` can be a relative symbol path within the file
                                  (e.g., 'MyClass.my_method', 'my_function', 'MODULE_CONSTANT').
                                  When `path` points to a directory, `target` can filter specific definitions within files.
                                  When `path` is a file and `target` is an empty string or None, it signifies operating on the file's root.

    *   `path` (str, optional): Optional parameter. Specifies the path to the Python file or directory
                              where the `target` resides, used when `target` is a relative symbol path.
                              Can be omitted if `target` is a fully importable symbol.
                              If it's a directory, 'list' mode will recursively scan all .py files.

    *   `list_detail_level` (str, optional): (For 'list' mode only)
                                            Controls the verbosity of the output. Options:
                                            'names_only' (default): Returns only definition names.
                                            'with_docstring': Returns definition names and their docstrings.
                                            'full_definition': Returns names, docstrings, and full source code blocks.
                                            (Note: Retrieving full definitions might depend on `inspect` module
                                            capabilities (when `path` is absent) or `libcst` source extraction (when `path` is present).)\n
    *   `operation` (str, optional): (For 'modify' mode only)
                                   The operation to perform: 'create', 'update', 'delete'.
    *   `target_type` (str, required for 'modify', optional for 'list'):
                                   - For 'modify' mode, required: Specifies the type of the target
                                     ('function', 'class', 'method', 'assignment', 'module').
                                   - For 'list' mode, optional: Filters definition types when scanning files/directories.
    *   `code_content` (str, optional): (For 'create' and 'update' operations in 'modify' mode)
                                      Provides the new code content to insert or replace.
                                      For functions/methods/classes, provide the full definition.
                                      For assignments, provide the full assignment statement, e.g., 'VAR = 10'.
    *   `modification_script` (str, optional): (For 'modify' mode only)
                                              A Python code string containing custom `libcst` transformation logic.
                                              This script executes in an environment providing:
                                              - `cst`: The libcst library itself.
                                              - `tree`: The current `libcst.Module` object being modified.\n                                              - `m`: The libcst matchers module (`libcst.matchers`).
                                              - `logging`: The Python logging module.
                                              The script should modify the `tree` object (required) or return a new `cst.Module`.
                                              If provided, `operation` and `code_content` parameters are ignored.

    **Internal Implementation and `libcst` Version Notes:**

    *   When using `mode='modify'` with `modification_script`, the tool loads the specified file,
        then executes the script to modify the AST. `libcst` handles code reformatting automatically.
    *   `libcst` APIs might change between versions (e.g., class or method name changes).\n        If encountering `AttributeError` or `TypeError` during script execution, check your
        `libcst` version and adjust the script according to its documentation.
    *   Here's an **example script** demonstrating how to update a function's docstring (assuming
        `libcst.matchers` is used to find the function and modify its `docstring` attribute, which might vary slightly across versions):\n
        ```python
        # --- modification_script example ---
        import libcst as cst
        import libcst.matchers as m
        import logging

        # Assume 'tree' is the libcst.Module object and 'target_func_name' is the name of the function to modify
        # 'target_func_name' would typically be passed from the tool's 'target' arg if dynamic.
        target_func_name = 'my_function' # This is just an example, replace with actual target

        class UpdateDocstringTransformer(cst.CSTTransformer):
            def __init__(self, func_name, new_docstring_content):
                self.func_name = func_name
                self.new_docstring_content = new_docstring_content
                self.updated = False
                self.found_target = False

            def leave_FunctionDef(self, original_node, updated_node):
                if original_node.name.value == self.func_name:
                    self.found_target = True
                    logging.info(f"Found function '{self.func_name}', attempting to update docstring.")

                    # Create a new docstring node
                    new_docstring_node = cst.Expr(
                        value=cst.SimpleString(f'\"\"\"{self.new_docstring_content}\"\"\"') # Use double quotes for consistency
                    )

                    new_body = list(updated_node.body.body)

                    # Check if the first statement is a docstring (SimpleString wrapped in Expr)
                    if new_body and isinstance(new_body[0], cst.Expr) \\\n                       and isinstance(new_body[0].value, cst.SimpleString):\n                        # Replace existing docstring
                        new_body[0] = new_docstring_node
                    else:
                        # Insert new docstring at the beginning of the body
                        # Ensure proper leading lines/indentation if necessary, libcst usually handles this.
                        new_body.insert(0, new_docstring_node)
                        # Add an extra newline if the body was not empty to separate docstring from first statement
                        if len(new_body) > 1 and not isinstance(new_body[1], cst.Newline):\n                            new_body.insert(1, cst.Newline())


                    self.updated = True
                    return updated_node.with_changes(body=updated_node.body.with_changes(body=tuple(new_body)))
                return updated_node

        # Example usage within the script:
        # These values would typically be passed from the tool call or derived.
        example_new_docstring = "This is the updated documentation for the function."

        # Instantiate the transformer and apply it to the 'tree'
        transformer = UpdateDocstringTransformer(target_func_name, example_new_docstring)
        tree = tree.visit(transformer) # Apply the transformation

        if not transformer.updated:
            logging.warning(f"Function '{target_func_name}' not found or docstring not updated by script.")
        else:
            logging.info(f"Docstring for '{target_func_name}' updated successfully by script.")
        # --- end of example script ---
        ```

    **Raises:**

    *   `ValueError`: Improper parameter usage, target not found, invalid code content, etc.
    *   `TypeError`: Incorrect parameter types.
    *   `FileNotFoundError`: Specified `path` does not exist.
    *   `ToolError`: A custom exception wrapping other exceptions raised by the tool, including `ImportError`, `AttributeError`, `SyntaxError`, `IOError`, `NotADirectoryError`, `IsADirectoryError`, and internal `libcst` related errors.
    """
    if mode not in ["list", "modify"]:
        raise ValueError(f"Invalid mode: '{mode}'. Mode must be 'list' or 'modify'.")

    if not isinstance(target, str):
        raise TypeError("Parameter 'target' must be of type string.")
    if path is not None and not isinstance(path, str):
        raise TypeError("Parameter 'path' must be of type string or None.")

    file_to_process = None
    internal_target_symbol_path = ""  # The symbol path relative to the file, derived from 'target' or passed directly

    if path:
        # If path is provided, 'target' is always treated as an internal symbol path
        file_to_process = path
        if not os.path.exists(file_to_process):
            raise FileNotFoundError(f"File or directory '{file_to_process}' does not exist.")
        internal_target_symbol_path = target
        # For list mode and path, an empty target means list all in the file/dir
        if mode == "list" and not target:
            internal_target_symbol_path = ""  # Ensure it's empty string for 'all'

    else:
        # If no path, 'target' must be a full importable symbol
        # The _resolve_symbol_path function now raises ToolError directly, so no try...except needed here.
        file_to_process, internal_target_symbol_path = _resolve_symbol_path(target)
        logging.info(
            f"Resolved target '{target}' to file: {file_to_process}, internal path: '{internal_target_symbol_path}'"
        )
        if not file_to_process.lower().endswith(".py"):
            raise ValueError(f"Target '{target}' resolved to file '{file_to_process}' which is not a Python file.")

    # --- Mode Dispatch ---
    if mode == "list":
        # If path is provided, use libcst (static analysis).
        # If no path, use inspect (dynamic import).
        if path:
            return _handle_list_mode_by_path(
                file_to_process,
                internal_target_symbol_path,
                target_type,
                list_detail_level,
            )
        else:  # No path, so target must be a full importable symbol
            return _handle_list_mode_by_symbol(
                target, target_type, list_detail_level
            )  # Use original 'target' for inspect

    elif mode == "modify":
        # For modify, target is always the internal path relative to the file_to_process
        return _handle_modify_mode(
            file_to_process,
            internal_target_symbol_path,
            operation,
            target_type,
            textwrap.dedent(code_content) if code_content else code_content,
            modification_script,
        )
    else:
        raise ValueError(f"Invalid mode: '{mode}'. Mode must be 'list' or 'modify'.")
