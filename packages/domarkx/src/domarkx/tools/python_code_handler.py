"""A comprehensive tool for handling Python code, using LibCST for lossless modifications."""

import difflib
import importlib.util
import inspect
import logging
import pathlib
import textwrap
from typing import Any

import libcst as cst
import libcst.matchers as m
from libcst.metadata import MetadataWrapper, PositionProvider

# Import the decorator and custom exception from the new module
from domarkx.tools.tool_factory import ToolError, tool_handler

logger = logging.getLogger(__name__)


# Helper for symbol resolution from modify_python_ast.py
def resolve_symbol_path(full_symbol: str) -> tuple[str, str]:
    """
    Resolve a fully qualified symbol path to its corresponding file path and the internal symbol path within that file.

    e.g., 'my_package.my_module.MyClass.my_method' -> ('/path/to/my_module.py', 'MyClass.my_method')
    If the symbol only points to a module, the internal symbol path is an empty string.
    """
    parts = full_symbol.split(".")
    file_path = None
    internal_symbol_parts: list[str] = []

    current_module_path_elements: list[str] = []
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
                logger.debug("Resolved module '%s' to file: %s", temp_module_name, file_path)
                break
        except Exception as e:
            logger.debug("Could not import %s: %s", temp_module_name, e)

    if not module_found:
        # Changed from ValueError to ToolError
        raise ToolError(
            f"Could not resolve file path for symbol '{full_symbol}'. No Python module found or accessible."
        )

    assert file_path is not None
    # Optional: Check if the resolved file is within the workspace
    workspace_root_path = pathlib.Path("/workspace/domarkx").resolve()
    file_path_resolved = pathlib.Path(file_path).resolve()
    if workspace_root_path not in file_path_resolved.parents and workspace_root_path != file_path_resolved:
        logger.warning(
            "Symbol '%s' resolves to a file outside the workspace: %s. Modifications might be restricted.",
            full_symbol,
            file_path,
        )

    return file_path, ".".join(internal_symbol_parts)


# --- New auxiliary function for LibCST 1.0+ code extraction ---
def _get_node_full_code(node: cst.CSTNode) -> str:
    """
    Generate the full source code for a LibCST node.

    This is the recommended way to get node source code in LibCST 1.0+ as node.code is removed.
    """
    temp_module = cst.Module(body=[node])  # type: ignore[list-item]
    return temp_module.code


class LibCstEditor:
    """A LibCST-based code editor for performing true lossless Python code modifications."""

    def __init__(self, file_path: str):
        """Initialize the editor with the path to a Python file."""
        self.file_path = file_path
        p = pathlib.Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"File '{file_path}' does not exist.")
        if not p.is_file():
            raise IsADirectoryError(f"Path '{file_path}' is a directory, not a file.")

        with p.open(encoding="utf-8") as f:
            self.source_code = f.read()
        self.module = cst.parse_module(self.source_code)
        logger.info("Initialized LibCstEditor for %s", file_path)

    def save(self, output_path: str | None = None) -> str:
        """Save the modified code, with LibCST handling formatting."""
        if output_path is None:
            output_path = self.file_path
        try:
            with pathlib.Path(output_path).open("w", encoding="utf-8") as f:
                f.write(self.module.code)
            logger.info("Code successfully saved to '%s'.", output_path)
            return f"Code successfully saved to '{output_path}'."
        except Exception as e:
            logger.error("Failed to save code to '%s': %s", output_path, e)
            raise ToolError(f"Failed to save code to '{output_path}': {e}", original_exception=e) from e

    def _apply_transformer(self, transformer: cst.CSTTransformer) -> None:
        """Apply a transformer and update the module."""
        modified_tree = self.module.visit(transformer)

        # Check if any changes were actually applied
        if modified_tree.deep_equals(self.module):
            operation_performed = False
            # Check for specific flags from the transformer if it indicates action
            if (
                getattr(transformer, "updated", False)
                or getattr(transformer, "created", False)
                or getattr(transformer, "deleted", False)
            ):
                operation_performed = True

            if not operation_performed:
                if not getattr(transformer, "found_target", True):
                    raise ValueError("Operation failed: Target node not found.")
                raise ValueError(
                    "Operation failed: Target node found but no changes were applied (content might be identical or no specific action flag set)."
                )
        self.module = modified_tree
        logger.info("Transformer applied.")

    def apply_script(self, script_code: str) -> None:
        """Execute a custom libcst transformation script."""
        exec_globals = {
            "cst": cst,
            "tree": self.module,
            "m": m,
            "logging": logger,
        }
        try:
            exec(script_code, exec_globals)
            if "tree" in exec_globals and isinstance(exec_globals["tree"], cst.Module):
                self.module = exec_globals["tree"]
                logger.info("Custom libcst script executed and module updated.")
            else:
                logger.warning("Custom script did not result in a valid cst.Module update in 'tree' variable.")
        except Exception as e:
            logger.error("Error executing custom libcst script: %s", e)
            raise ToolError(f"Error executing custom libcst script: {e}", original_exception=e) from e

    def _get_node_matcher(self, internal_symbol_path: str, target_type: str) -> m.BaseMatcherNode:
        """Construct a LibCST matcher based on the internal relative symbol path and target type."""
        if not internal_symbol_path:
            if target_type == "module":
                return m.Module()
            raise ValueError(
                f"Invalid target_type '{target_type}' for empty internal_symbol_path. Only 'module' is supported when internal_symbol_path is empty."
            )

        parts = internal_symbol_path.split(".")
        target_name = parts[-1]

        innermost_matcher: m.BaseMatcherNode
        if target_type in {"function", "method"}:
            innermost_matcher = m.FunctionDef(name=m.Name(target_name))
        elif target_type == "class":
            innermost_matcher = m.ClassDef(name=m.Name(target_name))
        elif target_type == "assignment":
            # Match top-level assignments for now, more complex cases can be added
            innermost_matcher = m.Assign(targets=[m.AssignTarget(target=m.Name(target_name))])
        else:
            raise ValueError(f"Unsupported target_type for matcher: {target_type}")

        current_matcher: m.BaseMatcherNode = innermost_matcher
        for i in range(len(parts) - 2, -1, -1):
            parent_name = parts[i]
            # Assumes parent is a class for nested definitions
            current_matcher = m.ClassDef(
                name=m.Name(parent_name),
                body=m.IndentedBlock(
                    body=[
                        m.ZeroOrMore(),
                        current_matcher,  # type: ignore[list-item]
                        m.ZeroOrMore(),
                    ]
                ),
            )

        return current_matcher

    def create_code(self, parent_internal_symbol_path: str, target_type: str, code_content: str) -> str:
        """Create a new class, function, or assignment statement."""
        parent_matcher = self._get_node_matcher(
            parent_internal_symbol_path,
            "module" if not parent_internal_symbol_path else "class",
        )

        class CreatorTransformer(cst.CSTTransformer):
            def __init__(self, target_type: str, code_content: str, parent_matcher: m.BaseMatcherNode) -> None:
                self.target_type = target_type
                self.code_content = code_content
                self.parent_matcher = parent_matcher
                self.created = False
                self.found_target = False

            def _get_new_node(self) -> cst.BaseStatement:
                try:
                    # Parse as a statement, which can be a function, class, or assignment
                    return cst.parse_statement(self.code_content)
                except cst.ParserSyntaxError as e:
                    raise ToolError(
                        f"Invalid code content for {self.target_type}: {e}",
                        original_exception=e,
                    ) from e

            def leave_Module(self, original_node: cst.Module, updated_node: cst.Module) -> cst.Module:  # noqa: N802
                if m.matches(original_node, self.parent_matcher):
                    self.found_target = True
                    if not self.created and self.target_type in ["function", "class", "assignment"]:
                        new_node = self._get_new_node()
                        new_body = list(updated_node.body)
                        # Ensure proper spacing for new code
                        if new_body and not isinstance(new_body[-1], cst.SimpleStatementLine):
                            new_body.append(cst.SimpleStatementLine(body=[]))
                        if isinstance(new_node, (cst.SimpleStatementLine, cst.BaseCompoundStatement)):
                            new_body.append(new_node)
                        new_body.append(cst.SimpleStatementLine(body=[]))
                        self.created = True
                        return updated_node.with_changes(body=tuple(new_body))  # type: ignore[assignment]
                return updated_node

            def leave_ClassDef(self, original_node: cst.ClassDef, updated_node: cst.ClassDef) -> cst.ClassDef:  # noqa: N802
                if m.matches(original_node, self.parent_matcher):
                    self.found_target = True
                    if not self.created and self.target_type == "method":
                        new_node = self._get_new_node()
                        if not isinstance(new_node, cst.FunctionDef):
                            # Change from ValueError to ToolError
                            raise ToolError("Code content for 'method' must be a function definition.")

                        existing_body = list(updated_node.body.body)
                        new_body_elements: list[cst.BaseStatement] = []
                        if not existing_body:
                            # For empty class body, ensure proper indentation is handled by libcst
                            new_body_elements = [
                                cst.SimpleStatementLine(body=[]),
                                new_node,
                                cst.SimpleStatementLine(body=[]),
                            ]
                        else:
                            new_body_elements = list(existing_body)  # type: ignore[arg-type]
                            if not isinstance(new_body_elements[-1], cst.SimpleStatementLine):
                                new_body_elements.append(cst.SimpleStatementLine(body=[]))
                            if isinstance(new_node, (cst.SimpleStatementLine, cst.BaseCompoundStatement)):
                                new_body_elements.append(new_node)
                            new_body_elements.append(cst.SimpleStatementLine(body=[]))

                        self.created = True
                        return updated_node.with_changes(
                            body=updated_node.body.with_changes(
                                body=tuple(new_body_elements)  # type: ignore[assignment]
                            )
                        )
                return updated_node

        self._apply_transformer(CreatorTransformer(target_type, code_content, parent_matcher))
        return f"Attempted to create {target_type} under '{parent_internal_symbol_path}'. Operation status: {'Created' if getattr(self._apply_transformer, 'created', False) else 'Failed to create (check logs)'}"

    def update_code(self, internal_symbol_path: str, target_type: str, new_code_content: str) -> str:
        """Update the code content of the specified symbol."""
        target_matcher = self._get_node_matcher(internal_symbol_path, target_type)

        class UpdaterTransformer(cst.CSTTransformer):
            def __init__(self, target_matcher: m.BaseMatcherNode, new_code_content: str, target_type: str) -> None:
                self.target_matcher = target_matcher
                self.new_code_content = new_code_content
                self.target_type = target_type
                self.updated = False
                self.found_target = False

            def on_leave(self, original_node: cst.CSTNode, updated_node: cst.CSTNode) -> Any:
                if m.matches(original_node, self.target_matcher):
                    self.found_target = True
                    if not self.updated:
                        try:
                            if self.target_type in ["function", "method", "class"]:
                                new_node = cst.parse_statement(self.new_code_content)
                                if self.target_type in {"function", "method"} and not isinstance(
                                    new_node, cst.FunctionDef
                                ):
                                    raise ToolError("New code content must be a function definition.")
                                if self.target_type == "class" and not isinstance(new_node, cst.ClassDef):
                                    raise ToolError("New code content must be a class definition.")
                                self.updated = True
                                return new_node
                            if self.target_type == "assignment":
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
                                    raise ToolError(
                                        "New code content for 'assignment' must be a single assignment statement (e.g., 'VAR = 10')."
                                    )

                                if not (
                                    isinstance(original_node, cst.Assign)
                                    and isinstance(original_node.targets[0].target, cst.Name)
                                    and original_node.targets[0].target.value
                                    == new_assignment_statement.targets[0].target.value
                                ):
                                    raise ToolError(
                                        "Cannot change the variable name during an assignment update. Provide the same variable name."
                                    )

                                new_value = new_assignment_statement.value

                                if not isinstance(updated_node, cst.Assign):
                                    raise ToolError("Target node is not an assignment.")
                                self.updated = True
                                return updated_node.with_changes(value=new_value)
                        except cst.ParserSyntaxError as e:
                            raise ToolError(
                                f"Invalid new code content for {self.target_type}: {e}",
                                original_exception=e,
                            ) from e
                        except Exception as e:
                            logger.error("Error updating node for %s: %s", self.target_type, e)
                            raise ToolError(
                                f"Error updating node for {self.target_type}: {e}",
                                original_exception=e,
                            ) from e
                return updated_node

        self._apply_transformer(UpdaterTransformer(target_matcher, new_code_content, target_type))
        return f"Attempted to update {target_type} at '{internal_symbol_path}'. Operation status: {'Updated' if getattr(self._apply_transformer, 'updated', False) else 'Failed to update (check logs)'}"

    def delete_code(self, internal_symbol_path: str, target_type: str) -> str:
        """Delete the code of the specified symbol."""
        target_matcher = self._get_node_matcher(internal_symbol_path, target_type)

        class DeleterTransformer(cst.CSTTransformer):
            def __init__(self, target_matcher: m.BaseMatcherNode, internal_symbol_path: str, target_type: str) -> None:
                self.target_matcher = target_matcher
                self.internal_symbol_path_parts = internal_symbol_path.split(".")
                self.target_type = target_type
                self.deleted = False
                self.found_target = False
                self.innermost_target_name = (
                    self.internal_symbol_path_parts[-1] if self.internal_symbol_path_parts else None
                )

            def on_leave(self, original_node: cst.CSTNode, updated_node: cst.CSTNode) -> Any:
                if self.deleted:  # Once deleted, propagate the change up
                    return updated_node

                if self.target_type == "method":
                    # For methods, we need to modify the parent ClassDef's body
                    if isinstance(original_node, cst.ClassDef) and m.matches(original_node, self.target_matcher):
                        # This matcher should ideally be for the class containing the method
                        self.found_target = True
                        logger.debug("Handling method deletion in class '%s'", original_node.name.value)
                        new_body_elements: list[cst.BaseStatement] = []
                        method_found_and_removed = False
                        if isinstance(updated_node, cst.ClassDef) and isinstance(updated_node.body, cst.IndentedBlock):
                            for element in updated_node.body.body:
                                # Check if the element is a function definition and matches the method name
                                if (
                                    isinstance(element, cst.FunctionDef)
                                    and element.name.value == self.innermost_target_name
                                ):
                                    logger.info("Found and removing method '%s'.", self.innermost_target_name)
                                    method_found_and_removed = True
                                    continue  # Skip adding this element to new_body_elements
                                new_body_elements.append(element)

                        if method_found_and_removed:
                            self.deleted = True
                            if isinstance(updated_node, cst.ClassDef):
                                return updated_node.with_changes(
                                    body=updated_node.body.with_changes(body=tuple(new_body_elements))
                                )
                    return updated_node

                # For function, class, assignment (top-level or nested non-method)
                if m.matches(original_node, self.target_matcher):
                    self.found_target = True
                    logger.info("Deleting node matching: %s (type: %s)", self.target_matcher, self.target_type)
                    self.deleted = True
                    return cst.RemoveFromParent()  # This is the key for deletion
                return updated_node

        self._apply_transformer(DeleterTransformer(target_matcher, internal_symbol_path, target_type))
        return f"Attempted to delete {target_type} at '{internal_symbol_path}'. Operation status: {'Deleted' if getattr(self._apply_transformer, 'deleted', False) else 'Failed to delete (check logs)'}"


def _handle_list_mode_by_symbol(full_symbol: str, target_type: str | None, list_detail_level: str) -> str:  # noqa: PLR0912
    """Handle the logic for list mode when the target is an importable symbol."""
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
            if target_type == "class" and not inspect.isclass(target_obj):
                return f"Symbol '{full_symbol}' is not a class."
            if target_type == "module" and not inspect.ismodule(target_obj):
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
        raise ToolError(
            f"Could not import symbol '{full_symbol}'. Ensure the module is installed and accessible: {e}",
            original_exception=e,
        ) from e
    except AttributeError as e:
        raise ToolError(f"Object not found in symbol '{full_symbol}': {e}", original_exception=e) from e
    except Exception as e:
        raise ToolError(
            f"An error occurred while querying symbol '{full_symbol}': {e}",
            original_exception=e,
        ) from e


def _handle_list_mode_by_path(
    file_path: str,
    internal_target_symbol_path: str,
    target_type: str | None,
    list_detail_level: str,
) -> str:
    """Handle the logic for list mode when the path argument is provided."""
    p = pathlib.Path(file_path)
    if p.is_file():
        files_to_process = [p]
    elif p.is_dir():
        files_to_process = list(p.rglob("*.py"))
    else:
        raise ValueError(f"Path '{file_path}' is neither a file nor a directory.")

    output: list[str] = []
    for current_file_path in files_to_process:
        if not internal_target_symbol_path:
            output.append(f"\n--- File: {current_file_path} ---")

        try:
            file_content = current_file_path.read_text(encoding="utf-8")
            parsed_cst = cst.parse_module(file_content)
        except cst.ParserSyntaxError as e:
            logger.error("Syntax error in file %s: %s", current_file_path, e)
            output.append(f"  Error: Syntax error in file {current_file_path} - {e}")
            continue

        class DefinitionCollector(cst.CSTVisitor):
            METADATA_DEPENDENCIES = (PositionProvider,)

            def __init__(self, target_symbol_path: str, target_type_filter: str | None, detail_level: str) -> None:
                self.definitions: list[str] = []
                self.target_symbol_path_parts = target_symbol_path.split(".") if target_symbol_path else []
                self.target_type_filter = target_type_filter
                self.detail_level = detail_level
                self.found_specific_target = False
                self.node_stack: list[cst.CSTNode] = []

            def on_visit(self, node: cst.CSTNode) -> bool:
                super().on_visit(node)
                self.node_stack.append(node)
                return True

            def on_leave(self, original_node: cst.CSTNode) -> None:
                super().on_leave(original_node)
                if self.node_stack and self.node_stack[-1] is original_node:
                    self.node_stack.pop()

            def _get_current_symbol_path(self, node_name: str) -> str:
                path_parts = [n.name.value for n in self.node_stack[:-1] if isinstance(n, cst.ClassDef)]
                if not path_parts:
                    logger.debug("_get_current_symbol_path: Module-level item '%s'", node_name)
                    return node_name
                path_parts.append(node_name)
                full_path = ".".join(path_parts)
                logger.debug("_get_current_symbol_path: Nested item '%s', full path: %s", node_name, full_path)
                return full_path

            def _is_target_match(self, node_name: str, node_type: str) -> bool:
                current_full_path = self._get_current_symbol_path(node_name)
                logger.debug(
                    "_is_target_match: Checking '%s' (type: %s) against target '%s' (filter: %s)",
                    current_full_path,
                    node_type,
                    ".".join(self.target_symbol_path_parts),
                    self.target_type_filter,
                )
                if self.target_symbol_path_parts:
                    return current_full_path == ".".join(self.target_symbol_path_parts)
                if not self.target_type_filter:
                    return node_type != "method" and node_type in ["function", "class", "assignment"]
                return self.target_type_filter == node_type

            def _add_definition(self, node: cst.CSTNode, type_name: str, name: str | None = None) -> None:
                def_name = name
                if def_name is None:
                    if isinstance(node, (cst.FunctionDef, cst.ClassDef)):
                        def_name = node.name.value
                    else:
                        raise TypeError(f"Cannot get name from node of type {type(node).__name__}")
                logger.debug("_add_definition: Adding definition: name=%s, type=%s", def_name, type_name)

                if self.target_symbol_path_parts and self._get_current_symbol_path(def_name) == ".".join(
                    self.target_symbol_path_parts
                ):
                    self.found_specific_target = True

                position_data = self.get_metadata(PositionProvider, node)
                start_line_num = getattr(getattr(position_data, "start", None), "line", -1)

                docstring, full_code = "", ""
                if self.detail_level in ["with_docstring", "full_definition"] and isinstance(
                    node, (cst.FunctionDef, cst.ClassDef, cst.Module)
                ):
                    docstring = f"\n    --- Docstring ---\n{node.get_docstring() or ''}"
                if self.detail_level == "full_definition":
                    try:
                        full_code = f"\n    --- Source Code ---\n{_get_node_full_code(node)}"
                    except Exception as e:
                        logger.warning("Could not get full source code for %s (%s): %s", def_name, type_name, e)
                        full_code = f"\n    --- Source Code Unavailable (Error: {e}) ---\n"
                self.definitions.append(f"  {type_name}: {def_name} (Line {start_line_num})\n{docstring}{full_code}")

            def visit_FunctionDef(self, node: cst.FunctionDef) -> None:  # noqa: N802
                is_method = any(isinstance(p, cst.ClassDef) for p in self.node_stack[:-1])
                node_type = "method" if is_method else "function"
                logger.debug("visit_FunctionDef: Found %s %s", node_type, node.name.value)
                if self._is_target_match(node.name.value, node_type):
                    self._add_definition(node, node_type, name=node.name.value)

            def visit_ClassDef(self, node: cst.ClassDef) -> None:  # noqa: N802
                logger.debug("visit_ClassDef: Found class %s", node.name.value)
                if self._is_target_match(node.name.value, "class"):
                    self._add_definition(node, "class")

            def visit_Assign(self, node: cst.Assign) -> None:  # noqa: N802
                logger.debug("visit_Assign: Node: %s", node)
                for target_node in node.targets:
                    if isinstance(target_node.target, cst.Name):
                        var_name = target_node.target.value
                        logger.debug("visit_Assign: Processing variable name: %s", var_name)
                        if self._is_target_match(var_name, "assignment"):
                            logger.debug("visit_Assign: Match found for %s. Calling _add_definition.", var_name)
                            self._add_definition(node, "assignment", name=var_name)
                        else:
                            logger.debug("visit_Assign: No match for %s (type: assignment).", var_name)
                    else:
                        logger.debug("visit_Assign: Skipping non-Name target: %s", type(target_node.target).__name__)

        collector = DefinitionCollector(internal_target_symbol_path, target_type, list_detail_level)
        wrapper = MetadataWrapper(parsed_cst)
        wrapper.visit(collector)

        if internal_target_symbol_path and not collector.found_specific_target:
            output.append(f"Symbol '{internal_target_symbol_path}' not found in file '{current_file_path}'.")
        elif not collector.definitions and not internal_target_symbol_path:
            output.append("  No code definitions found.")
        else:
            output.extend(collector.definitions)

    return "\n".join(output)


def _handle_modify_mode(  # noqa: PLR0913, PLR0912
    file_to_process: str,
    internal_target_symbol_path: str,
    operation: str | None,
    target_type: str | None,
    code_content: str | None,
    modification_script: str | None,
) -> str:
    """Handle the logic for modify mode."""
    if not file_to_process.lower().endswith(".py"):
        raise ValueError(f"Modify operations only support Python files. File '{file_to_process}' is not a Python file.")

    editor = LibCstEditor(file_to_process)
    initial_module_code = editor.module.code

    if modification_script:
        logger.info("Applying custom modification script to %s...", file_to_process)
        editor.apply_script(modification_script)
        result_message = "Custom libcst script executed."
    elif operation:
        valid_target_types = {"function", "class", "method", "assignment", "module"}
        if not target_type or target_type not in valid_target_types:
            raise ValueError(
                f"Invalid or missing 'target_type'. Must be one of {valid_target_types}. Got: {target_type}"
            )

        if operation != "create_code" and not internal_target_symbol_path:
            raise ValueError("'operation' requires 'target' (as an internal symbol path).")

        if operation == "create_code":
            if not code_content:
                raise ValueError("create_code operation requires 'code_content'.")
            result_message = editor.create_code(internal_target_symbol_path or "", target_type, code_content)
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
    if final_module_code != initial_module_code:
        logger.info("Code changes detected. Showing diff:")
        diff = difflib.unified_diff(
            initial_module_code.splitlines(keepends=True),
            final_module_code.splitlines(keepends=True),
            fromfile=f"a/{pathlib.Path(file_to_process).name}\n",
            tofile=f"b/{pathlib.Path(file_to_process).name}\n",
        )
        for line in diff:
            logger.info(line.strip())

    editor.save()
    return result_message


def _handle_diff_method_mode(target: str) -> str:
    """Handle the logic for diff_method mode."""
    try:
        try:
            module_and_class, method_name = target.rsplit(".", 1)
            module_name, class_name = module_and_class.rsplit(".", 1)
        except ValueError as e:
            raise ToolError(
                f"Invalid target format for diff_method: '{target}'. Expected 'pkg.module.Class.method'."
            ) from e

        module = importlib.import_module(module_name)
        subclass = getattr(module, class_name)
        if not inspect.isclass(subclass):
            raise ToolError(f"Target '{target}' does not point to a class.")

        subclass_method = getattr(subclass, method_name, None)
        if not subclass_method or not (inspect.isfunction(subclass_method) or inspect.ismethod(subclass_method)):
            raise ToolError(f"Method '{method_name}' not found in class '{class_name}'.")

        parent_method, parent_class = None, None
        for cls in inspect.getmro(subclass)[1:]:
            if hasattr(cls, method_name):
                parent_method_candidate = getattr(cls, method_name)
                if subclass_method is not parent_method_candidate:
                    parent_method = parent_method_candidate
                    for defining_cls in inspect.getmro(cls):
                        if method_name in defining_cls.__dict__:
                            parent_class = defining_cls
                            break
                    break

        if not parent_method:
            if any(method_name in bc.__dict__ for bc in inspect.getmro(subclass)[1:]):
                return f"Method '{class_name}.{method_name}' is not overridden from its parent class."
            return f"Method '{method_name}' is defined on '{class_name}' but not found in any parent class."

        subclass_source = textwrap.dedent(inspect.getsource(subclass_method))
        parent_source = textwrap.dedent(inspect.getsource(parent_method))

        assert parent_class is not None
        if subclass_source == parent_source:
            return f"Method '{class_name}.{method_name}' has identical implementation to '{parent_class.__name__}.{method_name}'."

        diff = difflib.unified_diff(
            parent_source.splitlines(keepends=True),
            subclass_source.splitlines(keepends=True),
            fromfile=f"a/{parent_class.__name__}.{method_name}",
            tofile=f"b/{class_name}.{method_name}",
        )
        return "".join(diff)

    except (ImportError, AttributeError, ValueError, TypeError) as e:
        raise ToolError(f"Error processing symbol '{target}': {e}", original_exception=e) from e


@tool_handler(log_level=logging.INFO)
def python_code_handler(  # noqa: PLR0913, PLR0912
    mode: str,
    target: str,
    path: str | None = None,
    list_detail_level: str = "names_only",
    operation: str | None = None,
    target_type: str | None = None,
    code_content: str | None = None,
    modification_script: str | None = None,
) -> str:
    r"""
    Handle Python code for listing, modifying, or diffing.

    This tool uses `libcst` for lossless AST manipulation. It can scan files or symbols
    to list definitions, perform modifications, or compare overridden methods.

    **Modes:**
    - `list`: List definitions in a file, directory, or from an importable symbol.
    - `modify`: Create, update, or delete code in a file using operations or a custom script.
    - `diff_method`: Compare an overridden method with its parent's implementation.

    **Parameters:**
    - `mode` (str): 'list', 'modify', or 'diff_method'.
    - `target` (str): The code target. A fully importable symbol (e.g., 'os.path.join')
      or a relative path within a file (e.g., 'MyClass.my_method').
    - `path` (str, optional): Path to a file or directory. Required for `modify` mode and
      `list` mode with relative targets.
    - `list_detail_level` (str, optional): 'names_only', 'with_docstring', 'full_definition'.
    - `operation` (str, optional): For 'modify' mode: 'create', 'update', 'delete'.
    - `target_type` (str, optional): Type of target: 'function', 'class', 'method', etc.
    - `code_content` (str, optional): New code for 'create' or 'update' operations.
    - `modification_script` (str, optional): Custom `libcst` script for complex modifications.

    **Raises:**
    - `ToolError`: For any errors during execution.
    """
    if mode not in ["list", "modify", "diff_method"]:
        raise ValueError(f"Invalid mode: '{mode}'. Must be 'list', 'modify', or 'diff_method'.")
    if not isinstance(target, str):
        raise TypeError("Parameter 'target' must be of type string.")
    if path is not None and not isinstance(path, str):
        raise TypeError("Parameter 'path' must be of type string or None.")

    file_to_process, internal_target_symbol_path = None, ""
    if path:
        file_to_process = path
        if not pathlib.Path(file_to_process).exists():
            raise FileNotFoundError(f"File or directory '{file_to_process}' does not exist.")
        internal_target_symbol_path = target
    elif mode != "diff_method":
        file_to_process, internal_target_symbol_path = resolve_symbol_path(target)
        logger.info(
            "Resolved target '%s' to file: %s, internal path: '%s'",
            target,
            file_to_process,
            internal_target_symbol_path,
        )
        if not file_to_process.lower().endswith(".py"):
            raise ValueError(f"Target '{target}' resolved to non-Python file '{file_to_process}'.")

    if mode == "list":
        if path and file_to_process:
            return _handle_list_mode_by_path(
                file_to_process, internal_target_symbol_path, target_type, list_detail_level
            )
        return _handle_list_mode_by_symbol(target, target_type, list_detail_level)

    if mode == "modify":
        if file_to_process is None:
            raise ValueError("The 'path' parameter is required for modify mode.")
        return _handle_modify_mode(
            file_to_process,
            internal_target_symbol_path,
            operation,
            target_type,
            textwrap.dedent(code_content) if code_content else code_content,
            modification_script,
        )
    if mode == "diff_method":
        if path:
            raise ValueError("The 'path' parameter is not supported for 'diff_method' mode.")
        return _handle_diff_method_mode(target)

    raise ValueError(f"Invalid mode: '{mode}'.")
