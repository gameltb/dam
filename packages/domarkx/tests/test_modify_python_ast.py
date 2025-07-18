import textwrap

import pytest

from domarkx.tool_call.run_tool_code.tool import REGISTERED_TOOLS, execute_tool_call
from domarkx.tools import python_code_handler
from domarkx.tools.tool_decorator import ToolError


# Setup and Teardown for temporary test files
@pytest.fixture
def temp_py_file(tmp_path):
    """Creates a temporary Python file for testing."""
    file_path = tmp_path / "test_module.py"
    initial_content = textwrap.dedent("""
        # This is a comment
        import os

        class MyClass:
            def __init__(self):
                pass

            def existing_method(self, a, b):
                return a + b

        def existing_function(x):
            return x * 2

        VAR = 10
    """)
    with open(file_path, "w") as f:
        f.write(initial_content)
    return file_path


@pytest.fixture
def invalid_file(tmp_path):
    """Creates a temporary Python file for testing."""
    file_path = tmp_path / "invalid_file.txt"
    initial_content = textwrap.dedent("""
        =this is not python
    """)
    with open(file_path, "w") as f:
        f.write(initial_content)
    return file_path


def read_file_content(file_path):
    with open(file_path, "r") as f:
        return f.read()


# Helper function to call the tool via execute_tool_call
def call_modify_python_ast_tool(file_path=None, operation=None, modification_script=None, symbol=None, **kwargs):
    REGISTERED_TOOLS["modify_python_ast"] = python_code_handler.python_code_handler
    parameters = {"mode": "modify"}
    if file_path:
        parameters["path"] = str(file_path)
    if symbol is not None:  # Use `is not None` to distinguish between None and empty string
        parameters["target"] = symbol
    else:
        parameters["target"] = ""

    if operation:
        parameters["operation"] = operation
        parameters.update(kwargs)  # Add specific operation parameters
    elif modification_script:
        parameters["modification_script"] = modification_script
    else:
        raise ValueError("Either operation or modification_script must be provided.")

    tool_call_dict = {"tool_name": "modify_python_ast", "parameters": parameters}
    tool_name, result = execute_tool_call(tool_call_dict, handle_exception=False)
    # If the tool returns an "Error:" string, raise it as a ValueError
    if result.startswith("Error:"):
        raise ValueError(result)
    return result


# Test Cases for existing modification_script tests
def test_add_function_with_script(temp_py_file):
    script = textwrap.dedent("""
        class AddFunctionTransformer(cst.CSTTransformer):
            def leave_Module(self, original_node, updated_node):
                new_function_module = cst.parse_module("def new_function(a, b):\\n    return a - b\\n")
                new_function_node = new_function_module.body[0]

                new_body = list(updated_node.body)
                if new_body and not isinstance(new_body[-1], (cst.EmptyLine, cst.Comment)):
                    new_body.append(cst.EmptyLine())
                new_body.append(new_function_node)
                return updated_node.with_changes(body=tuple(new_body))

        tree = tree.visit(AddFunctionTransformer())
    """)
    call_modify_python_ast_tool(temp_py_file, modification_script=script)
    content = read_file_content(temp_py_file)
    assert "def new_function(a, b):" in content
    assert "return a - b" in content
    assert "existing_function" in content


def test_modify_function_body_with_script(temp_py_file):
    script = textwrap.dedent("""
        class ModifyFunctionBodyTransformer(cst.CSTTransformer):
            def leave_FunctionDef(self, original_node, updated_node):
                if original_node.name.value == "existing_function":
                    dummy_func_module = cst.parse_module("def _dummy():\\n    return x * x * x\\n")
                    new_body_block = dummy_func_module.body[0].body
                    return updated_node.with_changes(body=new_body_block)
                return updated_node

        tree = tree.visit(ModifyFunctionBodyTransformer())
    """)
    call_modify_python_ast_tool(temp_py_file, modification_script=script)
    content = read_file_content(temp_py_file)
    assert "return x * x * x" in content
    assert "return x * 2" not in content


def test_add_class_with_script(temp_py_file):
    script = textwrap.dedent("""
        class AddClassTransformer(cst.CSTTransformer):
            def leave_Module(self, original_node, updated_node):
                new_class_module = cst.parse_module("class NewClass:\\n    def __init__(self):\\n        pass\\n")
                new_class_node = new_class_module.body[0]

                new_body = list(updated_node.body)
                if new_body and not isinstance(new_body[-1], (cst.EmptyLine, cst.Comment)):
                    new_body.append(cst.EmptyLine())
                new_body.append(new_class_node)
                return updated_node.with_changes(body=tuple(new_body))

        tree = tree.visit(AddClassTransformer())
    """)
    call_modify_python_ast_tool(temp_py_file, modification_script=script)
    content = read_file_content(temp_py_file)
    assert "class NewClass:" in content
    assert "def __init__(self):" in content


def test_add_import_with_script(temp_py_file):
    script = textwrap.dedent("""
        class AddImportTransformer(cst.CSTTransformer):
            def leave_Module(self, original_node, updated_node):
                new_import_line = cst.parse_module("import sys").body[0].with_changes(
                    trailing_whitespace=cst.TrailingWhitespace(newline=cst.Newline())
                )

                new_body = list(updated_node.body)

                insert_idx = -1
                for i, node in enumerate(new_body):
                    if isinstance(node, cst.SimpleStatementLine) and \\
                       node.body and \\
                       isinstance(node.body[0], cst.Import) and \\
                       node.body[0].names[0].name.value == "os":
                        insert_idx = i + 1
                        break

                if insert_idx != -1:
                    new_body.insert(insert_idx, new_import_line)

                    if insert_idx + 1 < len(new_body) and \\
                       not isinstance(new_body[insert_idx + 1], cst.EmptyLine) and \\
                       (not isinstance(new_body[insert_idx + 1], cst.SimpleStatementLine) or \\
                        not isinstance(new_body[insert_idx + 1].body[0], (cst.Import, cst.ImportFrom))):
                        new_body.insert(insert_idx + 1, cst.EmptyLine())
                else:
                    first_actual_code_idx = 0
                    for i, node in enumerate(new_body):
                        if not isinstance(node, (cst.Comment, cst.EmptyLine)):
                            first_actual_code_idx = i
                            break
                    new_body.insert(first_actual_code_idx, new_import_line)
                    if first_actual_code_idx + 1 < len(new_body) and not isinstance(new_body[first_actual_code_idx + 1], cst.EmptyLine):
                        new_body.insert(first_actual_code_idx + 1, cst.EmptyLine())

                return updated_node.with_changes(body=tuple(new_body))

        tree = tree.visit(AddImportTransformer())
    """)
    call_modify_python_ast_tool(temp_py_file, modification_script=script)
    content = read_file_content(temp_py_file)
    assert "import sys" in content
    import_os_idx = content.find("import os")
    import_sys_idx = content.find("import sys")
    class_myclass_idx = content.find("class MyClass:")
    assert import_sys_idx > import_os_idx
    assert import_sys_idx < class_myclass_idx
    assert content.lstrip().startswith("# This is a comment")


def test_remove_function_with_script(temp_py_file):
    script = textwrap.dedent("""
        class RemoveFunctionTransformer(cst.CSTTransformer):
            def leave_FunctionDef(self, original_node, updated_node):
                if original_node.name.value == "existing_function":
                    return cst.RemovalSentinel.REMOVE
                return updated_node

        tree = tree.visit(RemoveFunctionTransformer())
    """)
    call_modify_python_ast_tool(temp_py_file, modification_script=script)
    content = read_file_content(temp_py_file)
    assert "def existing_function(x):" not in content
    assert "return x * 2" not in content
    assert "class MyClass:" in content


# --- New Test Cases for create_code, update_code, delete_code operations ---


def test_create_function(temp_py_file):
    code_content = textwrap.dedent("""
        def new_top_level_function(a, b):
            return a * b
    """)
    call_modify_python_ast_tool(
        file_path=temp_py_file,
        operation="create_code",
        symbol="",  # Target the module level directly by passing empty string
        target_type="function",
        code_content=code_content,
    )
    content = read_file_content(temp_py_file)
    assert "def new_top_level_function(a, b):" in content
    assert "return a * b" in content


def test_create_method_in_class(temp_py_file):
    code_content = textwrap.dedent("""
        def new_method(self, value):
            return self.existing_method(value, 1) * 2
    """)
    call_modify_python_ast_tool(
        file_path=temp_py_file,
        operation="create_code",
        symbol="MyClass",  # Parent symbol is the class (relative path within the file)
        target_type="method",
        code_content=code_content,
    )
    content = read_file_content(temp_py_file)
    assert "class MyClass:" in content
    assert "def new_method(self, value):" in content
    assert "return self.existing_method(value, 1) * 2" in content


def test_create_class(temp_py_file):
    code_content = textwrap.dedent("""
        class AnotherClass:
            def greet(self):
                return "Hello from AnotherClass"
    """)
    call_modify_python_ast_tool(
        file_path=temp_py_file,
        operation="create_code",
        symbol="",  # Adding to module level
        target_type="class",
        code_content=code_content,
    )
    content = read_file_content(temp_py_file)
    assert "class AnotherClass:" in content
    assert "def greet(self):" in content
    assert 'return "Hello from AnotherClass"' in content


def test_create_assignment(temp_py_file):
    code_content = "NEW_CONSTANT = 'hello'"
    call_modify_python_ast_tool(
        file_path=temp_py_file,
        operation="create_code",
        symbol="",  # Adding to module level
        target_type="assignment",
        code_content=code_content,
    )
    content = read_file_content(temp_py_file)
    assert "NEW_CONSTANT = 'hello'" in content


def test_update_function_body(temp_py_file):
    new_code = textwrap.dedent("""
        def existing_function(x, y):
            # Updated function body
            return x * y + 10
    """)
    call_modify_python_ast_tool(
        file_path=temp_py_file,
        operation="update_code",
        symbol="existing_function",  # Relative path within the file
        target_type="function",
        code_content=new_code,
    )
    content = read_file_content(temp_py_file)
    assert "def existing_function(x, y):" in content
    assert "# Updated function body" in content
    assert "return x * y + 10" in content
    assert "return x * 2" not in content  # Ensure old content is gone


def test_update_method_body(temp_py_file):
    new_code = textwrap.dedent("""
        def existing_method(self, a, d, c):
            # Updated method body
            return a + d + c + 1
    """)
    call_modify_python_ast_tool(
        file_path=temp_py_file,
        operation="update_code",
        symbol="MyClass.existing_method",  # Relative path within the file
        target_type="method",
        code_content=new_code,
    )
    content = read_file_content(temp_py_file)
    assert "def existing_method(self, a, d, c):" in content
    assert "# Updated method body" in content
    assert "return a + d + c + 1" in content
    assert "return a + b" not in content  # Ensure old content is gone


def test_update_assignment(temp_py_file):
    new_value = "VAR = 200"
    call_modify_python_ast_tool(
        file_path=temp_py_file,
        operation="update_code",
        symbol="VAR",  # Relative path within the file
        target_type="assignment",
        code_content=new_value,
    )
    content = read_file_content(temp_py_file)
    assert "VAR = 200" in content
    assert "VAR = 10" not in content


def test_delete_function(temp_py_file):
    call_modify_python_ast_tool(
        file_path=temp_py_file,
        operation="delete_code",
        symbol="existing_function",  # Relative path within the file
        target_type="function",
    )
    content = read_file_content(temp_py_file)
    assert "def existing_function(x):" not in content
    assert "return x * 2" not in content


def test_delete_method(temp_py_file):
    call_modify_python_ast_tool(
        file_path=temp_py_file,
        operation="delete_code",
        symbol="MyClass.existing_method",  # Relative path within the file
        target_type="method",
    )
    content = read_file_content(temp_py_file)
    assert "def existing_method(self, a, b):" not in content
    assert "return a + b" not in content
    assert "class MyClass:" in content  # Class itself should remain


def test_delete_class(temp_py_file):
    call_modify_python_ast_tool(
        file_path=temp_py_file,
        operation="delete_code",
        symbol="MyClass",  # Relative path within the file
        target_type="class",
    )
    content = read_file_content(temp_py_file)
    assert "class MyClass:" not in content
    assert "def __init__(self):" not in content
    assert "def existing_method(self, a, b):" not in content


def test_delete_assignment(temp_py_file):
    call_modify_python_ast_tool(
        file_path=temp_py_file,
        operation="delete_code",
        symbol="VAR",  # Relative path within the file
        target_type="assignment",
    )
    content = read_file_content(temp_py_file)
    assert "VAR = 10" not in content


def test_create_code_error_invalid_type(temp_py_file):
    code_content = "def invalid_type_func(): pass"
    with pytest.raises(ToolError) as excinfo:
        call_modify_python_ast_tool(
            file_path=temp_py_file,
            operation="create_code",
            symbol="",  # module level
            target_type="invalid_type",
            code_content=code_content,
        )
    assert isinstance(excinfo.value.original_exception, ValueError)


def test_update_code_error_not_found(temp_py_file):
    new_code = "def non_existent_func(): pass"
    with pytest.raises(ToolError, match=r"Operation failed: Target node not found"):
        call_modify_python_ast_tool(
            file_path=temp_py_file,
            operation="update_code",
            symbol="non_existent_function",  # Relative path within the file
            target_type="function",
            code_content=new_code,
        )
    # Check that original file content remains unchanged
    content = read_file_content(temp_py_file)
    assert "def existing_function(x):" in content


def test_delete_code_error_not_found(temp_py_file):
    with pytest.raises(ToolError, match=r"Operation failed: Target node not found"):
        call_modify_python_ast_tool(
            file_path=temp_py_file,
            operation="delete_code",
            symbol="NonExistentClass",  # Relative path within the file
            target_type="class",
        )
    # Check that original file content remains unchanged
    content = read_file_content(temp_py_file)
    assert "class MyClass:" in content


def test_create_code_error_parent_not_found(temp_py_file):
    code_content = "def new_func_in_non_existent(): pass"
    with pytest.raises(ToolError, match=r"Operation failed: Target node not found"):
        call_modify_python_ast_tool(
            file_path=temp_py_file,
            operation="create_code",
            symbol="NonExistentClass",  # Parent does not exist (relative path)
            target_type="function",
            code_content=code_content,
        )
    content = read_file_content(temp_py_file)
    assert "def new_func_in_non_existent():" not in content  # Ensure no creation happened


# Generic error tests (kept as they are still relevant)
def test_invalid_python_file_path(invalid_file):
    with pytest.raises(ToolError) as excinfo:
        call_modify_python_ast_tool(
            file_path="non_existent_file.py",
            operation="create_code",
            symbol="",
            target_type="function",
            code_content="def func(): pass",
        )
    assert isinstance(excinfo.value.original_exception, FileNotFoundError)

    with pytest.raises(ToolError) as excinfo:
        call_modify_python_ast_tool(
            file_path=invalid_file,
            operation="create_code",
            symbol="",
            target_type="function",
            code_content="def func(): pass",
        )
    assert isinstance(excinfo.value.original_exception, ValueError)


def test_syntax_error_in_script(temp_py_file):
    script = "this is not python code"
    with pytest.raises(ToolError) as excinfo:
        call_modify_python_ast_tool(temp_py_file, modification_script=script)
    assert isinstance(excinfo.value.original_exception, ToolError)


def test_invalid_cst_manipulation_in_script(temp_py_file):
    script = textwrap.dedent("""
        class BadTransformer(cst.CSTTransformer):
            def leave_FunctionDef(self, original_node, updated_node):
                if original_node.name.value == "existing_function":
                    return "not a cst node"
                return updated_node

        tree = tree.visit(BadTransformer())
    """)
    with pytest.raises(ToolError) as excinfo:
        call_modify_python_ast_tool(temp_py_file, modification_script=script)
    assert isinstance(excinfo.value.original_exception, ToolError)


def test_operation_missing_target_type(temp_py_file):
    with pytest.raises(ToolError) as excinfo:
        call_modify_python_ast_tool(
            file_path=temp_py_file, operation="create_code", symbol="", code_content="def func(): pass"
        )
    assert isinstance(excinfo.value.original_exception, ValueError)


def test_operation_missing_code_content(temp_py_file):
    with pytest.raises(ToolError) as excinfo:
        call_modify_python_ast_tool(file_path=temp_py_file, operation="create_code", symbol="", target_type="function")
    assert isinstance(excinfo.value.original_exception, ValueError)


def test_unknown_operation_name(temp_py_file):
    with pytest.raises(ToolError) as excinfo:
        call_modify_python_ast_tool(
            file_path=temp_py_file,
            operation="non_existent_op",
            symbol="",
            target_type="function",
            code_content="def func(): pass",
        )
    assert isinstance(excinfo.value.original_exception, ValueError)
