import sys
import os
import pytest
import re
import textwrap

from domarkx.tools.python_code_handler import _resolve_symbol_path, python_code_handler
from domarkx.tools.tool_decorator import ToolError
import libcst as cst

# Original contents of example_module.py for resetting
ORIGINAL_EXAMPLE_MODULE_CONTENT = '''
"""
This is a docstring for example_module.
"""

MODULE_CONSTANT = 100

class MyClass:
    """
    This is MyClass.
    It has a constructor and a method.
    """
    def __init__(self, value):
        """
        Constructor for MyClass.
        """
        self.value = value

    def my_method(self, x):
        """
        A method in MyClass.
        """
        return self.value + x

def top_level_function(a, b):
    """A top-level function."""
    return a + b

# Another constant
ANOTHER_CONSTANT = "hello"
'''


# --- Test _resolve_symbol_path helper function ---
def test_resolve_symbol_path_top_level_module(tmp_path):
    # Create a temporary module for testing symbol resolution
    example_module_path = tmp_path / "example_module.py"
    example_module_path.write_text(ORIGINAL_EXAMPLE_MODULE_CONTENT)

    old_sys_path = sys.path[:]
    sys.path.insert(0, str(tmp_path))
    try:
        file_path, internal_path = _resolve_symbol_path("example_module")
        assert file_path == str(example_module_path)
        assert internal_path == ""
    finally:
        sys.path = old_sys_path


def test_resolve_symbol_path_function_in_module(tmp_path):
    example_module_path = tmp_path / "example_module.py"
    example_module_path.write_text(ORIGINAL_EXAMPLE_MODULE_CONTENT)

    old_sys_path = sys.path[:]
    sys.path.insert(0, str(tmp_path))
    try:
        file_path, internal_path = _resolve_symbol_path("example_module.top_level_function")
        assert file_path == str(example_module_path)
        assert internal_path == "top_level_function"
    finally:
        sys.path = old_sys_path


def test_resolve_symbol_path_method_in_class(tmp_path):
    example_module_path = tmp_path / "example_module.py"
    example_module_path.write_text(ORIGINAL_EXAMPLE_MODULE_CONTENT)

    old_sys_path = sys.path[:]
    sys.path.insert(0, str(tmp_path))
    try:
        file_path, internal_path = _resolve_symbol_path("example_module.MyClass.my_method")
        assert file_path == str(example_module_path)
        assert internal_path == "MyClass.my_method"
    finally:
        sys.path = old_sys_path


def test_resolve_symbol_path_non_existent_symbol(tmp_path):
    old_sys_path = sys.path[:]
    sys.path.insert(0, str(tmp_path))
    try:
        with pytest.raises(ToolError) as excinfo:
            _resolve_symbol_path("non_existent_module.func")
        assert "Could not resolve file path for symbol 'non_existent_module.func'" in str(excinfo.value)
    finally:
        sys.path = old_sys_path


# --- Test List Mode ---
def test_list_mode_path_all_names(tmp_path):
    example_module_path = tmp_path / "example_module.py"
    example_module_path.write_text(ORIGINAL_EXAMPLE_MODULE_CONTENT)

    result = python_code_handler(mode="list", path=str(example_module_path), target="", list_detail_level="names_only")
    assert "MODULE_CONSTANT" in result
    assert "MyClass" in result
    assert "__init__" not in result  # Methods are not listed by default for 'names_only' when target is empty
    assert "my_method" not in result
    assert "top_level_function" in result
    assert "ANOTHER_CONSTANT" in result


def test_list_mode_path_specific_function_full_definition(tmp_path):
    example_module_path = tmp_path / "example_module.py"
    example_module_path.write_text(ORIGINAL_EXAMPLE_MODULE_CONTENT)

    result = python_code_handler(mode="list", path=str(example_module_path), target="top_level_function", list_detail_level="full_definition")
    assert "function: top_level_function" in result
    assert "A top-level function." in result
    assert "return a + b" in result
    assert "Source Code" in result
    assert "MODULE_CONSTANT" not in result  # Should not include other definitions


def test_list_mode_path_specific_class_with_docstring(tmp_path):
    example_module_path = tmp_path / "example_module.py"
    example_module_path.write_text(ORIGINAL_EXAMPLE_MODULE_CONTENT)

    result = python_code_handler(mode="list", path=str(example_module_path), target="MyClass", list_detail_level="with_docstring")
    assert "class: MyClass" in result
    assert "This is MyClass." in result
    assert "It has a constructor and a method." in result
    assert "Docstring" in result
    assert "def __init__" not in result  # Should not include full code
    assert "Source Code" not in result


def test_list_mode_path_specific_method_full_definition(tmp_path):
    example_module_path = tmp_path / "example_module.py"
    example_module_path.write_text(ORIGINAL_EXAMPLE_MODULE_CONTENT)

    result = python_code_handler(mode="list", path=str(example_module_path), target="MyClass.my_method", list_detail_level="full_definition")
    assert "method: my_method" in result  # LibCST treats methods as FunctionDef
    assert "A method in MyClass." in result
    assert "return self.value + x" in result
    assert "Source Code" in result


def test_list_mode_path_non_existent_target(tmp_path):
    example_module_path = tmp_path / "example_module.py"
    example_module_path.write_text(ORIGINAL_EXAMPLE_MODULE_CONTENT)

    result = python_code_handler(mode="list", path=str(example_module_path), target="non_existent_func", list_detail_level="names_only")
    assert "Symbol 'non_existent_func' not found in file" in result


def test_list_mode_path_directory_scan(tmp_path):
    # Create example_module.py
    example_module_path = tmp_path / "example_module.py"
    example_module_path.write_text(ORIGINAL_EXAMPLE_MODULE_CONTENT)

    # Create another_module.py
    another_module_path = tmp_path / "another_module.py"
    another_module_path.write_text("def another_function(): pass\n")

    # Create empty_module.py
    empty_module_path = tmp_path / "empty_module.py"
    empty_module_path.write_text("# This is an empty module for testing purposes.\n")

    # Create syntax_error_module.py
    syntax_error_module_path = tmp_path / "syntax_error_module.py"
    syntax_error_module_path.write_text("def incomplete_function(\n    return 1\n")

    result = python_code_handler(
        mode="list",
        path=str(tmp_path),
        target="",  # Empty target for listing all in directory
        list_detail_level="names_only",
    )

    assert "example_module.py" in result
    assert "MODULE_CONSTANT" in result
    assert "MyClass" in result
    assert "top_level_function" in result
    assert "another_module.py" in result
    assert "another_function" in result
    assert "empty_module.py" in result
    assert "syntax_error_module.py" in result


def test_list_mode_symbol_function_with_docstring(tmp_path):
    example_module_path = tmp_path / "example_module.py"
    example_module_path.write_text(ORIGINAL_EXAMPLE_MODULE_CONTENT)

    old_sys_path = sys.path[:]
    sys.path.insert(0, str(tmp_path))
    try:
        result = python_code_handler(mode="list", target="example_module.top_level_function", list_detail_level="with_docstring")
        assert "--- Symbol: example_module.top_level_function ---" in result
        assert "Docstring" in result
        assert "A top-level function." in result
        assert "Source Code" not in result  # Only docstring
    finally:
        sys.path = old_sys_path


def test_list_mode_symbol_class_full_definition(tmp_path):
    example_module_path = tmp_path / "example_module.py"
    example_module_path.write_text(ORIGINAL_EXAMPLE_MODULE_CONTENT)

    old_sys_path = sys.path[:]
    sys.path.insert(0, str(tmp_path))
    try:
        result = python_code_handler(mode="list", target="example_module.MyClass", list_detail_level="full_definition")
        assert "--- Symbol: example_module.MyClass ---" in result
        assert "This is MyClass." in result
        assert "Source Code" in result
        assert "def __init__" in result
        assert "def my_method" in result
    finally:
        sys.path = old_sys_path


def test_list_mode_symbol_non_existent():
    with pytest.raises(ToolError) as excinfo:
        python_code_handler(mode="list", target="non_existent_module.SomeClass")
    assert "Could not resolve file path for symbol 'non_existent_module.SomeClass'" in str(excinfo.value.original_exception)


# --- Test Modify Mode ---
def test_modify_mode_create_function(tmp_path):
    example_module_path = tmp_path / "example_module.py"
    example_module_path.write_text(ORIGINAL_EXAMPLE_MODULE_CONTENT)

    initial_content = example_module_path.read_text()
    python_code_handler(
        mode="modify",
        path=str(example_module_path),
        target="",  # Create at module root
        operation="create_code",
        target_type="function",
        code_content='''def new_function(x):
    """A new function."""
    return x * 2''',
    )
    modified_content = example_module_path.read_text()
    assert "def new_function(x):" in modified_content
    assert '"""A new function."""' in modified_content
    assert "return x * 2" in modified_content
    # Ensure original content is still there
    assert "top_level_function" in modified_content


def test_modify_mode_update_function(tmp_path):
    example_module_path = tmp_path / "example_module.py"
    example_module_path.write_text(ORIGINAL_EXAMPLE_MODULE_CONTENT)

    new_func_content = '''def top_level_function(a, b):
    """
    Updated top-level function.
    Now it multiplies.
    """
    return a * b'''

    python_code_handler(
        mode="modify",
        path=str(example_module_path),
        target="top_level_function",
        operation="update_code",
        target_type="function",
        code_content=new_func_content,
    )
    modified_content = example_module_path.read_text()
    assert "return a * b" in modified_content
    assert "Updated top-level function." in modified_content
    assert "A top-level function." not in modified_content


def test_modify_mode_delete_function(tmp_path):
    example_module_path = tmp_path / "example_module.py"
    example_module_path.write_text(ORIGINAL_EXAMPLE_MODULE_CONTENT)

    python_code_handler(
        mode="modify",
        path=str(example_module_path),
        target="top_level_function",
        operation="delete_code",
        target_type="function",
    )
    modified_content = example_module_path.read_text()
    assert "def top_level_function(a, b):" not in modified_content
    assert "A top-level function." not in modified_content


def test_modify_mode_create_method_in_class(tmp_path):
    example_module_path = tmp_path / "example_module.py"
    example_module_path.write_text(ORIGINAL_EXAMPLE_MODULE_CONTENT)

    python_code_handler(
        mode="modify",
        path=str(example_module_path),
        target="MyClass",  # Target is the parent class
        operation="create_code",
        target_type="method",
        code_content='''    def new_method(self):
        return "new"''',
    )
    modified_content = example_module_path.read_text()
    assert "class MyClass:" in modified_content
    assert "def new_method(self):" in modified_content
    assert 'return "new"' in modified_content


def test_modify_mode_update_assignment(tmp_path):
    example_module_path = tmp_path / "example_module.py"
    example_module_path.write_text(ORIGINAL_EXAMPLE_MODULE_CONTENT)

    python_code_handler(
        mode="modify",
        path=str(example_module_path),
        target="MODULE_CONSTANT",
        operation="update_code",
        target_type="assignment",
        code_content="MODULE_CONSTANT = 200",
    )
    modified_content = example_module_path.read_text()
    assert "MODULE_CONSTANT = 200" in modified_content
    assert "MODULE_CONSTANT = 100" not in modified_content


def test_modify_mode_delete_assignment(tmp_path):
    example_module_path = tmp_path / "example_module.py"
    example_module_path.write_text(ORIGINAL_EXAMPLE_MODULE_CONTENT)

    python_code_handler(
        mode="modify",
        path=str(example_module_path),
        target="ANOTHER_CONSTANT",
        operation="delete_code",
        target_type="assignment",
    )
    modified_content = example_module_path.read_text()
    assert 'ANOTHER_CONSTANT = "hello"' not in modified_content


def test_modify_mode_custom_script_update_docstring(tmp_path):
    example_module_path = tmp_path / "example_module.py"
    example_module_path.write_text(ORIGINAL_EXAMPLE_MODULE_CONTENT)

    # This test will mimic the example script from the docstring
    script = '''
import libcst as cst
import libcst.matchers as m
import logging

target_func_name = 'top_level_function'
new_doc = 'This is an updated docstring via custom script.'

class UpdateDocstringTransformer(cst.CSTTransformer):
    def __init__(self, func_name, new_docstring_content):
        self.func_name = func_name
        self.new_docstring_content = new_docstring_content
        self.updated = False
        self.found_target = False

    def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.FunctionDef:
        if original_node.name.value == self.func_name:
            self.found_target = True
            logging.info(f"Found function '{self.func_name}', attempting to update docstring.")
            
            new_body = list(updated_node.body.body)
            
            new_docstring_node = cst.Expr(
                value=cst.SimpleString(f'"""{self.new_docstring_content}"""')
            )

            # Check if the first statement is a docstring and replace it
            if new_body and isinstance(new_body[0], cst.Expr) and isinstance(new_body[0].value, cst.SimpleString):
                    # If the original docstring is multi-line, it might be parsed differently
                    # and might have leading/trailing newlines. We'll replace it regardless.
                new_body[0] = new_docstring_node
            else:
                    # No docstring found, so insert the new one at the beginning of the function body.
                new_body.insert(0, new_docstring_node)

                # Remove the old docstring if it exists
                if len(new_body) > 1 and isinstance(new_body[1], cst.Expr) and isinstance(new_body[1].value, cst.SimpleString):
                    del new_body[1]

            self.updated = True
            # Here's the fix: we need to update the body of the 'updated_node'
            # and then return a new FunctionDef with this updated body.
            new_body_suite = updated_node.body.with_changes(body=tuple(new_body))
            return updated_node.with_changes(body=new_body_suite)
        return updated_node

transformer = UpdateDocstringTransformer(target_func_name, new_doc)
tree = tree.visit(transformer)
if not transformer.updated:
    logging.warning(f"Function '{target_func_name}' not found or docstring not updated by script.")
'''
    python_code_handler(
        mode="modify",
        path=str(example_module_path),
        target="top_level_function",  # target here is mostly for logging/context within the tool, script uses its own
        modification_script=script,
    )
    modified_content = example_module_path.read_text()
    assert "This is an updated docstring via custom script." in modified_content
    assert "A top-level function." not in modified_content


# --- Error Handling Tests ---
def test_invalid_mode():
    with pytest.raises(ToolError) as excinfo:
        python_code_handler(mode="invalid_mode", target="anything")
    assert "Invalid mode: 'invalid_mode'." in str(excinfo.value.original_exception)


def test_missing_target():
    with pytest.raises(ToolError) as excinfo:
        python_code_handler(mode="list", target=None)
    assert "Parameter 'target' must be of type string." in str(excinfo.value.original_exception)


def test_list_path_non_existent_file(tmp_path):
    non_existent_path = tmp_path / "non_existent.py"
    with pytest.raises(ToolError) as excinfo:
        python_code_handler(mode="list", path=str(non_existent_path), target="")
    assert "File or directory" in str(excinfo.value.original_exception)
    assert "does not exist" in str(excinfo.value.original_exception)


def test_list_path_syntax_error_file(tmp_path):
    syntax_error_module_path = tmp_path / "syntax_error_module.py"
    syntax_error_module_path.write_text("def incomplete_function(\n    return 1\n")

    # The decorator now wraps CST exceptions as ToolError
    result = python_code_handler(mode="list", path=str(syntax_error_module_path), target="")
    assert "Syntax error in file" in result


def test_modify_non_existent_target(tmp_path):
    example_module_path = tmp_path / "example_module.py"
    example_module_path.write_text(ORIGINAL_EXAMPLE_MODULE_CONTENT)

    with pytest.raises(ToolError, match="Operation failed: Target node not found."):
        python_code_handler(
            mode="modify",
            path=str(example_module_path),
            target="non_existent_func",
            operation="delete_code",
            target_type="function",
        )


def test_modify_invalid_target_type(tmp_path):
    example_module_path = tmp_path / "example_module.py"
    example_module_path.write_text(ORIGINAL_EXAMPLE_MODULE_CONTENT)

    with pytest.raises(ToolError, match="Invalid or missing 'target_type'. Must be one of"):
        python_code_handler(
            mode="modify",
            path=str(example_module_path),
            target="top_level_function",
            operation="update_code",
            target_type="invalid_type",
            code_content="def dummy(): pass",
        )


def test_modify_missing_code_content_for_create(tmp_path):
    example_module_path = tmp_path / "example_module.py"
    example_module_path.write_text(ORIGINAL_EXAMPLE_MODULE_CONTENT)

    with pytest.raises(ToolError, match="create_code operation requires 'code_content'."):
        python_code_handler(
            mode="modify",
            path=str(example_module_path),
            target="",
            operation="create_code",
            target_type="function",
            code_content=None,
        )


def test_modify_missing_code_content_for_update(tmp_path):
    example_module_path = tmp_path / "example_module.py"
    example_module_path.write_text(ORIGINAL_EXAMPLE_MODULE_CONTENT)

    with pytest.raises(ToolError, match="update_code operation requires 'code_content'."):
        python_code_handler(
            mode="modify",
            path=str(example_module_path),
            target="top_level_function",
            operation="update_code",
            target_type="function",
            code_content=None,
        )


# Test when path is provided but it's a directory and no target_type is specified
def test_list_mode_path_directory_no_target_type(tmp_path):
    # Create example_module.py
    example_module_path = tmp_path / "example_module.py"
    example_module_path.write_text(ORIGINAL_EXAMPLE_MODULE_CONTENT)

    # Create another_module.py
    another_module_path = tmp_path / "another_module.py"
    another_module_path.write_text("def another_function(): pass\n")

    # This should list all definitions found in all .py files in the directory
    result = python_code_handler(
        mode="list",
        path=str(tmp_path),
        target="",  # List all in directory
        list_detail_level="names_only",
    )
    assert "MODULE_CONSTANT" in result
    assert "MyClass" in result
    assert "top_level_function" in result
    assert "another_function" in result  # From another_module.py
