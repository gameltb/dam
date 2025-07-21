import linecache
import traceback


def execute_code_block(code: str, global_vars: dict = None, local_vars: dict = None, filename="<setup-script>"):
    """
    Executes a block of code and prints a traceback if an exception occurs.

    Args:
        code (str): The code to execute.
        global_vars (dict, optional): A dictionary of global variables. Defaults to None.
        local_vars (dict, optional): A dictionary of local variables. Defaults to None.
        filename (str, optional): The filename to use in the traceback. Defaults to "<setup-script>".
    """
    if global_vars is None:
        global_vars = {}
    if local_vars is None:
        local_vars = {}

    try:
        # Add the code to the linecache
        linecache.cache[filename] = (len(code), None, [line + "\n" for line in code.splitlines()], filename)

        # Compile and execute the code
        compiled_code = compile(code, filename, "exec")
        exec(compiled_code, global_vars, local_vars)
    except Exception:
        traceback.print_exc()
        raise
    finally:
        # Clean up the linecache
        if filename in linecache.cache:
            del linecache.cache[filename]
