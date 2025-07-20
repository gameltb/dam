import traceback


def execute_code_block(code: str, global_vars: dict = None, local_vars: dict = None):
    """
    Executes a block of code and prints a traceback if an exception occurs.

    Args:
        code (str): The code to execute.
        global_vars (dict, optional): A dictionary of global variables. Defaults to None.
        local_vars (dict, optional): A dictionary of local variables. Defaults to None.
    """
    if global_vars is None:
        global_vars = {}
    if local_vars is None:
        local_vars = {}

    try:
        exec(code, global_vars, local_vars)
    except Exception:
        traceback.print_exc()
        raise
