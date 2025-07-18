from domarkx.tools.tool_decorator import TOOL_REGISTRY, tool_handler


@tool_handler()
def list_available_tools():
    """
    Lists all available tool functions registered via tool_handler.
    Returns:
        List[dict]: Each dict contains 'name', 'doc', and 'signature' for a tool.
    """
    import inspect

    tool_infos = []
    for func in TOOL_REGISTRY:
        tool_infos.append(
            {
                "name": func.__name__,
                "doc": inspect.getdoc(func),
                "signature": str(inspect.signature(func)),
            }
        )
    return tool_infos
