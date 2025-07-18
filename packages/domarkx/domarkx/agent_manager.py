import ast
from domarkx.agents.resume_funcall_assistant_agent import ResumeFunCallAssistantAgent
from domarkx.utils.chat_doc_parser import ParsedDocument


def _get_last_expression(code):
    try:
        tree = ast.parse(code)
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last_expr = tree.body.pop()
            return ast.unparse(last_expr), ast.unparse(tree)
        return None, code
    except SyntaxError:
        return None, code


async def create_agent(doc: ParsedDocument, system_message: str, chat_agent_state: dict):
    setup_script_blocks = doc.get_code_blocks(language="python", attrs="setup-script")
    if not setup_script_blocks:
        raise ValueError("No setup-script code block found in the document.")

    setup_script = setup_script_blocks[0].code

    # Extract client and tools from the setup script
    local_vars = {}
    global_vars = {}

    # Try to get the last expression to evaluate
    last_expr, remaining_code = _get_last_expression(setup_script)

    exec(remaining_code, global_vars, local_vars)

    if last_expr:
        result = eval(last_expr, global_vars, local_vars)
    else:
        result = None

    client = local_vars.get("client")
    tools = local_vars.get("tools")
    tool_executors = local_vars.get("tool_executors", [])

    if client is None:
        raise ValueError("The 'client' variable was not defined in the setup-script.")
    if tools is None:
        raise ValueError("The 'tools' variable was not defined in the setup-script.")

    # Start tool executors
    for executor in tool_executors:
        if hasattr(executor, "start"):
            executor.start()

    agent = ResumeFunCallAssistantAgent(
        "assistant",
        model_client=client,
        system_message=system_message,
        model_client_stream=True,
        tools=tools,
    )
    await agent.load_state(chat_agent_state)
    return agent, tool_executors
