import asyncio
import copy
import pathlib
from typing import Annotated

import rich
import rich.markdown
import typer
from autogen_ext.models._utils.parse_r1_content import parse_r1_content
from prompt_toolkit import PromptSession
from rich.console import Console

import domarkx.ui.console
from domarkx.agent_manager import create_agent
from domarkx.document_parser import parse_document
from domarkx.utils.chat_doc_parser import Message, append_message


async def aexec_doc(doc: pathlib.Path, handle_one_toolcall: bool = False):
    console = Console(markup=False)
    parsed_doc = parse_document(doc)

    if parsed_doc.config.session_setup_code:
        session_setup_code = parsed_doc.config.session_setup_code
        console.print(rich.markdown.Markdown(f"```{session_setup_code.language}\n{session_setup_code.code}\n```"))
        local_vars = {}
        exec(session_setup_code.code, globals(), local_vars)

        client = local_vars["client"]

    console.print("".join(parsed_doc.raw_lines))

    chat_agent_state = parsed_doc.config.session_config

    system_message = parsed_doc.conversation[0].content

    messages = _process_initial_messages(parsed_doc)
    chat_agent_state["llm_context"]["messages"] = messages

    if system_message is None or len(system_message) == 0:
        system_message = "You are a helpful AI assistant. "

    chat_agent = await create_agent(client, system_message, chat_agent_state)

    # console.input("Press Enter to run stream, Ctrl+C to cancel.")

    while True:
        task_msg = None
        latest_msg = messages[-1] if len(messages) > 0 else None
        if len(messages) == 0 or (latest_msg.get("type", "") not in ["UserMessage"] and "content" in latest_msg):
            task_msg: str = await PromptSession().prompt_async(
                "task > ",
                multiline=True,
                bottom_toolbar="press Alt+Enter in order to accept the input. (Or Escape followed by Enter.)",
            )
            if latest_msg and latest_msg.get("type", "") in ["FunctionExecutionResultMessage"]:
                if len(task_msg.strip()) == 0:
                    task_msg = None

        response = await domarkx.ui.console.Console(chat_agent.run_stream(task=task_msg), output_stats=True, exit_after_one_toolcall=handle_one_toolcall)

        new_state = await chat_agent.save_state()

        _append_new_messages(doc, new_state, messages)

        messages = new_state["llm_context"]["messages"]

        if handle_one_toolcall:
            break
        user_input: str = await PromptSession().prompt_async("input r to continue > ")
        user_input = user_input.strip().lower()
        if len(user_input) != 0 and user_input != "r":
            break


def _append_new_messages(doc, new_state, messages):
    for message in new_state["llm_context"]["messages"][len(messages) :]:
        message: dict = copy.deepcopy(message)
        content = ""
        if "content" in message:
            if isinstance(message["content"], str):
                content = message.pop("content")
            elif isinstance(message["content"], list) and len(message["content"]) == 1:
                content = message["content"][0].pop("content", "")
        thought = message.pop("thought", "")
        if thought:
            thought = "\n".join("> " + line for line in f"""<think>{thought}</think>""".splitlines())
            content = f"""
{thought}

{content}"""
        with doc.open("a") as f:
            append_message(f, Message("assistant", content, message))


def _process_initial_messages(parsed_doc):
    messages = []
    for md_message in parsed_doc.conversation[1:]:
        message_dict = md_message.metadata
        thought, content = parse_r1_content(md_message.content)
        if "content" not in message_dict:
            message_dict["content"] = content
        elif isinstance(message_dict["content"], list) and len(message_dict["content"]) == 1:
            if "content" not in message_dict["content"][0] and "arguments" not in message_dict["content"][0]:
                message_dict["content"][0]["content"] = content
        if thought:
            message_dict["thought"] = "\n".join(line.removeprefix("> ") for line in thought.splitlines())
        messages.append(message_dict)
    return messages


def exec_doc(
    doc: Annotated[
        pathlib.Path,
        typer.Argument(exists=True, file_okay=True, dir_okay=False, writable=True, readable=True, resolve_path=True),
    ],
    handle_one_toolcall: bool = False,
):
    asyncio.run(aexec_doc(doc, handle_one_toolcall))


def register(main_app: typer.Typer):
    main_app.command()(exec_doc)
