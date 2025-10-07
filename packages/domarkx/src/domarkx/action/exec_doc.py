import asyncio
import pathlib
import re
from datetime import datetime
from typing import Annotated, Any, Optional

import aiofiles
import typer
from prompt_toolkit import PromptSession

import domarkx.ui.console
from domarkx.autogen_session import AutoGenSession
from domarkx.config import settings
from domarkx.macro_expander import DocExpander
from domarkx.ui.console import PROMPT_TOOLKIT_IS_MULTILINE_CONDITION


async def aexec_doc(  # noqa: PLR0912, PLR0915
    doc: pathlib.Path,
    handle_one_toolcall: bool = False,
    allow_user_message_in_function_execution: bool = True,
    overwrite: bool = False,
) -> None:
    # Read the content from the document
    async with aiofiles.open(doc) as f:
        content = await f.read()

    # Expand macros
    expander = DocExpander(base_dir=str(doc.parent))
    expanded_doc = expander.expand(content)
    expanded_content = expanded_doc.to_markdown()

    sessions_dir = pathlib.Path(settings.project_path) / "sessions"
    sessions_dir.mkdir(exist_ok=True)

    doc_to_exec = doc
    project_path = pathlib.Path(settings.project_path)
    timestamp_format = "%Y%m%d_%H%M%S"
    # Check if the file is under the project path
    if project_path in doc.parents and sessions_dir not in doc.parents:
        # Create a temporary file in the sessions folder
        now = datetime.now().strftime(timestamp_format)
        temp_filename = f"{now}_{doc.stem}.md"
        doc_to_exec = sessions_dir / temp_filename

        async with aiofiles.open(doc_to_exec, "w") as f:
            await f.write(expanded_content)
    else:
        if not overwrite:
            # Check for timestamp with optional letters at the end of the stem
            match = re.search(r"(_\d{8}_\d{6})([A-Z]*)$", doc.stem)

            if match:
                # Timestamp found.
                # Try to create a new filename with 'A' appended.
                new_stem_with_a = f"{doc.stem}A"
                path_with_a = doc.with_name(f"{new_stem_with_a}.md")

                if path_with_a.exists():
                    # File with 'A' exists, so we create a new timestamped file.
                    # We need to find the base name by stripping the found suffix.
                    base_name = doc.stem[: match.start()]
                    now = datetime.now().strftime(timestamp_format)
                    new_filename = f"{base_name}_{now}.md"
                    doc_to_exec = doc.with_name(new_filename)
                else:
                    # File with 'A' does not exist, so we use it.
                    doc_to_exec = path_with_a
            else:
                # No timestamp found, add one.
                now = datetime.now().strftime(timestamp_format)
                new_filename = f"{doc.stem}_{now}.md"
                doc_to_exec = doc.with_name(new_filename)

        async with aiofiles.open(doc_to_exec, "w") as f:
            await f.write(expanded_content)

    session = AutoGenSession(doc_to_exec)
    await session.setup()

    while True:
        task_msg: str | None = None
        latest_msg = session.messages[-1] if len(session.messages) > 0 else None
        if (
            not allow_user_message_in_function_execution
            and latest_msg
            and isinstance(latest_msg.get("content", ""), list)
        ):
            # If the last message is a FunctionExecutionMessage, we don't want to prompt for user input.
            task_msg = None
        elif len(session.messages) == 0 or (
            latest_msg is not None and latest_msg.get("type", "") not in ["UserMessage"] and "content" in latest_msg
        ):
            task_msg = await PromptSession[Optional[str]]().prompt_async(
                "task > ",
                multiline=PROMPT_TOOLKIT_IS_MULTILINE_CONDITION,
                bottom_toolbar=lambda: "press Alt+Enter in order to accept the input. (Or Escape followed by Enter.)"
                if PROMPT_TOOLKIT_IS_MULTILINE_CONDITION()
                else None,
            )
            if (
                latest_msg
                and latest_msg.get("type", "") in ["FunctionExecutionResultMessage"]
                and task_msg is not None
                and not task_msg.strip()
            ):
                task_msg = None

        await domarkx.ui.console.Console(
            session.agent.run_stream(task=task_msg), output_stats=True, exit_after_one_toolcall=handle_one_toolcall
        )

        new_state = await session.agent.save_state()

        session.append_new_messages(dict(new_state))

        session.messages = new_state["llm_context"]["messages"]

        if handle_one_toolcall:
            break
        user_input = await PromptSession[Optional[str]]().prompt_async("input r to continue > ")
        if user_input is not None:
            user_input = user_input.strip().lower()
            if len(user_input) != 0 and user_input != "r":
                break


def exec_doc(
    doc: Annotated[
        pathlib.Path,
        typer.Argument(exists=True, file_okay=True, dir_okay=False, writable=True, readable=True, resolve_path=True),
    ],
    handle_one_toolcall: bool = False,
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite the original file in the sessions folder."),
) -> None:
    asyncio.run(aexec_doc(doc, handle_one_toolcall, overwrite=overwrite))


def register(main_app: typer.Typer, _: dict[str, Any]) -> None:
    main_app.command()(exec_doc)
