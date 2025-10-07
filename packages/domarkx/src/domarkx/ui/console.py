"""A console rendering utility for AutoGen streams."""

import asyncio
import os
import sys
import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from inspect import iscoroutinefunction
from typing import TypeVar, cast

from autogen_agentchat.agents import UserProxyAgent
from autogen_agentchat.base import Response, TaskResult
from autogen_agentchat.messages import (
    BaseAgentEvent,
    BaseChatMessage,
    ModelClientStreamingChunkEvent,
    MultiModalMessage,
    ToolCallRequestEvent,
    ToolCallSummaryMessage,
    UserInputRequestedEvent,
)
from autogen_core import CancellationToken
from autogen_core.models import RequestUsage
from prompt_toolkit import PromptSession
from prompt_toolkit.application import get_app
from prompt_toolkit.filters import Condition

PROMPT_TOOLKIT_IS_MULTILINE_CONDITION = Condition(lambda: bool(get_app().current_buffer.text))


def _is_running_in_iterm() -> bool:
    return os.getenv("TERM_PROGRAM") == "iTerm.app"


def _is_output_a_tty() -> bool:
    return sys.stdout.isatty()


SyncInputFunc = Callable[[str], str]
AsyncInputFunc = Callable[[str, CancellationToken | None], Awaitable[str]]
InputFuncType = SyncInputFunc | AsyncInputFunc

T = TypeVar("T", bound=TaskResult | Response)


class UserInputManager:
    """A manager for handling user input events."""

    def __init__(self, callback: InputFuncType):
        """Initialize the UserInputManager."""
        self.input_events: dict[str, asyncio.Event] = {}
        self.callback = callback

    def get_wrapped_callback(self) -> AsyncInputFunc:
        """Get a wrapped callback function that waits for an event before calling the original callback."""

        async def user_input_func_wrapper(prompt: str, cancellation_token: CancellationToken | None) -> str:
            # Lookup the event for the prompt, if it exists wait for it.
            # If it doesn't exist, create it and store it.
            # Get request ID:
            request_id = UserProxyAgent.InputRequestContext.request_id()
            if request_id in self.input_events:
                event = self.input_events[request_id]
            else:
                event = asyncio.Event()
                self.input_events[request_id] = event

            await event.wait()

            del self.input_events[request_id]

            if iscoroutinefunction(self.callback):
                # Cast to AsyncInputFunc for proper typing
                async_func = cast(AsyncInputFunc, self.callback)
                return await async_func(prompt, cancellation_token)
            # Cast to SyncInputFunc for proper typing
            sync_func = cast(SyncInputFunc, self.callback)
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, sync_func, prompt)

        return user_input_func_wrapper

    def notify_event_received(self, request_id: str) -> None:
        """Notify that an event has been received."""
        if request_id in self.input_events:
            self.input_events[request_id].set()
        else:
            event = asyncio.Event()
            self.input_events[request_id] = event


def aprint(output: str, end: str = "\n", flush: bool = False) -> Awaitable[None]:
    """Asynchronously print to the console."""
    return asyncio.to_thread(print, output, end=end, flush=flush)


def ainput(prompt: str) -> Awaitable[str]:
    """Asynchronously get input from the console."""
    return asyncio.to_thread(input, prompt)


async def console_render[T: TaskResult | Response](  # noqa: PLR0912, PLR0915
    stream: AsyncGenerator[BaseAgentEvent | BaseChatMessage | T, None],
    *,
    no_inline_images: bool = False,
    output_stats: bool = False,
    user_input_manager: UserInputManager | None = None,
    exit_after_one_toolcall: bool = False,
) -> T | None:
    """
    Consume the message stream and render the messages to the console.

    This function consumes messages from a stream and renders them to the console,
    handling different message types and providing options for outputting stats and images.

    .. note::

        `output_stats` is experimental and the stats may not be accurate.
        It will be improved in future releases.

    Args:
        stream: Message stream to render.
        no_inline_images: If terminal is iTerm2 will render images inline. Use this to disable this behavior.
        output_stats: (Experimental) If True, will output a summary of the messages and inline token usage info.
        user_input_manager: A manager for handling user input events.
        exit_after_one_toolcall: If True, will exit after the first tool call.

    Returns:
        The last processed TaskResult or Response.

    """
    render_image_iterm = _is_running_in_iterm() and _is_output_a_tty() and not no_inline_images
    start_time = time.time()
    total_usage = RequestUsage(prompt_tokens=0, completion_tokens=0)

    last_processed: T | None = None

    streaming_chunks: list[str] = []

    async for message in stream:
        if isinstance(message, TaskResult):
            duration = time.time() - start_time
            if output_stats:
                output = (
                    f"{'-' * 10} Summary {'-' * 10}\n"
                    f"Number of messages: {len(message.messages)}\n"
                    f"Finish reason: {message.stop_reason}\n"
                    f"Total prompt tokens: {total_usage.prompt_tokens}\n"
                    f"Total completion tokens: {total_usage.completion_tokens}\n"
                    f"Duration: {duration:.2f} seconds\n"
                )
                await aprint(output, end="", flush=True)

            # mypy ignore
            last_processed = message  # type: ignore

        elif isinstance(message, Response):
            duration = time.time() - start_time

            # Print final response.
            if isinstance(message.chat_message, MultiModalMessage):
                final_content = message.chat_message.to_text(iterm=render_image_iterm)
            else:
                final_content = message.chat_message.to_text()
            output = f"{'-' * 10} {message.chat_message.source} {'-' * 10}\n{final_content}\n"
            if message.chat_message.models_usage:
                if output_stats:
                    output += f"[Prompt tokens: {message.chat_message.models_usage.prompt_tokens}, Completion tokens: {message.chat_message.models_usage.completion_tokens}]\n"
                total_usage.completion_tokens += message.chat_message.models_usage.completion_tokens
                total_usage.prompt_tokens += message.chat_message.models_usage.prompt_tokens
            await aprint(output, end="", flush=True)

            # Print summary.
            if output_stats:
                num_inner_messages = len(message.inner_messages) if message.inner_messages is not None else 0
                output = (
                    f"{'-' * 10} Summary {'-' * 10}\n"
                    f"Number of inner messages: {num_inner_messages}\n"
                    f"Total prompt tokens: {total_usage.prompt_tokens}\n"
                    f"Total completion tokens: {total_usage.completion_tokens}\n"
                    f"Duration: {duration:.2f} seconds\n"
                )
                await aprint(output, end="", flush=True)

            # mypy ignore
            last_processed = message  # type: ignore
        # We don't want to print UserInputRequestedEvent messages, we just use them to signal the user input event.
        elif isinstance(message, UserInputRequestedEvent):
            if user_input_manager is not None:
                user_input_manager.notify_event_received(message.request_id)
        else:
            # Cast required for mypy to be happy
            message = cast(BaseAgentEvent | BaseChatMessage, message)  # type: ignore
            if not streaming_chunks:
                # Print message sender.
                await aprint(
                    f"{'-' * 10} {message.__class__.__name__} ({message.source}) {'-' * 10}", end="\n", flush=True
                )
            if isinstance(message, ModelClientStreamingChunkEvent):
                await aprint(message.to_text(), end="", flush=True)
                streaming_chunks.append(message.content)
            else:
                if streaming_chunks:
                    streaming_chunks.clear()
                    # Chunked messages are already printed, so we just print a newline.
                    await aprint("", end="\n", flush=True)
                elif isinstance(message, MultiModalMessage):
                    await aprint(message.to_text(iterm=render_image_iterm), end="\n", flush=True)
                else:
                    await aprint(message.to_text(), end="\n", flush=True)
                if message.models_usage:
                    if output_stats:
                        await aprint(
                            f"[Prompt tokens: {message.models_usage.prompt_tokens}, Completion tokens: {message.models_usage.completion_tokens}]",
                            end="\n",
                            flush=True,
                        )
                    total_usage.completion_tokens += message.models_usage.completion_tokens
                    total_usage.prompt_tokens += message.models_usage.prompt_tokens
                if not exit_after_one_toolcall and isinstance(message, ToolCallRequestEvent):
                    user_input = await PromptSession[str]().prompt_async("ToolCallRequestEvent > ")
                    user_input = user_input.strip().lower()
                    if len(user_input) != 0 and user_input not in {"d", "do"}:
                        return last_processed
                if isinstance(message, ToolCallSummaryMessage) and exit_after_one_toolcall:
                    return last_processed

    return last_processed
