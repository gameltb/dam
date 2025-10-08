"""Custom OpenAI client for OpenRouter R1 models."""

import warnings
from collections.abc import AsyncGenerator, Mapping, Sequence
from typing import (
    Any,
    Literal,
    cast,
)

from autogen_core import (
    CancellationToken,
    FunctionCall,
)
from autogen_core.logging import LLMCallEvent, LLMStreamEndEvent, LLMStreamStartEvent
from autogen_core.models import (
    ChatCompletionTokenLogprob,
    CreateResult,
    LLMMessage,
    ModelFamily,
    RequestUsage,
    TopLogprob,
)
from autogen_core.tools import Tool, ToolSchema
from autogen_ext.models._utils.normalize_stop_reason import normalize_stop_reason
from autogen_ext.models._utils.parse_r1_content import parse_r1_content
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.models.openai._openai_client import (
    _add_usage,  # pyright: ignore[reportPrivateUsage]
    logger,
)
from openai.types.chat.chat_completion_chunk import Choice
from pydantic import BaseModel

EMPTY_CHUNK_WARNING_THRESHOLD = 10


class OpenRouterR1OpenAIChatCompletionClient(OpenAIChatCompletionClient):
    """A custom OpenAI Chat Completion client for OpenRouter R1 models."""

    component_provider_override = "domarkx.models.openrouter.OpenRouterR1OpenAIChatCompletionClient"

    def _process_reasoning_content(self, choice: Choice, is_reasoning: bool) -> tuple[str | None, bool]:
        """Process reasoning content from a choice delta."""
        reasoning_text: str | None = None
        if choice.delta.model_extra is not None:
            if "reasoning_content" in choice.delta.model_extra:
                reasoning_text = choice.delta.model_extra.get("reasoning_content")
            elif "reasoning" in choice.delta.model_extra:
                reasoning_text = choice.delta.model_extra.get("reasoning")

        if isinstance(reasoning_text, str) and len(reasoning_text) > 0:
            if not is_reasoning:
                return f"<think>{reasoning_text}", True
            return reasoning_text, True
        if is_reasoning:
            return "</think>", False

        return None, is_reasoning

    def _process_tool_calls(self, choice: Choice, full_tool_calls: dict[int, FunctionCall]) -> None:
        """Process tool calls from a choice delta."""
        if choice.delta.tool_calls is not None:
            for tool_call_chunk in choice.delta.tool_calls:
                idx = tool_call_chunk.index
                if idx not in full_tool_calls:
                    full_tool_calls[idx] = FunctionCall(id="", arguments="", name="")

                if tool_call_chunk.id is not None:
                    full_tool_calls[idx].id += tool_call_chunk.id

                if tool_call_chunk.function is not None:
                    if tool_call_chunk.function.name is not None:
                        full_tool_calls[idx].name += tool_call_chunk.function.name
                    if tool_call_chunk.function.arguments is not None:
                        full_tool_calls[idx].arguments += tool_call_chunk.function.arguments

    def _finalize_content(
        self,
        content_deltas: list[str],
        thought_deltas: list[str],
        full_tool_calls: dict[int, FunctionCall],
    ) -> tuple[str | list[FunctionCall], str | None]:
        """Finalize content and thought from collected deltas."""
        content: str | list[FunctionCall]
        thought: str | None = None
        if full_tool_calls:
            content = list(full_tool_calls.values())
            if content_deltas:
                thought = "".join(content_deltas)
        else:
            if content_deltas:
                content = "".join(content_deltas)
            else:
                warnings.warn("No text content or tool calls are available. Model returned empty result.", stacklevel=2)
                content = ""

            if thought_deltas:
                thought = "".join(thought_deltas).lstrip("<think>").rstrip("</think>")

            if isinstance(content, str) and self._model_info["family"] == ModelFamily.R1 and thought is None:
                thought, content = parse_r1_content(content)

        return content, thought

    async def _iterate_and_process_chunks(
        self, chunks: AsyncGenerator[Any, None], state: dict[str, Any]
    ) -> AsyncGenerator[str, None]:
        """Iterate over stream chunks, update state, and yield content."""
        async for chunk in chunks:
            if state["first_chunk"]:
                logger.info(LLMStreamStartEvent(messages=cast(list[dict[str, Any]], state["messages"])))
                state["first_chunk"] = False

            if not chunk.choices:
                state["empty_chunk_count"] += 1
                if (
                    not state["empty_chunk_warning_has_been_issued"]
                    and state["empty_chunk_count"] >= EMPTY_CHUNK_WARNING_THRESHOLD
                ):
                    warnings.warn("More than 10 consecutive empty chunks received.", stacklevel=2)
                    state["empty_chunk_warning_has_been_issued"] = True
                continue
            state["empty_chunk_count"] = 0

            if len(chunk.choices) > 1:
                warnings.warn("Multiple choices received, only using the first.", UserWarning, stacklevel=2)

            choice = chunk.choices[0]
            state["choice"] = choice
            if chunk.usage is None and state["stop_reason"] is None:
                state["stop_reason"] = choice.finish_reason
            state["chunk"] = chunk

            reasoning_content, state["is_reasoning"] = self._process_reasoning_content(choice, state["is_reasoning"])
            if reasoning_content:
                state["thought_deltas"].append(reasoning_content)
                yield reasoning_content

            if choice.delta.content:
                state["content_deltas"].append(choice.delta.content)
                yield choice.delta.content
                continue

            self._process_tool_calls(choice, state["full_tool_calls"])

            if choice.logprobs and choice.logprobs.content:
                state["logprobs"] = [
                    ChatCompletionTokenLogprob(
                        token=x.token,
                        logprob=x.logprob,
                        top_logprobs=[TopLogprob(logprob=y.logprob, bytes=y.bytes) for y in x.top_logprobs],
                        bytes=x.bytes,
                    )
                    for x in choice.logprobs.content
                ]

    async def create_stream(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Tool | ToolSchema] = [],
        tool_choice: Tool | Literal["auto", "required", "none"] = "auto",
        json_output: bool | type[BaseModel] | None = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: CancellationToken | None = None,
        max_consecutive_empty_chunk_tolerance: int = 0,
        include_usage: bool | None = None,
    ) -> AsyncGenerator[str | CreateResult, None]:
        """Create a stream of string chunks from the model ending with a :class:`~autogen_core.models.CreateResult`."""
        create_params = self._process_create_args(messages, tools, tool_choice, json_output, extra_create_args)
        if include_usage is not None:
            if (
                "stream_options" in create_params.create_args
                and create_params.create_args["stream_options"].get("include_usage") != include_usage
            ):
                raise ValueError("include_usage and extra_create_args['stream_options']['include_usage'] differ.")
            create_params.create_args.setdefault("stream_options", {})["include_usage"] = True

        if max_consecutive_empty_chunk_tolerance != 0:
            warnings.warn("max_consecutive_empty_chunk_tolerance is deprecated.", DeprecationWarning, stacklevel=2)

        chunks = (
            self._create_stream_chunks_beta_client(
                tool_params=create_params.tools,
                oai_messages=create_params.messages,
                response_format=create_params.response_format,
                create_args_no_response_format=create_params.create_args,
                cancellation_token=cancellation_token,
            )
            if create_params.response_format
            else self._create_stream_chunks(
                tool_params=create_params.tools,
                oai_messages=create_params.messages,
                create_args=create_params.create_args,
                cancellation_token=cancellation_token,
            )
        )

        state: dict[str, Any] = {
            "content_deltas": [],
            "thought_deltas": [],
            "full_tool_calls": {},
            "logprobs": None,
            "is_reasoning": False,
            "first_chunk": True,
            "stop_reason": None,
            "choice": None,
            "chunk": None,
            "empty_chunk_count": 0,
            "empty_chunk_warning_has_been_issued": False,
            "messages": create_params.messages,
        }

        async for content_chunk in self._iterate_and_process_chunks(chunks, state):
            yield content_chunk

        if state["stop_reason"] == "function_call":
            raise ValueError("Function calls are not supported in this context")

        chunk = state.get("chunk")
        usage = RequestUsage(
            prompt_tokens=chunk.usage.prompt_tokens if chunk and chunk.usage else 0,
            completion_tokens=chunk.usage.completion_tokens if chunk and chunk.usage else 0,
        )

        content, thought = self._finalize_content(
            state["content_deltas"], state["thought_deltas"], state["full_tool_calls"]
        )

        choice = state.get("choice")
        if choice:
            if isinstance(content, str):
                choice.delta.content = content
            if thought:
                choice.delta.reasoning = thought  # type: ignore[attr-defined]

        if chunk:
            logger.info(
                LLMCallEvent(
                    messages=cast(list[dict[str, Any]], create_params.messages),
                    response=chunk.model_dump(),
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                )
            )

        result = CreateResult(
            finish_reason=normalize_stop_reason(state["stop_reason"]),
            content=content,
            usage=usage,
            cached=False,
            logprobs=state.get("logprobs"),
            thought=thought,
        )
        logger.info(
            LLMStreamEndEvent(
                response=result.model_dump(),
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
            )
        )

        self._total_usage = _add_usage(self._total_usage, usage)
        self._actual_usage = _add_usage(self._actual_usage, usage)
        yield result
