import warnings
from typing import (
    Any,
    AsyncGenerator,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Union,
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
    _add_usage,
    logger,
)
from openai.types.chat import (
    ChatCompletionChunk,
)
from pydantic import BaseModel


class OpenRouterR1OpenAIChatCompletionClient(OpenAIChatCompletionClient):
    component_provider_override = "domarkx.models.openrouter.OpenRouterR1OpenAIChatCompletionClient"

    async def create_stream(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Tool | ToolSchema] = [],
        tool_choice: Tool | Literal["auto", "required", "none"] = "auto",
        json_output: Optional[bool | type[BaseModel]] = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: Optional[CancellationToken] = None,
        max_consecutive_empty_chunk_tolerance: int = 0,
        include_usage: Optional[bool] = None,
    ) -> AsyncGenerator[Union[str, CreateResult], None]:
        """Create a stream of string chunks from the model ending with a :class:`~autogen_core.models.CreateResult`.

        Extends :meth:`autogen_core.models.ChatCompletionClient.create_stream` to support OpenAI API.

        In streaming, the default behaviour is not return token usage counts.
        See: `OpenAI API reference for possible args <https://platform.openai.com/docs/api-reference/chat/create>`_.

        You can set set the `include_usage` flag to True or `extra_create_args={"stream_options": {"include_usage": True}}`. If both the flag and `stream_options` are set, but to different values, an exception will be raised.
        (if supported by the accessed API) to
        return a final chunk with usage set to a :class:`~autogen_core.models.RequestUsage` object
        with prompt and completion token counts,
        all preceding chunks will have usage as `None`.
        See: `OpenAI API reference for stream options <https://platform.openai.com/docs/api-reference/chat/create#chat-create-stream_options>`_.

        Other examples of supported arguments that can be included in `extra_create_args`:
            - `temperature` (float): Controls the randomness of the output. Higher values (e.g., 0.8) make the output more random, while lower values (e.g., 0.2) make it more focused and deterministic.
            - `max_tokens` (int): The maximum number of tokens to generate in the completion.
            - `top_p` (float): An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass.
            - `frequency_penalty` (float): A value between -2.0 and 2.0 that penalizes new tokens based on their existing frequency in the text so far, decreasing the likelihood of repeated phrases.
            - `presence_penalty` (float): A value between -2.0 and 2.0 that penalizes new tokens based on whether they appear in the text so far, encouraging the model to talk about new topics.
        """

        create_params = self._process_create_args(
            messages,
            tools,
            tool_choice,
            json_output,
            extra_create_args,
        )

        if include_usage is not None:
            if "stream_options" in create_params.create_args:
                stream_options = create_params.create_args["stream_options"]
                if "include_usage" in stream_options and stream_options["include_usage"] != include_usage:
                    raise ValueError(
                        "include_usage and extra_create_args['stream_options']['include_usage'] are both set, but differ in value."
                    )
            else:
                # If stream options are not present, add them.
                create_params.create_args["stream_options"] = {"include_usage": True}

        if max_consecutive_empty_chunk_tolerance != 0:
            warnings.warn(
                "The 'max_consecutive_empty_chunk_tolerance' parameter is deprecated and will be removed in the future releases. All of empty chunks will be skipped with a warning.",
                DeprecationWarning,
                stacklevel=2,
            )

        if create_params.response_format is not None:
            chunks = self._create_stream_chunks_beta_client(
                tool_params=create_params.tools,
                oai_messages=create_params.messages,
                response_format=create_params.response_format,
                create_args_no_response_format=create_params.create_args,
                cancellation_token=cancellation_token,
            )
        else:
            chunks = self._create_stream_chunks(
                tool_params=create_params.tools,
                oai_messages=create_params.messages,
                create_args=create_params.create_args,
                cancellation_token=cancellation_token,
            )

        # Prepare data to process streaming chunks.
        chunk: ChatCompletionChunk | None = None
        stop_reason = None
        maybe_model = None
        content_deltas: List[str] = []
        thought_deltas: List[str] = []
        full_tool_calls: Dict[int, FunctionCall] = {}
        logprobs: Optional[List[ChatCompletionTokenLogprob]] = None

        empty_chunk_warning_has_been_issued: bool = False
        empty_chunk_warning_threshold: int = 10
        empty_chunk_count = 0
        first_chunk = True
        is_reasoning = False

        # Process the stream of chunks.
        async for chunk in chunks:
            if first_chunk:
                first_chunk = False
                # Emit the start event.
                logger.info(
                    LLMStreamStartEvent(
                        messages=cast(List[Dict[str, Any]], create_params.messages),
                    )
                )

            # Set the model from the lastest chunk.
            maybe_model = chunk.model

            # Empty chunks has been observed when the endpoint is under heavy load.
            #  https://github.com/microsoft/autogen/issues/4213
            if len(chunk.choices) == 0:
                empty_chunk_count += 1
                if not empty_chunk_warning_has_been_issued and empty_chunk_count >= empty_chunk_warning_threshold:
                    empty_chunk_warning_has_been_issued = True
                    warnings.warn(
                        f"Received more than {empty_chunk_warning_threshold} consecutive empty chunks. Empty chunks are being ignored.",
                        stacklevel=2,
                    )
                continue
            else:
                empty_chunk_count = 0

            if len(chunk.choices) > 1:
                # This is a multi-choice chunk, we need to warn the user.
                warnings.warn(
                    f"Received a chunk with {len(chunk.choices)} choices. Only the first choice will be used.",
                    UserWarning,
                    stacklevel=2,
                )

            # Set the choice to the first choice in the chunk.
            choice = chunk.choices[0]

            # for liteLLM chunk usage, do the following hack keeping the pervious chunk.stop_reason (if set).
            # set the stop_reason for the usage chunk to the prior stop_reason
            stop_reason = choice.finish_reason if chunk.usage is None and stop_reason is None else stop_reason
            maybe_model = chunk.model

            reasoning_content: str | None = None
            if choice.delta.model_extra is not None:
                if "reasoning_content" in choice.delta.model_extra:
                    # If there is a reasoning_content field, then we populate the thought field. This is for models such as R1.
                    reasoning_content = choice.delta.model_extra.get("reasoning_content")
                elif "reasoning" in choice.delta.model_extra:
                    reasoning_content = choice.delta.model_extra.get("reasoning")

            if isinstance(reasoning_content, str) and len(reasoning_content) > 0:
                if not is_reasoning:
                    # Enter reasoning mode.
                    reasoning_content = "<think>" + reasoning_content
                    is_reasoning = True
                thought_deltas.append(reasoning_content)
                yield reasoning_content
            elif is_reasoning:
                # Exit reasoning mode.
                reasoning_content = "</think>"
                thought_deltas.append(reasoning_content)
                is_reasoning = False
                yield reasoning_content

            # First try get content
            if choice.delta.content:
                content_deltas.append(choice.delta.content)
                if len(choice.delta.content) > 0:
                    yield choice.delta.content
                # NOTE: for OpenAI, tool_calls and content are mutually exclusive it seems, so we can skip the rest of the loop.
                # However, this may not be the case for other APIs -- we should expect this may need to be updated.
                continue
            # Otherwise, get tool calls
            if choice.delta.tool_calls is not None:
                for tool_call_chunk in choice.delta.tool_calls:
                    idx = tool_call_chunk.index
                    if idx not in full_tool_calls:
                        # We ignore the type hint here because we want to fill in type when the delta provides it
                        full_tool_calls[idx] = FunctionCall(id="", arguments="", name="")

                    if tool_call_chunk.id is not None:
                        full_tool_calls[idx].id += tool_call_chunk.id

                    if tool_call_chunk.function is not None:
                        if tool_call_chunk.function.name is not None:
                            full_tool_calls[idx].name += tool_call_chunk.function.name
                        if tool_call_chunk.function.arguments is not None:
                            full_tool_calls[idx].arguments += tool_call_chunk.function.arguments
            if choice.logprobs and choice.logprobs.content:
                logprobs = [
                    ChatCompletionTokenLogprob(
                        token=x.token,
                        logprob=x.logprob,
                        top_logprobs=[TopLogprob(logprob=y.logprob, bytes=y.bytes) for y in x.top_logprobs],
                        bytes=x.bytes,
                    )
                    for x in choice.logprobs.content
                ]

        # Finalize the CreateResult.

        # TODO: can we remove this?
        if stop_reason == "function_call":
            raise ValueError("Function calls are not supported in this context")

        # We need to get the model from the last chunk, if available.
        model = maybe_model or create_params.create_args["model"]
        model = model.replace("gpt-35", "gpt-3.5")  # hack for Azure API

        # Because the usage chunk is not guaranteed to be the last chunk, we need to check if it is available.
        if chunk and chunk.usage:
            prompt_tokens = chunk.usage.prompt_tokens
            completion_tokens = chunk.usage.completion_tokens
        else:
            prompt_tokens = 0
            completion_tokens = 0
        usage = RequestUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

        # Detect whether it is a function call or just text.
        content: Union[str, List[FunctionCall]]
        thought: str | None = None
        # Determine the content and thought based on what was collected
        if full_tool_calls:
            # This is a tool call response
            content = list(full_tool_calls.values())
            if content_deltas:
                # Store any text alongside tool calls as thoughts
                thought = "".join(content_deltas)
        else:
            # This is a text response (possibly with thoughts)
            if content_deltas:
                content = "".join(content_deltas)
            else:
                warnings.warn(
                    "No text content or tool calls are available. Model returned empty result.",
                    stacklevel=2,
                )
                content = ""

            # Set thoughts if we have any reasoning content.
            if thought_deltas:
                thought = "".join(thought_deltas).lstrip("<think>").rstrip("</think>")

            # This is for local R1 models whose reasoning content is within the content string.
            if isinstance(content, str) and self._model_info["family"] == ModelFamily.R1 and thought is None:
                thought, content = parse_r1_content(content)

        choice.delta.content = content
        if thought is not None and len(thought) > 0:
            choice.delta.reasoning = thought

        logger.info(
            LLMCallEvent(
                messages=cast(Dict[str, Any], create_params.messages),
                response=chunk.model_dump(),
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
            )
        )

        # Create the result.
        result = CreateResult(
            finish_reason=normalize_stop_reason(stop_reason),
            content=content,
            usage=usage,
            cached=False,
            logprobs=logprobs,
            thought=thought,
        )

        # Log the end of the stream.
        logger.info(
            LLMStreamEndEvent(
                response=result.model_dump(),
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
            )
        )

        # Update the total usage.
        self._total_usage = _add_usage(self._total_usage, usage)
        self._actual_usage = _add_usage(self._actual_usage, usage)

        # Yield the CreateResult.
        yield result
