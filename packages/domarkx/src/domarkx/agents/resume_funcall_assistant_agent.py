"""An AssistantAgent that can resume from a previous function call."""
import uuid
from collections.abc import AsyncGenerator, Sequence
from typing import (
    cast,
)

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import (
    BaseAgentEvent,
    BaseChatMessage,
    ThoughtEvent,
)
from autogen_core import CancellationToken, FunctionCall
from autogen_core.models import (
    AssistantMessage,
    CreateResult,
    RequestUsage,
)
from rich.console import Console

console = Console(markup=False)


class ResumeFunCallAssistantAgent(AssistantAgent):
    """An AssistantAgent that can resume from a previous function call."""

    async def on_messages_stream(
        self,
        messages: Sequence[BaseChatMessage],
        cancellation_token: CancellationToken,
    ) -> AsyncGenerator[BaseAgentEvent | BaseChatMessage | Response, None]:
        """
        Process messages and stream the response.

        Args:
            messages: Sequence of messages to process
            cancellation_token: Token for cancelling operation

        Yields:
            Events, messages and final response during processing

        """
        # Gather all relevant state here
        agent_name = self.name
        model_context = self._model_context
        memory = self._memory
        system_messages = self._system_messages
        workbench = self._workbench
        handoff_tools = self._handoff_tools
        handoffs = self._handoffs
        model_client = self._model_client
        model_client_stream = self._model_client_stream
        reflect_on_tool_use = self._reflect_on_tool_use
        max_tool_iterations = self._max_tool_iterations
        tool_call_summary_format = self._tool_call_summary_format
        tool_call_summary_formatter = self._tool_call_summary_formatter
        output_content_type = self._output_content_type
        inner_messages: list[BaseAgentEvent | BaseChatMessage] = []

        # check if we have
        model_context_messages = await model_context.get_messages()
        if (
            len(model_context_messages) > 0
            and isinstance(model_context_messages[-1].content, list)
            and all(isinstance(item, FunctionCall) for item in model_context_messages[-1].content)
        ):
            fun_model_result = model_context_messages[-1]

            create_result = CreateResult(
                finish_reason="function_calls",
                content=cast(list[FunctionCall], fun_model_result.content),
                usage=RequestUsage(prompt_tokens=0, completion_tokens=0),
                cached=False,
                logprobs=None,
                thought=None,
            )
            message_id = str(uuid.uuid4())
            async for output_event in self._process_model_result(
                model_result=create_result,
                inner_messages=inner_messages,
                cancellation_token=cancellation_token,
                agent_name=agent_name,
                system_messages=system_messages,
                model_context=model_context,
                workbench=workbench,
                handoff_tools=handoff_tools,
                handoffs=handoffs,
                model_client=model_client,
                model_client_stream=model_client_stream,
                reflect_on_tool_use=reflect_on_tool_use,
                max_tool_iterations=max_tool_iterations,
                tool_call_summary_format=tool_call_summary_format,
                tool_call_summary_formatter=tool_call_summary_formatter,
                output_content_type=output_content_type,
                message_id=message_id,
                format_string=self._output_content_type_format,
            ):
                yield output_event

        # STEP 1: Add new user/handoff messages to the model context
        await self._add_messages_to_context(
            model_context=model_context,
            messages=messages,
        )

        # STEP 2: Update model context with any relevant memory
        for event_msg in await self._update_model_context_with_memory(
            memory=memory,
            model_context=model_context,
            agent_name=agent_name,
        ):
            inner_messages.append(event_msg)
            yield event_msg

        # STEP 3: Generate a message ID for correlation between streaming chunks and final message
        message_id = str(uuid.uuid4())

        # STEP 4: Run the first inference
        model_result = None
        async for inference_output in self._call_llm(
            model_client=model_client,
            model_client_stream=model_client_stream,
            system_messages=system_messages,
            model_context=model_context,
            workbench=workbench,
            handoff_tools=handoff_tools,
            agent_name=agent_name,
            cancellation_token=cancellation_token,
            output_content_type=output_content_type,
            message_id=message_id,
        ):
            if isinstance(inference_output, CreateResult):
                model_result = inference_output
            else:
                # Streaming chunk event
                yield inference_output

        assert model_result is not None, "No model result was produced."

        # --- NEW: If the model produced a hidden "thought," yield it as an event ---
        if model_result.thought:
            thought_event = ThoughtEvent(content=model_result.thought, source=agent_name)
            yield thought_event
            inner_messages.append(thought_event)

        # Add the assistant message to the model context (including thought if present)
        await model_context.add_message(
            AssistantMessage(
                content=model_result.content,
                source=agent_name,
                thought=getattr(model_result, "thought", None),
            )
        )

        # STEP 5: Process the model output
        async for output_event in self._process_model_result(
            model_result=model_result,
            inner_messages=inner_messages,
            cancellation_token=cancellation_token,
            agent_name=agent_name,
            system_messages=system_messages,
            model_context=model_context,
            workbench=workbench,
            handoff_tools=handoff_tools,
            handoffs=handoffs,
            model_client=model_client,
            model_client_stream=model_client_stream,
            reflect_on_tool_use=reflect_on_tool_use,
            max_tool_iterations=max_tool_iterations,
            tool_call_summary_format=tool_call_summary_format,
            tool_call_summary_formatter=tool_call_summary_formatter,
            output_content_type=output_content_type,
            message_id=message_id,
            format_string=self._output_content_type_format,
        ):
            yield output_event
