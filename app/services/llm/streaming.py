"""Server-Sent Events streaming handler for real-time chat."""

import json
from typing import AsyncGenerator, Optional
from dataclasses import dataclass
from enum import Enum


class StreamEventType(str, Enum):
    """Types of streaming events."""
    TEXT = "text"
    TOOL_START = "tool_start"
    TOOL_RESULT = "tool_result"
    ERROR = "error"
    DONE = "done"
    THINKING = "thinking"


@dataclass
class StreamEvent:
    """A single streaming event."""
    event_type: StreamEventType
    data: str
    tool_name: Optional[str] = None
    tool_id: Optional[str] = None

    def to_sse(self) -> str:
        """Format as Server-Sent Event."""
        payload = {
            "type": self.event_type.value,
            "data": self.data,
        }
        if self.tool_name:
            payload["tool_name"] = self.tool_name
        if self.tool_id:
            payload["tool_id"] = self.tool_id

        return f"data: {json.dumps(payload)}\n\n"


class StreamingHandler:
    """Handles SSE streaming for coaching chat."""

    def __init__(self):
        self.buffer = ""

    async def stream_response(
        self,
        text_stream: AsyncGenerator[str, None],
    ) -> AsyncGenerator[str, None]:
        """Convert text stream to SSE events."""
        try:
            async for chunk in text_stream:
                event = StreamEvent(
                    event_type=StreamEventType.TEXT,
                    data=chunk,
                )
                yield event.to_sse()

            # Signal completion
            done_event = StreamEvent(
                event_type=StreamEventType.DONE,
                data="",
            )
            yield done_event.to_sse()

        except Exception as e:
            error_event = StreamEvent(
                event_type=StreamEventType.ERROR,
                data=str(e),
            )
            yield error_event.to_sse()

    async def stream_with_tools(
        self,
        llm_client,
        messages: list[dict],
        system: str,
        tools: list[dict],
        tool_handler,
    ) -> AsyncGenerator[str, None]:
        """Stream response with tool call handling.

        This handles the agentic loop:
        1. Send message to LLM
        2. If tool calls, execute them and continue
        3. Stream final text response
        """
        current_messages = messages.copy()
        max_iterations = 5  # Prevent infinite loops

        for _ in range(max_iterations):
            # Get LLM response
            text, tool_calls = await llm_client.chat_with_tools(
                messages=current_messages,
                system=system,
                tools=tools,
            )

            if not tool_calls:
                # No more tool calls, stream the final text
                if text:
                    # Send text in chunks for streaming effect
                    words = text.split()
                    for i, word in enumerate(words):
                        chunk = word + (" " if i < len(words) - 1 else "")
                        event = StreamEvent(
                            event_type=StreamEventType.TEXT,
                            data=chunk,
                        )
                        yield event.to_sse()

                done_event = StreamEvent(
                    event_type=StreamEventType.DONE,
                    data="",
                )
                yield done_event.to_sse()
                return

            # Process tool calls
            tool_results = []
            for tool_call in tool_calls:
                # Notify client of tool start
                start_event = StreamEvent(
                    event_type=StreamEventType.TOOL_START,
                    data=f"Using {tool_call['name']}...",
                    tool_name=tool_call["name"],
                    tool_id=tool_call["id"],
                )
                yield start_event.to_sse()

                # Execute tool
                result = await tool_handler.handle_tool_call(
                    tool_call["name"],
                    tool_call["input"],
                )

                # Notify client of result
                result_event = StreamEvent(
                    event_type=StreamEventType.TOOL_RESULT,
                    data=json.dumps(result)[:500] + "...",  # Truncate for UI
                    tool_name=tool_call["name"],
                    tool_id=tool_call["id"],
                )
                yield result_event.to_sse()

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_call["id"],
                    "content": json.dumps(result),
                })

            # Add assistant message with tool calls
            current_messages.append({
                "role": "assistant",
                "content": [
                    {"type": "text", "text": text} if text else None,
                    *[
                        {
                            "type": "tool_use",
                            "id": tc["id"],
                            "name": tc["name"],
                            "input": tc["input"],
                        }
                        for tc in tool_calls
                    ]
                ]
            })
            # Filter out None
            current_messages[-1]["content"] = [
                c for c in current_messages[-1]["content"] if c is not None
            ]

            # Add tool results
            current_messages.append({
                "role": "user",
                "content": tool_results,
            })

        # Max iterations reached
        error_event = StreamEvent(
            event_type=StreamEventType.ERROR,
            data="Max tool iterations reached",
        )
        yield error_event.to_sse()
