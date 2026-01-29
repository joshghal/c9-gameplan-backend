"""Main coaching service orchestrator."""

from typing import AsyncGenerator, Optional
from dataclasses import dataclass
from datetime import datetime
import json

from ..llm import (
    get_anthropic_client,
    ContextBuilder,
    StreamingHandler,
    get_coaching_prompt,
)
from ..llm.prompts import COACHING_TOOLS
from ..llm.context_builder import ToolHandler


@dataclass
class CoachingMessage:
    """A message in a coaching conversation."""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime
    tool_calls: Optional[list] = None


@dataclass
class CoachingSession:
    """A coaching conversation session."""
    session_id: str
    messages: list[CoachingMessage]
    created_at: datetime
    map_context: Optional[str] = None
    team_context: Optional[str] = None


class CoachService:
    """Orchestrates AI coaching interactions."""

    def __init__(self):
        self.client = get_anthropic_client()
        self.context_builder = ContextBuilder()
        self.streaming_handler = StreamingHandler()
        self.tool_handler = ToolHandler()
        self._sessions: dict[str, CoachingSession] = {}

    def create_session(
        self,
        session_id: str,
        map_name: Optional[str] = None,
        team_name: Optional[str] = None,
    ) -> CoachingSession:
        """Create a new coaching session."""
        session = CoachingSession(
            session_id=session_id,
            messages=[],
            created_at=datetime.now(),
            map_context=map_name,
            team_context=team_name,
        )
        self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[CoachingSession]:
        """Get an existing session."""
        return self._sessions.get(session_id)

    def _build_messages(self, session: CoachingSession) -> list[dict]:
        """Convert session messages to API format."""
        return [
            {"role": msg.role, "content": msg.content}
            for msg in session.messages
        ]

    async def chat(
        self,
        session_id: str,
        message: str,
        use_tools: bool = True,
    ) -> str:
        """Send a message and get a response (non-streaming)."""
        session = self.get_session(session_id)
        if not session:
            session = self.create_session(session_id)

        # Build context based on session
        context = self.context_builder.build_coaching_context(
            query=message,
            map_name=session.map_context,
            team_name=session.team_context,
        )

        system_prompt = get_coaching_prompt("coach", context)

        # Add user message
        session.messages.append(CoachingMessage(
            role="user",
            content=message,
            timestamp=datetime.now(),
        ))

        messages = self._build_messages(session)

        if use_tools:
            text, tool_calls = await self.client.chat_with_tools(
                messages=messages,
                system=system_prompt,
                tools=COACHING_TOOLS,
            )

            # Handle tool calls in a loop
            while tool_calls:
                # Execute tools
                tool_results = []
                for tc in tool_calls:
                    result = await self.tool_handler.handle_tool_call(
                        tc["name"], tc["input"]
                    )
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tc["id"],
                        "content": json.dumps(result),
                    })

                # Continue conversation with tool results
                messages.append({
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
                        ],
                    ],
                })
                messages[-1]["content"] = [
                    c for c in messages[-1]["content"] if c is not None
                ]
                messages.append({"role": "user", "content": tool_results})

                text, tool_calls = await self.client.chat_with_tools(
                    messages=messages,
                    system=system_prompt,
                    tools=COACHING_TOOLS,
                )

            response_text = text
        else:
            response = await self.client.chat(
                messages=messages,
                system=system_prompt,
            )
            response_text = response.content[0].text

        # Save assistant response
        session.messages.append(CoachingMessage(
            role="assistant",
            content=response_text,
            timestamp=datetime.now(),
        ))

        return response_text

    async def stream_chat(
        self,
        session_id: str,
        message: str,
        use_tools: bool = True,
    ) -> AsyncGenerator[str, None]:
        """Stream a chat response with SSE."""
        session = self.get_session(session_id)
        if not session:
            session = self.create_session(session_id)

        # Build context
        context = self.context_builder.build_coaching_context(
            query=message,
            map_name=session.map_context,
            team_name=session.team_context,
        )

        system_prompt = get_coaching_prompt("coach", context)

        # Add user message
        session.messages.append(CoachingMessage(
            role="user",
            content=message,
            timestamp=datetime.now(),
        ))

        messages = self._build_messages(session)

        if use_tools:
            async for event in self.streaming_handler.stream_with_tools(
                llm_client=self.client,
                messages=messages,
                system=system_prompt,
                tools=COACHING_TOOLS,
                tool_handler=self.tool_handler,
            ):
                yield event
        else:
            async for event in self.streaming_handler.stream_response(
                self.client.stream_chat(
                    messages=messages,
                    system=system_prompt,
                )
            ):
                yield event

    async def analyze_position(
        self,
        map_name: str,
        positions: list[dict],
        side: str,
        phase: str,
    ) -> str:
        """Analyze a specific game position."""
        position_summary = json.dumps(positions, indent=2)

        prompt = f"""Analyze this game position:

Map: {map_name}
Side: {side}
Phase: {phase}

Player Positions:
{position_summary}

Provide tactical analysis including:
1. Current formation assessment
2. Likely opponent reads
3. Vulnerabilities
4. Recommended adjustments"""

        response = await self.client.chat(
            messages=[{"role": "user", "content": prompt}],
            system=get_coaching_prompt("coach"),
        )

        return response.content[0].text

    def clear_session(self, session_id: str) -> bool:
        """Clear a session's history."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False


# Singleton instance
_coach_service: Optional[CoachService] = None


def get_coach_service() -> CoachService:
    """Get the singleton coach service instance."""
    global _coach_service
    if _coach_service is None:
        _coach_service = CoachService()
    return _coach_service
