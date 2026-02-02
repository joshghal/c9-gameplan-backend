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
    simulation_context: Optional[dict] = None  # Snapshots, events, final state


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

        # Inject simulation context if available (from Command Center)
        if session.simulation_context:
            sim_ctx = session.simulation_context
            sim_summary = "\n\n## Active Simulation Context\n"
            sim_summary += "IMPORTANT: You have full simulation data below. For questions about THIS simulation (why a team won, what happened, key moments), answer directly from this data. Do NOT call tools like get_team_patterns — those return historical VCT data, not data about this simulation. Only use tools if the user asks about general team tendencies or patterns unrelated to this specific round.\n"
            sim_summary += "ALWAYS use real player names (e.g. 'Jakee', 'leaf') — NEVER use internal IDs like 'c9_1' or 'nrg_player_0'.\n\n"

            # Build player ID → name lookup from roster
            name_map: dict[str, str] = {}
            roster = sim_ctx.get("player_roster", [])
            if roster:
                # Group by team for clarity
                attack_players = [p for p in roster if p.get("side") == "attack"]
                defense_players = [p for p in roster if p.get("side") == "defense"]
                atk_team = sim_ctx.get("attack_team", "Attack")
                def_team = sim_ctx.get("defense_team", "Defense")

                sim_summary += f"PLAYERS IN THIS SIMULATION (use these names EXACTLY):\n"
                sim_summary += f"\n  {atk_team} (ATTACK):\n"
                for p in attack_players:
                    pid = p.get("id", "")
                    pname = p.get("name", pid)
                    name_map[pid] = pname
                    sim_summary += f"    {pname} — plays {p.get('agent', '?')}\n"
                sim_summary += f"\n  {def_team} (DEFENSE):\n"
                for p in defense_players:
                    pid = p.get("id", "")
                    pname = p.get("name", pid)
                    name_map[pid] = pname
                    sim_summary += f"    {pname} — plays {p.get('agent', '?')}\n"

                # Flat name list for quick lookup
                all_names = [name_map[pid] for pid in name_map if name_map[pid] != pid]
                sim_summary += f"\nALL PLAYER NAMES: {', '.join(all_names)}\n"
                sim_summary += "When the user mentions ANY of these names, that player IS in this simulation. Do not say they are not.\n\n"

            def resolve_name(pid: str) -> str:
                """Resolve player ID to real name."""
                return name_map.get(pid, pid)

            if sim_ctx.get("map_name"):
                sim_summary += f"Map: {sim_ctx['map_name']}"
                if sim_ctx.get("attack_team"):
                    sim_summary += f" | ATK: {sim_ctx['attack_team']} vs DEF: {sim_ctx.get('defense_team', '?')}"
                sim_summary += "\n"
            if sim_ctx.get("final_state"):
                fs = sim_ctx["final_state"]
                sim_summary += f"Result: ATK {fs.get('attack_alive', '?')} alive vs DEF {fs.get('defense_alive', '?')} alive"
                sim_summary += f" | Spike: {'planted' if fs.get('spike_planted') else 'not planted'}"
                sim_summary += f" | Duration: {fs.get('duration_ms', 0)}ms | Events: {fs.get('total_events', 0)}\n"
                # Include player final states
                positions = fs.get("positions", [])
                if positions:
                    sim_summary += "Final player states:\n"
                    for p in positions:
                        status = "ALIVE" if p.get("is_alive") else "DEAD"
                        pname = p.get("name") or resolve_name(p.get("player_id", "?"))
                        sim_summary += f"  - {pname} ({p.get('side')}, {p.get('agent', '?')}): {status}\n"
            def resolve_all_ids(text: str) -> str:
                """Replace all known player IDs with real names in a string."""
                for pid, pname in name_map.items():
                    if pid and pid in text:
                        text = text.replace(pid, pname)
                return text

            # Current narration context (from what-if questions)
            if sim_ctx.get("current_narration"):
                sim_summary += f"\n## CURRENT MOMENT (Moment {sim_ctx.get('current_moment_index', 0) + 1})\n"
                sim_summary += f"The user is asking about THIS specific moment in the match:\n"
                sim_summary += f"\"{sim_ctx['current_narration']}\"\n"
                sim_summary += "Answer based on this moment's context. You know all the players and teams involved.\n"

            # Match metadata
            match_ctx = sim_ctx.get("match_context")
            if match_ctx:
                teams = match_ctx.get("teams", [])
                if teams:
                    sim_summary += f"\nMatch: {' vs '.join(teams)}"
                if match_ctx.get("tournament"):
                    sim_summary += f" | Tournament: {match_ctx['tournament']}"
                if match_ctx.get("date"):
                    sim_summary += f" | Date: {match_ctx['date']}"
                if match_ctx.get("round_num"):
                    sim_summary += f" | Round {match_ctx['round_num']}"
                sim_summary += "\n"

            if sim_ctx.get("events"):
                events_list = sim_ctx["events"][:30]
                sim_summary += f"\nRound events ({len(events_list)} shown):\n"
                for ev in events_list:
                    # Handle both dict formats
                    time = ev.get("time_ms", ev.get("timestamp_ms", 0))
                    etype = ev.get("type", ev.get("event_type", "?"))
                    desc = ev.get("description", "")
                    if desc:
                        # Pre-built description may contain raw IDs — resolve them
                        desc = resolve_all_ids(desc)
                    else:
                        # Build description from event fields, resolving IDs to names
                        if etype == "kill":
                            killer = resolve_name(ev.get("killer", ev.get("player_id", "?")))
                            victim = resolve_name(ev.get("victim", ev.get("target_id", "?")))
                            desc = f"{killer} killed {victim}"
                            if ev.get("weapon"):
                                desc += f" with {ev['weapon']}"
                        elif etype == "spike_plant":
                            planter = resolve_name(ev.get("player_id", "?"))
                            desc = f"{planter} planted spike at {ev.get('site', '?')}"
                        else:
                            raw = json.dumps({k: v for k, v in ev.items() if k not in ('time_ms', 'timestamp_ms', 'type', 'event_type')})[:120]
                            desc = resolve_all_ids(raw)
                    sim_summary += f"  [{time}ms] {etype}: {desc}\n"
            if sim_ctx.get("snapshots"):
                snaps = sim_ctx["snapshots"]
                sim_summary += f"\nKey moments: {len(snaps)} snapshots captured\n"
                for i, snap in enumerate(snaps[:10]):
                    pc = snap.get("player_count", {})
                    sim_summary += f"  Moment {i}: [{snap.get('time_ms', 0)}ms] {snap.get('label', snap.get('phase', '?'))} — ATK {pc.get('attack', '?')}v{pc.get('defense', '?')} DEF"
                    if snap.get("spike_planted"):
                        sim_summary += f" | Spike planted"
                    sim_summary += "\n"
            # Final pass: resolve any remaining raw player IDs in the entire summary
            if name_map:
                sim_summary = resolve_all_ids(sim_summary)
            context += sim_summary

        system_prompt = get_coaching_prompt("coach", context)

        # Add user message
        session.messages.append(CoachingMessage(
            role="user",
            content=message,
            timestamp=datetime.now(),
        ))

        messages = self._build_messages(session)

        # Accumulate streamed text so we can save it to session history
        full_response = ""

        if use_tools:
            async for event in self.streaming_handler.stream_with_tools(
                llm_client=self.client,
                messages=messages,
                system=system_prompt,
                tools=COACHING_TOOLS,
                tool_handler=self.tool_handler,
            ):
                # Extract text chunks to accumulate full response
                if '"type": "text"' in event:
                    try:
                        payload = json.loads(event.split("data: ", 1)[1].strip())
                        if payload.get("type") == "text":
                            full_response += payload.get("data", "")
                    except (json.JSONDecodeError, IndexError):
                        pass
                yield event
        else:
            async for event in self.streaming_handler.stream_response(
                self.client.stream_chat(
                    messages=messages,
                    system=system_prompt,
                )
            ):
                if '"type": "text"' in event:
                    try:
                        payload = json.loads(event.split("data: ", 1)[1].strip())
                        if payload.get("type") == "text":
                            full_response += payload.get("data", "")
                    except (json.JSONDecodeError, IndexError):
                        pass
                yield event

        # Save assistant response for conversation continuity
        if full_response.strip():
            session.messages.append(CoachingMessage(
                role="assistant",
                content=full_response,
                timestamp=datetime.now(),
            ))

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
