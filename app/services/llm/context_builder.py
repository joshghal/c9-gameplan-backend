"""Context builder for RAG-enhanced coaching responses."""

import json
from typing import Optional
from pathlib import Path

from ...config import get_settings


class ContextBuilder:
    """Builds context from pattern data for LLM coaching."""

    def __init__(self, data_path: Optional[Path] = None):
        """Initialize with optional custom data path."""
        self.settings = get_settings()
        self.data_path = data_path or Path(__file__).parent.parent.parent.parent / "data"
        self._pattern_cache: dict = {}

    def _load_json(self, filename: str) -> dict:
        """Load JSON file from data directory with caching."""
        if filename in self._pattern_cache:
            return self._pattern_cache[filename]

        file_path = self.data_path / filename
        if file_path.exists():
            with open(file_path, "r") as f:
                data = json.load(f)
                self._pattern_cache[filename] = data
                return data
        return {}

    def get_position_patterns(
        self,
        team_name: Optional[str] = None,
        map_name: Optional[str] = None,
        side: Optional[str] = None,
    ) -> dict:
        """Get position pattern data for context."""
        patterns = self._load_json("position_patterns.json")

        if not patterns:
            return {"message": "Position patterns not loaded"}

        result = {}

        # Filter by map
        if map_name:
            result = patterns.get(map_name, {})
        else:
            result = patterns

        # Filter by side
        if side and isinstance(result, dict):
            if side in result:
                result = {side: result[side]}

        return result

    def get_economy_patterns(self, team_name: Optional[str] = None) -> dict:
        """Get economy pattern data."""
        patterns = self._load_json("economy_patterns.json")

        if not patterns:
            return {"message": "Economy patterns not loaded"}

        if team_name and team_name in patterns:
            return {team_name: patterns[team_name]}

        return patterns

    def get_trade_patterns(
        self,
        team_name: Optional[str] = None,
        map_name: Optional[str] = None,
    ) -> dict:
        """Get trade timing pattern data."""
        patterns = self._load_json("trade_patterns.json")

        if not patterns:
            return {"message": "Trade patterns not loaded"}

        if team_name and team_name in patterns:
            return {team_name: patterns[team_name]}

        return patterns

    def get_player_trajectories(
        self,
        player_name: Optional[str] = None,
        map_name: Optional[str] = None,
        limit: int = 100,
    ) -> list:
        """Get player movement trajectory samples."""
        trajectories = self._load_json("position_trajectories.json")

        if not trajectories:
            return []

        result = trajectories

        if isinstance(result, list):
            if player_name:
                result = [t for t in result if t.get("player") == player_name]
            if map_name:
                result = [t for t in result if t.get("map") == map_name]
            return result[:limit]

        return []

    def get_simulation_profiles(self) -> dict:
        """Get player behavior profiles for simulation."""
        return self._load_json("simulation_profiles.json")

    def build_coaching_context(
        self,
        query: str,
        map_name: Optional[str] = None,
        team_name: Optional[str] = None,
        side: Optional[str] = None,
        include_economy: bool = True,
        include_trades: bool = True,
    ) -> str:
        """Build comprehensive context for a coaching query.

        Args:
            query: The user's question
            map_name: Optional map to focus on
            team_name: Optional team to focus on
            side: Optional side (attack/defense)
            include_economy: Whether to include economy patterns
            include_trades: Whether to include trade patterns

        Returns:
            Formatted context string for the LLM
        """
        sections = []

        # Position patterns
        if map_name or team_name:
            patterns = self.get_position_patterns(
                team_name=team_name,
                map_name=map_name,
                side=side,
            )
            if patterns and patterns != {"message": "Position patterns not loaded"}:
                sections.append(
                    f"## Position Patterns\n```json\n{json.dumps(patterns, indent=2)[:2000]}\n```"
                )

        # Economy patterns
        if include_economy and team_name:
            economy = self.get_economy_patterns(team_name=team_name)
            if economy and economy != {"message": "Economy patterns not loaded"}:
                sections.append(
                    f"## Economy Patterns\n```json\n{json.dumps(economy, indent=2)[:1000]}\n```"
                )

        # Trade patterns
        if include_trades and team_name:
            trades = self.get_trade_patterns(team_name=team_name)
            if trades and trades != {"message": "Trade patterns not loaded"}:
                sections.append(
                    f"## Trade Patterns\n```json\n{json.dumps(trades, indent=2)[:1000]}\n```"
                )

        if not sections:
            return ""

        return "\n\n".join(sections)

    def build_scouting_context(
        self,
        team_name: str,
        map_name: Optional[str] = None,
    ) -> str:
        """Build comprehensive context for a scouting report."""
        sections = []

        # Get all relevant patterns for the team
        position_patterns = self.get_position_patterns(map_name=map_name)
        economy_patterns = self.get_economy_patterns(team_name=team_name)
        trade_patterns = self.get_trade_patterns(team_name=team_name)
        profiles = self.get_simulation_profiles()

        if position_patterns:
            sections.append(f"## Position Heatmaps\n{json.dumps(position_patterns, indent=2)[:3000]}")

        if economy_patterns:
            sections.append(f"## Economy Tendencies\n{json.dumps(economy_patterns, indent=2)[:1500]}")

        if trade_patterns:
            sections.append(f"## Trade Patterns\n{json.dumps(trade_patterns, indent=2)[:1500]}")

        if profiles and team_name in profiles:
            sections.append(f"## Player Profiles\n{json.dumps(profiles[team_name], indent=2)[:2000]}")

        return "\n\n".join(sections)


# Tool handler implementations
class ToolHandler:
    """Handles tool calls from the LLM."""

    def __init__(self):
        self.context_builder = ContextBuilder()

    async def handle_tool_call(self, tool_name: str, tool_input: dict) -> dict:
        """Route and execute a tool call."""
        handlers = {
            "get_team_patterns": self._get_team_patterns,
            "get_position_heatmap": self._get_position_heatmap,
            "get_trade_patterns": self._get_trade_patterns,
            "get_economy_patterns": self._get_economy_patterns,
            "run_what_if": self._run_what_if,
        }

        handler = handlers.get(tool_name)
        if not handler:
            return {"error": f"Unknown tool: {tool_name}"}

        return await handler(tool_input)

    async def _get_team_patterns(self, params: dict) -> dict:
        """Get team position patterns."""
        return self.context_builder.get_position_patterns(
            team_name=params.get("team_name"),
            map_name=params.get("map_name"),
            side=params.get("side"),
        )

    async def _get_position_heatmap(self, params: dict) -> dict:
        """Get position frequency heatmap."""
        return self.context_builder.get_position_patterns(
            map_name=params.get("map_name"),
            side=params.get("side"),
        )

    async def _get_trade_patterns(self, params: dict) -> dict:
        """Get trade timing patterns."""
        return self.context_builder.get_trade_patterns(
            team_name=params.get("team_name"),
            map_name=params.get("map_name"),
        )

    async def _get_economy_patterns(self, params: dict) -> dict:
        """Get economy decision patterns."""
        return self.context_builder.get_economy_patterns(
            team_name=params.get("team_name"),
        )

    async def _run_what_if(self, params: dict) -> dict:
        """Run a what-if simulation with the actual engine."""
        from ..simulation_engine import SimulationEngine
        from ...schemas.simulations import WhatIfScenario

        scenario_desc = params.get('scenario_description', '')
        map_name = params.get('map_name', 'ascent')
        attack_team = params.get('attack_team', 'cloud9')
        defense_team = params.get('defense_team', 'g2')

        try:
            # Create and run a quick simulation
            engine = SimulationEngine(db=None)

            class _MockSession:
                def __init__(self):
                    self.id = 'whatif_tool'
                    self.attack_team_id = attack_team
                    self.defense_team_id = defense_team
                    self.map_name = map_name
                    self.round_type = 'full'
                    self.status = 'created'
                    self.current_time_ms = 0
                    self.phase = 'opening'

            session = _MockSession()
            state = await engine.initialize(session, round_type='full')

            # Run to completion
            for _ in range(200):
                state = await engine.advance(session, 1)
                attack_alive = sum(1 for p in state.positions if p.side == 'attack' and p.is_alive)
                defense_alive = sum(1 for p in state.positions if p.side == 'defense' and p.is_alive)
                if attack_alive == 0 or defense_alive == 0 or state.current_time_ms >= 100000:
                    break

            winner = 'attack' if defense_alive == 0 else 'defense'
            kills = [e for e in state.events if hasattr(e, 'event_type') and e.event_type == 'kill']

            return {
                "scenario": scenario_desc,
                "map": map_name,
                "attack_team": attack_team,
                "defense_team": defense_team,
                "winner": winner,
                "attack_alive": attack_alive,
                "defense_alive": defense_alive,
                "duration_ms": state.current_time_ms,
                "spike_planted": state.spike_planted,
                "total_kills": len(kills),
                "status": "completed",
            }
        except Exception as e:
            return {
                "scenario": scenario_desc,
                "error": str(e),
                "status": "error",
            }
