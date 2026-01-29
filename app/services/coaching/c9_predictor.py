"""C9 action predictor - predicts what Cloud9 would do in a given situation."""

from typing import Optional
from dataclasses import dataclass
import json

from ..llm import get_anthropic_client, ContextBuilder, get_coaching_prompt


@dataclass
class C9Prediction:
    """A prediction of Cloud9's likely action."""
    primary_action: str
    confidence: float  # 0-1
    alternatives: list[dict]  # [{action, confidence}]
    key_player: str
    reasoning: str
    timing_expectation: str
    map_name: Optional[str] = None
    side: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "primary_action": self.primary_action,
            "confidence": self.confidence,
            "alternatives": self.alternatives,
            "key_player": self.key_player,
            "reasoning": self.reasoning,
            "timing_expectation": self.timing_expectation,
            "map_name": self.map_name,
            "side": self.side,
        }


class C9Predictor:
    """Predicts Cloud9's likely actions based on their patterns."""

    def __init__(self):
        self.client = get_anthropic_client()
        self.context_builder = ContextBuilder()

    async def predict_action(
        self,
        map_name: str,
        side: str,
        phase: str,
        game_state: dict,
        opponent_info: Optional[dict] = None,
    ) -> C9Prediction:
        """Predict what Cloud9 would do in this situation.

        Args:
            map_name: Current map
            side: 'attack' or 'defense'
            phase: Game phase (opening, mid_round, post_plant, retake)
            game_state: Current state (economy, alive players, etc.)
            opponent_info: Optional info about opponent tendencies

        Returns:
            C9Prediction with likely action and alternatives
        """
        # Get C9's historical patterns
        context = self.context_builder.build_coaching_context(
            query="C9 tendencies",
            map_name=map_name,
            team_name="cloud9",
            side=side,
        )

        prompt = f"""Given Cloud9's known patterns and this game state, predict their most likely action:

Map: {map_name}
Side: {side}
Phase: {phase}

Game State:
{json.dumps(game_state, indent=2)}

{f"Opponent Info: {json.dumps(opponent_info)}" if opponent_info else ""}

Cloud9 Historical Patterns:
{context}

Provide your prediction as JSON:
{{
    "primary_action": "Most likely action",
    "confidence": 0.0-1.0,
    "alternatives": [
        {{"action": "Alternative 1", "confidence": 0.X}},
        {{"action": "Alternative 2", "confidence": 0.X}}
    ],
    "key_player": "Player most likely to initiate",
    "reasoning": "Why this prediction based on C9's patterns",
    "timing_expectation": "When to expect this (e.g., '10-15s into round')"
}}"""

        response = await self.client.chat(
            messages=[{"role": "user", "content": prompt}],
            system=get_coaching_prompt("c9_predictor"),
            max_tokens=800,
        )

        try:
            result = json.loads(response.content[0].text)
        except json.JSONDecodeError:
            # Parse from natural language
            text = response.content[0].text
            result = {
                "primary_action": text[:100],
                "confidence": 0.6,
                "alternatives": [],
                "key_player": "Unknown",
                "reasoning": text,
                "timing_expectation": "Variable",
            }

        return C9Prediction(
            primary_action=result.get("primary_action", "Unknown"),
            confidence=min(1.0, max(0.0, result.get("confidence", 0.5))),
            alternatives=result.get("alternatives", []),
            key_player=result.get("key_player", "Unknown"),
            reasoning=result.get("reasoning", ""),
            timing_expectation=result.get("timing_expectation", "Variable"),
            map_name=map_name,
            side=side,
        )

    async def predict_opening(
        self,
        map_name: str,
        side: str,
        round_type: str,  # pistol, eco, force, full_buy
        opponent_team: Optional[str] = None,
    ) -> C9Prediction:
        """Predict C9's opening setup/execute for a round.

        This is a specialized prediction for round starts.
        """
        game_state = {
            "round_type": round_type,
            "phase": "opening",
            "players_alive": 5,
        }

        opponent_info = None
        if opponent_team:
            opponent_context = self.context_builder.get_position_patterns(
                team_name=opponent_team,
                map_name=map_name,
                side="defense" if side == "attack" else "attack",
            )
            opponent_info = {"team": opponent_team, "patterns": opponent_context}

        return await self.predict_action(
            map_name=map_name,
            side=side,
            phase="opening",
            game_state=game_state,
            opponent_info=opponent_info,
        )

    async def predict_post_plant(
        self,
        map_name: str,
        site: str,  # A or B
        attackers_alive: int,
        defenders_alive: int,
        time_remaining: int,
    ) -> C9Prediction:
        """Predict C9's post-plant positioning and retake hold."""
        game_state = {
            "phase": "post_plant",
            "site": site,
            "attackers_alive": attackers_alive,
            "defenders_alive": defenders_alive,
            "time_remaining_ms": time_remaining,
        }

        return await self.predict_action(
            map_name=map_name,
            side="attack",
            phase="post_plant",
            game_state=game_state,
        )

    async def counter_prediction(
        self,
        prediction: C9Prediction,
        your_team_side: str,
    ) -> str:
        """Get a counter-strategy against a C9 prediction."""
        prompt = f"""Cloud9 is predicted to: {prediction.primary_action}
Confidence: {prediction.confidence:.0%}
On {prediction.map_name} as {prediction.side}

Reasoning: {prediction.reasoning}

You are playing {your_team_side}. How should you counter this?
Be specific about positions and timing."""

        response = await self.client.chat(
            messages=[{"role": "user", "content": prompt}],
            system="You are a VALORANT counter-strategist. Provide specific, actionable counters.",
            max_tokens=400,
        )

        return response.content[0].text


# Singleton
_predictor: Optional[C9Predictor] = None


def get_c9_predictor() -> C9Predictor:
    global _predictor
    if _predictor is None:
        _predictor = C9Predictor()
    return _predictor
