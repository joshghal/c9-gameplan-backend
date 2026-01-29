"""Mistake analyzer with gravity scoring."""

from typing import Optional
from dataclasses import dataclass
from enum import Enum
import json

from ..llm import get_anthropic_client, get_coaching_prompt


class MistakeCategory(str, Enum):
    """Categories of mistakes."""
    POSITIONING = "positioning"
    TIMING = "timing"
    UTILITY = "utility"
    ECONOMY = "economy"
    COMMUNICATION = "communication"
    AIM = "aim"
    ROTATION = "rotation"
    TRADE = "trade"


class GravityLevel(str, Enum):
    """Mistake gravity levels."""
    CRITICAL = "critical"      # 8-10
    MAJOR = "major"            # 5-7
    MINOR = "minor"            # 2-4
    NEGLIGIBLE = "negligible"  # 1


@dataclass
class MistakeAnalysis:
    """Analysis of a single mistake."""
    description: str
    category: MistakeCategory
    gravity_score: int  # 1-10
    gravity_level: GravityLevel
    correct_play: str
    impact_explanation: str
    is_pattern: bool = False
    player_id: Optional[str] = None
    timestamp_ms: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "description": self.description,
            "category": self.category.value,
            "gravity_score": self.gravity_score,
            "gravity_level": self.gravity_level.value,
            "correct_play": self.correct_play,
            "impact_explanation": self.impact_explanation,
            "is_pattern": self.is_pattern,
            "player_id": self.player_id,
            "timestamp_ms": self.timestamp_ms,
        }


@dataclass
class RoundAnalysis:
    """Analysis of all mistakes in a round."""
    round_number: int
    total_mistakes: int
    total_gravity: int
    average_gravity: float
    mistakes: list[MistakeAnalysis]
    round_impact: str  # How mistakes affected round outcome

    def to_dict(self) -> dict:
        return {
            "round_number": self.round_number,
            "total_mistakes": self.total_mistakes,
            "total_gravity": self.total_gravity,
            "average_gravity": self.average_gravity,
            "mistakes": [m.to_dict() for m in self.mistakes],
            "round_impact": self.round_impact,
        }


class MistakeAnalyzer:
    """Analyzes gameplay for mistakes and assigns gravity scores."""

    def __init__(self):
        self.client = get_anthropic_client()

    def _score_to_level(self, score: int) -> GravityLevel:
        """Convert numeric score to gravity level."""
        if score >= 8:
            return GravityLevel.CRITICAL
        elif score >= 5:
            return GravityLevel.MAJOR
        elif score >= 2:
            return GravityLevel.MINOR
        else:
            return GravityLevel.NEGLIGIBLE

    async def analyze_death(
        self,
        player_id: str,
        death_context: dict,
        game_state: dict,
    ) -> MistakeAnalysis:
        """Analyze a death event for mistakes.

        Args:
            player_id: ID of the player who died
            death_context: Context around the death (positions, abilities, etc.)
            game_state: Current game state (economy, round, etc.)

        Returns:
            MistakeAnalysis with gravity scoring
        """
        prompt = f"""Analyze this death for mistakes:

Player: {player_id}
Death Context: {json.dumps(death_context, indent=2)}
Game State: {json.dumps(game_state, indent=2)}

Determine:
1. Was this death avoidable? If so, how?
2. What mistake(s) led to this death?
3. Gravity score (1-10) with justification
4. The correct play

Respond in JSON format:
{{
    "description": "What happened",
    "category": "positioning|timing|utility|economy|communication|aim|rotation|trade",
    "gravity_score": 1-10,
    "correct_play": "What should have been done",
    "impact_explanation": "How this affected the round"
}}"""

        response = await self.client.chat(
            messages=[{"role": "user", "content": prompt}],
            system=get_coaching_prompt("mistake_analyzer"),
            max_tokens=500,
        )

        try:
            result = json.loads(response.content[0].text)
        except json.JSONDecodeError:
            # Extract from text if not valid JSON
            result = {
                "description": "Analysis parsing error - manual review needed",
                "category": "positioning",
                "gravity_score": 5,
                "correct_play": "Review manually",
                "impact_explanation": response.content[0].text[:200],
            }

        return MistakeAnalysis(
            description=result.get("description", "Unknown"),
            category=MistakeCategory(result.get("category", "positioning")),
            gravity_score=result.get("gravity_score", 5),
            gravity_level=self._score_to_level(result.get("gravity_score", 5)),
            correct_play=result.get("correct_play", "Unknown"),
            impact_explanation=result.get("impact_explanation", ""),
            player_id=player_id,
        )

    async def analyze_round(
        self,
        round_number: int,
        events: list[dict],
        positions_history: list[dict],
        outcome: str,
    ) -> RoundAnalysis:
        """Analyze an entire round for mistakes.

        Args:
            round_number: Round number being analyzed
            events: List of events (kills, plants, etc.)
            positions_history: Position snapshots through the round
            outcome: Round outcome (attack_win, defense_win)

        Returns:
            RoundAnalysis with all identified mistakes
        """
        prompt = f"""Analyze this round for mistakes:

Round: {round_number}
Outcome: {outcome}

Events:
{json.dumps(events, indent=2)[:2000]}

Position History (samples):
{json.dumps(positions_history[:5], indent=2)[:1000]}

Identify all significant mistakes. For each:
1. Describe what happened
2. Categorize (positioning/timing/utility/economy/communication/aim/rotation/trade)
3. Assign gravity score (1-10)
4. Explain the correct play

Respond with a JSON array of mistakes and overall round impact analysis."""

        response = await self.client.chat(
            messages=[{"role": "user", "content": prompt}],
            system=get_coaching_prompt("mistake_analyzer"),
            max_tokens=2000,
        )

        try:
            result = json.loads(response.content[0].text)
            mistakes_data = result.get("mistakes", [])
            round_impact = result.get("round_impact", "Analysis incomplete")
        except json.JSONDecodeError:
            mistakes_data = []
            round_impact = response.content[0].text[:500]

        mistakes = []
        for m in mistakes_data:
            try:
                mistakes.append(MistakeAnalysis(
                    description=m.get("description", "Unknown"),
                    category=MistakeCategory(m.get("category", "positioning")),
                    gravity_score=m.get("gravity_score", 5),
                    gravity_level=self._score_to_level(m.get("gravity_score", 5)),
                    correct_play=m.get("correct_play", "Unknown"),
                    impact_explanation=m.get("impact_explanation", ""),
                    player_id=m.get("player_id"),
                    timestamp_ms=m.get("timestamp_ms"),
                ))
            except Exception:
                continue

        total_gravity = sum(m.gravity_score for m in mistakes)
        avg_gravity = total_gravity / len(mistakes) if mistakes else 0

        return RoundAnalysis(
            round_number=round_number,
            total_mistakes=len(mistakes),
            total_gravity=total_gravity,
            average_gravity=round(avg_gravity, 2),
            mistakes=mistakes,
            round_impact=round_impact,
        )

    async def quick_analysis(
        self,
        situation_description: str,
    ) -> MistakeAnalysis:
        """Quick analysis of a described situation."""
        prompt = f"""Analyze this VALORANT situation for mistakes:

{situation_description}

Rate on the Mistake Gravity Index (1-10) and explain."""

        response = await self.client.chat(
            messages=[{"role": "user", "content": prompt}],
            system=get_coaching_prompt("mistake_analyzer"),
            max_tokens=400,
        )

        text = response.content[0].text

        # Extract score from response
        import re
        score_match = re.search(r'(\d+)/10|score[:\s]+(\d+)|gravity[:\s]+(\d+)', text.lower())
        gravity_score = 5
        if score_match:
            gravity_score = int(next(g for g in score_match.groups() if g))

        return MistakeAnalysis(
            description=situation_description,
            category=MistakeCategory.POSITIONING,
            gravity_score=gravity_score,
            gravity_level=self._score_to_level(gravity_score),
            correct_play=text,
            impact_explanation="",
        )


# Singleton
_analyzer: Optional[MistakeAnalyzer] = None


def get_mistake_analyzer() -> MistakeAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = MistakeAnalyzer()
    return _analyzer
