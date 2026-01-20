"""
AI Decision System - Utility-Based AI with Monte Carlo Evaluation

This system uses proper AI techniques instead of hardcoded if-else rules:

1. UTILITY AI SYSTEM
   - Each action has a utility score calculated from multiple "considerations"
   - Considerations use response curves for non-linear mappings
   - Actions are selected probabilistically using softmax

2. RESPONSE CURVES
   - Linear: y = mx + b
   - Exponential: y = a * e^(bx)
   - Logistic (S-curve): y = 1 / (1 + e^(-k(x-x0)))
   - Quadratic: y = ax^2 + bx + c

3. MONTE CARLO EVALUATION
   - For strategic decisions (execute, rotate), simulate multiple outcomes
   - Weight outcomes by probability to get expected utility

4. SOFTMAX ACTION SELECTION
   - Convert utilities to probabilities: P(a) = e^(U(a)/T) / Σe^(U(i)/T)
   - Temperature T controls randomness (lower = more deterministic)

Key Principles:
- Decisions based on PlayerKnowledge, not omniscient state
- Risk assessment through utility calculations
- Emergent behavior from utility maximization
- Realistic variance through probabilistic selection
"""

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
from enum import Enum

from .information_system import (
    PlayerKnowledge, EnemyInfo, InfoConfidence, InformationManager
)


# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================

class Decision(Enum):
    """Possible decisions a player can make."""
    HOLD = "hold"
    ADVANCE = "advance"
    RETREAT = "retreat"
    EXECUTE = "execute"
    PLANT = "plant"
    DEFUSE = "defuse"
    ROTATE = "rotate"
    PEEK = "peek"
    TRADE = "trade"
    LURK = "lurk"
    FLASH_PUSH = "flash_push"
    SMOKE_SITE = "smoke_site"


@dataclass
class DecisionContext:
    """Context for making a decision - represents the player's world view."""
    player_id: str
    team: str
    role: str  # 'entry', 'support', 'controller', 'sentinel', 'lurk'
    position: Tuple[float, float]
    time_ms: int
    round_time_ms: int

    # Player state
    has_spike: bool
    is_alive: bool
    health: int
    armor: int
    has_flash: bool
    has_smoke: bool
    has_molly: bool
    has_util: bool

    # Knowledge
    knowledge: PlayerKnowledge

    # Team state (known)
    teammates_alive: int
    teammates_positions: Dict[str, Tuple[float, float]]

    # Spike state (if known)
    spike_planted: bool
    spike_site: Optional[str]
    spike_time_remaining_ms: Optional[int]

    # Site info
    target_site: Optional[str]
    site_position: Optional[Tuple[float, float]]
    site_radius: float = 0.08


@dataclass
class DecisionResult:
    """Result of AI decision making."""
    decision: Decision
    confidence: float  # 0.0 to 1.0
    utility: float  # Raw utility score
    target_position: Optional[Tuple[float, float]] = None
    reasoning: str = ""


# =============================================================================
# RESPONSE CURVES - Non-linear utility mappings
# =============================================================================

class ResponseCurve:
    """
    Response curves map input values to utility contributions.

    These provide non-linear relationships that better model
    real decision-making than linear if-else thresholds.
    """

    @staticmethod
    def linear(x: float, slope: float = 1.0, intercept: float = 0.0,
               x_min: float = 0.0, x_max: float = 1.0) -> float:
        """Linear response: y = slope * x + intercept"""
        x_norm = (x - x_min) / (x_max - x_min) if x_max != x_min else 0
        return max(0, min(1, slope * x_norm + intercept))

    @staticmethod
    def exponential(x: float, base: float = 2.0,
                    x_min: float = 0.0, x_max: float = 1.0) -> float:
        """Exponential response: accelerating increase"""
        x_norm = (x - x_min) / (x_max - x_min) if x_max != x_min else 0
        x_norm = max(0, min(1, x_norm))
        return (base ** x_norm - 1) / (base - 1)

    @staticmethod
    def logarithmic(x: float, steepness: float = 5.0,
                    x_min: float = 0.0, x_max: float = 1.0) -> float:
        """Logarithmic response: diminishing returns"""
        x_norm = (x - x_min) / (x_max - x_min) if x_max != x_min else 0
        x_norm = max(0.001, min(1, x_norm))  # Avoid log(0)
        return math.log(1 + steepness * x_norm) / math.log(1 + steepness)

    @staticmethod
    def sigmoid(x: float, k: float = 10.0, x0: float = 0.5,
                x_min: float = 0.0, x_max: float = 1.0) -> float:
        """S-curve response: threshold with smooth transition"""
        x_norm = (x - x_min) / (x_max - x_min) if x_max != x_min else 0
        return 1 / (1 + math.exp(-k * (x_norm - x0)))

    @staticmethod
    def quadratic(x: float, a: float = 1.0, peak: float = 0.5,
                  x_min: float = 0.0, x_max: float = 1.0) -> float:
        """Quadratic response: peaks at specified point"""
        x_norm = (x - x_min) / (x_max - x_min) if x_max != x_min else 0
        return max(0, 1 - a * (x_norm - peak) ** 2)

    @staticmethod
    def inverse(x: float, x_min: float = 0.0, x_max: float = 1.0) -> float:
        """Inverse response: high when input is low"""
        x_norm = (x - x_min) / (x_max - x_min) if x_max != x_min else 0
        return max(0, min(1, 1 - x_norm))

    @staticmethod
    def step(x: float, threshold: float = 0.5,
             smoothness: float = 20.0) -> float:
        """Smooth step function at threshold"""
        return 1 / (1 + math.exp(-smoothness * (x - threshold)))


# =============================================================================
# CONSIDERATION - A single factor in utility calculation
# =============================================================================

@dataclass
class Consideration:
    """
    A single consideration in utility calculation.

    Considerations are the building blocks of Utility AI.
    Each consideration evaluates one aspect of the situation
    and returns a utility score [0, 1].

    Example: "time_pressure" consideration for planting
    - Input: time_remaining_ratio (0 = no time, 1 = full time)
    - Curve: exponential (urgency increases rapidly late round)
    - Weight: 0.3 (30% of total plant utility)
    """
    name: str
    input_func: Callable[[DecisionContext], float]  # Returns raw input value
    curve: Callable[[float], float]  # Response curve
    weight: float = 1.0  # Importance weight

    def evaluate(self, ctx: DecisionContext) -> float:
        """Evaluate this consideration for the given context."""
        raw_input = self.input_func(ctx)
        curved = self.curve(raw_input)
        return curved * self.weight


# =============================================================================
# UTILITY REASONER - Core AI decision system
# =============================================================================

class UtilityReasoner:
    """
    Utility AI System for tactical decisions.

    Each action has a set of considerations. The action with
    the highest total utility is selected (with some randomness
    via softmax selection).

    This replaces hardcoded if-else with data-driven utility curves.
    """

    # Timing constants
    ROUND_TIME_MS = 100000
    SPIKE_TIME_MS = 45000
    DEFUSE_TIME_MS = 7000
    PLANT_TIME_MS = 4000

    # Softmax temperature (lower = more deterministic)
    TEMPERATURE = 0.15

    def __init__(self):
        # Initialize consideration sets for each action
        self.action_considerations: Dict[Decision, List[Consideration]] = {
            Decision.HOLD: self._build_hold_considerations(),
            Decision.ADVANCE: self._build_advance_considerations(),
            Decision.RETREAT: self._build_retreat_considerations(),
            Decision.PLANT: self._build_plant_considerations(),
            Decision.DEFUSE: self._build_defuse_considerations(),
            Decision.PEEK: self._build_peek_considerations(),
            Decision.ROTATE: self._build_rotate_considerations(),
            Decision.TRADE: self._build_trade_considerations(),
            Decision.LURK: self._build_lurk_considerations(),
            Decision.EXECUTE: self._build_execute_considerations(),
        }

        # Base utilities (action tendencies)
        # TUNED: Increased combat-oriented utilities to encourage more fights
        self.base_utilities: Dict[Decision, float] = {
            Decision.HOLD: 0.35,      # Holding angles is common
            Decision.ADVANCE: 0.30,   # TUNED: Higher - push for map control
            Decision.RETREAT: 0.1,
            Decision.PLANT: 0.0,      # Only high if at site with spike
            Decision.DEFUSE: 0.0,     # Only high if at spike
            Decision.PEEK: 0.35,      # TUNED: Higher - take fights
            Decision.ROTATE: 0.15,
            Decision.TRADE: 0.25,     # TUNED: Higher - follow up on teammate fights
            Decision.LURK: 0.15,
            Decision.EXECUTE: 0.0,    # Team-level decision
        }

    # =========================================================================
    # CONSIDERATION BUILDERS
    # =========================================================================

    def _build_hold_considerations(self) -> List[Consideration]:
        """Considerations for HOLD action."""
        return [
            # Good health = more willing to hold
            Consideration(
                name="health_confidence",
                input_func=lambda ctx: ctx.health / 100.0,
                curve=lambda x: ResponseCurve.linear(x, slope=0.5, intercept=0.25),
                weight=0.2
            ),
            # Unknown enemies = more cautious (hold)
            Consideration(
                name="uncertainty_caution",
                input_func=lambda ctx: self._get_uncertainty(ctx),
                curve=lambda x: ResponseCurve.sigmoid(x, k=8, x0=0.4),
                weight=0.3
            ),
            # Site anchor role = prefer hold
            Consideration(
                name="anchor_role",
                input_func=lambda ctx: 1.0 if ctx.role == 'sentinel' else 0.0,
                curve=lambda x: ResponseCurve.linear(x),
                weight=0.3
            ),
            # Post-plant attackers prefer hold
            Consideration(
                name="post_plant_hold",
                input_func=lambda ctx: 1.0 if ctx.spike_planted and ctx.team == 'attack' else 0.0,
                curve=lambda x: ResponseCurve.linear(x),
                weight=0.4
            ),
        ]

    def _build_advance_considerations(self) -> List[Consideration]:
        """Considerations for ADVANCE action.

        TUNED: More aggressive advancement for map control and fights.
        """
        return [
            # Time pressure increases advance urgency
            # TUNED: Lower base so early aggression happens
            Consideration(
                name="time_pressure",
                input_func=lambda ctx: ctx.time_ms / self.ROUND_TIME_MS,
                curve=lambda x: ResponseCurve.logarithmic(x, steepness=2.5),  # Front-loaded
                weight=0.30
            ),
            # Entry role prefers advancing
            Consideration(
                name="entry_aggression",
                input_func=lambda ctx: 1.0 if ctx.role == 'entry' else 0.4,
                curve=lambda x: ResponseCurve.linear(x),
                weight=0.25  # TUNED: Higher weight for role
            ),
            # Man advantage encourages push
            Consideration(
                name="man_advantage",
                input_func=lambda ctx: self._get_man_advantage(ctx),
                curve=lambda x: ResponseCurve.sigmoid(x, k=4, x0=0.45),  # Easier trigger
                weight=0.25
            ),
            # Low threat ahead = advance (but also advance to contest)
            Consideration(
                name="path_safety",
                input_func=lambda ctx: 1.0 - self._get_path_threat(ctx) * 0.5,  # Reduced threat penalty
                curve=lambda x: ResponseCurve.logarithmic(x, steepness=2),
                weight=0.20
            ),
        ]

    def _build_retreat_considerations(self) -> List[Consideration]:
        """Considerations for RETREAT action."""
        return [
            # Low health = retreat
            Consideration(
                name="low_health",
                input_func=lambda ctx: 1.0 - ctx.health / 100.0,
                curve=lambda x: ResponseCurve.exponential(x, base=4.0),
                weight=0.4
            ),
            # Outnumbered = retreat
            Consideration(
                name="outnumbered",
                input_func=lambda ctx: max(0, -self._get_man_advantage(ctx)),
                curve=lambda x: ResponseCurve.sigmoid(x, k=5, x0=0.3),
                weight=0.3
            ),
            # High nearby threat = retreat
            Consideration(
                name="high_threat",
                input_func=lambda ctx: self._get_immediate_threat(ctx),
                curve=lambda x: ResponseCurve.sigmoid(x, k=6, x0=0.6),
                weight=0.3
            ),
        ]

    def _build_plant_considerations(self) -> List[Consideration]:
        """Considerations for PLANT action.

        TUNED: Require higher site safety and more time pressure to delay planting.
        Pro teams clear site more thoroughly before planting.
        """
        return [
            # Must have spike (hard requirement)
            Consideration(
                name="has_spike",
                input_func=lambda ctx: 1.0 if ctx.has_spike else 0.0,
                curve=lambda x: x,  # Binary
                weight=10.0  # Critical weight - 0 if no spike
            ),
            # Must be at site
            Consideration(
                name="at_site",
                input_func=lambda ctx: 1.0 if self._is_at_site(ctx) else 0.0,
                curve=lambda x: x,
                weight=10.0  # Critical
            ),
            # Time pressure (TUNED: sigmoid starts later, less eager early)
            # Pro teams don't rush plant - they clear first
            Consideration(
                name="plant_urgency",
                input_func=lambda ctx: ctx.time_ms / self.ROUND_TIME_MS,
                curve=lambda x: ResponseCurve.sigmoid(x, k=6, x0=0.55),  # Kicks in around 55s
                weight=0.30
            ),
            # Site cleared = safe to plant (TUNED: higher requirement)
            # Need high confidence site is clear before planting
            Consideration(
                name="site_safety",
                input_func=lambda ctx: 1.0 - self._get_site_threat(ctx),
                curve=lambda x: ResponseCurve.sigmoid(x, k=8, x0=0.6),  # Need 60%+ safety
                weight=0.45  # Increased weight - safety is paramount
            ),
            # Teammate support present
            Consideration(
                name="teammate_support",
                input_func=lambda ctx: min(ctx.teammates_alive - 1, 3) / 3.0,
                curve=lambda x: ResponseCurve.logarithmic(x, steepness=2),
                weight=0.15
            ),
        ]

    def _build_defuse_considerations(self) -> List[Consideration]:
        """Considerations for DEFUSE action."""
        return [
            # Must be at spike (hard requirement)
            Consideration(
                name="at_spike",
                input_func=lambda ctx: 1.0 if self._is_at_spike(ctx) else 0.0,
                curve=lambda x: x,
                weight=10.0
            ),
            # Time pressure (must defuse)
            Consideration(
                name="defuse_urgency",
                input_func=lambda ctx: self._get_defuse_urgency(ctx),
                curve=lambda x: ResponseCurve.exponential(x, base=5.0),
                weight=0.4
            ),
            # Area cleared = safe to defuse
            Consideration(
                name="area_safety",
                input_func=lambda ctx: 1.0 - self._get_immediate_threat(ctx),
                curve=lambda x: ResponseCurve.sigmoid(x, k=4, x0=0.5),
                weight=0.3
            ),
            # Teammates covering
            Consideration(
                name="cover_available",
                input_func=lambda ctx: min(ctx.teammates_alive - 1, 2) / 2.0,
                curve=lambda x: ResponseCurve.linear(x, slope=0.8, intercept=0.2),
                weight=0.2
            ),
        ]

    def _build_peek_considerations(self) -> List[Consideration]:
        """Considerations for PEEK action.

        TUNED: Make players more willing to take fights, especially defenders
        contesting map control.
        """
        return [
            # Known enemy nearby = peek them
            # TUNED: Lower peak value makes peeking more likely at various ranges
            Consideration(
                name="enemy_nearby",
                input_func=lambda ctx: self._get_nearest_enemy_proximity(ctx),
                curve=lambda x: ResponseCurve.quadratic(x, a=1.5, peak=0.5),  # Lower peak
                weight=0.40  # Higher weight
            ),
            # Good health = more willing to peek
            Consideration(
                name="peek_confidence",
                input_func=lambda ctx: ctx.health / 100.0,
                curve=lambda x: ResponseCurve.sigmoid(x, k=5, x0=0.35),  # Lower threshold
                weight=0.20
            ),
            # Duelist/Entry role = more peek-oriented
            Consideration(
                name="duelist_aggression",
                input_func=lambda ctx: 1.0 if ctx.role == 'entry' else 0.5,  # Higher base
                curve=lambda x: ResponseCurve.linear(x),
                weight=0.20
            ),
            # Has flash = peek is more viable
            Consideration(
                name="has_flash_advantage",
                input_func=lambda ctx: 1.0 if ctx.has_flash else 0.3,  # Base utility even without flash
                curve=lambda x: ResponseCurve.linear(x),
                weight=0.15
            ),
            # Defender aggression - defenders should contest, not just hold
            Consideration(
                name="defender_contest",
                input_func=lambda ctx: 1.0 if ctx.team == 'defense' and not ctx.spike_planted else 0.0,
                curve=lambda x: ResponseCurve.linear(x, slope=0.6, intercept=0.1),
                weight=0.25  # Encourage defenders to fight
            ),
        ]

    def _build_rotate_considerations(self) -> List[Consideration]:
        """Considerations for ROTATE action."""
        return [
            # Spike planted elsewhere = rotate
            Consideration(
                name="spike_elsewhere",
                input_func=lambda ctx: self._spike_planted_elsewhere(ctx),
                curve=lambda x: ResponseCurve.step(x, threshold=0.5),
                weight=0.6
            ),
            # Rotator role
            Consideration(
                name="rotator_role",
                input_func=lambda ctx: 1.0 if ctx.role == 'rotator' else 0.3,
                curve=lambda x: ResponseCurve.linear(x),
                weight=0.2
            ),
            # No action at current position
            Consideration(
                name="idle_at_position",
                input_func=lambda ctx: 1.0 - self._get_position_value(ctx),
                curve=lambda x: ResponseCurve.logarithmic(x, steepness=2),
                weight=0.2
            ),
        ]

    def _build_trade_considerations(self) -> List[Consideration]:
        """Considerations for TRADE action (follow up teammate's fight)."""
        return [
            # Teammate in fight nearby
            Consideration(
                name="teammate_engaged",
                input_func=lambda ctx: self._teammate_fighting_nearby(ctx),
                curve=lambda x: ResponseCurve.step(x, threshold=0.3),
                weight=0.5
            ),
            # Support role
            Consideration(
                name="support_role",
                input_func=lambda ctx: 1.0 if ctx.role == 'support' else 0.3,
                curve=lambda x: ResponseCurve.linear(x),
                weight=0.3
            ),
            # Position allows trade
            Consideration(
                name="trade_position",
                input_func=lambda ctx: self._can_trade(ctx),
                curve=lambda x: ResponseCurve.linear(x),
                weight=0.2
            ),
        ]

    def _build_lurk_considerations(self) -> List[Consideration]:
        """Considerations for LURK action (play slow, catch rotates)."""
        return [
            # Lurker role
            Consideration(
                name="lurker_role",
                input_func=lambda ctx: 1.0 if ctx.role == 'lurk' else 0.0,
                curve=lambda x: ResponseCurve.linear(x),
                weight=0.5
            ),
            # Early round = good lurk time
            Consideration(
                name="early_round",
                input_func=lambda ctx: 1.0 - ctx.time_ms / self.ROUND_TIME_MS,
                curve=lambda x: ResponseCurve.sigmoid(x, k=5, x0=0.5),
                weight=0.3
            ),
            # Not needed at site
            Consideration(
                name="not_needed",
                input_func=lambda ctx: 1.0 if ctx.teammates_alive >= 3 else 0.3,
                curve=lambda x: ResponseCurve.linear(x),
                weight=0.2
            ),
        ]

    def _build_execute_considerations(self) -> List[Consideration]:
        """Considerations for EXECUTE action (team commit to site)."""
        return [
            # Time pressure
            Consideration(
                name="execute_urgency",
                input_func=lambda ctx: ctx.time_ms / self.ROUND_TIME_MS,
                curve=lambda x: ResponseCurve.exponential(x, base=3.5),
                weight=0.35
            ),
            # Info quality (know where enemies are)
            Consideration(
                name="info_quality",
                input_func=lambda ctx: self._get_info_quality(ctx),
                curve=lambda x: ResponseCurve.sigmoid(x, k=4, x0=0.4),
                weight=0.25
            ),
            # Man advantage
            Consideration(
                name="numbers_advantage",
                input_func=lambda ctx: (self._get_man_advantage(ctx) + 2) / 4,
                curve=lambda x: ResponseCurve.sigmoid(x, k=5, x0=0.5),
                weight=0.25
            ),
            # Utility available
            Consideration(
                name="util_available",
                input_func=lambda ctx: 1.0 if ctx.has_util else 0.3,
                curve=lambda x: ResponseCurve.linear(x),
                weight=0.15
            ),
        ]

    # =========================================================================
    # INPUT HELPER FUNCTIONS
    # =========================================================================

    def _get_uncertainty(self, ctx: DecisionContext) -> float:
        """How uncertain is the player about enemy positions?"""
        known = ctx.knowledge.get_known_enemy_count()
        dead = len(ctx.knowledge.confirmed_dead)
        unknown = 5 - known - dead
        return unknown / 5.0

    def _get_man_advantage(self, ctx: DecisionContext) -> float:
        """Calculate man advantage (-1 to 1 normalized)."""
        dead_enemies = len(ctx.knowledge.confirmed_dead)
        enemies_alive = 5 - dead_enemies
        advantage = ctx.teammates_alive - enemies_alive
        return (advantage + 4) / 8.0  # Normalize -4 to +4 → 0 to 1

    def _get_path_threat(self, ctx: DecisionContext) -> float:
        """Threat level along path to objective."""
        if ctx.site_position:
            return ctx.knowledge.get_site_threat_level(
                ctx.site_position[0], ctx.site_position[1],
                ctx.site_radius * 2, ctx.time_ms
            )
        return 0.5

    def _get_immediate_threat(self, ctx: DecisionContext) -> float:
        """Immediate threat at current position."""
        nearby = ctx.knowledge.get_enemies_near(
            ctx.position[0], ctx.position[1], 0.15, ctx.time_ms
        )
        if not nearby:
            return 0.1
        # More enemies = higher threat, decays with distance
        threat = 0
        for enemy in nearby:
            dist = math.sqrt(
                (enemy.last_known_x - ctx.position[0])**2 +
                (enemy.last_known_y - ctx.position[1])**2
            )
            threat += 0.4 * (1 - dist / 0.15)
        return min(1.0, threat)

    def _get_site_threat(self, ctx: DecisionContext) -> float:
        """Threat at target site."""
        if ctx.site_position:
            return ctx.knowledge.get_site_threat_level(
                ctx.site_position[0], ctx.site_position[1],
                ctx.site_radius, ctx.time_ms
            )
        return 0.5

    def _is_at_site(self, ctx: DecisionContext) -> bool:
        """Is player at the target site?"""
        if not ctx.site_position:
            return False
        dist = math.sqrt(
            (ctx.position[0] - ctx.site_position[0])**2 +
            (ctx.position[1] - ctx.site_position[1])**2
        )
        return dist < ctx.site_radius

    def _is_at_spike(self, ctx: DecisionContext) -> bool:
        """Is player at the planted spike?"""
        if not ctx.spike_planted or not ctx.site_position:
            return False
        dist = math.sqrt(
            (ctx.position[0] - ctx.site_position[0])**2 +
            (ctx.position[1] - ctx.site_position[1])**2
        )
        return dist < 0.10

    def _get_defuse_urgency(self, ctx: DecisionContext) -> float:
        """Urgency to defuse based on time remaining."""
        if not ctx.spike_time_remaining_ms:
            return 0.0
        # Urgency increases as spike time decreases
        time_ratio = 1.0 - ctx.spike_time_remaining_ms / self.SPIKE_TIME_MS
        return time_ratio

    def _get_nearest_enemy_proximity(self, ctx: DecisionContext) -> float:
        """Proximity to nearest known enemy (0 = far, 1 = very close)."""
        nearby = ctx.knowledge.get_enemies_near(
            ctx.position[0], ctx.position[1], 0.30, ctx.time_ms
        )
        if not nearby:
            return 0.0
        closest_dist = float('inf')
        for enemy in nearby:
            dist = math.sqrt(
                (enemy.last_known_x - ctx.position[0])**2 +
                (enemy.last_known_y - ctx.position[1])**2
            )
            closest_dist = min(closest_dist, dist)
        return max(0, 1 - closest_dist / 0.30)

    def _spike_planted_elsewhere(self, ctx: DecisionContext) -> float:
        """Is spike planted at a different site than where player is?"""
        if not ctx.spike_planted or not ctx.knowledge.knows_spike_planted:
            return 0.0
        if ctx.site_position:
            dist = math.sqrt(
                (ctx.position[0] - ctx.site_position[0])**2 +
                (ctx.position[1] - ctx.site_position[1])**2
            )
            return 1.0 if dist > ctx.site_radius * 2 else 0.0
        return 0.5

    def _get_position_value(self, ctx: DecisionContext) -> float:
        """How valuable is current position?"""
        # Anchoring site = high value
        if self._is_at_site(ctx) and ctx.role == 'sentinel':
            return 0.8
        # Near known enemy = high value
        if self._get_nearest_enemy_proximity(ctx) > 0.5:
            return 0.6
        return 0.3

    def _teammate_fighting_nearby(self, ctx: DecisionContext) -> float:
        """Is a teammate engaged in a fight nearby?"""
        # This would need event data - simplified version
        for tid, tpos in ctx.teammates_positions.items():
            dist_to_teammate = math.sqrt(
                (ctx.position[0] - tpos[0])**2 +
                (ctx.position[1] - tpos[1])**2
            )
            if dist_to_teammate < 0.20:
                # Check if there's an enemy near that teammate
                enemies_near_teammate = ctx.knowledge.get_enemies_near(
                    tpos[0], tpos[1], 0.15, ctx.time_ms
                )
                if enemies_near_teammate:
                    return 1.0
        return 0.0

    def _can_trade(self, ctx: DecisionContext) -> float:
        """Can player trade if teammate dies?"""
        # Simplified - check if positioned behind teammates
        if len(ctx.teammates_positions) == 0:
            return 0.0
        # Just return whether there are teammates ahead
        return 0.5 if ctx.teammates_alive > 1 else 0.0

    def _get_info_quality(self, ctx: DecisionContext) -> float:
        """Quality of information about enemy positions."""
        known = ctx.knowledge.get_known_enemy_count()
        dead = len(ctx.knowledge.confirmed_dead)
        # Weight known + dead more heavily
        return (known + dead * 2) / 10.0

    # =========================================================================
    # UTILITY CALCULATION
    # =========================================================================

    def calculate_utility(self, action: Decision, ctx: DecisionContext) -> float:
        """
        Calculate total utility for an action.

        Utility = base_utility + sum(consideration.evaluate(ctx))

        Considerations with weight >= 10 are "critical" - if they evaluate to 0,
        the action is not viable (utility = 0).
        """
        considerations = self.action_considerations.get(action, [])
        base = self.base_utilities.get(action, 0.1)

        total = base
        for cons in considerations:
            value = cons.evaluate(ctx)
            # Critical considerations (weight >= 10) gate the action
            if cons.weight >= 10.0 and value < 0.01:
                return 0.0
            total += value

        return total

    def softmax_select(self, utilities: Dict[Decision, float],
                       temperature: float = None) -> Decision:
        """
        Select action using softmax probability distribution.

        P(action) = e^(utility/T) / sum(e^(utility/T))

        Temperature controls randomness:
        - Low T (0.1): Nearly deterministic, picks highest utility
        - High T (1.0): More random, all actions more equally likely
        """
        if temperature is None:
            temperature = self.TEMPERATURE

        # Filter to viable actions (utility > 0)
        viable = {a: u for a, u in utilities.items() if u > 0}
        if not viable:
            return Decision.HOLD

        # Calculate softmax probabilities (pure Python)
        actions = list(viable.keys())
        utils = [viable[a] for a in actions]

        # Softmax with temperature - use max subtraction for numerical stability
        max_util = max(utils)
        exp_utils = [math.exp((u - max_util) / temperature) for u in utils]
        sum_exp = sum(exp_utils)
        probs = [e / sum_exp for e in exp_utils]

        # Sample from distribution using inverse transform sampling
        r = random.random()
        cumulative = 0.0
        for i, p in enumerate(probs):
            cumulative += p
            if r <= cumulative:
                return actions[i]

        # Fallback to last action (shouldn't normally reach here)
        return actions[-1]

    def select_best_action(self, ctx: DecisionContext,
                           available_actions: List[Decision] = None
                           ) -> Tuple[Decision, float, Dict[Decision, float]]:
        """
        Select the best action for a context.

        Returns: (selected_action, utility, all_utilities)
        """
        if available_actions is None:
            # Filter by context
            if ctx.team == 'attack':
                if ctx.spike_planted:
                    available_actions = [Decision.HOLD, Decision.PEEK, Decision.RETREAT]
                else:
                    available_actions = [
                        Decision.HOLD, Decision.ADVANCE, Decision.RETREAT,
                        Decision.PLANT, Decision.PEEK, Decision.TRADE, Decision.LURK
                    ]
            else:
                if ctx.spike_planted:
                    available_actions = [
                        Decision.HOLD, Decision.ADVANCE, Decision.RETREAT,
                        Decision.DEFUSE, Decision.PEEK, Decision.ROTATE
                    ]
                else:
                    available_actions = [
                        Decision.HOLD, Decision.ADVANCE, Decision.RETREAT,
                        Decision.PEEK, Decision.ROTATE
                    ]

        # Calculate utilities for all available actions
        utilities = {}
        for action in available_actions:
            utilities[action] = self.calculate_utility(action, ctx)

        # Select using softmax
        selected = self.softmax_select(utilities)

        return selected, utilities[selected], utilities


# =============================================================================
# MONTE CARLO EVALUATOR - For strategic decisions
# =============================================================================

class MonteCarloEvaluator:
    """
    Monte Carlo evaluation for strategic decisions.

    For complex decisions like "should the team execute?",
    simulates multiple outcomes and evaluates expected value.
    """

    def __init__(self, num_simulations: int = 50):
        self.num_simulations = num_simulations

    def evaluate_execute(self,
                         team_contexts: List[DecisionContext],
                         site_defenders: int,
                         utility_advantage: float) -> Tuple[float, str]:
        """
        Evaluate expected outcome of executing on a site.

        Runs Monte Carlo simulations of potential outcomes.

        Returns: (expected_value, reasoning)
        """
        if not team_contexts:
            return 0.0, "No team context"

        attackers = len([c for c in team_contexts if c.is_alive])

        # Outcome weights based on situation
        wins = 0
        losses = 0

        for _ in range(self.num_simulations):
            # Simulate outcome
            outcome = self._simulate_execute_outcome(
                attackers, site_defenders, utility_advantage
            )
            if outcome > 0:
                wins += 1
            else:
                losses += 1

        win_rate = wins / self.num_simulations
        expected_value = win_rate * 2 - 1  # Map to [-1, 1]

        reasoning = f"MC eval: {wins}/{self.num_simulations} wins ({win_rate:.1%})"
        return expected_value, reasoning

    def _simulate_execute_outcome(self, attackers: int, defenders: int,
                                   utility_advantage: float) -> float:
        """
        Simulate a single execute outcome.

        Returns: positive if attackers win, negative if defenders win
        """
        # Base win probability from numbers
        base_prob = 0.5 + (attackers - defenders) * 0.1

        # Utility advantage bonus
        base_prob += utility_advantage * 0.15

        # Defensive advantage (holding angles)
        base_prob -= 0.05

        # Clamp and add randomness
        prob = max(0.1, min(0.9, base_prob))

        if random.random() < prob:
            return 1.0
        return -1.0

    def evaluate_defuse_attempt(self,
                                ctx: DecisionContext,
                                enemies_nearby: int) -> Tuple[float, str]:
        """
        Evaluate expected outcome of attempting defuse.

        Returns: (expected_value, reasoning)
        """
        if ctx.spike_time_remaining_ms is None:
            return -1.0, "No spike time info"

        time_remaining = ctx.spike_time_remaining_ms
        defuse_time = 7000  # 7 seconds

        # Can we even complete the defuse?
        can_full_defuse = time_remaining > defuse_time + 2000

        success_count = 0
        for _ in range(self.num_simulations):
            success = self._simulate_defuse_outcome(
                enemies_nearby,
                ctx.teammates_alive - 1,  # Others covering
                can_full_defuse,
                time_remaining
            )
            if success:
                success_count += 1

        success_rate = success_count / self.num_simulations
        expected_value = success_rate * 2 - 1

        reasoning = f"Defuse success: {success_count}/{self.num_simulations} ({success_rate:.1%})"
        return expected_value, reasoning

    def _simulate_defuse_outcome(self, enemies: int, cover: int,
                                  can_full: bool, time_ms: int) -> bool:
        """Simulate a single defuse attempt."""
        # Survival probability during defuse
        survival_prob = 0.8 - enemies * 0.15 + cover * 0.1

        if not can_full:
            # Half defuse is riskier
            survival_prob *= 0.7

        survival_prob = max(0.1, min(0.9, survival_prob))

        return random.random() < survival_prob


# =============================================================================
# AI DECISION SYSTEM - Main interface
# =============================================================================

class AIDecisionSystem:
    """
    Main AI decision system using Utility AI.

    This replaces the old if-else based system with:
    1. Utility-based action evaluation
    2. Response curves for non-linear factors
    3. Softmax selection for natural variance
    4. Monte Carlo evaluation for strategic decisions
    """

    ROUND_TIME_MS = 100000
    SPIKE_TIME_MS = 45000
    DEFUSE_TIME_MS = 7000
    PLANT_TIME_MS = 4000

    def __init__(self):
        self.reasoner = UtilityReasoner()
        self.mc_evaluator = MonteCarloEvaluator(num_simulations=30)

    def make_decision(self, ctx: DecisionContext) -> DecisionResult:
        """
        Make a decision for a player using Utility AI.

        This is the main entry point for AI decision making.
        """
        # Get best action from utility reasoner
        action, utility, all_utilities = self.reasoner.select_best_action(ctx)

        # Get target position based on action
        target = self._get_action_target(action, ctx)

        # Generate reasoning
        top_3 = sorted(all_utilities.items(), key=lambda x: -x[1])[:3]
        reasoning = f"Top actions: " + ", ".join(
            f"{a.value}={u:.2f}" for a, u in top_3
        )

        return DecisionResult(
            decision=action,
            confidence=min(1.0, utility / 2.0),  # Normalize utility to confidence
            utility=utility,
            target_position=target,
            reasoning=reasoning
        )

    def _get_action_target(self, action: Decision,
                           ctx: DecisionContext) -> Optional[Tuple[float, float]]:
        """Get target position for an action."""
        if action == Decision.HOLD:
            return ctx.position

        elif action in [Decision.ADVANCE, Decision.PLANT, Decision.EXECUTE]:
            return ctx.site_position

        elif action == Decision.RETREAT:
            # Move away from site
            if ctx.site_position:
                dx = ctx.position[0] - ctx.site_position[0]
                dy = ctx.position[1] - ctx.site_position[1]
                dist = math.sqrt(dx*dx + dy*dy) or 0.01
                return (
                    ctx.position[0] + dx/dist * 0.15,
                    ctx.position[1] + dy/dist * 0.15
                )
            return ctx.position

        elif action in [Decision.DEFUSE, Decision.ROTATE]:
            return ctx.site_position

        elif action == Decision.PEEK:
            # Target nearest known enemy
            nearby = ctx.knowledge.get_enemies_near(
                ctx.position[0], ctx.position[1], 0.30, ctx.time_ms
            )
            if nearby:
                closest = nearby[0]
                return (closest.last_known_x, closest.last_known_y)
            return ctx.site_position

        return None

    def should_team_execute(self, team_contexts: List[DecisionContext],
                           info_manager: InformationManager
                           ) -> Tuple[bool, Optional[str], float]:
        """
        Decide if the attack team should execute using Monte Carlo evaluation.

        Returns: (should_execute, target_site, confidence)
        """
        if not team_contexts:
            return False, None, 0.0

        ctx = team_contexts[0]
        time_pressure = ctx.time_ms / self.ROUND_TIME_MS

        # Estimate site defenders
        site_threat = 0.5
        if ctx.site_position:
            site_threat = ctx.knowledge.get_site_threat_level(
                ctx.site_position[0], ctx.site_position[1],
                ctx.site_radius * 2, ctx.time_ms
            )

        estimated_defenders = int(site_threat * 3) + 1

        # Calculate utility advantage (smokes, flashes)
        util_count = sum(1 for c in team_contexts if c.has_util)
        util_advantage = (util_count / len(team_contexts)) - 0.5

        # Monte Carlo evaluation
        expected_value, mc_reasoning = self.mc_evaluator.evaluate_execute(
            team_contexts, estimated_defenders, util_advantage
        )

        # Calculate execute utility
        execute_utilities = {}
        for c in team_contexts:
            _, util, _ = self.reasoner.select_best_action(
                c, available_actions=[Decision.EXECUTE]
            )
            execute_utilities[c.player_id] = util

        avg_execute_utility = sum(execute_utilities.values()) / len(execute_utilities)

        # Combine MC evaluation with utility
        combined_score = (expected_value + 1) / 2 * 0.4 + avg_execute_utility * 0.6

        # Time-based threshold (TUNED: higher thresholds to delay executes)
        # Pro teams typically execute around 50-60s into round, not immediately
        if time_pressure < 0.4:
            threshold = 0.85  # Very high early round - need strong reason to execute
        elif time_pressure < 0.55:
            threshold = 0.70  # Mid round - still cautious
        elif time_pressure < 0.7:
            threshold = 0.55  # Late round - start considering
        elif time_pressure < 0.85:
            threshold = 0.40  # Very late - more willing
        else:
            threshold = 0.25  # Crunch time - just go

        should_execute = combined_score > threshold
        target_site = ctx.target_site or 'A'

        return should_execute, target_site, combined_score


class AIBehaviorIntegration:
    """
    Integrates AI decisions with the simulation engine.

    Provides methods to get movement targets and actions
    based on Utility AI decisions.
    """

    def __init__(self, info_manager: InformationManager):
        self.info_manager = info_manager
        self.decision_system = AIDecisionSystem()
        self.current_decisions: Dict[str, DecisionResult] = {}

    def update_player_decision(self, ctx: DecisionContext) -> DecisionResult:
        """Update and get decision for a player."""
        decision = self.decision_system.make_decision(ctx)
        self.current_decisions[ctx.player_id] = decision
        return decision

    def get_movement_target(self, player_id: str,
                           current_pos: Tuple[float, float],
                           site_positions: Dict[str, Tuple[float, float]]
                           ) -> Optional[Tuple[float, float]]:
        """Get movement target based on current decision."""
        decision = self.current_decisions.get(player_id)
        if not decision:
            return None

        if decision.target_position:
            return decision.target_position

        # Decision-based default targets
        if decision.decision == Decision.HOLD:
            return current_pos

        elif decision.decision == Decision.RETREAT:
            return (current_pos[0] * 0.9, current_pos[1] * 1.1)

        elif decision.decision in [Decision.ADVANCE, Decision.ROTATE]:
            if site_positions:
                return list(site_positions.values())[0]

        return None

    def should_plant(self, player_id: str) -> bool:
        """Check if player should plant based on decision."""
        decision = self.current_decisions.get(player_id)
        return decision and decision.decision == Decision.PLANT

    def should_defuse(self, player_id: str) -> bool:
        """Check if player should defuse based on decision."""
        decision = self.current_decisions.get(player_id)
        return decision and decision.decision == Decision.DEFUSE

    def should_peek(self, player_id: str) -> Tuple[bool, Optional[Tuple[float, float]]]:
        """Check if player should peek based on decision."""
        decision = self.current_decisions.get(player_id)
        if decision and decision.decision == Decision.PEEK:
            return (True, decision.target_position)
        return (False, None)

    def get_decision_for_player(self, player_id: str) -> Optional[DecisionResult]:
        """Get the current decision for a player."""
        return self.current_decisions.get(player_id)
