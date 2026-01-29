"""Behavior Adaptation for VALORANT tactical simulations.

Modifies player behavior (aggression, positioning, movement speed) based on
round state including man advantage, post-plant scenarios, and trade opportunities.

Now integrates VCT phase_behaviors from behavioral_patterns.json for data-driven
phase-specific behavior modifications.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
import math
import json
from pathlib import Path

from .round_state import RoundState


# Load VCT behavioral patterns for phase-based behaviors
_VCT_PHASE_BEHAVIORS: Optional[Dict] = None


def _load_vct_phase_behaviors() -> Dict:
    """Load phase behaviors from VCT behavioral patterns JSON."""
    global _VCT_PHASE_BEHAVIORS
    if _VCT_PHASE_BEHAVIORS is not None:
        return _VCT_PHASE_BEHAVIORS

    try:
        data_path = Path(__file__).parent.parent / 'data' / 'behavioral_patterns.json'
        with open(data_path, 'r') as f:
            data = json.load(f)
        _VCT_PHASE_BEHAVIORS = data.get('phase_behaviors', {})
    except Exception:
        # Fallback to default behaviors if file not found
        _VCT_PHASE_BEHAVIORS = {
            'early': {
                'movement_speed': 0.9,
                'rotation_probability': 0.1,
                'engagement_likelihood': 0.3,
                'peek_frequency': 0.4,
                'utility_usage_rate': 0.5
            },
            'mid': {
                'movement_speed': 0.7,
                'rotation_probability': 0.3,
                'engagement_likelihood': 0.6,
                'peek_frequency': 0.6,
                'utility_usage_rate': 0.3
            },
            'late': {
                'movement_speed': 0.5,
                'rotation_probability': 0.5,
                'engagement_likelihood': 0.8,
                'peek_frequency': 0.8,
                'utility_usage_rate': 0.2
            }
        }
    return _VCT_PHASE_BEHAVIORS


@dataclass
class BehaviorModifiers:
    """Modifiers applied to player behavior."""
    aggression: float = 0.0       # -1.0 to 1.0 (negative = passive, positive = aggressive)
    movement_speed: float = 1.0   # Multiplier (0.5 = slow/careful, 1.5 = rushing)
    peek_willingness: float = 0.5 # 0-1 likelihood to peek angles
    rotation_urgency: float = 0.5 # 0-1 how urgently to rotate
    trade_priority: float = 0.5   # 0-1 priority on trading kills


@dataclass
class PlayerTendencies:
    """Individual player tendencies that affect behavior."""
    base_aggression: float = 0.5  # 0-1 base aggression level
    clutch_factor: float = 0.5    # 0-1 performance under pressure
    trade_awareness: float = 0.5  # 0-1 tendency to trade
    rotation_speed: float = 0.5   # 0-1 how fast they rotate
    peek_style: str = "normal"    # 'wide', 'jiggle', 'normal', 'passive'


@dataclass
class SpikeCarrierTendencies:
    """Per-player spike carrier behavior modifiers.

    VCT-derived defaults vary by role:
    - Duelists: Higher aggression (0.7), faster plants
    - Initiators: Moderate (0.4), info-focused
    - Controllers: Lower aggression (0.3), utility-focused
    - Sentinels: Lowest aggression (0.2), safe plants, may lurk

    Validation targets from VCT:
    - Carrier death rate: ~12% (±4%)
    - Plant rate: ~65% of attack rounds
    - Avg plant time: ~56s (±8s)
    """
    carrier_aggression: float = 0.3       # 0=passive, 1=aggressive
    early_engagement_mult: float = 0.3    # SETUP/MAP_CONTROL engagement modifier
    execute_engagement_mult: float = 0.5  # EXECUTE phase engagement modifier
    post_plant_hold_mult: float = 0.4     # Post-plant aggression modifier
    drop_spike_threshold: float = 0.7     # Man disadvantage threshold to drop spike
    lurk_probability: float = 0.1         # Chance to lurk with spike (sentinel higher)
    fast_plant_tendency: float = 0.5      # 0=slow/default, 1=rush plant

    def get_engagement_modifier(self, phase: str, man_advantage: int, time_ms: int) -> float:
        """Calculate engagement modifier based on context.

        Args:
            phase: Current round phase (setup, control, execute, post_plant)
            man_advantage: Team player count difference (+2 = 5v3, -1 = 4v5)
            time_ms: Current round time in milliseconds

        Returns:
            Engagement probability multiplier (0.1 to 1.0)
        """
        # Base modifier from phase
        if phase in ['setup', 'SETUP']:
            base_mod = self.early_engagement_mult
        elif phase in ['control', 'MAP_CONTROL', 'map_control']:
            base_mod = self.early_engagement_mult * 1.2  # Slightly higher during control
        elif phase in ['execute', 'EXECUTE']:
            base_mod = self.execute_engagement_mult
        elif phase in ['post_plant', 'POST_PLANT']:
            base_mod = self.post_plant_hold_mult
        else:
            base_mod = 0.4  # Default

        # Adjust for man advantage
        if man_advantage >= 2:
            # Strong advantage - can be more aggressive
            base_mod *= 1.3
        elif man_advantage == 1:
            base_mod *= 1.1
        elif man_advantage == -1:
            # Slight disadvantage - more cautious
            base_mod *= 0.8
        elif man_advantage <= -2:
            # Heavy disadvantage - very passive
            base_mod *= 0.5

        # Time pressure adjustment (attack without plant)
        if time_ms > 70000 and phase not in ['post_plant', 'POST_PLANT']:
            # After 70s, need to be more aggressive to plant
            time_pressure = min(0.5, (time_ms - 70000) / 30000)  # 0-0.5 bonus
            base_mod += time_pressure * self.fast_plant_tendency

        return max(0.1, min(1.0, base_mod))


class BehaviorAdapter:
    """Adapts player behavior based on round state and tendencies."""

    # Aggression modifiers based on man count
    # CALIBRATED: Stronger penalties for being outnumbered to match VCT outcomes
    # VCT shows 4v5 → ~35% win for disadvantaged, 3v5 → ~15% win
    # Outnumbered players should be VERY passive to avoid crossfire deaths
    MAN_ADVANTAGE_AGGRESSION = {
        2: 0.30,   # +2 players = aggressive but controlled hunt
        1: 0.15,   # +1 player = slight aggression, take map control
        0: 0.0,    # Even = neutral
        -1: -0.35, # -1 player = passive, avoid multi-enemy fights
        -2: -0.55, # -2 players = very passive, isolate fights or save
        -3: -0.70, # -3 players = save mode, play for exit frags
        -4: -0.80, # -4 players = full save, hide and save weapon
    }

    # Post-plant behavior
    POST_PLANT_ATTACK_AGGRESSION = -0.30  # Hold angles, play passive
    POST_PLANT_DEFENSE_AGGRESSION = 0.25  # Must retake, urgent but not frantic

    # Trade window behavior
    TRADE_WINDOW_AGGRESSION = 0.30  # Push for trade but don't suicide
    TRADE_WINDOW_SPEED = 1.3  # Move faster to trade

    # Time pressure modifiers (for attack without plant)
    TIME_PRESSURE_THRESHOLDS = [
        (60000, 0.0),    # Before 60s - no pressure
        (80000, 0.15),   # 60-80s - slight aggression
        (90000, 0.30),   # 80-90s - more aggressive
        (95000, 0.50),   # 90-95s - very aggressive (must execute)
    ]

    # Economy-based modifiers
    ECO_ROUND_AGGRESSION = {
        'pistol': 0.10,  # Pistol rounds slightly aggressive
        'eco': -0.20,    # Eco = passive, try to save
        'force': 0.15,   # Force buy = aggressive to get value
        'half': 0.0,     # Half buy = neutral
        'full': 0.0,     # Full buy = play default
    }

    @classmethod
    def _map_phase_to_vct_key(cls, phase: Optional[str]) -> str:
        """Map expanded phase names to VCT behavioral_patterns.json keys.

        The simulation uses 6 phases, but VCT data has 3 (early, mid, late).
        This maps the expanded phases to the corresponding VCT keys.

        Args:
            phase: Expanded phase name (e.g., 'post_plant_early', 'opening')

        Returns:
            VCT phase key ('early', 'mid', or 'late')
        """
        if phase is None:
            return 'mid'

        # Map expanded phases to VCT keys
        phase_mapping = {
            'opening': 'early',
            'early': 'early',
            'mid_round': 'mid',
            'mid': 'mid',
            'late': 'late',
            'post_plant': 'mid',  # Default post-plant behavior
            'post_plant_early': 'early',  # Setup phase - similar to early
            'post_plant_mid': 'mid',  # Hold phase
            'post_plant_late': 'late',  # Late pressure
            'retake': 'late',  # Retakes are high-pressure like late game
        }

        return phase_mapping.get(phase, 'mid')

    @classmethod
    def calculate_behavior_modifiers(
        cls,
        player_id: str,
        player_team: str,  # 'attack' or 'defense'
        round_state: RoundState,
        time_ms: int,
        tendencies: Optional[PlayerTendencies] = None,
        distance_to_last_kill: Optional[float] = None,
        phase: Optional[str] = None
    ) -> BehaviorModifiers:
        """Calculate behavior modifiers for a player.

        Args:
            player_id: ID of the player
            player_team: 'attack' or 'defense'
            round_state: Current round state
            time_ms: Current time in milliseconds
            tendencies: Player's individual tendencies
            distance_to_last_kill: Distance to where last kill occurred
            phase: Current game phase (early, mid, late, post_plant_early, etc.)

        Returns:
            BehaviorModifiers for the player
        """
        if tendencies is None:
            tendencies = PlayerTendencies()

        base_aggression = tendencies.base_aggression - 0.5  # Convert 0-1 to -0.5 to 0.5
        modifiers = BehaviorModifiers(aggression=base_aggression)

        # Apply VCT phase-based behavior modifiers
        vct_phase_behaviors = _load_vct_phase_behaviors()
        # Map expanded phases to VCT phase keys
        phase_key = cls._map_phase_to_vct_key(phase) if phase else 'mid'
        phase_data = vct_phase_behaviors.get(phase_key, {})

        if phase_data:
            # Apply VCT movement speed modifier
            vct_movement = phase_data.get('movement_speed', 1.0)
            modifiers.movement_speed *= vct_movement

            # Map engagement_likelihood to aggression modifier
            engagement = phase_data.get('engagement_likelihood', 0.5)
            modifiers.aggression += (engagement - 0.5) * 0.3  # Scale to -0.15 to +0.15

            # Map peek_frequency to peek_willingness
            modifiers.peek_willingness = phase_data.get('peek_frequency', 0.5)

            # Map rotation_probability to rotation_urgency (higher rotation prob = more urgent)
            modifiers.rotation_urgency = phase_data.get('rotation_probability', 0.3)

        # Man advantage modifier
        is_attack = player_team == 'attack'
        man_diff = round_state.attack_alive - round_state.defense_alive
        if not is_attack:
            man_diff = -man_diff  # Flip for defense perspective

        aggression_mod = cls.MAN_ADVANTAGE_AGGRESSION.get(
            max(-4, min(2, man_diff)), 0.0
        )
        modifiers.aggression += aggression_mod

        # Post-plant behavior
        if round_state.spike_planted:
            if is_attack:
                modifiers.aggression += cls.POST_PLANT_ATTACK_AGGRESSION
                modifiers.movement_speed = 0.7  # Slow, hold angles
                modifiers.peek_willingness = 0.2  # Don't peek, let them come
            else:
                modifiers.aggression += cls.POST_PLANT_DEFENSE_AGGRESSION
                modifiers.movement_speed = 1.2  # Urgent rotation but not sprinting
                modifiers.rotation_urgency = 0.9  # Very urgent
                modifiers.peek_willingness = 0.7  # Need to clear angles

        else:
            # Opening phase: Defenders sprint to positions
            # FIX: Extended speed periods so defenders can reach positions before attackers
            if not is_attack:
                if time_ms < 20000:
                    # Sprint to position - defenders need to reach sites before attackers
                    # Extended from 15s to 20s to allow full setup
                    modifiers.movement_speed = 1.5  # Faster movement to reach position
                    modifiers.rotation_urgency = 0.8
                elif time_ms < 35000:
                    # Slowing down, getting into position
                    # Extended from 25s to 35s
                    modifiers.movement_speed = 1.1
                else:
                    # In position, hold angles
                    modifiers.movement_speed = 0.7

            # Time pressure for attack (need to plant)
            if is_attack:
                for threshold, mod in cls.TIME_PRESSURE_THRESHOLDS:
                    if time_ms < threshold:
                        break
                    modifiers.aggression += mod
                    modifiers.movement_speed = min(1.5, 1.0 + mod)

        # Trade window behavior
        if round_state.potential_trade_window and distance_to_last_kill is not None:
            # Close enough to trade
            if distance_to_last_kill < 0.15:  # Within 15% of map
                # Only if we lost the player
                if round_state.last_killed_team == player_team:
                    modifiers.aggression += cls.TRADE_WINDOW_AGGRESSION * tendencies.trade_awareness
                    modifiers.movement_speed *= cls.TRADE_WINDOW_SPEED
                    modifiers.trade_priority = 0.9

        # Economy modifier
        buy_type = round_state.attack_buy_type if is_attack else round_state.defense_buy_type
        eco_mod = cls.ECO_ROUND_AGGRESSION.get(buy_type, 0.0)
        modifiers.aggression += eco_mod

        # Apply clutch factor if in clutch situation
        team_alive = round_state.attack_alive if is_attack else round_state.defense_alive
        opponent_alive = round_state.defense_alive if is_attack else round_state.attack_alive

        if team_alive == 1 and opponent_alive > 1:
            # In clutch - behavior depends on clutch_factor
            clutch_aggression = (tendencies.clutch_factor - 0.5) * 0.4
            modifiers.aggression += clutch_aggression

            # Low clutch factor = very passive, try to isolate fights
            if tendencies.clutch_factor < 0.4:
                modifiers.movement_speed = 0.6
                modifiers.peek_willingness = 0.2
            # High clutch factor = confident, aggressive plays
            elif tendencies.clutch_factor > 0.7:
                modifiers.peek_willingness = 0.7

        # Clamp aggression to valid range
        # Defenders cap lower — even in retakes, pros play controlled
        if not is_attack:
            modifiers.aggression = max(-1.0, min(0.6, modifiers.aggression))
        else:
            modifiers.aggression = max(-1.0, min(1.0, modifiers.aggression))

        return modifiers

    @classmethod
    def should_attempt_trade(
        cls,
        player_id: str,
        player_team: str,
        round_state: RoundState,
        player_position: Tuple[float, float],
        tendencies: Optional[PlayerTendencies] = None
    ) -> Tuple[bool, float]:
        """Determine if a player should attempt to trade a teammate's death.

        Args:
            player_id: ID of the player
            player_team: 'attack' or 'defense'
            round_state: Current round state
            player_position: Current position of the player
            tendencies: Player's individual tendencies

        Returns:
            Tuple of (should_trade, urgency)
        """
        if tendencies is None:
            tendencies = PlayerTendencies()

        # Not in trade window
        if not round_state.potential_trade_window:
            return (False, 0.0)

        # Wrong team (enemy died, not teammate)
        if round_state.last_killed_team != player_team:
            return (False, 0.0)

        # No position info
        if round_state.last_kill_position is None:
            return (False, 0.0)

        # Calculate distance to kill
        dx = player_position[0] - round_state.last_kill_position[0]
        dy = player_position[1] - round_state.last_kill_position[1]
        distance = math.sqrt(dx * dx + dy * dy)

        # Too far to trade effectively
        TRADE_DISTANCE_THRESHOLD = 0.2  # 20% of map
        if distance > TRADE_DISTANCE_THRESHOLD:
            return (False, 0.0)

        # Calculate trade urgency based on distance and tendencies
        urgency = (1.0 - distance / TRADE_DISTANCE_THRESHOLD) * tendencies.trade_awareness

        # Higher urgency if spike is planted and we're defense
        if round_state.spike_planted and player_team == 'defense':
            urgency *= 1.3

        # Lower urgency if we have man advantage
        man_diff = round_state.attack_alive - round_state.defense_alive
        if player_team == 'defense':
            man_diff = -man_diff

        if man_diff > 0:
            urgency *= 0.8  # Less urgent if we have numbers

        return (urgency > 0.3, min(1.0, urgency))

    @classmethod
    def calculate_rotation_urgency(
        cls,
        player_team: str,
        round_state: RoundState,
        time_ms: int,
        current_site: Optional[str],
        spike_site: Optional[str]
    ) -> float:
        """Calculate how urgently a player should rotate.

        Args:
            player_team: 'attack' or 'defense'
            round_state: Current round state
            time_ms: Current time in milliseconds
            current_site: Site player is currently at ('A', 'B', or None)
            spike_site: Site where spike is planted or being taken

        Returns:
            Urgency value 0-1 (higher = should rotate faster)
        """
        urgency = 0.5  # Base urgency

        is_attack = player_team == 'attack'

        # Post-plant defense needs to rotate
        if round_state.spike_planted:
            if not is_attack:
                # Defense must rotate to spike site
                if current_site != round_state.spike_site:
                    urgency = 0.95  # Very urgent

                    # Even more urgent if low time
                    SPIKE_TIME_MS = 45000
                    time_since_plant = time_ms - round_state.spike_plant_time_ms
                    time_remaining = SPIKE_TIME_MS - time_since_plant

                    if time_remaining < 25000:
                        urgency = 1.0
                else:
                    urgency = 0.3  # Already at site
            else:
                # Attack doesn't need to rotate post-plant
                urgency = 0.2
        else:
            # Pre-plant: check man advantage
            if not is_attack:
                # Defense rotation based on enemy numbers
                # If enemies showing at a site, rotate to help
                man_diff = round_state.defense_alive - round_state.attack_alive
                if man_diff < 0:
                    urgency = 0.7  # Rotate faster when outnumbered
            else:
                # Attack rotation based on time
                if time_ms > 60000:
                    urgency = 0.6  # Start rotating for execute
                if time_ms > 80000:
                    urgency = 0.8  # Need to commit

        return min(1.0, max(0.0, urgency))

    # === SPIKE CARRIER BEHAVIOR METHODS ===
    # VCT-derived per-player spike carrier tendencies

    @classmethod
    def create_spike_carrier_tendencies(
        cls,
        role: str,
        player_profile: Optional[Dict] = None
    ) -> SpikeCarrierTendencies:
        """Create spike carrier tendencies from role defaults + VCT profile.

        Args:
            role: Player role (duelist, initiator, controller, sentinel)
            player_profile: Optional VCT player profile with aggression, first_kill_rate, etc.

        Returns:
            SpikeCarrierTendencies configured for this player
        """
        # Role-based defaults from VCT extraction
        # See validated_parameters.py for derivation
        ROLE_DEFAULTS = {
            'duelist': {
                'carrier_aggression': 0.7,
                'early_engagement_mult': 0.5,
                'execute_engagement_mult': 0.7,
                'post_plant_hold_mult': 0.5,
                'drop_spike_threshold': 0.6,
                'lurk_probability': 0.05,
                'fast_plant_tendency': 0.7,
            },
            'initiator': {
                'carrier_aggression': 0.4,
                'early_engagement_mult': 0.3,
                'execute_engagement_mult': 0.5,
                'post_plant_hold_mult': 0.4,
                'drop_spike_threshold': 0.7,
                'lurk_probability': 0.15,
                'fast_plant_tendency': 0.5,
            },
            'controller': {
                'carrier_aggression': 0.3,
                'early_engagement_mult': 0.2,
                'execute_engagement_mult': 0.4,
                'post_plant_hold_mult': 0.3,
                'drop_spike_threshold': 0.75,
                'lurk_probability': 0.10,
                'fast_plant_tendency': 0.4,
            },
            'sentinel': {
                'carrier_aggression': 0.2,
                'early_engagement_mult': 0.15,
                'execute_engagement_mult': 0.3,
                'post_plant_hold_mult': 0.25,
                'drop_spike_threshold': 0.8,
                'lurk_probability': 0.25,
                'fast_plant_tendency': 0.3,
            },
        }

        # Get role defaults
        defaults = ROLE_DEFAULTS.get(role.lower(), ROLE_DEFAULTS['controller'])
        tendencies = SpikeCarrierTendencies(**defaults)

        # Adjust based on VCT player profile if available
        if player_profile:
            # High aggression players are more aggressive carriers
            aggression = player_profile.get('aggression', 0.5)
            tendencies.carrier_aggression = min(0.9, tendencies.carrier_aggression + (aggression - 0.5) * 0.4)

            # High first_kill_rate indicates aggressive entry style
            first_kill_rate = player_profile.get('first_kill_rate', 0.1)
            if first_kill_rate > 0.15:
                tendencies.early_engagement_mult *= 1.2
                tendencies.fast_plant_tendency = min(0.9, tendencies.fast_plant_tendency + 0.15)

            # High clutch_potential means better post-plant performance
            clutch = player_profile.get('clutch_potential', 0.5)
            if clutch > 0.6:
                tendencies.post_plant_hold_mult = min(0.7, tendencies.post_plant_hold_mult + 0.15)

            # Low first_death_rate means safer carry style
            first_death_rate = player_profile.get('first_death_rate', 0.1)
            if first_death_rate < 0.08:
                tendencies.carrier_aggression = max(0.1, tendencies.carrier_aggression - 0.1)
                tendencies.drop_spike_threshold = min(0.9, tendencies.drop_spike_threshold + 0.05)

        return tendencies

    @classmethod
    def get_spike_carrier_engagement_modifier(
        cls,
        player,  # SimulatedPlayer (avoid circular import)
        phase: str,
        round_state: RoundState,
        time_ms: int
    ) -> float:
        """Get engagement probability modifier for spike carrier.

        Replaces the fixed 0.3/0.5 modifiers with per-player calculation.

        Args:
            player: The spike carrier (SimulatedPlayer)
            phase: Current round phase
            round_state: Current round state
            time_ms: Current round time

        Returns:
            Engagement multiplier (0.1 to 1.0)
        """
        # Get spike carrier tendencies from player
        tendencies = getattr(player, 'spike_carrier_tendencies', None)
        if tendencies is None:
            # Fallback to default if not initialized
            tendencies = SpikeCarrierTendencies()

        # Calculate man advantage
        man_advantage = round_state.attack_alive - round_state.defense_alive

        # Use tendencies to calculate modifier
        return tendencies.get_engagement_modifier(phase, man_advantage, time_ms)

    @classmethod
    def should_drop_spike(
        cls,
        carrier,  # SimulatedPlayer
        teammates: List,  # List[SimulatedPlayer]
        round_state: RoundState,
        phase: str
    ) -> Tuple[bool, Optional[str]]:
        """Determine if carrier should drop spike for a teammate.

        Args:
            carrier: Current spike carrier
            teammates: List of alive teammates
            round_state: Current round state
            phase: Current round phase

        Returns:
            Tuple of (should_drop, target_player_id or None)
        """
        tendencies = getattr(carrier, 'spike_carrier_tendencies', None)
        if tendencies is None:
            tendencies = SpikeCarrierTendencies()

        # Only consider dropping in certain conditions
        man_advantage = round_state.attack_alive - round_state.defense_alive

        # Heavy disadvantage - consider dropping to safer player
        if man_advantage <= -2 and tendencies.drop_spike_threshold > 0.5:
            # Find a sentinel/controller teammate who might be safer
            for mate in teammates:
                if mate.player_id == carrier.player_id or not mate.is_alive:
                    continue
                mate_role = cls._get_role_from_agent(mate.agent)
                if mate_role in ['sentinel', 'controller']:
                    return (True, mate.player_id)

        # Duelist carrier during execute should drop to support
        carrier_role = cls._get_role_from_agent(carrier.agent)
        if carrier_role == 'duelist' and phase in ['execute', 'EXECUTE']:
            # Duelists need to entry - drop spike to support
            for mate in teammates:
                if mate.player_id == carrier.player_id or not mate.is_alive:
                    continue
                mate_role = cls._get_role_from_agent(mate.agent)
                if mate_role in ['controller', 'sentinel']:
                    return (True, mate.player_id)

        return (False, None)

    @classmethod
    def should_lurk_with_spike(
        cls,
        carrier,  # SimulatedPlayer
        round_state: RoundState,
        time_ms: int
    ) -> Tuple[bool, str]:
        """Determine if carrier should lurk with spike.

        Only sentinels and some controllers lurk with spike.

        Args:
            carrier: Spike carrier
            round_state: Current round state
            time_ms: Current round time

        Returns:
            Tuple of (should_lurk, reason)
        """
        tendencies = getattr(carrier, 'spike_carrier_tendencies', None)
        if tendencies is None:
            tendencies = SpikeCarrierTendencies()

        # Only lurk early in round
        if time_ms > 50000:
            return (False, "too_late")

        # Only lurk with man advantage
        man_advantage = round_state.attack_alive - round_state.defense_alive
        if man_advantage < 1:
            return (False, "no_advantage")

        # Check lurk probability based on tendencies
        import random
        if random.random() < tendencies.lurk_probability:
            return (True, "tendencies")

        return (False, "random_check_failed")

    @staticmethod
    def _get_role_from_agent(agent: str) -> str:
        """Map agent to role."""
        duelists = ['jett', 'raze', 'reyna', 'phoenix', 'neon', 'yoru', 'iso']
        initiators = ['sova', 'skye', 'breach', 'kayo', 'fade', 'gekko']
        controllers = ['omen', 'brimstone', 'astra', 'viper', 'harbor', 'clove']
        sentinels = ['killjoy', 'cypher', 'sage', 'chamber', 'deadlock', 'vyse']

        agent_lower = agent.lower()
        if agent_lower in duelists:
            return 'duelist'
        elif agent_lower in initiators:
            return 'initiator'
        elif agent_lower in controllers:
            return 'controller'
        elif agent_lower in sentinels:
            return 'sentinel'
        return 'initiator'


@dataclass
class PositioningHints:
    """Hints for positioning based on round state."""
    should_hold_angle: bool = False
    preferred_angle_direction: Optional[Tuple[float, float]] = None
    should_seek_cover: bool = False
    cover_direction: Optional[Tuple[float, float]] = None
    should_fall_back: bool = False
    fallback_position: Optional[Tuple[float, float]] = None


class PositioningAdvisor:
    """Provides positioning advice based on round state."""

    @classmethod
    def get_positioning_hints(
        cls,
        player_team: str,
        round_state: RoundState,
        player_position: Tuple[float, float],
        time_ms: int,
        behavior: BehaviorModifiers
    ) -> PositioningHints:
        """Get positioning hints for a player.

        Args:
            player_team: 'attack' or 'defense'
            round_state: Current round state
            player_position: Current player position
            time_ms: Current time
            behavior: Current behavior modifiers

        Returns:
            PositioningHints for the player
        """
        hints = PositioningHints()
        is_attack = player_team == 'attack'

        # Passive behavior = hold angles
        if behavior.aggression < -0.2:
            hints.should_hold_angle = True
            hints.should_seek_cover = True

        # Very passive = fall back
        if behavior.aggression < -0.5:
            hints.should_fall_back = True
            # Fall back toward spawn
            if is_attack:
                hints.fallback_position = (0.2, 0.8)  # Attack spawn
            else:
                hints.fallback_position = (0.8, 0.2)  # Defense spawn

        # Post-plant attack should hold angles toward site
        if round_state.spike_planted and is_attack:
            hints.should_hold_angle = True
            # Face toward spike site
            if round_state.spike_site == 'A':
                hints.preferred_angle_direction = (0.3, 0.3)
            else:
                hints.preferred_angle_direction = (0.7, 0.3)

        # Post-plant defense should push toward spike
        if round_state.spike_planted and not is_attack:
            hints.should_hold_angle = False
            hints.should_seek_cover = False  # Need to push

        # Man disadvantage = seek cover
        man_diff = round_state.attack_alive - round_state.defense_alive
        if not is_attack:
            man_diff = -man_diff

        if man_diff < -1:
            hints.should_seek_cover = True
            hints.should_fall_back = True

        return hints
