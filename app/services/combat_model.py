"""
Realistic Combat Model for VALORANT Simulation

Models real gunfight mechanics including:
- Reaction time (visual, cognitive, motor response)
- Crosshair placement (pre-aim quality affects first shot timing)
- Time-to-kill based on weapon fire rate and damage
- Accuracy degradation during spray
- Counter-strafe mechanics
- Peeker's advantage from netcode

Based on validated Valorant mechanics and pro player data.
"""

import math
import random
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List
from enum import Enum


class EngagementType(Enum):
    """Type of engagement affects combat dynamics."""
    PEEK_VS_HOLD = "peek_vs_hold"  # Peeker has advantage
    HOLD_VS_PEEK = "hold_vs_peek"  # Holder has pre-aim
    BOTH_PEEKING = "both_peeking"  # Both moving
    BOTH_HOLDING = "both_holding"  # Static fight
    TRADE = "trade"  # Trading a kill


class MovementState(Enum):
    """Player movement state affects accuracy timing."""
    STATIONARY = "stationary"      # Full accuracy
    COUNTER_STRAFING = "counter_strafing"  # Recovering accuracy
    WALKING = "walking"            # Reduced accuracy, quiet
    RUNNING = "running"            # Very low accuracy


@dataclass
class PlayerCombatProfile:
    """Individual player's combat characteristics.

    These values should be derived from real player data.
    Pro players: faster reactions, better crosshair placement, higher HS%

    VCT Calibration Notes:
    - Crosshair placement calibrated from hold_angles.json std_angle data
      - Tight angles (std < 0.5 rad): Pro-level pre-aim = 0.90+
      - Medium angles (std 0.5-1.0 rad): Good pre-aim = 0.70-0.85
      - Loose angles (std 1.0-1.5 rad): Average = 0.50-0.70
      - Very loose (std > 1.5 rad): Poor = <0.50
    - VCT average std_angle: 1.08 rad (simple avg), 1.39 rad (weighted)
    - This suggests pros hold WIDER angles than expected (watching corridors)
    """
    # Reaction time (milliseconds)
    # Pro: 150-180ms, Good: 180-220ms, Average: 220-280ms
    # Note: General FPS research, not VCT-specific
    base_reaction_ms: float = 200.0
    reaction_variance: float = 30.0  # Standard deviation

    # Crosshair placement quality (0-1)
    # VCT CALIBRATION: Based on hold_angles.json std_angle analysis
    # - 12.6% of zones have tight angles (<0.5 rad) = elite pre-aim
    # - 33.2% medium (0.5-1.0 rad) = good corridor watching
    # - 32.5% loose (1.0-1.5 rad) = watching multiple angles
    # - 21.7% very loose (>1.5 rad) = reactive positioning
    # Pro default: 0.70 reflects watching corridors (weighted avg ~1.39 rad std)
    crosshair_placement: float = 0.70

    # Headshot rate under pressure
    # VCT-verified: Pro average 25%, top performers 31-34%
    headshot_rate: float = 0.25

    # Spray control (0-1)
    # 1.0 = perfect spray transfer, 0.0 = random after first shot
    # Note: Could derive from hits-to-kill data (VCT avg 3.9 hits)
    spray_control: float = 0.6

    # First shot accuracy modifier
    # How well they wait for accuracy reset vs panic shooting
    first_shot_discipline: float = 0.8

    # Counter-strafe proficiency (0-1)
    # Pro: 0.9+, Good: 0.7-0.85, Average: 0.5-0.7
    # Community verified timing: 104ms to deadzone
    counter_strafe_skill: float = 0.75

    # Clutch factor - performance under pressure
    # >1.0 = performs better, <1.0 = chokes
    clutch_factor: float = 1.0

    # Peek style preference
    # Higher = wider swings, Lower = shoulder peeks
    peek_aggression: float = 0.5


@dataclass
class WeaponCombatStats:
    """Combat-relevant weapon statistics."""
    name: str
    fire_rate_rpm: float  # Rounds per minute
    headshot_damage: float
    body_damage: float
    leg_damage: float
    first_shot_accuracy: float  # Degrees of spread
    spray_accuracy_decay: float  # Accuracy loss per shot
    recoil_reset_ms: float  # Time to reset to first shot accuracy

    @property
    def ms_between_shots(self) -> float:
        """Milliseconds between shots based on fire rate."""
        return 60000.0 / self.fire_rate_rpm

    def accuracy_at_shot(self, shot_number: int, spray_control: float = 0.5) -> float:
        """Get accuracy for nth shot in a spray.

        Args:
            shot_number: Which shot (1 = first shot)
            spray_control: Player's spray control skill (0-1)

        Returns:
            Accuracy modifier (0-1)
        """
        if shot_number <= 1:
            return 1.0 - (self.first_shot_accuracy / 10.0)  # Convert degrees to modifier

        # Accuracy degrades with each shot, mitigated by spray control
        base_decay = self.spray_accuracy_decay * (shot_number - 1)
        controlled_decay = base_decay * (1.0 - spray_control * 0.7)

        return max(0.1, 1.0 - controlled_decay)


# Weapon database with combat stats
WEAPON_COMBAT_STATS: Dict[str, WeaponCombatStats] = {
    # Rifles
    'vandal': WeaponCombatStats(
        name='Vandal',
        fire_rate_rpm=600,  # 9.75 rounds/sec
        headshot_damage=160,
        body_damage=40,
        leg_damage=34,
        first_shot_accuracy=0.25,  # Very accurate
        spray_accuracy_decay=0.08,
        recoil_reset_ms=375
    ),
    'phantom': WeaponCombatStats(
        name='Phantom',
        fire_rate_rpm=660,  # 11 rounds/sec
        headshot_damage=156,  # 140 at range
        body_damage=39,  # 35 at range
        leg_damage=33,
        first_shot_accuracy=0.20,  # Slightly more accurate
        spray_accuracy_decay=0.06,
        recoil_reset_ms=350
    ),

    # SMGs
    'spectre': WeaponCombatStats(
        name='Spectre',
        fire_rate_rpm=780,
        headshot_damage=78,
        body_damage=26,
        leg_damage=22,
        first_shot_accuracy=0.40,
        spray_accuracy_decay=0.04,
        recoil_reset_ms=300
    ),
    'stinger': WeaponCombatStats(
        name='Stinger',
        fire_rate_rpm=1000,
        headshot_damage=67,
        body_damage=27,
        leg_damage=23,
        first_shot_accuracy=0.60,
        spray_accuracy_decay=0.03,
        recoil_reset_ms=250
    ),

    # Snipers
    'operator': WeaponCombatStats(
        name='Operator',
        fire_rate_rpm=36,  # 0.6 rounds/sec
        headshot_damage=255,
        body_damage=150,
        leg_damage=120,
        first_shot_accuracy=0.10,
        spray_accuracy_decay=0.0,  # Single shot
        recoil_reset_ms=1500
    ),
    'marshal': WeaponCombatStats(
        name='Marshal',
        fire_rate_rpm=90,
        headshot_damage=202,
        body_damage=101,
        leg_damage=85,
        first_shot_accuracy=0.15,
        spray_accuracy_decay=0.0,
        recoil_reset_ms=1000
    ),

    # Pistols
    'classic': WeaponCombatStats(
        name='Classic',
        fire_rate_rpm=400,
        headshot_damage=78,
        body_damage=26,
        leg_damage=22,
        first_shot_accuracy=0.40,
        spray_accuracy_decay=0.10,
        recoil_reset_ms=400
    ),
    'sheriff': WeaponCombatStats(
        name='Sheriff',
        fire_rate_rpm=240,
        headshot_damage=160,
        body_damage=55,
        leg_damage=47,
        first_shot_accuracy=0.25,
        spray_accuracy_decay=0.15,
        recoil_reset_ms=600
    ),
    'ghost': WeaponCombatStats(
        name='Ghost',
        fire_rate_rpm=400,
        headshot_damage=105,
        body_damage=30,
        leg_damage=26,
        first_shot_accuracy=0.30,
        spray_accuracy_decay=0.08,
        recoil_reset_ms=350
    ),

    # Shotguns
    'judge': WeaponCombatStats(
        name='Judge',
        fire_rate_rpm=210,
        headshot_damage=34,  # Per pellet
        body_damage=17,
        leg_damage=14,
        first_shot_accuracy=0.80,  # Spread weapon
        spray_accuracy_decay=0.02,
        recoil_reset_ms=200
    ),
    'bucky': WeaponCombatStats(
        name='Bucky',
        fire_rate_rpm=110,
        headshot_damage=40,  # Per pellet
        body_damage=20,
        leg_damage=17,
        first_shot_accuracy=0.75,
        spray_accuracy_decay=0.0,
        recoil_reset_ms=800
    ),

    # Machine guns
    'odin': WeaponCombatStats(
        name='Odin',
        fire_rate_rpm=720,
        headshot_damage=95,
        body_damage=38,
        leg_damage=32,
        first_shot_accuracy=0.80,
        spray_accuracy_decay=0.02,  # Tightens while spraying
        recoil_reset_ms=500
    ),
    'ares': WeaponCombatStats(
        name='Ares',
        fire_rate_rpm=600,
        headshot_damage=72,
        body_damage=30,
        leg_damage=25,
        first_shot_accuracy=0.90,
        spray_accuracy_decay=0.02,
        recoil_reset_ms=450
    ),
}


@dataclass
class CombatState:
    """Current state of a combat encounter."""
    player_a_id: str
    player_b_id: str

    # Timing
    engagement_start_ms: int = 0
    current_time_ms: int = 0

    # Player A state
    a_reaction_complete: bool = False
    a_reaction_time_ms: float = 0
    a_shots_fired: int = 0
    a_last_shot_ms: int = 0
    a_damage_dealt: float = 0
    a_hits: int = 0

    # Player B state
    b_reaction_complete: bool = False
    b_reaction_time_ms: float = 0
    b_shots_fired: int = 0
    b_last_shot_ms: int = 0
    b_damage_dealt: float = 0
    b_hits: int = 0


@dataclass
class CombatResult:
    """Result of a combat engagement."""
    winner_id: str
    loser_id: str
    time_to_kill_ms: float
    shots_fired_winner: int
    shots_fired_loser: int
    headshot_kill: bool
    damage_dealt_winner: float
    damage_dealt_loser: float
    was_trade: bool = False
    engagement_type: EngagementType = EngagementType.BOTH_HOLDING


class RealisticCombatModel:
    """
    Models realistic VALORANT combat encounters.

    Combat flow:
    1. Detection - Players become aware of each other
    2. Reaction - Visual processing + cognitive + motor response
    3. Aim adjustment - Crosshair movement to target
    4. Fire - Shots with accuracy based on weapon/movement/spray
    5. Resolution - First to deal lethal damage wins
    """

    # Peeker's advantage (netcode delay)
    PEEKERS_ADVANTAGE_MS = 50.0  # 40-70ms typical

    # Counter-strafe timing (from validated parameters)
    COUNTER_STRAFE_TO_WALK_MS = 55    # Time to walking accuracy
    COUNTER_STRAFE_TO_FULL_MS = 160   # Time to full accuracy

    # Hitbox sizes for accuracy calculation
    HEAD_HITBOX_DEGREES = 2.5   # Head is ~2.5 degrees at 20m
    BODY_HITBOX_DEGREES = 8.0   # Body is ~8 degrees at 20m

    # VCT Distance-Damage Pattern (from 27,292 damage events)
    # ACTUAL DATA shows medium range has HIGHEST damage (optimal rifle range)
    # Source: trade_patterns.json - distance_damage section
    # Note: This reflects pro play reality - medium range = clean rifle headshots
    VCT_DISTANCE_AVG_DAMAGE = {
        # (max_distance_units, avg_damage, samples)
        500: (33.5, 1222),     # Close: shotgun/SMG territory, spray fights
        1000: (37.1, 4254),    # Short: transitioning to rifle range
        2000: (38.8, 10572),   # Medium: OPTIMAL rifle range, highest damage
        3000: (33.3, 7631),    # Long: damage falloff begins
        99999: (29.8, 3613),   # Very long: significant falloff
    }

    # Normalized to medium range (peak damage = 1.0)
    # This reflects that pros deal most damage at optimal rifle range
    VCT_DISTANCE_DAMAGE_MODIFIER = {
        5: 0.86,    # 0-5m: 86% (33.5/38.8) - close range spray fights
        10: 0.96,   # 5-10m: 96% (37.1/38.8) - approaching optimal
        20: 1.00,   # 10-20m: 100% - OPTIMAL rifle range
        30: 0.86,   # 20-30m: 86% (33.3/38.8) - falloff starting
        999: 0.77,  # 30m+: 77% (29.8/38.8) - significant falloff
    }

    def __init__(self):
        self.default_profile = PlayerCombatProfile()

    def calculate_distance_damage_modifier(self, distance_meters: float) -> float:
        """Calculate damage modifier based on distance using VCT data.

        VCT Data shows significant damage falloff at range:
        - Close (0-5m): 100%
        - Short (5-10m): 86%
        - Medium (10-20m): 71%
        - Long (20-30m): 63%
        - Very long (30m+): 54%

        Args:
            distance_meters: Distance in meters

        Returns:
            Damage modifier (0.54 to 1.0)
        """
        for max_dist, modifier in sorted(self.VCT_DISTANCE_DAMAGE_MODIFIER.items()):
            if distance_meters <= max_dist:
                return modifier
        return 0.54  # Minimum falloff

    def calculate_reaction_time(
        self,
        profile: PlayerCombatProfile,
        engagement_type: EngagementType,
        is_peeking: bool,
        is_flashed: bool = False,
        distance_meters: float = 15.0
    ) -> float:
        """Calculate actual reaction time for this engagement.

        Components:
        - Base reaction (visual processing): 120-180ms for pros
        - Cognitive processing: 30-60ms
        - Motor response: 30-50ms
        - Situational modifiers
        """
        # Base reaction with variance
        base = profile.base_reaction_ms
        variance = random.gauss(0, profile.reaction_variance)
        reaction = base + variance

        # Peeker vs holder modifiers
        if engagement_type == EngagementType.PEEK_VS_HOLD and is_peeking:
            # Peeker is ready, holder must react
            reaction *= 0.85  # Peeker is mentally prepared
        elif engagement_type == EngagementType.HOLD_VS_PEEK and not is_peeking:
            # Holder has crosshair placed, faster aim adjustment
            reaction *= 0.90  # Pre-aimed

        # Flash heavily impacts reaction
        if is_flashed:
            reaction *= 2.5  # Can't see properly

        # Distance affects target acquisition
        # Closer = easier to see, faster reaction
        if distance_meters < 10:
            reaction *= 0.95
        elif distance_meters > 25:
            reaction *= 1.10

        # Clutch factor
        reaction *= (2.0 - profile.clutch_factor)  # Inverted - lower reaction for higher clutch

        return max(100, reaction)  # Minimum 100ms (inhuman below this)

    def calculate_aim_adjustment_time(
        self,
        profile: PlayerCombatProfile,
        distance_meters: float,
        crosshair_offset_degrees: float,
        is_peeking: bool
    ) -> float:
        """Calculate time to move crosshair onto target.

        Based on crosshair placement quality and angular distance.
        """
        # Perfect crosshair placement = 0 adjustment needed
        # Poor placement = full angular adjustment needed

        # Calculate effective offset based on crosshair placement skill
        base_offset = crosshair_offset_degrees * (1.0 - profile.crosshair_placement * 0.8)

        # Peeker already knows where to aim (pre-aimed the angle)
        if is_peeking:
            base_offset *= 0.5

        # Mouse movement speed (degrees per ms)
        # Pro players: ~0.3-0.5 deg/ms, Average: 0.15-0.25 deg/ms
        mouse_speed = 0.25 + (profile.crosshair_placement * 0.25)

        adjustment_time = base_offset / mouse_speed

        return max(0, adjustment_time)

    def calculate_counter_strafe_delay(
        self,
        profile: PlayerCombatProfile,
        is_running: bool,
        is_walking: bool
    ) -> Tuple[float, float]:
        """Calculate time to reach shooting accuracy when counter-strafing.

        Returns:
            (time_to_shoot_ms, accuracy_modifier)
        """
        if not is_running and not is_walking:
            # Already stationary
            return (0, 1.0)

        if is_walking:
            # Walking has partial accuracy, can shoot sooner
            return (20, 0.65)

        # Running - need full counter-strafe
        skill = profile.counter_strafe_skill

        # Better skill = faster counter-strafe
        # Pro: 55ms to walking acc, 160ms to full
        # Poor: adds 30-50ms
        time_to_walk_acc = self.COUNTER_STRAFE_TO_WALK_MS * (1.0 + (1.0 - skill) * 0.5)

        # If player has good discipline, they wait for accuracy
        # If not, they shoot early with reduced accuracy
        if profile.first_shot_discipline > 0.7:
            # Patient player waits for accuracy
            return (time_to_walk_acc, 0.85)
        else:
            # Impatient player shoots immediately
            return (10, 0.3 * skill)

    def calculate_hit_probability(
        self,
        weapon: WeaponCombatStats,
        profile: PlayerCombatProfile,
        distance_meters: float,
        shot_number: int,
        accuracy_modifier: float = 1.0,
        aiming_for_head: bool = True
    ) -> Tuple[bool, str]:
        """Calculate if a shot hits and where.

        Returns:
            (hit, region) where region is 'head', 'body', or 'miss'
        """
        # Base accuracy from weapon and shot number
        weapon_accuracy = weapon.accuracy_at_shot(shot_number, profile.spray_control)

        # Apply accuracy modifier (movement, flash, etc.)
        total_accuracy = weapon_accuracy * accuracy_modifier

        # Distance affects accuracy (spread grows with distance)
        distance_penalty = 1.0 - (max(0, distance_meters - 15) * 0.01)
        total_accuracy *= max(0.3, distance_penalty)

        # Calculate spread in degrees
        spread_degrees = weapon.first_shot_accuracy * (1.0 / max(0.1, total_accuracy))

        # Random shot placement within spread cone
        shot_offset = random.gauss(0, spread_degrees / 3)  # 3 sigma = 99.7% within spread

        if aiming_for_head:
            # Aiming at head - hit head if within head hitbox
            if abs(shot_offset) < self.HEAD_HITBOX_DEGREES:
                return (True, 'head')
            elif abs(shot_offset) < self.BODY_HITBOX_DEGREES:
                # Missed head, but hit body
                return (True, 'body')
            else:
                return (False, 'miss')
        else:
            # Aiming at body - larger target
            if abs(shot_offset) < self.BODY_HITBOX_DEGREES:
                # Check if accidentally got headshot
                if abs(shot_offset) < self.HEAD_HITBOX_DEGREES and random.random() < 0.1:
                    return (True, 'head')
                return (True, 'body')
            else:
                return (False, 'miss')

    def simulate_engagement(
        self,
        player_a_id: str,
        player_a_profile: PlayerCombatProfile,
        player_a_weapon: str,
        player_a_health: float,
        player_a_armor: float,
        player_a_movement: MovementState,
        player_b_id: str,
        player_b_profile: PlayerCombatProfile,
        player_b_weapon: str,
        player_b_health: float,
        player_b_armor: float,
        player_b_movement: MovementState,
        distance_meters: float,
        engagement_type: EngagementType,
        a_is_flashed: bool = False,
        b_is_flashed: bool = False,
        max_duration_ms: float = 3000.0
    ) -> CombatResult:
        """Simulate a full combat engagement tick-by-tick.

        Returns detailed result of the combat.
        """
        # Get weapon stats
        weapon_a = WEAPON_COMBAT_STATS.get(player_a_weapon.lower(), WEAPON_COMBAT_STATS['vandal'])
        weapon_b = WEAPON_COMBAT_STATS.get(player_b_weapon.lower(), WEAPON_COMBAT_STATS['vandal'])

        # Determine who is peeking
        a_is_peeking = engagement_type in [EngagementType.PEEK_VS_HOLD, EngagementType.BOTH_PEEKING]
        b_is_peeking = engagement_type in [EngagementType.HOLD_VS_PEEK, EngagementType.BOTH_PEEKING]

        # Calculate reaction times
        a_reaction = self.calculate_reaction_time(
            player_a_profile, engagement_type, a_is_peeking, a_is_flashed, distance_meters
        )
        b_reaction = self.calculate_reaction_time(
            player_b_profile, engagement_type, b_is_peeking, b_is_flashed, distance_meters
        )

        # Apply peeker's advantage
        if a_is_peeking and not b_is_peeking:
            b_reaction += self.PEEKERS_ADVANTAGE_MS
        elif b_is_peeking and not a_is_peeking:
            a_reaction += self.PEEKERS_ADVANTAGE_MS

        # Calculate crosshair adjustment time
        # Random initial offset based on crosshair placement
        a_offset = random.uniform(0, 15) * (1.0 - player_a_profile.crosshair_placement)
        b_offset = random.uniform(0, 15) * (1.0 - player_b_profile.crosshair_placement)

        a_aim_time = self.calculate_aim_adjustment_time(player_a_profile, distance_meters, a_offset, a_is_peeking)
        b_aim_time = self.calculate_aim_adjustment_time(player_b_profile, distance_meters, b_offset, b_is_peeking)

        # Counter-strafe delay
        a_strafe_delay, a_strafe_acc = self.calculate_counter_strafe_delay(
            player_a_profile,
            player_a_movement == MovementState.RUNNING,
            player_a_movement == MovementState.WALKING
        )
        b_strafe_delay, b_strafe_acc = self.calculate_counter_strafe_delay(
            player_b_profile,
            player_b_movement == MovementState.RUNNING,
            player_b_movement == MovementState.WALKING
        )

        # Total time to first shot
        a_first_shot_time = a_reaction + a_aim_time + a_strafe_delay
        b_first_shot_time = b_reaction + b_aim_time + b_strafe_delay

        # Combat state
        a_health = player_a_health + player_a_armor
        b_health = player_b_health + player_b_armor

        a_shots = 0
        b_shots = 0
        a_damage = 0.0
        b_damage = 0.0
        a_next_shot = a_first_shot_time
        b_next_shot = b_first_shot_time

        current_time = 0.0
        headshot_kill = False

        # Simulate tick by tick
        while current_time < max_duration_ms:
            # Determine who shoots next
            if a_health <= 0 or b_health <= 0:
                break

            if a_next_shot <= b_next_shot:
                # Player A shoots
                current_time = a_next_shot
                a_shots += 1

                # Calculate accuracy
                acc_mod = a_strafe_acc if a_shots == 1 else 1.0
                if a_is_flashed:
                    acc_mod *= 0.1

                # Shoot
                hit, region = self.calculate_hit_probability(
                    weapon_a, player_a_profile, distance_meters,
                    a_shots, acc_mod, aiming_for_head=True
                )

                if hit:
                    # Apply VCT distance-damage falloff
                    distance_mod = self.calculate_distance_damage_modifier(distance_meters)
                    if region == 'head':
                        damage = weapon_a.headshot_damage * distance_mod
                        headshot_kill = True
                    else:
                        damage = weapon_a.body_damage * distance_mod

                    b_health -= damage
                    a_damage += damage

                # Schedule next shot
                a_next_shot = current_time + weapon_a.ms_between_shots
            else:
                # Player B shoots
                current_time = b_next_shot
                b_shots += 1

                acc_mod = b_strafe_acc if b_shots == 1 else 1.0
                if b_is_flashed:
                    acc_mod *= 0.1

                hit, region = self.calculate_hit_probability(
                    weapon_b, player_b_profile, distance_meters,
                    b_shots, acc_mod, aiming_for_head=True
                )

                if hit:
                    # Apply VCT distance-damage falloff
                    distance_mod = self.calculate_distance_damage_modifier(distance_meters)
                    if region == 'head':
                        damage = weapon_b.headshot_damage * distance_mod
                        headshot_kill = True
                    else:
                        damage = weapon_b.body_damage * distance_mod

                    a_health -= damage
                    b_damage += damage

                b_next_shot = current_time + weapon_b.ms_between_shots

        # Determine winner
        if b_health <= 0 and a_health > 0:
            winner_id = player_a_id
            loser_id = player_b_id
            winner_damage = a_damage
            loser_damage = b_damage
            winner_shots = a_shots
            loser_shots = b_shots
        elif a_health <= 0 and b_health > 0:
            winner_id = player_b_id
            loser_id = player_a_id
            winner_damage = b_damage
            loser_damage = a_damage
            winner_shots = b_shots
            loser_shots = a_shots
            headshot_kill = headshot_kill and b_health > 0
        else:
            # Both alive (timeout) or both dead - whoever has more health wins
            if a_health >= b_health:
                winner_id = player_a_id
                loser_id = player_b_id
                winner_damage = a_damage
                loser_damage = b_damage
                winner_shots = a_shots
                loser_shots = b_shots
            else:
                winner_id = player_b_id
                loser_id = player_a_id
                winner_damage = b_damage
                loser_damage = a_damage
                winner_shots = b_shots
                loser_shots = a_shots

        return CombatResult(
            winner_id=winner_id,
            loser_id=loser_id,
            time_to_kill_ms=current_time,
            shots_fired_winner=winner_shots,
            shots_fired_loser=loser_shots,
            headshot_kill=headshot_kill and (a_health <= 0 or b_health <= 0),
            damage_dealt_winner=winner_damage,
            damage_dealt_loser=loser_damage,
            was_trade=engagement_type == EngagementType.TRADE,
            engagement_type=engagement_type
        )

    def quick_engagement(
        self,
        player_a_skill: float,  # 0-1 skill rating
        player_a_weapon: str,
        player_a_health: float,
        player_b_skill: float,
        player_b_weapon: str,
        player_b_health: float,
        distance_meters: float,
        a_is_peeking: bool = False,
        b_is_peeking: bool = False,
        a_is_flashed: bool = False,
        b_is_flashed: bool = False
    ) -> CombatResult:
        """Quick combat resolution using skill ratings.

        Converts skill (0-1) to a full combat profile for simulation.
        """
        # Convert skill to profile
        a_profile = self._skill_to_profile(player_a_skill)
        b_profile = self._skill_to_profile(player_b_skill)

        # Determine engagement type
        if a_is_peeking and not b_is_peeking:
            eng_type = EngagementType.PEEK_VS_HOLD
        elif b_is_peeking and not a_is_peeking:
            eng_type = EngagementType.HOLD_VS_PEEK
        elif a_is_peeking and b_is_peeking:
            eng_type = EngagementType.BOTH_PEEKING
        else:
            eng_type = EngagementType.BOTH_HOLDING

        # Movement state from peeking
        a_movement = MovementState.RUNNING if a_is_peeking else MovementState.STATIONARY
        b_movement = MovementState.RUNNING if b_is_peeking else MovementState.STATIONARY

        return self.simulate_engagement(
            player_a_id="player_a",
            player_a_profile=a_profile,
            player_a_weapon=player_a_weapon,
            player_a_health=player_a_health,
            player_a_armor=50,  # Default light armor
            player_a_movement=a_movement,
            player_b_id="player_b",
            player_b_profile=b_profile,
            player_b_weapon=player_b_weapon,
            player_b_health=player_b_health,
            player_b_armor=50,
            player_b_movement=b_movement,
            distance_meters=distance_meters,
            engagement_type=eng_type,
            a_is_flashed=a_is_flashed,
            b_is_flashed=b_is_flashed
        )

    def _skill_to_profile(self, skill: float) -> PlayerCombatProfile:
        """Convert a 0-1 skill rating to a full combat profile.

        VCT Calibration Notes:
        - Skill 1.0 = VCT pro player level
        - Skill 0.5 = Diamond/Immortal level
        - Skill 0.0 = Iron/Bronze level

        Parameters calibrated against:
        - Headshot rate: VCT avg 25%, range 15-34%
        - Crosshair placement: VCT hold angle std_angle analysis
        - Counter-strafe: Community verified 104ms timing
        """
        skill = max(0.0, min(1.0, skill))

        return PlayerCombatProfile(
            # Reaction time: Pro (1.0): 160ms, Bad (0.0): 280ms
            # Note: General FPS research values
            base_reaction_ms=280 - (skill * 120),
            reaction_variance=40 - (skill * 20),

            # Crosshair placement: VCT calibrated from hold_angles std_angle
            # Pro (1.0): 0.90 (tight angles, std <0.5 rad)
            # Average (0.5): 0.62 (medium angles, std ~1.0 rad)
            # Bad (0.0): 0.35 (very loose, std >1.5 rad)
            crosshair_placement=0.35 + (skill * 0.55),

            # Headshot rate: VCT verified (avg 25%, top 31-34%)
            # Pro (1.0): 30%, Bad (0.0): 15%
            headshot_rate=0.15 + (skill * 0.15),

            # Spray control: Could derive from VCT avg 3.9 hits to kill
            # Pro (1.0): 0.90, Bad (0.0): 0.30
            spray_control=0.30 + (skill * 0.60),

            # First shot discipline
            # Pro (1.0): 0.95, Bad (0.0): 0.50
            first_shot_discipline=0.50 + (skill * 0.45),

            # Counter-strafe: Community verified 104ms timing
            # Pro (1.0): 0.95, Bad (0.0): 0.45
            counter_strafe_skill=0.45 + (skill * 0.50),

            # Clutch factor: neutral by default
            clutch_factor=0.9 + (skill * 0.2),

            peek_aggression=0.5
        )


# Singleton instance for easy access
combat_model = RealisticCombatModel()


def resolve_combat(
    attacker_skill: float,
    attacker_weapon: str,
    attacker_health: float,
    defender_skill: float,
    defender_weapon: str,
    defender_health: float,
    distance_meters: float,
    attacker_peeking: bool = True,
    defender_peeking: bool = False,
    attacker_flashed: bool = False,
    defender_flashed: bool = False
) -> CombatResult:
    """Convenience function to resolve combat between two players."""
    return combat_model.quick_engagement(
        player_a_skill=attacker_skill,
        player_a_weapon=attacker_weapon,
        player_a_health=attacker_health,
        player_b_skill=defender_skill,
        player_b_weapon=defender_weapon,
        player_b_health=defender_health,
        distance_meters=distance_meters,
        a_is_peeking=attacker_peeking,
        b_is_peeking=defender_peeking,
        a_is_flashed=attacker_flashed,
        b_is_flashed=defender_flashed
    )
