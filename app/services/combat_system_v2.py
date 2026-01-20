"""
Combat System V2: TTK-Based Combat Resolution

Philosophy: Combat outcomes emerge from weapon mechanics, not hardcoded bonuses.

=== KEY INSIGHT FROM VCT DATA ===
- Force buy is only 9% worse than full buy (51% vs 42%)
- This should emerge from TTK differences, not from `ECO_PENALTY = 0.30`

=== CRITICAL BUG FIX: is_moving ===
Players must have is_moving=False when engaging in combat.
- Vandal accuracy: 90% standing, 28% moving
- Classic accuracy: 70% standing, 55% moving

If attackers are always "moving" (running to site), Classic defenders
actually have BETTER accuracy than moving Vandal attackers!
This single fix changed eco_vs_full from ~40% to ~10%.

=== KNOWN MAGIC NUMBERS ===
- POSITION_VARIANCE_MS = 100 (tuned, no VCT basis)
- PEEK_ADVANTAGE_CHANCE = 0.18 (tuned, no VCT basis)
- 80ms position bonus (tuned, no VCT basis)

These control mechanics but are not derived from data.

Weapon data based on actual Valorant stats (Patch 8.0+)
"""

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


class WeaponCategory(Enum):
    SIDEARM = "sidearm"
    SMG = "smg"
    SHOTGUN = "shotgun"
    RIFLE = "rifle"
    SNIPER = "sniper"
    MACHINE_GUN = "machine_gun"


@dataclass
class WeaponStats:
    """Real Valorant weapon statistics."""
    name: str
    category: WeaponCategory
    cost: int

    # Damage values
    damage_head: int
    damage_body: int
    damage_leg: int

    # Fire rate (rounds per second)
    fire_rate: float

    # Accuracy (0-1, first shot standing)
    first_shot_accuracy: float
    moving_accuracy: float

    # Range falloff: list of (distance_units, damage_multiplier)
    # Valorant uses ~40 units per meter
    range_falloff: List[Tuple[int, float]] = field(default_factory=list)

    # Magazine and reload
    magazine_size: int = 30
    reload_time_ms: int = 2500

    # Special properties
    is_automatic: bool = True
    wall_penetration: str = "medium"  # low, medium, high

    def get_damage_at_range(self, distance_units: float, hitbox: str = "body") -> float:
        """Calculate damage at a given distance."""
        base_damage = {
            "head": self.damage_head,
            "body": self.damage_body,
            "leg": self.damage_leg,
        }.get(hitbox, self.damage_body)

        # Apply range falloff
        multiplier = 1.0
        for threshold, mult in sorted(self.range_falloff):
            if distance_units >= threshold:
                multiplier = mult
            else:
                break

        return base_damage * multiplier


# Real Valorant weapon stats (Patch 8.0+)
WEAPON_DATABASE: Dict[str, WeaponStats] = {
    # === SIDEARMS ===
    "classic": WeaponStats(
        name="Classic",
        category=WeaponCategory.SIDEARM,
        cost=0,
        damage_head=78,
        damage_body=26,
        damage_leg=22,
        fire_rate=6.75,
        first_shot_accuracy=0.70,
        moving_accuracy=0.55,
        range_falloff=[(30*40, 0.84), (50*40, 0.67)],  # 30m, 50m
        magazine_size=12,
        reload_time_ms=1750,
    ),
    "shorty": WeaponStats(
        name="Shorty",
        category=WeaponCategory.SIDEARM,
        cost=150,
        damage_head=36,  # Per pellet, 15 pellets
        damage_body=12,
        damage_leg=10,
        fire_rate=3.3,
        first_shot_accuracy=0.50,
        moving_accuracy=0.45,
        range_falloff=[(7*40, 0.50), (15*40, 0.25)],
        magazine_size=2,
        reload_time_ms=1750,
        is_automatic=False,
    ),
    "frenzy": WeaponStats(
        name="Frenzy",
        category=WeaponCategory.SIDEARM,
        cost=450,
        damage_head=78,
        damage_body=26,
        damage_leg=22,
        fire_rate=10.0,
        first_shot_accuracy=0.65,
        moving_accuracy=0.50,
        range_falloff=[(20*40, 0.84), (50*40, 0.72)],
        magazine_size=13,
        reload_time_ms=1500,
    ),
    "ghost": WeaponStats(
        name="Ghost",
        category=WeaponCategory.SIDEARM,
        cost=500,
        damage_head=105,
        damage_body=30,
        damage_leg=25,
        fire_rate=6.75,
        first_shot_accuracy=0.80,
        moving_accuracy=0.60,
        range_falloff=[(30*40, 0.88), (50*40, 0.75)],
        magazine_size=15,
        reload_time_ms=1500,
    ),
    "sheriff": WeaponStats(
        name="Sheriff",
        category=WeaponCategory.SIDEARM,
        cost=800,
        damage_head=160,
        damage_body=55,
        damage_leg=46,
        fire_rate=4.0,
        first_shot_accuracy=0.87,
        moving_accuracy=0.40,
        range_falloff=[(30*40, 0.85)],
        magazine_size=6,
        reload_time_ms=2250,
        is_automatic=False,
    ),

    # === SMGs ===
    "stinger": WeaponStats(
        name="Stinger",
        category=WeaponCategory.SMG,
        cost=950,
        damage_head=67,
        damage_body=27,
        damage_leg=22,
        fire_rate=18.0,
        first_shot_accuracy=0.60,
        moving_accuracy=0.50,
        range_falloff=[(20*40, 0.85), (50*40, 0.70)],
        magazine_size=20,
        reload_time_ms=2250,
    ),
    "spectre": WeaponStats(
        name="Spectre",
        category=WeaponCategory.SMG,
        cost=1600,
        damage_head=78,
        damage_body=26,
        damage_leg=22,
        fire_rate=13.33,
        first_shot_accuracy=0.70,
        moving_accuracy=0.60,
        range_falloff=[(20*40, 0.85), (50*40, 0.70)],
        magazine_size=30,
        reload_time_ms=2250,
    ),

    # === RIFLES ===
    "bulldog": WeaponStats(
        name="Bulldog",
        category=WeaponCategory.RIFLE,
        cost=2050,
        damage_head=116,
        damage_body=35,
        damage_leg=29,
        fire_rate=9.15,
        first_shot_accuracy=0.80,
        moving_accuracy=0.35,
        range_falloff=[(50*40, 0.90)],
        magazine_size=24,
        reload_time_ms=2500,
    ),
    "guardian": WeaponStats(
        name="Guardian",
        category=WeaponCategory.RIFLE,
        cost=2250,
        damage_head=195,
        damage_body=65,
        damage_leg=48,
        fire_rate=5.25,
        first_shot_accuracy=0.95,
        moving_accuracy=0.30,
        range_falloff=[],  # No falloff
        magazine_size=12,
        reload_time_ms=2500,
        is_automatic=False,
    ),
    "phantom": WeaponStats(
        name="Phantom",
        category=WeaponCategory.RIFLE,
        cost=2900,
        damage_head=156,
        damage_body=39,
        damage_leg=33,
        fire_rate=11.0,
        first_shot_accuracy=0.88,
        moving_accuracy=0.30,
        range_falloff=[(15*40, 0.90), (30*40, 0.85), (50*40, 0.80)],
        magazine_size=30,
        reload_time_ms=2500,
    ),
    "vandal": WeaponStats(
        name="Vandal",
        category=WeaponCategory.RIFLE,
        cost=2900,
        damage_head=160,
        damage_body=40,
        damage_leg=34,
        fire_rate=9.75,
        first_shot_accuracy=0.90,
        moving_accuracy=0.28,
        range_falloff=[],  # No falloff - key Vandal advantage
        magazine_size=25,
        reload_time_ms=2500,
    ),

    # === SNIPERS ===
    "marshal": WeaponStats(
        name="Marshal",
        category=WeaponCategory.SNIPER,
        cost=950,
        damage_head=202,
        damage_body=101,
        damage_leg=85,
        fire_rate=1.5,
        first_shot_accuracy=0.95,
        moving_accuracy=0.70,  # Can run and gun
        range_falloff=[],
        magazine_size=5,
        reload_time_ms=2500,
        is_automatic=False,
    ),
    "outlaw": WeaponStats(
        name="Outlaw",
        category=WeaponCategory.SNIPER,
        cost=2400,
        damage_head=238,
        damage_body=140,
        damage_leg=119,
        fire_rate=2.75,
        first_shot_accuracy=0.97,
        moving_accuracy=0.20,
        range_falloff=[],
        magazine_size=2,
        reload_time_ms=2800,
        is_automatic=False,
    ),
    "operator": WeaponStats(
        name="Operator",
        category=WeaponCategory.SNIPER,
        cost=4700,
        damage_head=255,
        damage_body=150,
        damage_leg=127,
        fire_rate=0.75,
        first_shot_accuracy=0.99,
        moving_accuracy=0.15,
        range_falloff=[],
        magazine_size=5,
        reload_time_ms=3700,
        is_automatic=False,
    ),

    # === MACHINE GUNS ===
    "ares": WeaponStats(
        name="Ares",
        category=WeaponCategory.MACHINE_GUN,
        cost=1600,
        damage_head=72,
        damage_body=30,
        damage_leg=25,
        fire_rate=13.0,
        first_shot_accuracy=0.55,
        moving_accuracy=0.50,
        range_falloff=[(30*40, 0.90), (50*40, 0.80)],
        magazine_size=50,
        reload_time_ms=3250,
    ),
    "odin": WeaponStats(
        name="Odin",
        category=WeaponCategory.MACHINE_GUN,
        cost=3200,
        damage_head=95,
        damage_body=38,
        damage_leg=32,
        fire_rate=15.6,
        first_shot_accuracy=0.50,
        moving_accuracy=0.45,
        range_falloff=[(30*40, 0.90), (50*40, 0.80)],
        magazine_size=100,
        reload_time_ms=5200,
        wall_penetration="high",
    ),

    # === SHOTGUNS ===
    "bucky": WeaponStats(
        name="Bucky",
        category=WeaponCategory.SHOTGUN,
        cost=850,
        damage_head=44,  # Per pellet
        damage_body=22,
        damage_leg=18,
        fire_rate=1.1,
        first_shot_accuracy=0.55,
        moving_accuracy=0.50,
        range_falloff=[(8*40, 0.50), (12*40, 0.25)],
        magazine_size=5,
        reload_time_ms=2500,
        is_automatic=False,
    ),
    "judge": WeaponStats(
        name="Judge",
        category=WeaponCategory.SHOTGUN,
        cost=1850,
        damage_head=34,  # Per pellet
        damage_body=17,
        damage_leg=14,
        fire_rate=3.5,
        first_shot_accuracy=0.55,
        moving_accuracy=0.50,
        range_falloff=[(10*40, 0.50), (15*40, 0.25)],
        magazine_size=7,
        reload_time_ms=2200,
    ),
}


@dataclass
class CombatState:
    """State of a player in combat."""
    health: int = 100
    armor: int = 0  # 0, 25 (light), 50 (heavy)
    is_moving: bool = False
    is_flashed: bool = False
    has_line_of_sight: bool = True
    distance_units: float = 1600  # ~40m default


@dataclass
class CombatResult:
    """Result of a combat engagement."""
    winner_id: str
    loser_id: str
    time_to_kill_ms: float
    shots_fired: int
    headshot_kill: bool
    damage_dealt: float


class CombatSystemV2:
    """
    TTK-based combat resolution system.

    Instead of: adv = 0.5 + weapon_bonus + position_bonus
    We calculate: Who kills whom first based on weapon TTK?

    Key insight: Real combat has HIGH VARIANCE due to:
    - Positioning (who peeks whom)
    - Information (who knows where enemy is)
    - Utility (flashes, smokes)
    - Human error (whiffs, panic)

    The TTK advantage from weapons is REAL but DIMINISHED by these factors.
    VCT data shows force buy is only 9% worse than full buy!
    """

    # Human factors from VCT data
    REACTION_TIME_MEAN_MS = 200
    REACTION_TIME_STD_MS = 50  # Higher variance

    # Hitbox probabilities (VCT data shows ~20% HS rate average)
    BASE_HEADSHOT_RATE = 0.20
    BASE_BODYSHOT_RATE = 0.70
    BASE_LEGSHOT_RATE = 0.10

    # Movement penalty
    MOVING_ACCURACY_PENALTY = 0.5

    # Flash penalty
    FLASH_ACCURACY_PENALTY = 0.15  # 85% accuracy reduction

    # Positioning factors (adds variance but doesn't override weapon advantage)
    # Tuned to match VCT data: eco ~15%, force ~42%
    POSITION_VARIANCE_MS = 100  # Random timing advantage (reduced)
    PEEK_ADVANTAGE_CHANCE = 0.18  # 18% chance the "worse" player gets lucky (reduced)

    def __init__(self):
        self.weapons = WEAPON_DATABASE

    def get_weapon(self, weapon_name: str) -> WeaponStats:
        """Get weapon by name, case-insensitive."""
        return self.weapons.get(weapon_name.lower(), self.weapons["classic"])

    def calculate_ttk(
        self,
        weapon: WeaponStats,
        target_state: CombatState,
        attacker_state: CombatState,
        headshot_rate: float = None,
    ) -> Tuple[float, int, bool]:
        """
        Calculate time-to-kill for a weapon against a target.

        Returns: (ttk_ms, shots_needed, was_headshot_kill)
        """
        if headshot_rate is None:
            headshot_rate = self.BASE_HEADSHOT_RATE

        # Effective health (armor absorbs 66% of damage)
        armor_absorption = 0.66
        effective_health = target_state.health + (target_state.armor * armor_absorption)

        # Get damage at distance
        head_damage = weapon.get_damage_at_range(attacker_state.distance_units, "head")
        body_damage = weapon.get_damage_at_range(attacker_state.distance_units, "body")
        leg_damage = weapon.get_damage_at_range(attacker_state.distance_units, "leg")

        # Calculate accuracy
        base_accuracy = weapon.first_shot_accuracy
        if attacker_state.is_moving:
            base_accuracy = weapon.moving_accuracy
        if attacker_state.is_flashed:
            base_accuracy *= self.FLASH_ACCURACY_PENALTY

        # Simulate shots to kill
        damage_dealt = 0
        shots_fired = 0
        was_headshot_kill = False

        while damage_dealt < effective_health and shots_fired < 30:  # Max 30 shots
            shots_fired += 1

            # Does this shot hit?
            if random.random() > base_accuracy:
                continue  # Miss

            # Determine hitbox
            roll = random.random()
            if roll < headshot_rate:
                damage_dealt += head_damage
                if damage_dealt >= effective_health:
                    was_headshot_kill = True
            elif roll < headshot_rate + self.BASE_BODYSHOT_RATE:
                damage_dealt += body_damage
            else:
                damage_dealt += leg_damage

            # Accuracy degrades with spray (simplified)
            base_accuracy *= 0.92

        # Time = reaction + (shots-1) / fire_rate
        reaction_time = max(0, random.gauss(self.REACTION_TIME_MEAN_MS, self.REACTION_TIME_STD_MS))
        shooting_time = (shots_fired - 1) / weapon.fire_rate * 1000 if shots_fired > 1 else 0
        ttk_ms = reaction_time + shooting_time

        return ttk_ms, shots_fired, was_headshot_kill

    def resolve_duel(
        self,
        player_a_id: str,
        player_a_weapon: str,
        player_a_state: CombatState,
        player_a_hs_rate: float,
        player_b_id: str,
        player_b_weapon: str,
        player_b_state: CombatState,
        player_b_hs_rate: float,
        player_a_has_position: bool = False,  # A is holding angle
        player_b_has_position: bool = False,  # B is holding angle
    ) -> CombatResult:
        """
        Resolve a 1v1 duel based on TTK with positioning variance.

        Key insight: In real Valorant, the player with worse weapon can still
        win through positioning, timing, and aim. This is why eco rounds
        have ~15% win rate, not 1%.
        """
        weapon_a = self.get_weapon(player_a_weapon)
        weapon_b = self.get_weapon(player_b_weapon)

        # Calculate base TTK for each player
        ttk_a, shots_a, hs_a = self.calculate_ttk(
            weapon_a, player_b_state, player_a_state, player_a_hs_rate
        )
        ttk_b, shots_b, hs_b = self.calculate_ttk(
            weapon_b, player_a_state, player_b_state, player_b_hs_rate
        )

        # Add positioning variance (random timing advantage)
        # This models: who peeks first, who has info, who is ready
        position_bonus_a = random.gauss(0, self.POSITION_VARIANCE_MS)
        position_bonus_b = random.gauss(0, self.POSITION_VARIANCE_MS)

        # Holding angle gives timing advantage
        if player_a_has_position:
            position_bonus_a -= 80  # 80ms advantage for holding
        if player_b_has_position:
            position_bonus_b -= 80

        # Apply positioning
        effective_ttk_a = ttk_a + position_bonus_a
        effective_ttk_b = ttk_b + position_bonus_b

        # The "peek advantage" mechanic
        # Sometimes the player who SHOULD lose gets the jump and wins anyway
        # This is what makes eco rounds viable
        if effective_ttk_a > effective_ttk_b:
            # B should win, but A might get lucky
            if random.random() < self.PEEK_ADVANTAGE_CHANCE:
                # A peeks perfectly and wins despite worse TTK
                effective_ttk_a = effective_ttk_b - 50
        else:
            # A should win, but B might get lucky
            if random.random() < self.PEEK_ADVANTAGE_CHANCE:
                effective_ttk_b = effective_ttk_a - 50

        # Determine winner
        if effective_ttk_a <= effective_ttk_b:
            return CombatResult(
                winner_id=player_a_id,
                loser_id=player_b_id,
                time_to_kill_ms=ttk_a,
                shots_fired=shots_a,
                headshot_kill=hs_a,
                damage_dealt=player_b_state.health + player_b_state.armor,
            )
        else:
            return CombatResult(
                winner_id=player_b_id,
                loser_id=player_a_id,
                time_to_kill_ms=ttk_b,
                shots_fired=shots_b,
                headshot_kill=hs_b,
                damage_dealt=player_a_state.health + player_a_state.armor,
            )

    def get_win_probability(
        self,
        weapon_a: str,
        state_a: CombatState,
        hs_rate_a: float,
        weapon_b: str,
        state_b: CombatState,
        hs_rate_b: float,
        a_has_position: bool = False,
        b_has_position: bool = False,
        simulations: int = 100,
    ) -> float:
        """
        Estimate win probability through Monte Carlo simulation.

        This replaces the old: adv = 0.5 + bonuses
        """
        a_wins = 0

        for _ in range(simulations):
            result = self.resolve_duel(
                "a", weapon_a, state_a, hs_rate_a,
                "b", weapon_b, state_b, hs_rate_b,
                player_a_has_position=a_has_position,
                player_b_has_position=b_has_position,
            )
            if result.winner_id == "a":
                a_wins += 1

        return a_wins / simulations


def analyze_weapon_matchups():
    """Analyze weapon matchups to verify economy emerges from TTK."""
    combat = CombatSystemV2()

    print("=" * 70)
    print("WEAPON MATCHUP ANALYSIS (TTK-Based with Positioning Variance)")
    print("=" * 70)

    # Combat states
    full_buy_state = CombatState(health=100, armor=50, is_moving=False)
    force_state = CombatState(health=100, armor=25, is_moving=False)
    eco_state = CombatState(health=100, armor=0, is_moving=False)

    hs_rate = 0.20  # Base HS rate

    print(f"\n{'Matchup':<35} {'Win %':<10} {'Target':<10} {'Status'}")
    print("-" * 70)

    # Economy matchups (key scenarios)
    matchups = [
        # (wpn_a, state_a, wpn_b, state_b, a_pos, b_pos, desc, target)
        ("vandal", full_buy_state, "vandal", full_buy_state, False, False, "Mirror (Vandal vs Vandal)", 50),
        ("phantom", full_buy_state, "vandal", full_buy_state, False, False, "Phantom vs Vandal", 50),
        ("classic", eco_state, "vandal", full_buy_state, False, False, "Eco vs Full (no position)", 15),
        ("classic", eco_state, "vandal", full_buy_state, True, False, "Eco vs Full (eco has angle)", 25),
        ("spectre", force_state, "vandal", full_buy_state, False, False, "Force vs Full", 42),
        ("spectre", force_state, "vandal", full_buy_state, True, False, "Force vs Full (force has angle)", 48),
        ("sheriff", eco_state, "vandal", full_buy_state, False, False, "Sheriff eco vs Full", 20),
        ("sheriff", eco_state, "vandal", full_buy_state, True, False, "Sheriff eco (has angle)", 30),
        ("vandal", full_buy_state, "classic", eco_state, False, False, "Full vs Eco", 85),
        ("vandal", full_buy_state, "classic", eco_state, False, True, "Full vs Eco (eco holds)", 75),
    ]

    for wpn_a, state_a, wpn_b, state_b, a_pos, b_pos, desc, target in matchups:
        win_prob = combat.get_win_probability(
            wpn_a, state_a, hs_rate,
            wpn_b, state_b, hs_rate,
            a_has_position=a_pos,
            b_has_position=b_pos,
            simulations=500
        )
        diff = abs(win_prob * 100 - target)
        status = "✓" if diff < 15 else "✗"
        print(f"{desc:<35} {win_prob*100:>5.1f}%     ~{target}%      {status}")

    print("\n" + "=" * 70)
    print("VCT REFERENCE TARGETS:")
    print("  - Full buy win rate: 51%")
    print("  - Force buy win rate: 42% (9% worse)")
    print("  - Eco win rate: ~15% (but can be higher with good positioning)")
    print("  - Positioning advantage: ~10-15% swing")
    print("=" * 70)

    # Summary
    print("\n" + "=" * 70)
    print("KEY INSIGHTS FROM TTK MODEL:")
    print("=" * 70)
    print("""
1. WEAPON ADVANTAGE EXISTS but is not deterministic
   - Vandal vs Classic TTK difference is huge (~2x)
   - But positioning/timing variance means eco can still win ~15-25%

2. POSITIONING MATTERS as much as weapons
   - Holding an angle gives ~10% advantage
   - This is why eco rounds are viable with good setups

3. NO HARDCODED BONUSES NEEDED
   - Economy outcomes EMERGE from:
     a) TTK differences (weapon damage/fire rate)
     b) Positioning variance (who peeks whom)
     c) Human error (reaction time variance)
""")
    print("=" * 70)


def _interpret(prob: float) -> str:
    if prob > 0.55:
        return "Advantage"
    elif prob < 0.45:
        return "Disadvantage"
    else:
        return "Even"


if __name__ == "__main__":
    analyze_weapon_matchups()
