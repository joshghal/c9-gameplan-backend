"""Weapon System for VALORANT tactical simulations.

Contains weapon statistics, damage calculations, and kill probability computations
based on actual VALORANT weapon data.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum
import math


class WeaponCategory(Enum):
    SIDEARM = "sidearm"
    SMG = "smg"
    SHOTGUN = "shotgun"
    RIFLE = "rifle"
    SNIPER = "sniper"
    LMG = "lmg"


@dataclass
class WeaponStats:
    """Statistics for a single weapon."""
    weapon_id: str
    name: str
    category: WeaponCategory
    cost: int
    head_damage: int
    body_damage: int
    leg_damage: int
    # [(max_distance_meters, damage_multiplier)] - falloff ranges
    damage_falloff: List[Tuple[float, float]]
    fire_rate: float  # rounds per second
    first_shot_accuracy: float  # 0-1, higher is more accurate
    magazine_size: int
    wall_penetration: str  # 'low', 'medium', 'high'

    def get_accuracy_at_distance(self, distance_meters: float) -> float:
        """Get effective accuracy at a given distance.

        Combines first_shot_accuracy with distance-based falloff to produce
        an overall accuracy modifier (0-1).

        Args:
            distance_meters: Distance to target in meters

        Returns:
            Accuracy modifier between 0 and 1
        """
        # Start with base first shot accuracy
        accuracy = self.first_shot_accuracy

        # Apply distance falloff - accuracy degrades at range
        # Use damage falloff as proxy for accuracy falloff
        falloff_mult = 1.0
        for max_dist, mult in self.damage_falloff:
            if distance_meters <= max_dist:
                falloff_mult = mult
                break
            falloff_mult = mult  # Use last value if beyond all ranges

        # Accuracy also drops at extreme range (beyond falloff)
        if distance_meters > 30:
            # Long range penalty
            range_penalty = 1.0 - ((distance_meters - 30) * 0.01)  # -1% per meter over 30
            range_penalty = max(0.5, range_penalty)  # Floor at 50%
            accuracy *= range_penalty

        # Apply weapon falloff effect on accuracy
        # Weapons with damage falloff also have accuracy falloff
        accuracy *= (0.5 + 0.5 * falloff_mult)  # Blend base accuracy with falloff

        return max(0.1, min(1.0, accuracy))  # Clamp between 10% and 100%


@dataclass
class ArmorStats:
    """Statistics for armor types."""
    armor_id: str
    name: str
    cost: int
    shield_value: int  # 25 light, 50 heavy
    damage_reduction: float  # multiplier applied to damage (0.66)


class WeaponDatabase:
    """Database of all VALORANT weapons and armor."""

    WEAPONS: Dict[str, WeaponStats] = {
        # Sidearms
        "classic": WeaponStats(
            weapon_id="classic",
            name="Classic",
            category=WeaponCategory.SIDEARM,
            cost=0,
            head_damage=78,
            body_damage=26,
            leg_damage=22,
            damage_falloff=[(30, 1.0), (50, 0.77)],
            fire_rate=6.75,
            first_shot_accuracy=0.4,
            magazine_size=12,
            wall_penetration="low"
        ),
        "shorty": WeaponStats(
            weapon_id="shorty",
            name="Shorty",
            category=WeaponCategory.SIDEARM,
            cost=150,
            head_damage=36,  # per pellet, 12 pellets
            body_damage=12,
            leg_damage=10,
            damage_falloff=[(7, 1.0), (15, 0.5)],
            fire_rate=3.33,
            first_shot_accuracy=0.3,
            magazine_size=2,
            wall_penetration="low"
        ),
        "frenzy": WeaponStats(
            weapon_id="frenzy",
            name="Frenzy",
            category=WeaponCategory.SIDEARM,
            cost=450,
            head_damage=78,
            body_damage=26,
            leg_damage=22,
            damage_falloff=[(20, 1.0), (50, 0.77)],
            fire_rate=10.0,
            first_shot_accuracy=0.35,
            magazine_size=13,
            wall_penetration="low"
        ),
        "ghost": WeaponStats(
            weapon_id="ghost",
            name="Ghost",
            category=WeaponCategory.SIDEARM,
            cost=500,
            head_damage=105,
            body_damage=30,
            leg_damage=26,
            damage_falloff=[(30, 1.0), (50, 0.86)],
            fire_rate=6.75,
            first_shot_accuracy=0.6,
            magazine_size=15,
            wall_penetration="medium"
        ),
        "sheriff": WeaponStats(
            weapon_id="sheriff",
            name="Sheriff",
            category=WeaponCategory.SIDEARM,
            cost=800,
            head_damage=159,
            body_damage=55,
            leg_damage=47,
            damage_falloff=[(30, 1.0), (50, 0.84)],
            fire_rate=4.0,
            first_shot_accuracy=0.75,
            magazine_size=6,
            wall_penetration="high"
        ),

        # SMGs
        "stinger": WeaponStats(
            weapon_id="stinger",
            name="Stinger",
            category=WeaponCategory.SMG,
            cost=1100,
            head_damage=67,
            body_damage=27,
            leg_damage=23,
            damage_falloff=[(20, 1.0), (50, 0.75)],
            fire_rate=18.0,
            first_shot_accuracy=0.35,
            magazine_size=20,
            wall_penetration="low"
        ),
        "spectre": WeaponStats(
            weapon_id="spectre",
            name="Spectre",
            category=WeaponCategory.SMG,
            cost=1600,
            head_damage=78,
            body_damage=26,
            leg_damage=22,
            damage_falloff=[(20, 1.0), (50, 0.77)],
            fire_rate=13.33,
            first_shot_accuracy=0.5,
            magazine_size=30,
            wall_penetration="medium"
        ),

        # Shotguns
        "bucky": WeaponStats(
            weapon_id="bucky",
            name="Bucky",
            category=WeaponCategory.SHOTGUN,
            cost=850,
            head_damage=44,  # per pellet, 15 pellets
            body_damage=22,
            leg_damage=19,
            damage_falloff=[(8, 1.0), (12, 0.5)],
            fire_rate=1.1,
            first_shot_accuracy=0.4,
            magazine_size=5,
            wall_penetration="low"
        ),
        "judge": WeaponStats(
            weapon_id="judge",
            name="Judge",
            category=WeaponCategory.SHOTGUN,
            cost=1850,
            head_damage=34,  # per pellet, 12 pellets
            body_damage=17,
            leg_damage=14,
            damage_falloff=[(10, 1.0), (15, 0.5)],
            fire_rate=3.5,
            first_shot_accuracy=0.35,
            magazine_size=7,
            wall_penetration="low"
        ),

        # Rifles
        "bulldog": WeaponStats(
            weapon_id="bulldog",
            name="Bulldog",
            category=WeaponCategory.RIFLE,
            cost=2050,
            head_damage=116,
            body_damage=35,
            leg_damage=30,
            damage_falloff=[(50, 1.0)],
            fire_rate=9.15,
            first_shot_accuracy=0.65,
            magazine_size=24,
            wall_penetration="medium"
        ),
        "guardian": WeaponStats(
            weapon_id="guardian",
            name="Guardian",
            category=WeaponCategory.RIFLE,
            cost=2250,
            head_damage=195,
            body_damage=65,
            leg_damage=49,
            damage_falloff=[(50, 1.0)],
            fire_rate=4.75,
            first_shot_accuracy=0.9,
            magazine_size=12,
            wall_penetration="high"
        ),
        "phantom": WeaponStats(
            weapon_id="phantom",
            name="Phantom",
            category=WeaponCategory.RIFLE,
            cost=2900,
            head_damage=156,
            body_damage=39,
            leg_damage=33,
            damage_falloff=[(15, 1.0), (30, 0.875), (50, 0.79)],
            fire_rate=11.0,
            first_shot_accuracy=0.85,
            magazine_size=30,
            wall_penetration="medium"
        ),
        "vandal": WeaponStats(
            weapon_id="vandal",
            name="Vandal",
            category=WeaponCategory.RIFLE,
            cost=2900,
            head_damage=160,
            body_damage=40,
            leg_damage=34,
            damage_falloff=[(50, 1.0)],  # No damage falloff
            fire_rate=9.75,
            first_shot_accuracy=0.8,
            magazine_size=25,
            wall_penetration="medium"
        ),

        # Snipers
        "marshal": WeaponStats(
            weapon_id="marshal",
            name="Marshal",
            category=WeaponCategory.SNIPER,
            cost=950,
            head_damage=202,
            body_damage=101,
            leg_damage=85,
            damage_falloff=[(50, 1.0)],
            fire_rate=1.5,
            first_shot_accuracy=0.95,
            magazine_size=5,
            wall_penetration="medium"
        ),
        "outlaw": WeaponStats(
            weapon_id="outlaw",
            name="Outlaw",
            category=WeaponCategory.SNIPER,
            cost=2400,
            head_damage=238,
            body_damage=140,
            leg_damage=119,
            damage_falloff=[(50, 1.0)],
            fire_rate=2.75,
            first_shot_accuracy=0.95,
            magazine_size=2,
            wall_penetration="high"
        ),
        "operator": WeaponStats(
            weapon_id="operator",
            name="Operator",
            category=WeaponCategory.SNIPER,
            cost=4700,
            head_damage=255,
            body_damage=150,
            leg_damage=120,
            damage_falloff=[(50, 1.0)],
            fire_rate=0.6,
            first_shot_accuracy=0.98,
            magazine_size=5,
            wall_penetration="high"
        ),

        # LMGs
        "ares": WeaponStats(
            weapon_id="ares",
            name="Ares",
            category=WeaponCategory.LMG,
            cost=1600,
            head_damage=72,
            body_damage=30,
            leg_damage=26,
            damage_falloff=[(30, 1.0), (50, 0.83)],
            fire_rate=13.0,
            first_shot_accuracy=0.4,
            magazine_size=50,
            wall_penetration="high"
        ),
        "odin": WeaponStats(
            weapon_id="odin",
            name="Odin",
            category=WeaponCategory.LMG,
            cost=3200,
            head_damage=95,
            body_damage=38,
            leg_damage=32,
            damage_falloff=[(30, 1.0), (50, 0.84)],
            fire_rate=12.0,
            first_shot_accuracy=0.45,
            magazine_size=100,
            wall_penetration="high"
        ),

        # Bandit (renamed Spectre from API data mapping)
        "bandit": WeaponStats(
            weapon_id="bandit",
            name="Bandit",
            category=WeaponCategory.SMG,
            cost=950,
            head_damage=72,
            body_damage=24,
            leg_damage=20,
            damage_falloff=[(20, 1.0), (50, 0.75)],
            fire_rate=14.0,
            first_shot_accuracy=0.4,
            magazine_size=20,
            wall_penetration="low"
        ),
    }

    ARMOR: Dict[str, ArmorStats] = {
        "none": ArmorStats(
            armor_id="none",
            name="none",
            cost=0,
            shield_value=0,
            damage_reduction=1.0  # Full damage taken
        ),
        "light": ArmorStats(
            armor_id="light",
            name="Light Armor",
            cost=400,
            shield_value=25,
            damage_reduction=0.66
        ),
        "heavy": ArmorStats(
            armor_id="heavy",
            name="Heavy Armor",
            cost=1000,
            shield_value=50,
            damage_reduction=0.66
        ),
        "regen": ArmorStats(
            armor_id="regen",
            name="Regen Shield",
            cost=1000,
            shield_value=50,
            damage_reduction=0.66
        ),
    }

    # Armor name normalization mapping
    ARMOR_NAME_MAP: Dict[str, str] = {
        "none": "none",
        "light armor": "light",
        "heavy armor": "heavy",
        "regen shield": "regen",
    }

    @classmethod
    def get_weapon(cls, weapon_name: str) -> Optional[WeaponStats]:
        """Get weapon stats by name (case-insensitive)."""
        normalized = weapon_name.lower().replace(" ", "").replace("-", "")
        for weapon_id, weapon in cls.WEAPONS.items():
            if weapon_id == normalized or weapon.name.lower().replace(" ", "") == normalized:
                return weapon
        return None

    @classmethod
    def get_armor(cls, armor_name: str) -> Optional[ArmorStats]:
        """Get armor stats by name (case-insensitive)."""
        normalized = armor_name.lower().strip()
        armor_id = cls.ARMOR_NAME_MAP.get(normalized, normalized)
        return cls.ARMOR.get(armor_id)

    @classmethod
    def calculate_damage(
        cls,
        weapon: WeaponStats,
        distance_meters: float,
        hit_region: str,  # 'head', 'body', 'leg'
        armor: ArmorStats,
        current_health: int,
        current_shield: int
    ) -> Tuple[int, int, int]:
        """Calculate damage dealt considering armor and distance.

        Returns:
            Tuple of (damage_to_health, damage_to_shield, new_shield_value)
        """
        # Get base damage for hit region
        if hit_region == 'head':
            base_damage = weapon.head_damage
        elif hit_region == 'leg':
            base_damage = weapon.leg_damage
        else:
            base_damage = weapon.body_damage

        # Apply distance falloff
        damage_multiplier = 1.0
        for max_dist, mult in weapon.damage_falloff:
            if distance_meters <= max_dist:
                damage_multiplier = mult
                break
            damage_multiplier = mult  # Use last falloff if beyond all ranges

        raw_damage = int(base_damage * damage_multiplier)

        # Apply armor
        if current_shield > 0 and armor.shield_value > 0:
            # Armor reduces damage by damage_reduction factor
            reduced_damage = int(raw_damage * armor.damage_reduction)
            shield_damage = min(reduced_damage, current_shield)
            health_damage = reduced_damage - shield_damage
            new_shield = current_shield - shield_damage
        else:
            health_damage = raw_damage
            shield_damage = 0
            new_shield = 0

        return (health_damage, shield_damage, new_shield)

    @classmethod
    def calculate_kill_probability(
        cls,
        attacker_weapon: WeaponStats,
        defender_weapon: WeaponStats,
        distance_meters: float,
        attacker_headshot_rate: float,
        defender_headshot_rate: float,
        attacker_armor: ArmorStats,
        defender_armor: ArmorStats,
        attacker_health: int = 100,
        defender_health: int = 100,
        attacker_shield: int = 0,
        defender_shield: int = 0
    ) -> Tuple[float, float]:
        """Calculate probability of each player winning the duel.

        Uses a simplified model based on:
        - Weapon damage output (time to kill)
        - First shot accuracy
        - Headshot rates
        - Distance effectiveness

        Returns:
            Tuple of (attacker_win_prob, defender_win_prob)
        """
        def calculate_ttk(
            weapon: WeaponStats,
            distance: float,
            headshot_rate: float,
            target_health: int,
            target_shield: int,
            target_armor: ArmorStats
        ) -> float:
            """Estimate time to kill in seconds."""
            # Calculate expected damage per shot
            head_dmg, _, _ = cls.calculate_damage(
                weapon, distance, 'head', target_armor, target_health, target_shield
            )
            body_dmg, _, _ = cls.calculate_damage(
                weapon, distance, 'body', target_armor, target_health, target_shield
            )

            avg_damage_per_shot = (head_dmg * headshot_rate) + (body_dmg * (1 - headshot_rate))

            # Factor in accuracy (simplified)
            effective_damage = avg_damage_per_shot * weapon.first_shot_accuracy

            if effective_damage <= 0:
                return float('inf')

            total_hp = target_health + target_shield
            shots_needed = math.ceil(total_hp / effective_damage)
            time_between_shots = 1.0 / weapon.fire_rate

            return shots_needed * time_between_shots

        # Calculate time to kill for both players
        attacker_ttk = calculate_ttk(
            attacker_weapon, distance_meters, attacker_headshot_rate,
            defender_health, defender_shield, defender_armor
        )
        defender_ttk = calculate_ttk(
            defender_weapon, distance_meters, defender_headshot_rate,
            attacker_health, attacker_shield, attacker_armor
        )

        # Edge cases
        if attacker_ttk == float('inf') and defender_ttk == float('inf'):
            return (0.5, 0.5)
        if attacker_ttk == float('inf'):
            return (0.0, 1.0)
        if defender_ttk == float('inf'):
            return (1.0, 0.0)

        # Convert TTK ratio to probability
        # Lower TTK = higher win probability
        total_ttk = attacker_ttk + defender_ttk
        attacker_win_prob = defender_ttk / total_ttk
        defender_win_prob = attacker_ttk / total_ttk

        # Apply weapon category bonuses
        # Snipers have advantage at long range
        if distance_meters > 25:
            if attacker_weapon.category == WeaponCategory.SNIPER:
                attacker_win_prob *= 1.3
            if defender_weapon.category == WeaponCategory.SNIPER:
                defender_win_prob *= 1.3

        # Shotguns have advantage at close range
        if distance_meters < 8:
            if attacker_weapon.category == WeaponCategory.SHOTGUN:
                attacker_win_prob *= 1.4
            if defender_weapon.category == WeaponCategory.SHOTGUN:
                defender_win_prob *= 1.4

        # SMGs are mobile advantage close-mid range
        if distance_meters < 20:
            if attacker_weapon.category == WeaponCategory.SMG:
                attacker_win_prob *= 1.1
            if defender_weapon.category == WeaponCategory.SMG:
                defender_win_prob *= 1.1

        # Normalize probabilities
        total = attacker_win_prob + defender_win_prob
        if total > 0:
            return (attacker_win_prob / total, defender_win_prob / total)
        return (0.5, 0.5)

    @classmethod
    def get_weapon_tier(cls, weapon: WeaponStats) -> int:
        """Get weapon tier for economy calculations (1-5)."""
        if weapon.category == WeaponCategory.SIDEARM:
            return 1 if weapon.cost == 0 else 2
        elif weapon.category == WeaponCategory.SMG:
            return 2
        elif weapon.category == WeaponCategory.SHOTGUN:
            return 2
        elif weapon.category == WeaponCategory.RIFLE:
            if weapon.cost >= 2900:
                return 4
            return 3
        elif weapon.category == WeaponCategory.SNIPER:
            if weapon.cost >= 4000:
                return 5
            return 3
        elif weapon.category == WeaponCategory.LMG:
            return 3
        return 1


def meters_from_normalized(normalized_distance: float, map_size_meters: float = 100.0) -> float:
    """Convert normalized distance (0-1) to meters.

    Typical VALORANT map dimensions are roughly 100m x 100m.
    """
    return normalized_distance * map_size_meters
