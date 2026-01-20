"""Economy Engine for VALORANT tactical simulations.

Handles buy type classification, loadout generation, and economic decision-making
based on actual VALORANT economy rules and observed data patterns.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import random

from .weapon_system import WeaponDatabase, WeaponStats, ArmorStats, WeaponCategory
from .data_loader import get_data_loader


class BuyType(Enum):
    """Classification of team buy types based on loadout value."""
    PISTOL = "pistol"   # Round 1 or < 1000 loadout
    ECO = "eco"         # 1000-2000 loadout (save round)
    FORCE = "force"     # 2000-3500 loadout (buying suboptimal)
    HALF = "half"       # 3500-4500 loadout (partial buy)
    FULL = "full"       # 4500+ loadout (full rifles + heavy)


@dataclass
class Loadout:
    """A player's loadout for a round."""
    weapon: WeaponStats
    armor: ArmorStats
    total_value: int

    @property
    def weapon_name(self) -> str:
        return self.weapon.name

    @property
    def armor_name(self) -> str:
        return self.armor.name


@dataclass
class TeamEconomy:
    """Economic state for a team."""
    credits: List[int] = field(default_factory=lambda: [800] * 5)  # Per player
    loss_streak: int = 0  # Consecutive losses (0-4 for bonus)
    round_wins: int = 0
    round_losses: int = 0

    @property
    def total_credits(self) -> int:
        return sum(self.credits)

    @property
    def average_credits(self) -> int:
        return self.total_credits // len(self.credits)

    @property
    def min_credits(self) -> int:
        return min(self.credits)


class EconomyEngine:
    """Engine for handling VALORANT economic decisions.

    Now integrated with VCT economy patterns for map-specific insights.
    """

    # Loss bonus progression (after 1st, 2nd, 3rd, 4th+ losses)
    LOSS_BONUS = [1900, 2400, 2900, 3400, 3400]

    # VCT Buy Win Rates (from 33 VCT matches, 3600+ rounds)
    # Source: economy_patterns.json
    VCT_BUY_WIN_RATES = {
        'full_buy': 0.509,   # 50.9% win rate (1418/2782)
        'half_buy': 0.498,   # 49.8% win rate (131/263)
        'force_buy': 0.422,  # 42.2% win rate (160/379)
        'eco': 0.540,        # 54.0% win rate (27/50) - small sample
    }

    # VCT Side-Specific Buy Distribution (attack vs defense)
    # Source: economy_patterns.json - by_side section
    # Shows attackers force more, defenders eco less
    VCT_SIDE_BUY_DISTRIBUTION = {
        'attacker': {
            'full_buy': 1297,   # 74.7%
            'half_buy': 192,    # 11.0%
            'force_buy': 199,   # 11.5%
            'eco': 49,          # 2.8%
        },
        'defender': {
            'full_buy': 1485,   # 85.5%
            'half_buy': 71,     # 4.1%
            'force_buy': 180,   # 10.4%
            'eco': 1,           # 0.06% (defenders almost never eco)
        },
    }

    @classmethod
    def get_map_economy_data(cls, map_name: str, side: str = 'attack') -> Optional[Dict]:
        """Get VCT map-specific economy data.

        Returns average loadout values and buy distributions for the map.
        """
        try:
            data_loader = get_data_loader()
            return data_loader.get_economy_by_map(map_name)
        except Exception:
            return None

    @classmethod
    def get_vct_buy_win_rate(cls, buy_type: 'BuyType') -> float:
        """Get VCT-derived win rate for a buy type.

        This helps inform economic decisions - e.g., force buying has lower
        win rate than full buying, but sometimes necessary.
        """
        type_map = {
            BuyType.FULL: 'full_buy',
            BuyType.HALF: 'half_buy',
            BuyType.FORCE: 'force_buy',
            BuyType.ECO: 'eco',
            BuyType.PISTOL: 'eco',  # Pistol rounds similar to eco
        }
        key = type_map.get(buy_type, 'eco')
        return cls.VCT_BUY_WIN_RATES.get(key, 0.45)

    # Base round income
    WIN_BONUS = 3000
    KILL_BONUS = 200
    SPIKE_PLANT_BONUS = 300  # For attackers
    SPIKE_DEFUSE_BONUS = 300  # For defenders

    # Buy thresholds
    BUY_THRESHOLDS = {
        BuyType.PISTOL: 1000,
        BuyType.ECO: 2000,
        BuyType.FORCE: 3500,
        BuyType.HALF: 4500,
        BuyType.FULL: float('inf'),
    }

    # Weapon preferences by buy type and role
    WEAPON_PREFERENCES = {
        BuyType.PISTOL: {
            'default': ['classic', 'ghost', 'frenzy', 'shorty'],
            'duelist': ['ghost', 'frenzy', 'classic'],
            'controller': ['ghost', 'classic'],
            'sentinel': ['ghost', 'classic', 'shorty'],
            'initiator': ['ghost', 'classic'],
        },
        BuyType.ECO: {
            'default': ['sheriff', 'marshal', 'ghost', 'spectre'],
            'duelist': ['sheriff', 'marshal', 'ghost'],
            'controller': ['sheriff', 'ghost'],
            'sentinel': ['marshal', 'sheriff', 'ghost'],
            'initiator': ['sheriff', 'ghost'],
        },
        BuyType.FORCE: {
            'default': ['spectre', 'marshal', 'bulldog', 'stinger', 'bandit'],
            'duelist': ['spectre', 'marshal', 'bulldog'],
            'controller': ['spectre', 'stinger'],
            'sentinel': ['marshal', 'spectre', 'judge'],
            'initiator': ['spectre', 'bulldog'],
        },
        BuyType.HALF: {
            'default': ['bulldog', 'guardian', 'spectre', 'marshal'],
            'duelist': ['guardian', 'bulldog', 'spectre'],
            'controller': ['bulldog', 'spectre'],
            'sentinel': ['marshal', 'guardian', 'bulldog'],
            'initiator': ['bulldog', 'guardian'],
        },
        BuyType.FULL: {
            'default': ['vandal', 'phantom', 'operator'],
            'duelist': ['vandal', 'phantom', 'operator'],
            'controller': ['phantom', 'vandal'],
            'sentinel': ['operator', 'vandal', 'phantom'],
            'initiator': ['phantom', 'vandal'],
        },
    }

    # Weapon distribution (VCT-calibrated from opponent_profiles.json)
    # Based on C9 player weapon preferences (kill-weighted)
    # Note: Limited sample - only 10 players analyzed
    # Vandal dominance aligns with VCT meta (46% usage)
    WEAPON_DISTRIBUTION = {
        'vandal': 0.462,     # VCT: 46.2% - dominant rifle
        'phantom': 0.317,    # VCT: 31.7% - secondary rifle
        'ghost': 0.070,      # VCT: 7.0% - pistol round/eco
        'classic': 0.057,    # VCT: 5.7% - pistol round
        'bulldog': 0.035,    # VCT: 3.5% - force buy
        'sheriff': 0.030,    # VCT: 3.0% - eco round
        'operator': 0.021,   # VCT: 2.1% - AWP role
        'spectre': 0.002,    # VCT: 0.2% - force buy
        'stinger': 0.004,    # Estimated - rarely seen in VCT
        'guardian': 0.003,   # Estimated - rarely seen in VCT
        'marshal': 0.003,    # Estimated - eco sniper
        'ares': 0.002,       # Estimated - rarely used
        'odin': 0.001,       # Estimated - wall bang strat
        'judge': 0.001,      # Estimated - corner hold
        'bucky': 0.001,      # Estimated - eco cheese
    }

    # Armor preferences
    ARMOR_BY_BUY_TYPE = {
        BuyType.PISTOL: {'none': 0.6, 'light': 0.4, 'heavy': 0.0},
        BuyType.ECO: {'none': 0.4, 'light': 0.5, 'heavy': 0.1},
        BuyType.FORCE: {'none': 0.1, 'light': 0.5, 'heavy': 0.4},
        BuyType.HALF: {'none': 0.0, 'light': 0.3, 'heavy': 0.7},
        BuyType.FULL: {'none': 0.0, 'light': 0.0, 'heavy': 1.0},
    }

    @classmethod
    def classify_buy_type(
        cls,
        team_loadout_value: int,
        round_num: int,
        is_pistol_round: bool = False
    ) -> BuyType:
        """Classify the buy type based on total team loadout value."""
        if is_pistol_round or round_num == 0:
            return BuyType.PISTOL

        avg_per_player = team_loadout_value / 5

        if avg_per_player < cls.BUY_THRESHOLDS[BuyType.PISTOL]:
            return BuyType.ECO
        elif avg_per_player < cls.BUY_THRESHOLDS[BuyType.ECO]:
            return BuyType.ECO
        elif avg_per_player < cls.BUY_THRESHOLDS[BuyType.FORCE]:
            return BuyType.FORCE
        elif avg_per_player < cls.BUY_THRESHOLDS[BuyType.HALF]:
            return BuyType.HALF
        else:
            return BuyType.FULL

    @classmethod
    def generate_team_loadout(
        cls,
        team_economy: TeamEconomy,
        round_num: int,
        agent_roles: Optional[List[str]] = None,
        map_name: Optional[str] = None,
        side: str = 'attack',
        forced_buy_type: Optional[BuyType] = None
    ) -> List[Loadout]:
        """Generate loadouts for a team based on their economy.

        Args:
            team_economy: Current economic state of the team
            round_num: Current round number (0-indexed)
            agent_roles: List of agent roles for each player
            map_name: Name of the map (for map-specific preferences)
            side: 'attack' or 'defense'
            forced_buy_type: Force a specific buy type (optional)

        Returns:
            List of 5 Loadout objects for the team
        """
        is_pistol = round_num == 0 or round_num == 12  # First round each half

        if forced_buy_type:
            buy_type = forced_buy_type
        else:
            # Determine buy type based on economy
            total_value = sum(
                cls._estimate_max_loadout_value(credits, is_pistol)
                for credits in team_economy.credits
            )
            buy_type = cls.classify_buy_type(total_value, round_num, is_pistol)

        # Generate loadout for each player
        loadouts = []
        roles = agent_roles or ['default'] * 5

        for i, credits in enumerate(team_economy.credits):
            role = roles[i] if i < len(roles) else 'default'
            loadout = cls._generate_player_loadout(
                credits, buy_type, role, is_pistol
            )
            loadouts.append(loadout)

        return loadouts

    @classmethod
    def _estimate_max_loadout_value(cls, credits: int, is_pistol: bool) -> int:
        """Estimate maximum loadout value a player can afford."""
        if is_pistol:
            return min(credits, 800)  # Pistol round cap
        return credits

    @classmethod
    def _generate_player_loadout(
        cls,
        credits: int,
        buy_type: BuyType,
        role: str,
        is_pistol: bool
    ) -> Loadout:
        """Generate a single player's loadout."""
        # Get weapon preferences for this buy type and role
        role_key = role.lower() if role.lower() in cls.WEAPON_PREFERENCES[buy_type] else 'default'
        weapon_prefs = cls.WEAPON_PREFERENCES[buy_type][role_key]

        # Get armor preferences
        armor_probs = cls.ARMOR_BY_BUY_TYPE[buy_type]

        # Select armor first
        armor = cls._select_armor(credits, armor_probs, is_pistol)
        remaining_credits = credits - armor.cost

        # Select weapon
        weapon = cls._select_weapon(remaining_credits, weapon_prefs, is_pistol)

        total_value = weapon.cost + armor.cost

        return Loadout(weapon=weapon, armor=armor, total_value=total_value)

    @classmethod
    def _select_armor(
        cls,
        credits: int,
        probabilities: Dict[str, float],
        is_pistol: bool
    ) -> ArmorStats:
        """Select armor based on credits and probabilities."""
        db = WeaponDatabase

        # Filter to affordable armors
        affordable = []
        weights = []

        for armor_id, prob in probabilities.items():
            armor = db.ARMOR.get(armor_id)
            if armor and armor.cost <= credits:
                affordable.append(armor)
                weights.append(prob)

        if not affordable:
            return db.ARMOR['none']

        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]

        # Random selection
        return random.choices(affordable, weights=weights, k=1)[0]

    @classmethod
    def _select_weapon(
        cls,
        credits: int,
        preferences: List[str],
        is_pistol: bool
    ) -> WeaponStats:
        """Select weapon based on credits and preferences."""
        db = WeaponDatabase

        # Filter to affordable weapons in preference order
        for weapon_id in preferences:
            weapon = db.WEAPONS.get(weapon_id)
            if weapon and weapon.cost <= credits:
                # Add some randomness - 70% chance to pick preferred, 30% to continue
                if random.random() < 0.7:
                    return weapon

        # Fallback: pick best affordable weapon from preferences
        for weapon_id in preferences:
            weapon = db.WEAPONS.get(weapon_id)
            if weapon and weapon.cost <= credits:
                return weapon

        # Ultimate fallback: classic
        return db.WEAPONS['classic']

    @classmethod
    def calculate_round_income(
        cls,
        won: bool,
        loss_streak: int,
        kills: int = 0,
        spike_planted: bool = False,
        spike_defused: bool = False,
        is_attacker: bool = True
    ) -> int:
        """Calculate income for end of round.

        Args:
            won: Whether the team won the round
            loss_streak: Current loss streak before this round
            kills: Number of kills the player got
            spike_planted: Whether spike was planted (attack bonus)
            spike_defused: Whether spike was defused (defense bonus)
            is_attacker: Whether this is the attacking team

        Returns:
            Credits earned this round
        """
        income = 0

        if won:
            income += cls.WIN_BONUS
        else:
            # Loss bonus based on streak
            streak_index = min(loss_streak, len(cls.LOSS_BONUS) - 1)
            income += cls.LOSS_BONUS[streak_index]

        # Kill bonus
        income += kills * cls.KILL_BONUS

        # Objective bonuses
        if spike_planted and is_attacker:
            income += cls.SPIKE_PLANT_BONUS

        if spike_defused and not is_attacker:
            income += cls.SPIKE_DEFUSE_BONUS

        return income

    @classmethod
    def update_team_economy(
        cls,
        team_economy: TeamEconomy,
        won: bool,
        player_kills: List[int],
        spike_planted: bool = False,
        spike_defused: bool = False,
        is_attacker: bool = True
    ) -> TeamEconomy:
        """Update team economy after a round.

        Args:
            team_economy: Current team economy state
            won: Whether team won the round
            player_kills: List of kills per player
            spike_planted: Whether spike was planted
            spike_defused: Whether spike was defused
            is_attacker: Whether this is attacking team

        Returns:
            Updated TeamEconomy
        """
        # Update loss streak
        if won:
            new_loss_streak = 0
            new_wins = team_economy.round_wins + 1
            new_losses = team_economy.round_losses
        else:
            new_loss_streak = min(team_economy.loss_streak + 1, 4)
            new_wins = team_economy.round_wins
            new_losses = team_economy.round_losses + 1

        # Calculate income for each player
        new_credits = []
        for i, current_credits in enumerate(team_economy.credits):
            kills = player_kills[i] if i < len(player_kills) else 0
            income = cls.calculate_round_income(
                won=won,
                loss_streak=team_economy.loss_streak,
                kills=kills,
                spike_planted=spike_planted,
                spike_defused=spike_defused,
                is_attacker=is_attacker
            )
            # Cap at 9000 credits
            new_credits.append(min(current_credits + income, 9000))

        return TeamEconomy(
            credits=new_credits,
            loss_streak=new_loss_streak,
            round_wins=new_wins,
            round_losses=new_losses
        )

    @classmethod
    def should_eco(cls, team_economy: TeamEconomy, round_num: int) -> bool:
        """Determine if team should eco based on economy state."""
        # Average credits needed for a full buy
        FULL_BUY_COST = 3900  # Rifle + Heavy

        # If everyone can afford full buy, no need to eco
        if all(c >= FULL_BUY_COST for c in team_economy.credits):
            return False

        # If average is low and it's not pistol round
        if team_economy.average_credits < 2500 and round_num > 0:
            return True

        # If minimum player credits is very low
        if team_economy.min_credits < 1500:
            # Check if forcing would leave team broke
            if team_economy.average_credits < 3000:
                return True

        return False

    @classmethod
    def get_recommended_buy_type(
        cls,
        team_economy: TeamEconomy,
        round_num: int,
        opponent_economy: Optional[TeamEconomy] = None
    ) -> BuyType:
        """Get recommended buy type based on economy state."""
        is_pistol = round_num == 0 or round_num == 12

        if is_pistol:
            return BuyType.PISTOL

        avg_credits = team_economy.average_credits

        # Full buy if affordable
        if avg_credits >= 4500:
            return BuyType.FULL

        # Half buy if can afford decent weapons
        if avg_credits >= 3500:
            return BuyType.HALF

        # Force if opponent is weak or loss streak is high
        if opponent_economy and opponent_economy.average_credits < 2000:
            if avg_credits >= 2000:
                return BuyType.FORCE

        # Force on bonus round (high loss streak)
        if team_economy.loss_streak >= 2 and avg_credits >= 2500:
            return BuyType.FORCE

        # Eco to save for next round
        if avg_credits < 2500:
            return BuyType.ECO

        return BuyType.FORCE
