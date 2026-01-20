"""Round State tracking for VALORANT tactical simulations.

Tracks first blood, man advantage, trades, and calculates real-time win probability
based on observed patterns from professional VALORANT matches.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime


@dataclass
class KillEvent:
    """Record of a kill during the round."""
    time_ms: int
    killer_id: str
    killer_team: str  # 'attack' or 'defense'
    victim_id: str
    victim_team: str
    position: Tuple[float, float]
    weapon: str
    is_headshot: bool
    is_wallbang: bool = False
    is_first_blood: bool = False
    is_trade: bool = False


@dataclass
class RoundState:
    """Current state of a round for win probability calculations."""

    # Kill tracking
    first_blood_team: Optional[str] = None  # 'attack' or 'defense'
    first_blood_time_ms: int = 0
    kills: List[KillEvent] = field(default_factory=list)

    # Alive counts
    attack_alive: int = 5
    defense_alive: int = 5

    # Trade tracking
    last_kill_time_ms: int = 0
    last_kill_position: Optional[Tuple[float, float]] = None
    last_killed_team: Optional[str] = None
    potential_trade_window: bool = False  # Within trade window

    # Spike state
    spike_planted: bool = False
    spike_plant_time_ms: int = 0
    spike_site: Optional[str] = None
    spike_defused: bool = False
    spike_defuse_start_ms: int = 0
    spike_defuser_id: Optional[str] = None
    spike_exploded: bool = False

    # Round timing constants (in ms)
    ROUND_TIME_MS: int = 100000  # 1:40 round time
    SPIKE_PLANT_TIME_MS: int = 4000  # 4 seconds to plant
    SPIKE_DEFUSE_TIME_MS: int = 7000  # 7 seconds to defuse
    SPIKE_HALF_DEFUSE_MS: int = 3500  # 3.5 seconds for half defuse
    SPIKE_EXPLOSION_TIME_MS: int = 45000  # 45 seconds after plant

    # Economy context
    attack_buy_type: str = "full"  # pistol, eco, force, half, full
    defense_buy_type: str = "full"

    def record_kill(
        self,
        time_ms: int,
        killer_id: str,
        killer_team: str,
        victim_id: str,
        victim_team: str,
        position: Tuple[float, float],
        weapon: str = "unknown",
        is_headshot: bool = False,
        is_wallbang: bool = False
    ) -> KillEvent:
        """Record a kill and update round state."""
        # Check if this is first blood
        is_first_blood = len(self.kills) == 0

        # Check if this is a trade (kill within 5 seconds of teammate death)
        TRADE_WINDOW_MS = 5000
        is_trade = (
            self.last_killed_team == killer_team and
            time_ms - self.last_kill_time_ms <= TRADE_WINDOW_MS and
            self.potential_trade_window
        )

        kill_event = KillEvent(
            time_ms=time_ms,
            killer_id=killer_id,
            killer_team=killer_team,
            victim_id=victim_id,
            victim_team=victim_team,
            position=position,
            weapon=weapon,
            is_headshot=is_headshot,
            is_wallbang=is_wallbang,
            is_first_blood=is_first_blood,
            is_trade=is_trade
        )

        self.kills.append(kill_event)

        # Update first blood
        if is_first_blood:
            self.first_blood_team = killer_team
            self.first_blood_time_ms = time_ms

        # Update alive counts
        if victim_team == 'attack':
            self.attack_alive -= 1
        else:
            self.defense_alive -= 1

        # Update trade tracking
        self.last_kill_time_ms = time_ms
        self.last_kill_position = position
        self.last_killed_team = victim_team
        self.potential_trade_window = True

        return kill_event

    def update_trade_window(self, time_ms: int):
        """Update trade window status based on time."""
        TRADE_WINDOW_MS = 5000
        if time_ms - self.last_kill_time_ms > TRADE_WINDOW_MS:
            self.potential_trade_window = False

    def plant_spike(self, time_ms: int, site: str):
        """Record spike plant."""
        self.spike_planted = True
        self.spike_plant_time_ms = time_ms
        self.spike_site = site

    def start_defuse(self, time_ms: int, defuser_id: str):
        """Start spike defuse."""
        self.spike_defuse_start_ms = time_ms
        self.spike_defuser_id = defuser_id

    def cancel_defuse(self):
        """Cancel spike defuse (defuser killed or moved)."""
        self.spike_defuse_start_ms = 0
        self.spike_defuser_id = None

    def complete_defuse(self, time_ms: int):
        """Complete spike defuse."""
        self.spike_defused = True
        self.spike_defuser_id = None

    def defuse_spike(self, time_ms: int):
        """Record successful spike defuse (alias for complete_defuse).

        Called by simulation engine when defuse is completed.
        """
        self.complete_defuse(time_ms)

    def check_spike_explosion(self, time_ms: int) -> bool:
        """Check if spike should explode. Returns True if exploded."""
        if self.spike_planted and not self.spike_defused and not self.spike_exploded:
            time_since_plant = time_ms - self.spike_plant_time_ms
            if time_since_plant >= self.SPIKE_EXPLOSION_TIME_MS:
                self.spike_exploded = True
                return True
        return False

    def check_defuse_complete(self, time_ms: int, half_defuse: bool = False) -> bool:
        """Check if defuse should complete. Returns True if defused."""
        if self.spike_defuse_start_ms > 0 and self.spike_defuser_id:
            defuse_time = self.SPIKE_HALF_DEFUSE_MS if half_defuse else self.SPIKE_DEFUSE_TIME_MS
            time_defusing = time_ms - self.spike_defuse_start_ms
            if time_defusing >= defuse_time:
                self.complete_defuse(time_ms)
                return True
        return False

    def get_round_end_condition(self, time_ms: int) -> Optional[Dict[str, str]]:
        """Check if round should end and return the condition.

        Returns:
            None if round continues, or Dict with 'winner' and 'reason'
        """
        # Check eliminations
        if self.attack_alive == 0 and self.defense_alive == 0:
            # Mutual elimination (rare)
            return {'winner': 'defense', 'reason': 'elimination_tie'}
        if self.attack_alive == 0:
            return {'winner': 'defense', 'reason': 'elimination'}
        if self.defense_alive == 0:
            if self.spike_planted:
                return {'winner': 'attack', 'reason': 'elimination_post_plant'}
            return {'winner': 'attack', 'reason': 'elimination'}

        # Check spike explosion
        if self.spike_exploded:
            return {'winner': 'attack', 'reason': 'spike_exploded'}

        # Check spike defused
        if self.spike_defused:
            return {'winner': 'defense', 'reason': 'spike_defused'}

        # Check timeout (only if spike not planted)
        if not self.spike_planted and time_ms >= self.ROUND_TIME_MS:
            return {'winner': 'defense', 'reason': 'timeout'}

        return None

    def get_time_remaining(self, time_ms: int) -> Dict[str, int]:
        """Get remaining time for various conditions.

        Returns:
            Dict with 'round_time_ms' and 'spike_time_ms' (if planted)
        """
        result = {
            'round_time_ms': max(0, self.ROUND_TIME_MS - time_ms)
        }
        if self.spike_planted and not self.spike_defused and not self.spike_exploded:
            time_since_plant = time_ms - self.spike_plant_time_ms
            result['spike_time_ms'] = max(0, self.SPIKE_EXPLOSION_TIME_MS - time_since_plant)
        return result

    def get_man_advantage(self) -> Tuple[str, int]:
        """Get which team has man advantage and by how many.

        Returns:
            Tuple of (team with advantage or 'even', advantage amount)
        """
        diff = self.attack_alive - self.defense_alive
        if diff > 0:
            return ('attack', diff)
        elif diff < 0:
            return ('defense', -diff)
        return ('even', 0)


class WinProbabilityCalculator:
    """Calculates real-time win probability based on round state.

    Based on observed statistics from professional VALORANT matches:
    - Man advantage multipliers from GRID data analysis
    - First blood impact from tournament statistics
    - Post-plant scenarios from bomb timer analysis
    """

    # Win probability multipliers based on alive player count
    # Format: attacker_alive -> defender_alive -> attack_win_multiplier
    MAN_ADVANTAGE_MULTIPLIERS = {
        5: {5: 1.0, 4: 1.35, 3: 2.1, 2: 4.5, 1: 12.0, 0: float('inf')},
        4: {5: 0.74, 4: 1.0, 3: 1.55, 2: 3.3, 1: 8.9, 0: float('inf')},
        3: {5: 0.48, 4: 0.65, 3: 1.0, 2: 2.13, 1: 5.7, 0: float('inf')},
        2: {5: 0.22, 4: 0.30, 3: 0.47, 2: 1.0, 1: 2.7, 0: float('inf')},
        1: {5: 0.08, 4: 0.11, 3: 0.18, 2: 0.37, 1: 1.0, 0: float('inf')},
        0: {5: 0.0, 4: 0.0, 3: 0.0, 2: 0.0, 1: 0.0, 0: 0.0},
    }

    # First blood impact on win probability
    FIRST_BLOOD_ATTACK_BONUS = 0.12  # +12% win rate for attack getting FB
    FIRST_BLOOD_DEFENSE_BONUS = 0.08  # +8% win rate for defense getting FB

    # Base win rates (pre-first-blood, 5v5)
    BASE_ATTACK_WIN_RATE = 0.48  # Slightly defender-sided
    BASE_DEFENSE_WIN_RATE = 0.52

    # Post-plant modifiers
    POST_PLANT_ATTACK_BONUS = 0.15  # Spike planted is significant advantage

    # Time-based modifiers (attackers lose time pressure as round progresses)
    TIME_PRESSURE_THRESHOLDS = [
        (60000, 0.0),   # Before 60s - no pressure
        (80000, -0.05), # 60-80s - slight pressure
        (90000, -0.10), # 80-90s - medium pressure
        (100000, -0.20), # 90-100s - high pressure (round timer)
    ]

    # Economy advantage modifiers
    ECONOMY_MODIFIERS = {
        ('full', 'eco'): 0.25,    # Full buy vs eco = +25%
        ('full', 'force'): 0.15,  # Full buy vs force = +15%
        ('full', 'half'): 0.08,   # Full buy vs half = +8%
        ('full', 'full'): 0.0,    # Even buy
        ('half', 'eco'): 0.15,
        ('half', 'force'): 0.08,
        ('force', 'eco'): 0.10,
        ('eco', 'eco'): 0.0,
        ('pistol', 'pistol'): 0.0,
    }

    @classmethod
    def calculate_win_probability(
        cls,
        round_state: RoundState,
        time_ms: int
    ) -> Dict[str, float]:
        """Calculate current win probability for both teams.

        Args:
            round_state: Current round state
            time_ms: Current round time in milliseconds

        Returns:
            Dict with 'attack' and 'defense' probabilities (sum to 1.0)
        """
        # Start with base rates
        attack_prob = cls.BASE_ATTACK_WIN_RATE
        defense_prob = cls.BASE_DEFENSE_WIN_RATE

        # Apply man advantage
        attack_alive = round_state.attack_alive
        defense_alive = round_state.defense_alive

        # Handle elimination
        if attack_alive == 0:
            return {'attack': 0.0, 'defense': 1.0}
        if defense_alive == 0:
            return {'attack': 1.0, 'defense': 0.0}

        # Get multiplier from table
        multiplier = cls.MAN_ADVANTAGE_MULTIPLIERS.get(
            attack_alive, {}
        ).get(defense_alive, 1.0)

        if multiplier == float('inf'):
            attack_prob = 1.0
            defense_prob = 0.0
        else:
            # Apply multiplier to attack probability
            attack_prob = attack_prob * multiplier
            # Ensure valid range
            attack_prob = max(0.05, min(0.95, attack_prob))
            defense_prob = 1.0 - attack_prob

        # Apply first blood bonus
        if round_state.first_blood_team == 'attack':
            attack_prob += cls.FIRST_BLOOD_ATTACK_BONUS
        elif round_state.first_blood_team == 'defense':
            defense_prob += cls.FIRST_BLOOD_DEFENSE_BONUS

        # Apply post-plant bonus
        if round_state.spike_planted:
            attack_prob += cls.POST_PLANT_ATTACK_BONUS

            # Time until explosion matters
            SPIKE_TIME_MS = 45000
            time_since_plant = time_ms - round_state.spike_plant_time_ms
            time_remaining = SPIKE_TIME_MS - time_since_plant

            # Less time = better for attack
            if time_remaining < 15000:  # Under 15 seconds
                attack_prob += 0.10
            elif time_remaining < 25000:  # Under 25 seconds
                attack_prob += 0.05

        else:
            # Apply time pressure for attack (need to plant before time runs out)
            for threshold, modifier in cls.TIME_PRESSURE_THRESHOLDS:
                if time_ms < threshold:
                    break
                attack_prob += modifier

        # Apply economy modifier
        econ_key = (round_state.attack_buy_type, round_state.defense_buy_type)
        econ_mod = cls.ECONOMY_MODIFIERS.get(econ_key, 0.0)
        attack_prob += econ_mod

        # Reverse key for defense advantage
        econ_key_rev = (round_state.defense_buy_type, round_state.attack_buy_type)
        econ_mod_rev = cls.ECONOMY_MODIFIERS.get(econ_key_rev, 0.0)
        defense_prob += econ_mod_rev

        # Normalize
        total = attack_prob + defense_prob
        if total > 0:
            attack_prob /= total
            defense_prob /= total
        else:
            attack_prob = 0.5
            defense_prob = 0.5

        # Final bounds check
        attack_prob = max(0.01, min(0.99, attack_prob))
        defense_prob = 1.0 - attack_prob

        return {
            'attack': round(attack_prob, 3),
            'defense': round(defense_prob, 3)
        }

    @classmethod
    def get_clutch_probability(
        cls,
        clutcher_team: str,
        clutcher_alive: int,
        opponent_alive: int,
        spike_planted: bool,
        time_remaining_ms: int,
        clutcher_is_attacker: bool
    ) -> float:
        """Calculate probability of winning a clutch situation (1vX).

        Args:
            clutcher_team: Team of the clutcher
            clutcher_alive: Number alive on clutcher team (usually 1)
            opponent_alive: Number alive on opponent team
            spike_planted: Whether spike is planted
            time_remaining_ms: Time remaining in round/spike
            clutcher_is_attacker: Whether clutcher is on attack

        Returns:
            Probability of clutcher winning (0-1)
        """
        # Base clutch probabilities (from pro data)
        CLUTCH_BASE_PROBS = {
            (1, 1): 0.45,  # 1v1
            (1, 2): 0.18,  # 1v2
            (1, 3): 0.06,  # 1v3
            (1, 4): 0.02,  # 1v4
            (1, 5): 0.005, # 1v5
        }

        base_prob = CLUTCH_BASE_PROBS.get((clutcher_alive, opponent_alive), 0.01)

        # Spike modifiers
        if spike_planted:
            if clutcher_is_attacker:
                # Attack clutching post-plant - favorable
                base_prob *= 1.5
                # More time = better odds
                if time_remaining_ms > 20000:
                    base_prob *= 1.2
            else:
                # Defense clutching post-plant - must defuse
                if time_remaining_ms < 7000:
                    # Not enough time to defuse
                    base_prob *= 0.3
                elif time_remaining_ms < 15000:
                    base_prob *= 0.6
        else:
            if clutcher_is_attacker:
                # Attack must plant - time pressure
                if time_remaining_ms < 20000:
                    base_prob *= 0.5

        return min(0.95, max(0.01, base_prob))


@dataclass
class TradeAnalysis:
    """Analysis of trade patterns in a round."""
    total_kills: int = 0
    trades: int = 0
    trade_rate: float = 0.0  # trades / opportunities
    average_trade_time_ms: float = 0.0
    attack_trades: int = 0
    defense_trades: int = 0

    @classmethod
    def analyze(cls, round_state: RoundState) -> 'TradeAnalysis':
        """Analyze trade patterns from round state."""
        kills = round_state.kills
        if not kills:
            return cls()

        trades = [k for k in kills if k.is_trade]
        trade_times = [k.time_ms - round_state.kills[i-1].time_ms
                       for i, k in enumerate(kills) if k.is_trade and i > 0]

        attack_trades = len([t for t in trades if t.killer_team == 'attack'])
        defense_trades = len([t for t in trades if t.killer_team == 'defense'])

        # Trade opportunities = deaths that could be traded
        opportunities = len([k for k in kills[:-1]])  # All deaths except last

        return cls(
            total_kills=len(kills),
            trades=len(trades),
            trade_rate=len(trades) / max(1, opportunities),
            average_trade_time_ms=sum(trade_times) / max(1, len(trade_times)),
            attack_trades=attack_trades,
            defense_trades=defense_trades
        )
