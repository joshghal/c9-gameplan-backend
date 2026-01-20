"""
Trade System: VCT Data-Driven Trade Mechanics (Readiness-Based)

From VCT analysis (12,029 kills, 3,036 trades):
- Trade rate: 25.2% of kills get traded
- Average trade time: 1.72 seconds
- Trade time distribution:
  - Under 1s: 40% (pre-aimed trades)
  - 1-2s: 26% (normal reaction)
  - 2-3s: 14% (slow/repositioning)
  - 3-5s: 21% (rotating from far)

=== KEY FINDING: Distance Doesn't Matter, Readiness Does ===

Correlation (trade_time vs distance): -0.013  ← Almost ZERO!

The fastest trades (42-214ms) happen at ALL distances - even 2347 units.
Trade time is determined by READINESS STATE, not physical distance:
  - Pre-aimed (watching angle): 42-200ms (19% of trades)
  - Normal (heard, turned, shot): 500-1500ms (44% of trades)
  - Repositioning (had to move): 2000-5000ms (37% of trades)

=== READINESS-BASED MODEL ===

Instead of checking distance, we:
1. Roll for readiness state (19% pre-aimed, 44% normal, 37% repositioning)
2. Determine trade timing from readiness state
3. Apply success probability based on readiness

This better matches VCT data where fast trades happen at any distance.

=== DESIGN PRINCIPLE ===

Numbers advantage should EMERGE from trade opportunities,
not from hardcoded "5v4 bonus = +10%".

More teammates = more trade opportunities = higher round win rate
"""

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


class ReadinessState(Enum):
    """Player readiness state for trading."""
    PRE_AIMED = "pre_aimed"      # Already watching the angle (19%)
    NORMAL = "normal"            # Heard gunshot, need to turn (44%)
    REPOSITIONING = "repositioning"  # Need to move for angle (37%)


@dataclass
class KillEvent:
    """Records a kill for trade tracking."""
    timestamp_ms: int
    killer_id: str
    killer_team: str
    killer_position: Tuple[float, float]
    victim_id: str
    victim_team: str
    victim_position: Tuple[float, float]


@dataclass
class TradeOpportunity:
    """A potential trade opportunity."""
    trader_id: str
    trader_position: Tuple[float, float]
    trader_role: str  # duelist, initiator, controller, sentinel
    original_kill: KillEvent
    distance_to_killer: float
    time_since_kill_ms: int
    has_los: bool
    readiness_state: ReadinessState  # NEW: Determined by roll, not distance


@dataclass
class TradeResult:
    """Result of a trade attempt."""
    success: bool
    trader_id: str
    target_id: str  # The original killer
    trade_time_ms: int
    readiness_state: ReadinessState  # NEW: Track what readiness led to trade


class TradeSystem:
    """
    Models trading mechanics based on VCT pro match data.

    Uses READINESS-BASED model instead of distance-based:
    - 19% pre-aimed (instant trades)
    - 44% normal reaction
    - 37% repositioning (delayed trades)

    Instead of: if team_size > enemy_size: bonus += 0.10
    We model: More teammates = more trade opportunities = higher survival
    """

    # From VCT data - validated_parameters.py
    TRADE_WINDOW_MS = 3000  # VCT: 79.3% of trades within 3s
    MAX_TRADE_WINDOW_MS = 5000  # Extended for repositioning trades

    # ✅ VCT-EXTRACTED: Overall trade rate constraint
    # VCT shows 25.2% of kills get traded - NOT every kill has a trade opportunity
    # This is the base probability that a kill CAN be traded (teammate in position)
    BASE_TRADE_OPPORTUNITY_PROB = 0.35  # ~35% of kills have tradeable position

    # ✅ VCT-EXTRACTED: Readiness state probabilities
    # These apply ONLY to kills that have a trade opportunity
    READINESS_PROBS = {
        ReadinessState.PRE_AIMED: 0.19,      # 19% instant trades
        ReadinessState.NORMAL: 0.44,          # 44% normal reaction
        ReadinessState.REPOSITIONING: 0.37,   # 37% delayed trades
    }

    # ✅ VCT-EXTRACTED: Trade timing ranges by readiness (ms)
    READINESS_TIMING = {
        ReadinessState.PRE_AIMED: (42, 200),       # Fastest observed: 42ms
        ReadinessState.NORMAL: (500, 1500),        # Normal reaction time
        ReadinessState.REPOSITIONING: (2000, 5000), # Need to move
    }

    # Trade success probability by readiness state
    # Combined with BASE_TRADE_OPPORTUNITY_PROB gives ~25% overall trade rate
    # 0.35 * 0.72 = 0.25 (where 0.72 is weighted avg of success rates)
    READINESS_SUCCESS = {
        ReadinessState.PRE_AIMED: 0.85,      # Pre-aimed very likely to succeed
        ReadinessState.NORMAL: 0.70,          # Normal reaction fair chance
        ReadinessState.REPOSITIONING: 0.45,   # Repositioning harder
    }

    # Role modifiers for readiness probabilities
    ROLE_READINESS_MODIFIERS = {
        'duelist': {ReadinessState.PRE_AIMED: 0.10, ReadinessState.REPOSITIONING: -0.05},
        'initiator': {ReadinessState.NORMAL: 0.05},
        'controller': {},  # No modifier
        'sentinel': {ReadinessState.PRE_AIMED: -0.05, ReadinessState.REPOSITIONING: 0.10},
    }

    # Human reaction time (still needed for minimum trade time)
    MIN_REACTION_TIME_MS = 42  # VCT: Fastest observed trade

    def __init__(self):
        self.pending_kills: List[KillEvent] = []
        self.completed_trades: List[TradeResult] = []

    def reset(self):
        """Reset state for new round."""
        self.pending_kills = []
        self.completed_trades = []

    def record_kill(
        self,
        timestamp_ms: int,
        killer_id: str,
        killer_team: str,
        killer_pos: Tuple[float, float],
        victim_id: str,
        victim_team: str,
        victim_pos: Tuple[float, float],
    ) -> KillEvent:
        """Record a kill for potential trading."""
        kill = KillEvent(
            timestamp_ms=timestamp_ms,
            killer_id=killer_id,
            killer_team=killer_team,
            killer_position=killer_pos,
            victim_id=victim_id,
            victim_team=victim_team,
            victim_position=victim_pos,
        )
        self.pending_kills.append(kill)
        return kill

    def _determine_readiness_state(self, role: str = 'initiator') -> ReadinessState:
        """
        Determine readiness state probabilistically.

        VCT data shows:
        - 19% of trades are pre-aimed (instant)
        - 44% are normal reaction
        - 37% require repositioning

        Role modifies these probabilities slightly.
        """
        # Get base probabilities
        probs = dict(self.READINESS_PROBS)

        # Apply role modifiers
        modifiers = self.ROLE_READINESS_MODIFIERS.get(role.lower(), {})
        for state, mod in modifiers.items():
            probs[state] = max(0.05, min(0.50, probs[state] + mod))

        # Normalize probabilities
        total = sum(probs.values())
        probs = {k: v / total for k, v in probs.items()}

        # Roll for readiness state
        roll = random.random()
        cumulative = 0.0
        for state, prob in probs.items():
            cumulative += prob
            if roll < cumulative:
                return state

        return ReadinessState.NORMAL  # Fallback

    def _get_trade_timing(self, readiness: ReadinessState) -> int:
        """Get trade timing based on readiness state."""
        min_ms, max_ms = self.READINESS_TIMING[readiness]
        return random.randint(min_ms, max_ms)

    def get_trade_opportunities(
        self,
        current_time_ms: int,
        potential_traders: List[Tuple[str, Tuple[float, float], str, str]],  # (id, pos, team, role)
        has_los_func=None,  # Function to check LOS: (pos1, pos2) -> bool (optional)
    ) -> List[TradeOpportunity]:
        """
        Find all potential trade opportunities using readiness-based model.

        potential_traders: List of (player_id, position, team, role) for alive players
        has_los_func: Optional function to check line of sight

        Note: Only ~35% of kills have teammates in tradeable positions (BASE_TRADE_OPPORTUNITY_PROB).
        This is checked per kill to limit overall trade rate to ~25%.
        """
        opportunities = []

        for kill in self.pending_kills:
            time_since_kill = current_time_ms - kill.timestamp_ms

            # Skip if outside max trade window
            if time_since_kill > self.MAX_TRADE_WINDOW_MS:
                continue

            # === CONSTRAINT: Not every kill has a tradeable opportunity ===
            # VCT shows 25% of kills get traded. This means many kills happen
            # when teammates aren't positioned to trade (solo death, caught out, etc.)
            # We model this with a base probability check per kill.
            if not hasattr(kill, '_trade_eligible'):
                kill._trade_eligible = random.random() < self.BASE_TRADE_OPPORTUNITY_PROB
            if not kill._trade_eligible:
                continue  # This kill cannot be traded (no teammates in position)

            # Find ONE teammate who could trade (only one trade per kill)
            trade_found = False
            for trader_info in potential_traders:
                if trade_found:
                    break

                # Handle both 3-tuple and 4-tuple formats
                if len(trader_info) == 4:
                    trader_id, trader_pos, trader_team, trader_role = trader_info
                else:
                    trader_id, trader_pos, trader_team = trader_info
                    trader_role = 'initiator'  # Default

                # Must be on victim's team
                if trader_team != kill.victim_team:
                    continue

                # Can't trade yourself
                if trader_id == kill.victim_id:
                    continue

                # Determine readiness state (NOT based on distance)
                readiness = self._determine_readiness_state(trader_role)

                # Check if enough time has passed for this readiness state
                min_time, max_time = self.READINESS_TIMING[readiness]
                if time_since_kill < min_time:
                    continue  # Not ready yet for this type of trade

                # Calculate distance (for logging/debugging, not probability)
                distance = math.sqrt(
                    (trader_pos[0] - kill.killer_position[0])**2 +
                    (trader_pos[1] - kill.killer_position[1])**2
                )

                # Check LOS if function provided
                has_los = True
                if has_los_func:
                    has_los = has_los_func(trader_pos, kill.killer_position)

                opportunities.append(TradeOpportunity(
                    trader_id=trader_id,
                    trader_position=trader_pos,
                    trader_role=trader_role,
                    original_kill=kill,
                    distance_to_killer=distance,
                    time_since_kill_ms=time_since_kill,
                    has_los=has_los,
                    readiness_state=readiness,
                ))
                trade_found = True  # Only one trade opportunity per kill

        return opportunities

    def calculate_trade_probability(self, opportunity: TradeOpportunity) -> float:
        """
        Calculate probability of successful trade based on READINESS STATE.

        Key change from old system:
        - OLD: Distance affected probability
        - NEW: Readiness state determines probability (VCT shows distance has no correlation)
        """
        if not opportunity.has_los:
            return 0.0

        # Base probability from readiness state
        base_prob = self.READINESS_SUCCESS[opportunity.readiness_state]

        # Time decay within window (slight penalty for very late trades)
        time_ms = opportunity.time_since_kill_ms
        min_time, max_time = self.READINESS_TIMING[opportunity.readiness_state]

        if time_ms > max_time:
            # Past optimal window, reduce probability
            overage = (time_ms - max_time) / 1000  # Seconds over
            base_prob *= max(0.3, 1.0 - overage * 0.2)

        return base_prob

    def attempt_trade(
        self,
        opportunity: TradeOpportunity,
        current_time_ms: int,
    ) -> Optional[TradeResult]:
        """
        Attempt to execute a trade.

        Returns TradeResult if successful, None if failed.
        """
        probability = self.calculate_trade_probability(opportunity)

        if random.random() < probability:
            # Successful trade
            result = TradeResult(
                success=True,
                trader_id=opportunity.trader_id,
                target_id=opportunity.original_kill.killer_id,
                trade_time_ms=opportunity.time_since_kill_ms,
                readiness_state=opportunity.readiness_state,
            )
            self.completed_trades.append(result)

            # Remove the kill from pending (it's been traded)
            if opportunity.original_kill in self.pending_kills:
                self.pending_kills.remove(opportunity.original_kill)

            return result

        return None

    def check_and_attempt_trade(
        self,
        current_time_ms: int,
        potential_traders: List[Tuple[str, Tuple[float, float], str, str]],
        has_los_func=None,
    ) -> List[TradeResult]:
        """
        Convenience method: Find opportunities and attempt all trades.

        Returns list of successful trades.
        """
        self.cleanup_old_kills(current_time_ms)

        opportunities = self.get_trade_opportunities(
            current_time_ms, potential_traders, has_los_func
        )

        successful_trades = []
        for opp in opportunities:
            result = self.attempt_trade(opp, current_time_ms)
            if result:
                successful_trades.append(result)

        return successful_trades

    def cleanup_old_kills(self, current_time_ms: int):
        """Remove kills that are too old to trade."""
        self.pending_kills = [
            k for k in self.pending_kills
            if current_time_ms - k.timestamp_ms <= self.MAX_TRADE_WINDOW_MS
        ]

    def get_trade_stats(self) -> Dict:
        """Get statistics about trades this round."""
        if not self.completed_trades:
            return {
                'total_trades': 0,
                'avg_trade_time_ms': 0,
                'fast_trades': 0,
                'readiness_distribution': {},
            }

        total = len(self.completed_trades)
        avg_time = sum(t.trade_time_ms for t in self.completed_trades) / total
        fast = sum(1 for t in self.completed_trades if t.trade_time_ms < 1000)

        # Count by readiness state
        readiness_counts = {}
        for t in self.completed_trades:
            state = t.readiness_state.value
            readiness_counts[state] = readiness_counts.get(state, 0) + 1

        return {
            'total_trades': total,
            'avg_trade_time_ms': avg_time,
            'fast_trades': fast,
            'fast_trade_rate': fast / total if total > 0 else 0,
            'readiness_distribution': readiness_counts,
        }


class NumbersAdvantageCalculator:
    """
    Calculate numbers advantage through trade mechanics.

    Key insight: 5v4 advantage comes from having more trade opportunities,
    not from a flat bonus.
    """

    @staticmethod
    def estimate_round_advantage(
        team_a_size: int,
        team_b_size: int,
        trade_rate: float = 0.25,  # VCT: 25% of kills get traded
    ) -> float:
        """
        Estimate win probability based on team sizes and trade potential.

        This replaces: if team_a > team_b: return 0.5 + 0.05 * (team_a - team_b)

        Instead, we model:
        - More players = more chances to trade
        - Trades even out fights
        - Excess players after trades = advantage
        """
        if team_a_size == team_b_size:
            return 0.5

        # Simulate multiple fight sequences
        a_wins = 0
        simulations = 1000

        for _ in range(simulations):
            a_alive = team_a_size
            b_alive = team_b_size

            # Simulate fights until one team is eliminated
            while a_alive > 0 and b_alive > 0:
                # One fight happens
                # Larger team has slight advantage in individual fights due to crossfire
                crossfire_bonus = 0.02 * abs(a_alive - b_alive)

                if a_alive > b_alive:
                    a_win_prob = 0.5 + crossfire_bonus
                elif b_alive > a_alive:
                    a_win_prob = 0.5 - crossfire_bonus
                else:
                    a_win_prob = 0.5

                if random.random() < a_win_prob:
                    # A wins the duel
                    b_alive -= 1
                    # B might get a trade
                    if b_alive > 0 and random.random() < trade_rate:
                        a_alive -= 1
                else:
                    # B wins the duel
                    a_alive -= 1
                    # A might get a trade
                    if a_alive > 0 and random.random() < trade_rate:
                        b_alive -= 1

            if a_alive > 0:
                a_wins += 1

        return a_wins / simulations


def test_readiness_based_trades():
    """Test the readiness-based trade system."""
    print("=" * 70)
    print("READINESS-BASED TRADE SYSTEM TEST")
    print("=" * 70)

    trade_sys = TradeSystem()

    # Test readiness state distribution
    print("\n1. Testing Readiness State Distribution (1000 samples):")
    print("-" * 50)

    state_counts = {state: 0 for state in ReadinessState}
    for _ in range(1000):
        state = trade_sys._determine_readiness_state('initiator')
        state_counts[state] += 1

    print(f"{'State':<20} {'Observed':<12} {'Target':<12} {'Status'}")
    print("-" * 50)
    targets = {
        ReadinessState.PRE_AIMED: 0.19,
        ReadinessState.NORMAL: 0.44,
        ReadinessState.REPOSITIONING: 0.37,
    }
    for state, count in state_counts.items():
        observed = count / 1000
        target = targets[state]
        diff = abs(observed - target)
        status = "✓" if diff < 0.05 else "✗"
        print(f"{state.value:<20} {observed*100:>8.1f}%     {target*100:>8.1f}%     {status}")

    # Test role modifiers
    print("\n2. Testing Role Modifiers (1000 samples each):")
    print("-" * 50)

    for role in ['duelist', 'sentinel', 'controller', 'initiator']:
        state_counts = {state: 0 for state in ReadinessState}
        for _ in range(1000):
            state = trade_sys._determine_readiness_state(role)
            state_counts[state] += 1

        pre_aimed_pct = state_counts[ReadinessState.PRE_AIMED] / 10
        print(f"{role:<12}: Pre-aimed={pre_aimed_pct:>5.1f}%  "
              f"Normal={state_counts[ReadinessState.NORMAL]/10:>5.1f}%  "
              f"Reposition={state_counts[ReadinessState.REPOSITIONING]/10:>5.1f}%")

    # Test trade timing
    print("\n3. Testing Trade Timing by Readiness State:")
    print("-" * 50)

    for state in ReadinessState:
        timings = [trade_sys._get_trade_timing(state) for _ in range(100)]
        min_time, max_time = trade_sys.READINESS_TIMING[state]
        avg_time = sum(timings) / len(timings)
        print(f"{state.value:<15}: Range [{min_time:>4}ms - {max_time:>4}ms]  "
              f"Avg: {avg_time:>6.0f}ms")

    # Test simulated trades
    print("\n4. Simulating 100 Trade Attempts:")
    print("-" * 50)

    trade_sys.reset()

    # Record a kill
    kill = trade_sys.record_kill(
        timestamp_ms=10000,
        killer_id="attacker_1",
        killer_team="attack",
        killer_pos=(0.5, 0.5),
        victim_id="defender_1",
        victim_team="defense",
        victim_pos=(0.5, 0.4),
    )

    # Simulate trades at different times
    trade_results = []
    for i in range(100):
        trade_sys.reset()
        trade_sys.pending_kills.append(kill)

        # Potential traders at various times
        current_time = 10000 + random.randint(100, 4000)
        traders = [
            ("defender_2", (0.6, 0.4), "defense", "duelist"),
            ("defender_3", (0.4, 0.5), "defense", "sentinel"),
        ]

        results = trade_sys.check_and_attempt_trade(current_time, traders)
        if results:
            trade_results.extend(results)

    if trade_results:
        trade_rate = len(trade_results) / 100
        avg_trade_time = sum(t.trade_time_ms for t in trade_results) / len(trade_results)
        readiness_dist = {}
        for t in trade_results:
            state = t.readiness_state.value
            readiness_dist[state] = readiness_dist.get(state, 0) + 1

        print(f"Trade Rate: {trade_rate*100:.1f}%  (VCT target: ~25%)")
        print(f"Avg Trade Time: {avg_trade_time:.0f}ms  (VCT target: 1720ms)")
        print(f"Readiness Distribution: {readiness_dist}")
    else:
        print("No successful trades (this is unusual)")

    print("\n" + "=" * 70)
    print("KEY INSIGHT: Distance Doesn't Matter, Readiness Does")
    print("Trade timing is determined by player readiness state (pre-aimed,")
    print("normal reaction, or repositioning) NOT by physical distance.")
    print("VCT data shows correlation between distance and time = -0.013")
    print("=" * 70)


def test_trade_system():
    """Test the trade system mechanics."""
    print("=" * 70)
    print("TRADE SYSTEM TEST")
    print("=" * 70)

    # First test readiness-based trades
    test_readiness_based_trades()

    print("\n")

    # Then test numbers advantage
    calc = NumbersAdvantageCalculator()

    print("=" * 70)
    print("NUMBERS ADVANTAGE TEST")
    print("=" * 70)

    print("\nNumbers Advantage Simulation (trade_rate=0.25):")
    print("-" * 50)

    scenarios = [
        (5, 5, 50),
        (5, 4, 60),
        (4, 5, 35),
        (5, 3, 75),
        (3, 5, 15),
        (2, 1, 75),
        (1, 2, 25),
        (1, 1, 50),
    ]

    print(f"{'Scenario':<15} {'Simulated':<12} {'Target':<10} {'Status'}")
    print("-" * 50)

    for a_size, b_size, target in scenarios:
        result = calc.estimate_round_advantage(a_size, b_size)
        diff = abs(result * 100 - target)
        status = "✓" if diff < 15 else "✗"
        print(f"{a_size}v{b_size:<10} {result*100:>8.1f}%     ~{target}%      {status}")

    print("\n" + "=" * 70)
    print("KEY INSIGHT:")
    print("Numbers advantage EMERGES from trade opportunities.")
    print("5v4 wins ~60% because the 5-team has more chances to trade.")
    print("=" * 70)


if __name__ == "__main__":
    test_trade_system()
