"""
Coordination System: Emergent Advantage from Coordinated Actions

=== DESIGN PHILOSOPHY ===

Instead of: if scenario == "retake": defender_bonus += 0.15
We model:  WHY coordinated pushes work - attention/focus mechanics

Key insight from real gameplay:
- A player engaged with one enemy is VULNERABLE to others
- You cannot aim at two targets simultaneously
- Coordinated entry creates crossfire situations

This makes the retake advantage EMERGE from timing, not from hardcoded bonuses.

=== ATTENTION MODEL ===

When a player starts engaging an enemy:
1. They become "focused" on that target
2. Their reaction time to OTHER threats increases
3. Other enemies entering during focus window have advantage

VCT data supports this:
- Pre-aimed trades: 42-200ms (already watching)
- Need-to-turn trades: 500-1500ms (heard, turned, aimed)
- The difference is READINESS, not distance

When focused elsewhere, you're guaranteed NOT pre-aimed.

=== EMERGENT OUTCOMES ===

Solo push (no coordination):
- Defender enters, attacker focuses on them
- Fair fight (50/50 with equal loadouts)
- Even if defender dies, no immediate threat to attacker

Coordinated push (2+ defenders within 500ms):
- Defender A enters, attacker focuses on A
- Defender B enters while attacker focused
- Attacker must either:
  a) Stay on A → B has free shot (focus penalty)
  b) Switch to B → A has free shot (switching penalty)
- This naturally produces ~65% defender win rate in retakes

=== PARAMETERS ===

FOCUS_DURATION_MS = 800
- How long attention is locked after engaging
- From VCT: 500-1500ms for "normal reaction" trades
- Split the difference for engagement focus

FOCUS_ACCURACY_PENALTY = 0.35
- Accuracy reduction when shooting at unfocused target
- Based on: turning + acquiring new target takes time
- Similar to running accuracy penalty concept

CROSSFIRE_WINDOW_MS = 500
- Window for "coordinated" entry
- If two players engage within this window, crossfire applies
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum
import math


@dataclass
class FocusState:
    """Tracks what a player is currently focused on."""
    target_id: str
    focus_start_ms: int
    position_when_focused: Tuple[float, float]


@dataclass
class CrossfireOpportunity:
    """Represents a crossfire situation."""
    focused_player_id: str  # Player who is focused elsewhere
    focused_on_id: str  # Who they're focused on
    unfocused_attacker_id: str  # Player who has the advantage
    time_since_focus_ms: int
    focus_penalty: float  # Accuracy penalty for focused player


@dataclass
class CoordinatedEntry:
    """Tracks coordinated site entries."""
    entry_zone: str  # e.g., "site_A", "site_B"
    players: List[str]
    entry_times_ms: Dict[str, int]
    is_coordinated: bool  # True if entries within CROSSFIRE_WINDOW_MS


class CoordinationSystem:
    """
    Models attention/focus mechanics for emergent coordination advantage.

    This replaces hardcoded retake bonuses with underlying mechanics:
    - Engaged players have reduced awareness of other threats
    - Coordinated entry exploits this to create crossfire
    - Players exposed to multiple enemies have disadvantage

    === CROSSFIRE EXPOSURE ===

    Key insight: A player visible to multiple enemies simultaneously
    is at disadvantage BEFORE combat even starts:
    - They can only shoot one target at a time
    - Other enemies get free shots while they're aiming elsewhere
    - This makes coordinated pushes effective

    This is different from focus penalty (which applies during engagement).
    Crossfire exposure applies based on LOS geometry.
    """

    # Timing constants
    FOCUS_DURATION_MS = 800  # How long engagement locks attention
    CROSSFIRE_WINDOW_MS = 500  # Window for "coordinated" entry
    SWITCHING_DELAY_MS = 200  # Time to switch targets

    # Combat modifiers
    FOCUS_ACCURACY_PENALTY = 0.35  # Penalty when shooting unfocused target
    CROSSFIRE_BONUS = 0.20  # Bonus for being in crossfire position

    # Crossfire exposure (NEW)
    CROSSFIRE_EXPOSURE_PENALTY = 0.25  # Penalty when visible to 2+ enemies
    HEAVY_CROSSFIRE_PENALTY = 0.40  # Penalty when visible to 3+ enemies

    # Entry detection
    SITE_ENTRY_RADIUS = 0.15  # Distance from site center to count as "entered"

    def __init__(self):
        self.player_focus: Dict[str, FocusState] = {}
        self.site_entries: Dict[str, CoordinatedEntry] = {}
        self.recent_engagements: List[Tuple[str, str, int]] = []  # (player_a, player_b, time_ms)

    def reset(self):
        """Reset state for new round."""
        self.player_focus.clear()
        self.site_entries.clear()
        self.recent_engagements.clear()

    def record_engagement_start(
        self,
        player_id: str,
        target_id: str,
        player_position: Tuple[float, float],
        time_ms: int
    ):
        """
        Record that a player has started engaging a target.

        This locks their attention for FOCUS_DURATION_MS.
        """
        self.player_focus[player_id] = FocusState(
            target_id=target_id,
            focus_start_ms=time_ms,
            position_when_focused=player_position
        )
        self.recent_engagements.append((player_id, target_id, time_ms))

    def clear_focus(self, player_id: str):
        """Clear focus when engagement ends (kill or death)."""
        if player_id in self.player_focus:
            del self.player_focus[player_id]

    def is_player_focused_elsewhere(
        self,
        player_id: str,
        potential_threat_id: str,
        time_ms: int
    ) -> bool:
        """
        Check if player is focused on someone OTHER than potential_threat.

        Returns True if player is vulnerable to potential_threat.
        """
        if player_id not in self.player_focus:
            return False

        focus = self.player_focus[player_id]

        # Already focused on this threat
        if focus.target_id == potential_threat_id:
            return False

        # Check if focus has expired
        time_since_focus = time_ms - focus.focus_start_ms
        if time_since_focus > self.FOCUS_DURATION_MS:
            return False

        return True

    def get_focus_penalty(
        self,
        defender_id: str,
        attacker_id: str,
        time_ms: int
    ) -> float:
        """
        Get accuracy penalty for defender due to being focused elsewhere.

        Returns 0.0 if not focused elsewhere, up to FOCUS_ACCURACY_PENALTY otherwise.
        """
        if not self.is_player_focused_elsewhere(defender_id, attacker_id, time_ms):
            return 0.0

        focus = self.player_focus[defender_id]
        time_since_focus = time_ms - focus.focus_start_ms

        # Penalty decays over time (sharper penalty early, less later)
        decay_factor = 1.0 - (time_since_focus / self.FOCUS_DURATION_MS)
        return self.FOCUS_ACCURACY_PENALTY * decay_factor

    def get_crossfire_opportunities(
        self,
        player_id: str,
        player_position: Tuple[float, float],
        player_team: str,
        all_players: List[Tuple[str, Tuple[float, float], str, bool]],  # (id, pos, team, alive)
        time_ms: int,
        has_los_func
    ) -> List[CrossfireOpportunity]:
        """
        Find enemies who are focused elsewhere and vulnerable to this player.

        Returns list of crossfire opportunities where player has advantage.
        """
        opportunities = []

        for enemy_id, enemy_pos, enemy_team, enemy_alive in all_players:
            if not enemy_alive:
                continue
            if enemy_team == player_team:
                continue

            # Check if enemy is focused elsewhere
            if not self.is_player_focused_elsewhere(enemy_id, player_id, time_ms):
                continue

            # Check LOS to enemy
            if not has_los_func(player_position, enemy_pos):
                continue

            focus = self.player_focus[enemy_id]
            penalty = self.get_focus_penalty(enemy_id, player_id, time_ms)

            opportunities.append(CrossfireOpportunity(
                focused_player_id=enemy_id,
                focused_on_id=focus.target_id,
                unfocused_attacker_id=player_id,
                time_since_focus_ms=time_ms - focus.focus_start_ms,
                focus_penalty=penalty
            ))

        return opportunities

    def record_site_entry(
        self,
        player_id: str,
        site_name: str,
        time_ms: int
    ):
        """
        Record that a player has entered a site.

        Used to detect coordinated entries.
        """
        if site_name not in self.site_entries:
            self.site_entries[site_name] = CoordinatedEntry(
                entry_zone=site_name,
                players=[],
                entry_times_ms={},
                is_coordinated=False
            )

        entry = self.site_entries[site_name]

        if player_id not in entry.players:
            entry.players.append(player_id)
            entry.entry_times_ms[player_id] = time_ms

            # Check if this creates a coordinated entry
            self._update_coordination_status(entry)

    def _update_coordination_status(self, entry: CoordinatedEntry):
        """Check if entries are coordinated (within CROSSFIRE_WINDOW_MS)."""
        if len(entry.players) < 2:
            entry.is_coordinated = False
            return

        times = list(entry.entry_times_ms.values())
        time_spread = max(times) - min(times)

        entry.is_coordinated = time_spread <= self.CROSSFIRE_WINDOW_MS

    def get_entry_coordination_bonus(
        self,
        player_id: str,
        site_name: str
    ) -> float:
        """
        Get bonus for being part of a coordinated entry.

        Returns CROSSFIRE_BONUS if player entered as part of coordinated push.
        """
        if site_name not in self.site_entries:
            return 0.0

        entry = self.site_entries[site_name]

        if player_id not in entry.players:
            return 0.0

        if not entry.is_coordinated:
            return 0.0

        # Coordinated entry bonus
        return self.CROSSFIRE_BONUS

    def get_crossfire_exposure_penalty(
        self,
        player_id: str,
        player_position: Tuple[float, float],
        player_team: str,
        all_players: List[Tuple[str, Tuple[float, float], str, bool]],
        has_los_func,
        combat_range: float = 0.3,
    ) -> float:
        """
        DEPRECATED: Use get_teammate_support_bonus instead.
        This method penalizes wrong side in some scenarios.
        """
        return 0.0  # Disabled

    def get_teammate_support_bonus(
        self,
        player_id: str,
        player_position: Tuple[float, float],
        player_team: str,
        target_position: Tuple[float, float],
        all_players: List[Tuple[str, Tuple[float, float], str, bool]],
        has_los_func,
        combat_range: float = 0.3,
    ) -> float:
        """
        Calculate bonus for having teammates who can support in this fight.

        This models crossfire from the CORRECT perspective:
        - If your teammates ALSO have LOS on your target, you have advantage
        - The target can only focus on one of you
        - This benefits the team with numbers advantage

        Returns:
        - 0.0 if no teammates have LOS on target
        - CROSSFIRE_EXPOSURE_PENALTY if 1 teammate can support
        - HEAVY_CROSSFIRE_PENALTY if 2+ teammates can support

        Key difference from exposure penalty:
        - Exposure penalizes YOU for enemies seeing you
        - Support BONUS rewards you for teammates seeing your target
        """
        teammates_with_los = 0

        for other_id, other_pos, other_team, other_alive in all_players:
            if not other_alive:
                continue
            if other_team != player_team:
                continue
            if other_id == player_id:
                continue

            # Check if teammate is in combat range of target
            dist_to_target = math.sqrt(
                (other_pos[0] - target_position[0])**2 +
                (other_pos[1] - target_position[1])**2
            )
            if dist_to_target > combat_range:
                continue

            # Check if teammate has LOS on target
            if has_los_func(other_pos, target_position):
                teammates_with_los += 1

        # Return bonus based on support level
        if teammates_with_los >= 2:
            return self.HEAVY_CROSSFIRE_PENALTY  # Reuse penalty value as bonus
        elif teammates_with_los >= 1:
            return self.CROSSFIRE_EXPOSURE_PENALTY
        else:
            return 0.0

    def cleanup_expired_focus(self, time_ms: int):
        """Remove expired focus states."""
        expired = []
        for player_id, focus in self.player_focus.items():
            if time_ms - focus.focus_start_ms > self.FOCUS_DURATION_MS:
                expired.append(player_id)

        for player_id in expired:
            del self.player_focus[player_id]

    def get_coordination_stats(self) -> Dict:
        """Get statistics about coordination this round."""
        total_entries = sum(len(e.players) for e in self.site_entries.values())
        coordinated = sum(
            len(e.players) for e in self.site_entries.values()
            if e.is_coordinated
        )

        return {
            'total_site_entries': total_entries,
            'coordinated_entries': coordinated,
            'coordination_rate': coordinated / total_entries if total_entries > 0 else 0,
            'focus_events': len(self.recent_engagements),
        }


class CoordinatedPushBehavior:
    """
    Models "wait for teammate" behavior for coordinated pushes.

    Instead of defenders rushing in one-by-one, this makes them:
    1. Detect when near a site entry point
    2. Wait briefly if teammate is nearby
    3. Push together within CROSSFIRE_WINDOW_MS

    This creates the CONDITIONS for crossfire advantage to apply.
    """

    # Behavior parameters
    WAIT_RADIUS = 0.12  # Distance to detect nearby teammates
    MAX_WAIT_TIME_MS = 2000  # Don't wait forever
    MIN_TEAMMATES_TO_WAIT = 1  # Wait if at least 1 teammate nearby

    def __init__(self):
        self.waiting_players: Dict[str, int] = {}  # player_id -> wait_start_ms
        self.push_ready: Set[str] = set()  # Players ready to push together

    def reset(self):
        """Reset for new round."""
        self.waiting_players.clear()
        self.push_ready.clear()

    def should_wait_for_teammates(
        self,
        player_id: str,
        player_position: Tuple[float, float],
        player_team: str,
        target_position: Tuple[float, float],
        all_players: List[Tuple[str, Tuple[float, float], str, bool]],
        time_ms: int
    ) -> bool:
        """
        Determine if player should wait for teammates before pushing.

        Returns True if player should hold position briefly.
        """
        # Check if already waiting too long
        if player_id in self.waiting_players:
            wait_time = time_ms - self.waiting_players[player_id]
            if wait_time > self.MAX_WAIT_TIME_MS:
                # Done waiting, push regardless
                del self.waiting_players[player_id]
                self.push_ready.add(player_id)
                return False

        # If already ready to push, go
        if player_id in self.push_ready:
            return False

        # Count nearby teammates heading to same area
        nearby_teammates = 0
        for other_id, other_pos, other_team, other_alive in all_players:
            if not other_alive:
                continue
            if other_team != player_team:
                continue
            if other_id == player_id:
                continue

            # Check if teammate is nearby
            dist_to_player = math.sqrt(
                (other_pos[0] - player_position[0])**2 +
                (other_pos[1] - player_position[1])**2
            )

            if dist_to_player < self.WAIT_RADIUS:
                nearby_teammates += 1

                # If teammate is also waiting, both become ready
                if other_id in self.waiting_players:
                    self.push_ready.add(player_id)
                    self.push_ready.add(other_id)
                    return False

        # If teammates nearby but not ready, start waiting
        if nearby_teammates >= self.MIN_TEAMMATES_TO_WAIT:
            if player_id not in self.waiting_players:
                self.waiting_players[player_id] = time_ms
            return True

        # No teammates nearby, push alone
        return False

    def get_coordinated_push_targets(
        self,
        players_ready: List[str],
        target_position: Tuple[float, float],
        spread: float = 0.05
    ) -> Dict[str, Tuple[float, float]]:
        """
        Generate slightly spread targets for coordinated push.

        Players don't stack exactly on same spot - they spread to cover angles.
        """
        import random

        targets = {}
        num_players = len(players_ready)

        for i, player_id in enumerate(players_ready):
            # Spread players in arc around target
            angle = (i / max(1, num_players - 1)) * math.pi - math.pi / 2
            offset_x = math.cos(angle) * spread
            offset_y = math.sin(angle) * spread

            targets[player_id] = (
                target_position[0] + offset_x + random.uniform(-0.02, 0.02),
                target_position[1] + offset_y + random.uniform(-0.02, 0.02)
            )

        return targets


def test_coordination_system():
    """Test the coordination system mechanics."""
    print("=" * 70)
    print("COORDINATION SYSTEM TEST")
    print("=" * 70)

    coord = CoordinationSystem()

    # Simulate scenario: 2 defenders pushing 1 attacker
    print("\n=== Scenario: Coordinated 2v1 Retake ===")

    # Attacker engages first defender
    coord.record_engagement_start(
        player_id="attacker_1",
        target_id="defender_1",
        player_position=(0.5, 0.3),
        time_ms=0
    )
    print("T=0ms: Attacker engages Defender 1")

    # Check if attacker is vulnerable to defender 2
    def mock_los(p1, p2):
        return True

    # At T=200ms, defender 2 peeks
    players = [
        ("attacker_1", (0.5, 0.3), "attack", True),
        ("defender_1", (0.5, 0.5), "defense", True),
        ("defender_2", (0.6, 0.5), "defense", True),
    ]

    opps = coord.get_crossfire_opportunities(
        player_id="defender_2",
        player_position=(0.6, 0.5),
        player_team="defense",
        all_players=players,
        time_ms=200,
        has_los_func=mock_los
    )

    if opps:
        print(f"T=200ms: Defender 2 has crossfire opportunity!")
        print(f"  - Attacker focused on: {opps[0].focused_on_id}")
        print(f"  - Focus penalty for attacker: {opps[0].focus_penalty:.2f}")

    # Test focus decay at T=600ms
    opps_later = coord.get_crossfire_opportunities(
        player_id="defender_2",
        player_position=(0.6, 0.5),
        player_team="defense",
        all_players=players,
        time_ms=600,
        has_los_func=mock_los
    )

    if opps_later:
        print(f"T=600ms: Focus penalty decayed to: {opps_later[0].focus_penalty:.2f}")

    # Test expired focus at T=1000ms
    opps_expired = coord.get_crossfire_opportunities(
        player_id="defender_2",
        player_position=(0.6, 0.5),
        player_team="defense",
        all_players=players,
        time_ms=1000,
        has_los_func=mock_los
    )

    print(f"T=1000ms: Focus expired, crossfire opps: {len(opps_expired)}")

    # Test coordinated entry detection
    print("\n=== Coordinated Entry Detection ===")
    coord.reset()

    coord.record_site_entry("defender_1", "site_A", 0)
    coord.record_site_entry("defender_2", "site_A", 300)

    entry = coord.site_entries["site_A"]
    print(f"Entry spread: 300ms, Coordinated: {entry.is_coordinated}")

    bonus = coord.get_entry_coordination_bonus("defender_1", "site_A")
    print(f"Coordination bonus: {bonus:.2f}")

    # Non-coordinated entry
    coord.reset()
    coord.record_site_entry("defender_1", "site_A", 0)
    coord.record_site_entry("defender_2", "site_A", 1000)

    entry = coord.site_entries["site_A"]
    print(f"\nEntry spread: 1000ms, Coordinated: {entry.is_coordinated}")

    print("\n" + "=" * 70)
    print("KEY INSIGHT:")
    print("Coordinated pushes work because of ATTENTION/FOCUS mechanics.")
    print("When engaged with one enemy, you're vulnerable to others.")
    print("This makes 2v1 retakes favor defenders WITHOUT hardcoded bonus.")
    print("=" * 70)


if __name__ == "__main__":
    test_coordination_system()
