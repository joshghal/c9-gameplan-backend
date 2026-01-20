"""
GRID Match Data Parser

Parses GRID JSONL event data to extract player profiles and match statistics.
This creates realistic player profiles for use in simulations.
"""

import json
import math
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from collections import defaultdict


logger = logging.getLogger(__name__)


@dataclass
class KillData:
    """Data for a single kill event."""
    round_num: int
    time_ms: int
    killer_id: str
    killer_name: str
    victim_id: str
    victim_name: str
    weapon: str
    headshot: bool
    killer_pos: Tuple[float, float]
    victim_pos: Tuple[float, float]
    killer_team: str
    victim_team: str

    @property
    def distance(self) -> float:
        """Calculate kill distance in game units."""
        dx = self.killer_pos[0] - self.victim_pos[0]
        dy = self.killer_pos[1] - self.victim_pos[1]
        return math.sqrt(dx * dx + dy * dy)

    @property
    def distance_meters(self) -> float:
        """Convert to approximate meters (1 meter ≈ 100 game units)."""
        return self.distance / 100.0


@dataclass
class RoundStats:
    """Statistics for a single round."""
    round_num: int
    winner_team: str
    win_reason: str
    spike_planted: bool
    plant_time_ms: Optional[int]
    kills: List[KillData] = field(default_factory=list)
    first_blood_team: Optional[str] = None
    first_blood_player: Optional[str] = None


@dataclass
class PlayerStats:
    """Accumulated player statistics from match data."""
    player_id: str
    player_name: str
    team_id: str
    team_name: str

    # Combat stats
    kills: int = 0
    deaths: int = 0
    assists: int = 0
    headshot_kills: int = 0
    bodyshot_kills: int = 0

    # First blood
    first_bloods: int = 0
    first_deaths: int = 0

    # Kill details
    kill_distances: List[float] = field(default_factory=list)
    kill_weapons: Dict[str, int] = field(default_factory=dict)
    kill_times_ms: List[int] = field(default_factory=list)

    # Positioning
    positions: List[Tuple[float, float]] = field(default_factory=list)

    # Agent
    agent: str = ""

    @property
    def headshot_rate(self) -> float:
        """Calculate headshot percentage."""
        total_kills = self.kills
        if total_kills == 0:
            return 0.0
        return self.headshot_kills / total_kills

    @property
    def kd_ratio(self) -> float:
        """Calculate K/D ratio."""
        if self.deaths == 0:
            return float(self.kills)
        return self.kills / self.deaths

    @property
    def avg_kill_distance(self) -> float:
        """Average kill distance in meters."""
        if not self.kill_distances:
            return 15.0
        return sum(self.kill_distances) / len(self.kill_distances)

    @property
    def entry_rate(self) -> float:
        """Rate of being in first blood (kill or death)."""
        total = self.first_bloods + self.first_deaths
        if total == 0:
            return 0.0
        return self.first_bloods / total


@dataclass
class PlayerProfile:
    """Derived player profile for simulation use."""
    player_id: str
    player_name: str
    team: str

    # Combat characteristics
    avg_headshot_rate: float = 0.25
    headshot_rate_variance: float = 0.05

    # Reaction time estimate (ms)
    estimated_reaction_ms: float = 180.0

    # Aggression (0-1)
    aggression: float = 0.5

    # Crosshair placement quality (0-1)
    crosshair_placement: float = 0.7

    # Entry frequency
    entry_frequency: float = 0.15

    # Consistency (0-1)
    consistency: float = 0.7

    # Preferred engagement range (meters)
    avg_kill_distance: float = 15.0

    # Clutch conversion
    clutch_rate: float = 0.0

    # Primary weapons
    primary_weapons: Dict[str, float] = field(default_factory=dict)

    # Agent pool
    agents_played: Dict[str, int] = field(default_factory=dict)
    primary_agent: str = ""


class GRIDMatchParser:
    """Parse GRID JSONL match data."""

    def __init__(self):
        self.players: Dict[str, PlayerStats] = {}
        self.teams: Dict[str, str] = {}  # team_id -> team_name
        self.rounds: List[RoundStats] = []
        self.kills: List[KillData] = []
        self.current_round = 0
        self.round_start_time = 0

    def parse_file(self, filepath: str) -> None:
        """Parse a GRID JSONL file."""
        logger.info(f"Parsing GRID data from {filepath}")

        with open(filepath, 'r') as f:
            for line_num, line in enumerate(f):
                try:
                    data = json.loads(line)
                    self._process_events(data.get('events', []))
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON at line {line_num}: {e}")
                except Exception as e:
                    logger.warning(f"Error processing line {line_num}: {e}")

        logger.info(f"Parsed {len(self.players)} players, {len(self.rounds)} rounds, {len(self.kills)} kills")

    def _process_events(self, events: List[Dict]) -> None:
        """Process a list of events."""
        for event in events:
            event_type = event.get('type', '')

            if event_type == 'game-started-round':
                self._handle_round_start(event)
            elif event_type == 'game-ended-round':
                self._handle_round_end(event)
            elif event_type == 'team-won-round':
                self._handle_round_won(event)
            elif event_type == 'player-killed-player':
                self._handle_kill(event)
            elif event_type == 'player-completed-plantBomb':
                self._handle_plant(event)

            # Always extract player/team info
            self._extract_entity_info(event.get('actor', {}))
            self._extract_entity_info(event.get('target', {}))

    def _extract_entity_info(self, entity: Dict) -> None:
        """Extract player/team information from an entity."""
        if not entity:
            return

        entity_type = entity.get('type')

        if entity_type == 'player':
            state = entity.get('state', {})
            player_id = state.get('id')
            player_name = state.get('name')
            team_id = state.get('teamId')

            if player_id and player_name:
                if player_id not in self.players:
                    self.players[player_id] = PlayerStats(
                        player_id=player_id,
                        player_name=player_name,
                        team_id=team_id or '',
                        team_name=self.teams.get(team_id, '')
                    )

                # Update position if available
                game_state = state.get('game', {})
                pos = game_state.get('position', {})
                if pos:
                    x = pos.get('x', 0)
                    y = pos.get('y', 0)
                    if x != 0 or y != 0:
                        self.players[player_id].positions.append((x, y))

                # Extract cumulative headshots from series state
                series_state = state.get('series', {})
                if series_state:
                    headshots = series_state.get('headshots', 0)
                    if headshots > 0:
                        # Update max headshots seen (cumulative)
                        self.players[player_id].headshot_kills = max(
                            self.players[player_id].headshot_kills,
                            headshots
                        )

        elif entity_type == 'team':
            state = entity.get('state', {})
            team_id = state.get('id')
            team_name = state.get('name')
            if team_id and team_name:
                self.teams[team_id] = team_name
                # Update player team names
                for player in self.players.values():
                    if player.team_id == team_id:
                        player.team_name = team_name

    def _handle_round_start(self, event: Dict) -> None:
        """Handle round start event."""
        target = event.get('target', {})
        state = target.get('state', {})
        self.current_round = state.get('number', self.current_round + 1)
        self.round_start_time = 0  # Will be set by clock

    def _handle_round_end(self, event: Dict) -> None:
        """Handle round end event."""
        pass  # Most info comes from team-won-round

    def _handle_round_won(self, event: Dict) -> None:
        """Handle team winning a round."""
        actor = event.get('actor', {})
        state = actor.get('state', {})
        team_name = state.get('name', '')

        action = event.get('action', '')
        win_reason = 'unknown'
        if 'elimination' in action:
            win_reason = 'elimination'
        elif 'spike' in action or 'explod' in action:
            win_reason = 'spike_exploded'
        elif 'defus' in action:
            win_reason = 'spike_defused'
        elif 'time' in action:
            win_reason = 'time'

        round_kills = [k for k in self.kills if k.round_num == self.current_round]

        round_stats = RoundStats(
            round_num=self.current_round,
            winner_team=team_name,
            win_reason=win_reason,
            spike_planted=any(k.round_num == self.current_round for k in self.kills),  # Approximation
            plant_time_ms=None,
            kills=round_kills
        )

        # Determine first blood
        if round_kills:
            first_kill = min(round_kills, key=lambda k: k.time_ms)
            round_stats.first_blood_player = first_kill.killer_name
            round_stats.first_blood_team = first_kill.killer_team

            # Update player first blood stats
            killer = self._find_player_by_name(first_kill.killer_name)
            victim = self._find_player_by_name(first_kill.victim_name)
            if killer:
                killer.first_bloods += 1
            if victim:
                victim.first_deaths += 1

        self.rounds.append(round_stats)

    def _handle_kill(self, event: Dict) -> None:
        """Handle a kill event."""
        actor = event.get('actor', {})
        target = event.get('target', {})

        actor_state = actor.get('state', {})
        target_state = target.get('state', {})

        killer_name = actor_state.get('name', '')
        victim_name = target_state.get('name', '')
        killer_team = self.teams.get(actor_state.get('teamId', ''), '')
        victim_team = self.teams.get(target_state.get('teamId', ''), '')

        # Get positions
        killer_game = actor_state.get('game', {})
        victim_game = target_state.get('game', {})
        killer_pos = killer_game.get('position', {})
        victim_pos = victim_game.get('position', {})

        killer_xy = (killer_pos.get('x', 0), killer_pos.get('y', 0))
        victim_xy = (victim_pos.get('x', 0), victim_pos.get('y', 0))

        # Determine weapon and headshot
        # Check weapon from state delta
        actor_delta = actor.get('stateDelta', {})
        round_delta = actor_delta.get('round', {})
        weapon_kills = round_delta.get('weaponKills', {})

        weapon = 'unknown'
        if weapon_kills:
            weapon = list(weapon_kills.keys())[0]

        # Check for headshot indicator in the event
        action = event.get('action', '')
        headshot = 'headshot' in action.lower()

        # Sometimes headshot info is in damage type
        damage_info = event.get('damage', {})
        if damage_info.get('type') == 'headshot':
            headshot = True

        kill_data = KillData(
            round_num=self.current_round,
            time_ms=0,  # Would need to track from clock events
            killer_id=actor_state.get('id', ''),
            killer_name=killer_name,
            victim_id=target_state.get('id', ''),
            victim_name=victim_name,
            weapon=weapon,
            headshot=headshot,
            killer_pos=killer_xy,
            victim_pos=victim_xy,
            killer_team=killer_team,
            victim_team=victim_team
        )

        self.kills.append(kill_data)

        # Update player stats
        killer = self._find_player_by_name(killer_name)
        victim = self._find_player_by_name(victim_name)

        if killer:
            killer.kills += 1
            if headshot:
                killer.headshot_kills += 1
            else:
                killer.bodyshot_kills += 1

            killer.kill_distances.append(kill_data.distance_meters)
            killer.kill_weapons[weapon] = killer.kill_weapons.get(weapon, 0) + 1

        if victim:
            victim.deaths += 1

    def _handle_plant(self, event: Dict) -> None:
        """Handle spike plant event."""
        # Would track plant times for round data
        pass

    def _find_player_by_name(self, name: str) -> Optional[PlayerStats]:
        """Find player by name."""
        for player in self.players.values():
            if player.player_name.lower() == name.lower():
                return player
        return None

    def build_profiles(self) -> Dict[str, PlayerProfile]:
        """Build player profiles from accumulated stats."""
        profiles = {}

        for player_id, stats in self.players.items():
            if stats.kills == 0 and stats.deaths == 0:
                continue

            # Calculate derived stats
            hs_rate = stats.headshot_rate

            # Estimate reaction time from first blood performance
            # High first blood rate = faster reactions
            fb_rate = stats.first_bloods / max(1, len(self.rounds))
            base_reaction = 180
            reaction_bonus = fb_rate * 100  # Up to -30ms for high FB rate
            estimated_reaction = max(150, min(220, base_reaction - reaction_bonus))

            # Aggression from first blood frequency
            entry_rate = (stats.first_bloods + stats.first_deaths) / max(1, len(self.rounds))
            aggression = min(1.0, entry_rate * 3)

            # Crosshair placement from headshot rate
            # Pro HS rate ~25% = 0.75 placement
            crosshair_placement = 0.4 + (hs_rate * 1.5)

            # Calculate weapon preferences
            total_weapon_kills = sum(stats.kill_weapons.values())
            weapon_prefs = {}
            if total_weapon_kills > 0:
                for weapon, count in stats.kill_weapons.items():
                    weapon_prefs[weapon] = count / total_weapon_kills

            profile = PlayerProfile(
                player_id=player_id,
                player_name=stats.player_name,
                team=stats.team_name,
                avg_headshot_rate=hs_rate,
                headshot_rate_variance=0.05,
                estimated_reaction_ms=estimated_reaction,
                aggression=aggression,
                crosshair_placement=min(0.95, crosshair_placement),
                entry_frequency=stats.entry_rate,
                consistency=0.75,  # Would need multiple matches to calculate
                avg_kill_distance=stats.avg_kill_distance,
                primary_weapons=weapon_prefs
            )

            profiles[stats.player_name.lower()] = profile

        return profiles

    def get_team_profiles(self, team_name: str) -> Dict[str, PlayerProfile]:
        """Get profiles for players on a specific team."""
        all_profiles = self.build_profiles()
        team_profiles = {}

        for name, profile in all_profiles.items():
            if profile.team.lower() == team_name.lower():
                team_profiles[name] = profile

        return team_profiles

    def print_summary(self) -> None:
        """Print a summary of parsed data."""
        print("\n" + "=" * 60)
        print("GRID MATCH DATA SUMMARY")
        print("=" * 60)

        print(f"\nTeams: {', '.join(self.teams.values())}")
        print(f"Rounds: {len(self.rounds)}")
        print(f"Total Kills: {len(self.kills)}")

        for team_name in self.teams.values():
            print(f"\n{team_name}:")
            for player in self.players.values():
                if player.team_name == team_name:
                    hs_pct = player.headshot_rate * 100
                    print(f"  {player.player_name:12s} | K:{player.kills:2d} D:{player.deaths:2d} | "
                          f"HS:{hs_pct:4.1f}% | FB:{player.first_bloods} FD:{player.first_deaths} | "
                          f"Avg Dist: {player.avg_kill_distance:.1f}m")


def parse_grid_data(filepath: str) -> GRIDMatchParser:
    """Parse GRID data file and return parser with results."""
    parser = GRIDMatchParser()
    parser.parse_file(filepath)
    return parser


def build_profiles_from_grid(filepath: str) -> Dict[str, PlayerProfile]:
    """Build player profiles from GRID data file."""
    parser = parse_grid_data(filepath)
    return parser.build_profiles()


# Main execution for testing
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python grid_parser.py <path_to_jsonl>")
        sys.exit(1)

    filepath = sys.argv[1]
    parser = parse_grid_data(filepath)
    parser.print_summary()

    print("\n" + "=" * 60)
    print("PLAYER PROFILES")
    print("=" * 60)

    profiles = parser.build_profiles()
    for name, profile in profiles.items():
        print(f"\n{profile.player_name} ({profile.team}):")
        print(f"  Headshot Rate: {profile.avg_headshot_rate*100:.1f}%")
        print(f"  Est. Reaction: {profile.estimated_reaction_ms:.0f}ms")
        print(f"  Aggression: {profile.aggression:.2f}")
        print(f"  Crosshair: {profile.crosshair_placement:.2f}")
        print(f"  Avg Distance: {profile.avg_kill_distance:.1f}m")
