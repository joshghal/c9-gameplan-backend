#!/usr/bin/env python3
"""
Extract Spike Carrier Data from GRID JSONL Files

Processes VCT match data to extract spike carrier patterns including:
- Pickup frequency per player and role
- Plant success rates
- Carrier death rates
- Plant timing distribution
- Site preferences

Output: backend/app/data/spike_carrier_patterns.json
"""

import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import statistics

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class PlayerSpikeStats:
    """Spike carrier statistics for a single player."""
    player_id: str
    player_name: str
    team: str
    role: str = "unknown"

    # Pickup and carry stats
    pickup_count: int = 0
    rounds_with_spike: int = 0

    # Plant stats
    plant_count: int = 0
    plant_times_ms: List[int] = field(default_factory=list)

    # Death stats
    dropped_on_death: int = 0

    # Site preferences
    plant_sites: Dict[str, int] = field(default_factory=lambda: {"a": 0, "b": 0, "c": 0})

    # Post-plant survival
    alive_after_plant: int = 0

    @property
    def plant_success_rate(self) -> float:
        """Ratio of plants to pickups."""
        return self.plant_count / max(1, self.pickup_count)

    @property
    def carrier_death_rate(self) -> float:
        """Ratio of deaths while carrying to pickups."""
        return self.dropped_on_death / max(1, self.pickup_count)

    @property
    def avg_plant_time_ms(self) -> float:
        """Average time to plant after round start."""
        if not self.plant_times_ms:
            return 56000  # Default from VCT stats
        return statistics.mean(self.plant_times_ms)

    @property
    def post_plant_survival_rate(self) -> float:
        """Survival rate after planting."""
        return self.alive_after_plant / max(1, self.plant_count)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "player_id": self.player_id,
            "player_name": self.player_name,
            "team": self.team,
            "role": self.role,
            "pickup_count": self.pickup_count,
            "plant_count": self.plant_count,
            "dropped_on_death": self.dropped_on_death,
            "avg_plant_time_ms": int(self.avg_plant_time_ms),
            "plant_success_rate": round(self.plant_success_rate, 3),
            "carrier_death_rate": round(self.carrier_death_rate, 3),
            "post_plant_survival": round(self.post_plant_survival_rate, 3),
            "plant_sites": dict(self.plant_sites),
        }


@dataclass
class RoleSpikeStats:
    """Aggregate spike stats by role."""
    role: str
    total_pickups: int = 0
    total_plants: int = 0
    total_carrier_deaths: int = 0
    plant_times_ms: List[int] = field(default_factory=list)
    rounds_played: int = 0

    @property
    def pickup_rate(self) -> float:
        """Pickups per round played."""
        return self.total_pickups / max(1, self.rounds_played)

    @property
    def plant_rate(self) -> float:
        """Plants per pickup."""
        return self.total_plants / max(1, self.total_pickups)

    @property
    def carrier_death_rate(self) -> float:
        """Deaths while carrying per pickup."""
        return self.total_carrier_deaths / max(1, self.total_pickups)

    @property
    def avg_plant_time_ms(self) -> float:
        """Average plant time."""
        if not self.plant_times_ms:
            return 56000
        return statistics.mean(self.plant_times_ms)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "pickup_rate": round(self.pickup_rate, 3),
            "plant_rate": round(self.plant_rate, 3),
            "carrier_death_rate": round(self.carrier_death_rate, 3),
            "avg_plant_time_ms": int(self.avg_plant_time_ms),
            "total_pickups": self.total_pickups,
            "total_plants": self.total_plants,
        }


class SpikeCarrierExtractor:
    """Extract spike carrier patterns from GRID JSONL files."""

    # Agent to role mapping
    AGENT_ROLES = {
        # Duelists
        'jett': 'duelist', 'raze': 'duelist', 'reyna': 'duelist',
        'phoenix': 'duelist', 'neon': 'duelist', 'yoru': 'duelist', 'iso': 'duelist',
        # Initiators
        'sova': 'initiator', 'skye': 'initiator', 'breach': 'initiator',
        'kayo': 'initiator', 'fade': 'initiator', 'gekko': 'initiator',
        # Controllers
        'omen': 'controller', 'brimstone': 'controller', 'astra': 'controller',
        'viper': 'controller', 'harbor': 'controller', 'clove': 'controller',
        # Sentinels
        'killjoy': 'sentinel', 'cypher': 'sentinel', 'sage': 'sentinel',
        'chamber': 'sentinel', 'deadlock': 'sentinel', 'vyse': 'sentinel',
    }

    def __init__(self, grid_data_dir: str):
        self.grid_data_dir = Path(grid_data_dir)
        self.player_stats: Dict[str, PlayerSpikeStats] = {}
        self.role_stats: Dict[str, RoleSpikeStats] = {
            "duelist": RoleSpikeStats("duelist"),
            "initiator": RoleSpikeStats("initiator"),
            "controller": RoleSpikeStats("controller"),
            "sentinel": RoleSpikeStats("sentinel"),
            "unknown": RoleSpikeStats("unknown"),
        }

        # Track per-round state
        self.current_round_carriers: Dict[str, str] = {}  # game_id -> player_id
        self.current_round_start: Dict[str, int] = {}  # game_id -> timestamp
        self.player_agents: Dict[str, str] = {}  # player_id -> agent
        self.player_teams: Dict[str, str] = {}  # player_id -> team_name

        # Metadata
        self.total_rounds = 0
        self.total_plants = 0
        self.total_pickups = 0
        self.total_carrier_deaths = 0

    def _get_role(self, agent: str) -> str:
        """Get role from agent name."""
        return self.AGENT_ROLES.get(agent.lower(), "unknown")

    def _get_or_create_player(self, player_id: str, player_name: str, team: str) -> PlayerSpikeStats:
        """Get or create player stats entry."""
        if player_id not in self.player_stats:
            agent = self.player_agents.get(player_id, "unknown")
            self.player_stats[player_id] = PlayerSpikeStats(
                player_id=player_id,
                player_name=player_name,
                team=team,
                role=self._get_role(agent),
            )
        return self.player_stats[player_id]

    def _determine_site(self, x: float, y: float, map_name: str) -> str:
        """Determine plant site from coordinates (simplified)."""
        # This is a simplified heuristic - actual implementation would use map-specific bounds
        # Most maps have A site at lower x coordinates
        if x < 0:
            return "a"
        elif x > 0 and y < 0:
            return "b"
        else:
            return "c" if map_name.lower() == "haven" else "b"

    def process_event(self, event: Dict, game_id: str, timestamp: str):
        """Process a single event from GRID data."""
        event_type = event.get("type", "")

        if event_type == "game-started-round":
            # Track round start for timing calculations
            self.current_round_start[game_id] = self._parse_timestamp(timestamp)
            self.total_rounds += 1

            # Reset carrier tracking for new round
            self.current_round_carriers[game_id] = None

        elif event_type == "player-pickedUp-item":
            target = event.get("target", {})
            if target.get("id") == "spike" or target.get("name", "").lower() == "spike":
                actor = event.get("actor", {})
                player_id = str(actor.get("id", ""))
                player_name = actor.get("state", {}).get("name", actor.get("name", "Unknown"))
                team_id = actor.get("state", {}).get("teamId", "")

                # Get team name from state
                team_name = self.player_teams.get(player_id, team_id)

                if player_id:
                    player = self._get_or_create_player(player_id, player_name, team_name)
                    player.pickup_count += 1
                    self.current_round_carriers[game_id] = player_id
                    self.total_pickups += 1

                    # Update role stats
                    role = player.role
                    self.role_stats[role].total_pickups += 1

        elif event_type == "player-dropped-item":
            target = event.get("target", {})
            if target.get("id") == "spike" or target.get("name", "").lower() == "spike":
                # Spike dropped - check if it was due to death
                actor = event.get("actor", {})
                player_id = str(actor.get("id", ""))

                # Check actor state to see if they died
                actor_state = actor.get("state", {}).get("game", {})
                is_alive = actor_state.get("alive", True)

                if player_id and player_id in self.player_stats and not is_alive:
                    player = self.player_stats[player_id]
                    player.dropped_on_death += 1
                    self.total_carrier_deaths += 1
                    self.role_stats[player.role].total_carrier_deaths += 1

        elif event_type == "player-completed-plantBomb":
            actor = event.get("actor", {})
            player_id = str(actor.get("id", ""))
            player_name = actor.get("state", {}).get("name", actor.get("name", "Unknown"))
            team_id = actor.get("state", {}).get("teamId", "")
            team_name = self.player_teams.get(player_id, team_id)

            if player_id:
                player = self._get_or_create_player(player_id, player_name, team_name)
                player.plant_count += 1
                self.total_plants += 1

                # Calculate plant time
                round_start = self.current_round_start.get(game_id, 0)
                current_time = self._parse_timestamp(timestamp)
                if round_start > 0:
                    plant_time_ms = current_time - round_start
                    # Clamp to reasonable values (3s to 100s)
                    if 3000 < plant_time_ms < 100000:
                        player.plant_times_ms.append(plant_time_ms)
                        self.role_stats[player.role].plant_times_ms.append(plant_time_ms)

                # Determine site from position
                position = actor.get("state", {}).get("game", {}).get("position", {})
                x = position.get("x", 0)
                y = position.get("y", 0)
                site = self._determine_site(x, y, "unknown")
                player.plant_sites[site] = player.plant_sites.get(site, 0) + 1

                # Check if alive after plant (at end of round, need to track separately)
                is_alive = actor.get("state", {}).get("game", {}).get("alive", True)
                if is_alive:
                    player.alive_after_plant += 1

                # Update role stats
                self.role_stats[player.role].total_plants += 1

        elif event_type == "player-killed-player":
            # Track if spike carrier was killed
            target = event.get("target", {})
            victim_id = str(target.get("id", ""))

            # Check if victim had spike
            inventory = target.get("state", {}).get("game", {}).get("inventory", {})
            items = inventory.get("items", [])
            had_spike = any(item.get("id") == "spike" or item.get("name", "").lower() == "spike"
                          for item in items)

            if had_spike and victim_id in self.player_stats:
                player = self.player_stats[victim_id]
                player.dropped_on_death += 1
                self.total_carrier_deaths += 1
                self.role_stats[player.role].total_carrier_deaths += 1

        elif event_type == "series-started-game":
            # Extract team and player info from game start
            series_state = event.get("seriesStateDelta", event.get("target", {}).get("state", {}))
            teams = series_state.get("teams", [])

            for team in teams:
                team_name = team.get("name", "")
                players = team.get("players", [])
                for player in players:
                    player_id = str(player.get("id", ""))
                    player_name = player.get("name", "")
                    agent = player.get("character", {}).get("name", "")

                    if player_id:
                        self.player_teams[player_id] = team_name
                        if agent:
                            self.player_agents[player_id] = agent.lower()

    def _parse_timestamp(self, timestamp: str) -> int:
        """Parse ISO timestamp to milliseconds since epoch."""
        from datetime import datetime
        try:
            # Handle various timestamp formats
            if timestamp.endswith('Z'):
                timestamp = timestamp[:-1] + '+00:00'
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            return int(dt.timestamp() * 1000)
        except:
            return 0

    def process_file(self, filepath: Path) -> int:
        """Process a single GRID JSONL file. Returns number of events processed."""
        events_processed = 0

        print(f"  Processing: {filepath.name}")

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        game_id = data.get("seriesId", "")
                        timestamp = data.get("occurredAt", "")

                        for event in data.get("events", []):
                            self.process_event(event, game_id, timestamp)
                            events_processed += 1

                    except json.JSONDecodeError:
                        continue

        except Exception as e:
            print(f"    Error processing {filepath.name}: {e}")

        return events_processed

    def process_all_files(self):
        """Process all GRID JSONL files in the directory."""
        jsonl_files = list(self.grid_data_dir.glob("*.jsonl"))
        print(f"\nFound {len(jsonl_files)} GRID JSONL files")

        total_events = 0
        for filepath in sorted(jsonl_files):
            events = self.process_file(filepath)
            total_events += events

        print(f"\nProcessed {total_events:,} events total")

    def update_role_round_counts(self):
        """Update role stats with round counts from player data."""
        # Approximate rounds per role from player data
        for player_id, player in self.player_stats.items():
            role = player.role
            # Estimate rounds played based on pickups (each round has 1 carrier)
            self.role_stats[role].rounds_played += player.pickup_count

    def generate_output(self) -> Dict:
        """Generate the output JSON structure."""
        self.update_role_round_counts()

        # Calculate role-based carrier aggression defaults
        role_carrier_defaults = {}
        for role, stats in self.role_stats.items():
            if stats.total_pickups > 0:
                # Derive carrier aggression from death rate (lower death rate = more passive, safer)
                # VCT target carrier death rate is ~12%
                death_rate = stats.carrier_death_rate

                # Map death rate to aggression:
                # - 0.05 death rate (very safe) -> 0.2 aggression
                # - 0.12 death rate (average) -> 0.4 aggression
                # - 0.20 death rate (aggressive) -> 0.7 aggression
                if death_rate < 0.08:
                    carrier_aggression = 0.2
                elif death_rate < 0.12:
                    carrier_aggression = 0.3
                elif death_rate < 0.16:
                    carrier_aggression = 0.5
                else:
                    carrier_aggression = 0.7

                role_carrier_defaults[role] = {
                    "carrier_aggression": carrier_aggression,
                    "early_engagement_mult": 0.3 if carrier_aggression < 0.4 else 0.5,
                    "execute_engagement_mult": 0.4 + carrier_aggression * 0.3,
                    "post_plant_hold_mult": 0.3 + carrier_aggression * 0.2,
                    "drop_spike_threshold": 0.8 - carrier_aggression * 0.2,
                    "lurk_probability": 0.25 - carrier_aggression * 0.2,
                    "fast_plant_tendency": 0.3 + carrier_aggression * 0.4,
                }

        # Build player stats dict (only include players with significant data)
        player_spike_stats = {}
        for player_id, player in self.player_stats.items():
            if player.pickup_count >= 3:  # Minimum sample size
                player_spike_stats[player.player_name] = player.to_dict()

        return {
            "metadata": {
                "total_rounds": self.total_rounds,
                "total_plants": self.total_plants,
                "total_pickups": self.total_pickups,
                "total_carrier_deaths": self.total_carrier_deaths,
                "plant_rate": round(self.total_plants / max(1, self.total_rounds), 3),
                "carrier_death_rate": round(self.total_carrier_deaths / max(1, self.total_pickups), 3),
                "avg_plant_time_ms": int(statistics.mean(
                    [t for stats in self.player_stats.values() for t in stats.plant_times_ms]
                ) if any(stats.plant_times_ms for stats in self.player_stats.values()) else 56000),
                "players_analyzed": len(player_spike_stats),
            },
            "player_spike_stats": player_spike_stats,
            "role_spike_stats": {role: stats.to_dict() for role, stats in self.role_stats.items()},
            "role_carrier_defaults": role_carrier_defaults,
            "vct_validation_targets": {
                "plant_rate": {"target": 0.65, "tolerance": 0.05, "description": "Plants per attack round"},
                "avg_plant_time_ms": {"target": 56000, "tolerance": 8000, "description": "Average plant time"},
                "carrier_death_rate": {"target": 0.12, "tolerance": 0.04, "description": "Carrier deaths per pickup"},
                "attack_win_rate": {"target": 0.48, "tolerance": 0.03, "description": "Attack side win rate"},
            }
        }


def main():
    """Main entry point."""
    # Paths
    script_dir = Path(__file__).parent
    backend_dir = script_dir.parent
    grid_data_dir = backend_dir.parent / "grid_data"
    output_path = backend_dir / "app" / "data" / "spike_carrier_patterns.json"

    print("=" * 60)
    print("Spike Carrier Data Extraction from VCT GRID Data")
    print("=" * 60)

    if not grid_data_dir.exists():
        print(f"\nError: GRID data directory not found: {grid_data_dir}")
        print("Please ensure VCT GRID JSONL files are in the grid_data directory.")
        sys.exit(1)

    # Process data
    extractor = SpikeCarrierExtractor(str(grid_data_dir))
    extractor.process_all_files()

    # Generate output
    output = extractor.generate_output()

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"\nOutput: {output_path}")
    print(f"\nSummary:")
    print(f"  Total rounds analyzed: {output['metadata']['total_rounds']:,}")
    print(f"  Total plants: {output['metadata']['total_plants']:,}")
    print(f"  Total spike pickups: {output['metadata']['total_pickups']:,}")
    print(f"  Plant rate: {output['metadata']['plant_rate']:.1%}")
    print(f"  Carrier death rate: {output['metadata']['carrier_death_rate']:.1%}")
    print(f"  Avg plant time: {output['metadata']['avg_plant_time_ms']/1000:.1f}s")
    print(f"  Players with data: {output['metadata']['players_analyzed']}")

    print("\nRole-based defaults:")
    for role, defaults in output.get("role_carrier_defaults", {}).items():
        print(f"  {role}:")
        print(f"    carrier_aggression: {defaults.get('carrier_aggression', 'N/A')}")
        print(f"    early_engagement_mult: {defaults.get('early_engagement_mult', 'N/A')}")


if __name__ == "__main__":
    main()
