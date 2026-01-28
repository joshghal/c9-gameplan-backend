#!/usr/bin/env python3
"""
Simulation Runner - Uses main SimulationEngine

Runs simulations using the actual simulation_engine.py logic,
outputs JSON snapshots that can be visualized with visualize.py.

Usage:
    cd backend
    source venv/bin/activate

    # Run single simulation
    python scripts/simulate.py --map lotus --c9-side attack

    # Run multiple rounds
    python scripts/simulate.py --map lotus --c9-side attack --rounds 5

    # Specify output
    python scripts/simulate.py --map lotus -o output/sim_001.json

    # Run and visualize
    python scripts/simulate.py --map lotus --visualize panels
"""

import sys
import os
import json
import asyncio
import argparse
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.services.simulation_engine import SimulationEngine
from app.services.c9_realism import get_c9_realism, C9_ROSTER
from app.services.opponent_realism import set_opponent_team, get_available_teams

# Default output directory
OUTPUT_DIR = Path(__file__).parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


@dataclass
class MockTeam:
    """Mock team for simulation."""
    id: str
    name: str
    players: List[Dict[str, Any]] = field(default_factory=list)

    def get(self, key: str, default=None):
        """Dict-like access for compatibility."""
        if key == 'players':
            return self.players
        return getattr(self, key, default)


@dataclass
class MockSession:
    """Mock SimulationSession for running without database."""
    id: uuid.UUID = field(default_factory=uuid.uuid4)
    attack_team_id: str = "attack_team"
    defense_team_id: str = "defense_team"
    map_name: str = "lotus"
    round_type: str = "full"
    current_time_ms: int = 0
    phase: str = "opening"
    status: str = "running"
    snapshots: List[Dict] = field(default_factory=list)
    events_log: List[Dict] = field(default_factory=list)

    # These are set during initialization
    attack_team: Optional[MockTeam] = None
    defense_team: Optional[MockTeam] = None


class MockScalars:
    """Mock scalars result."""
    def __init__(self, values=None):
        self._values = values or []

    def all(self):
        return self._values

    def first(self):
        return self._values[0] if self._values else None


class MockResult:
    """Mock database query result."""
    def __init__(self, value=None, values=None):
        self._value = value
        self._values = values or []

    def scalar_one_or_none(self):
        return self._value

    def scalars(self):
        return MockScalars(self._values)


class MockPlayer:
    """Mock player object with attributes."""
    def __init__(self, id: str, name: str, agent: str):
        self.id = id
        self.name = name
        self.agent = agent
        self.role = agent  # Use agent as role fallback


class MockTeamDB:
    """Mock team object for database returns."""
    def __init__(self, team: 'MockTeam'):
        self.id = team.id
        self.name = team.name
        self.players = [
            MockPlayer(p['id'], p['name'], p['agent'])
            for p in team.players
        ]


class MockDB:
    """Mock async database session."""
    def __init__(self):
        self._teams = {}
        self._team_queue = []  # Teams to return in order

    def register_team(self, team: 'MockTeam'):
        """Register a team so it can be looked up by ID."""
        self._teams[team.id] = team
        self._team_queue.append(MockTeamDB(team))

    async def execute(self, query):
        """Mock execute that returns registered teams in order."""
        # Return teams in the order they were registered
        if self._team_queue:
            team = self._team_queue.pop(0)
            return MockResult(value=team)
        return MockResult(None)

    async def commit(self):
        pass

    async def refresh(self, obj):
        pass


def create_c9_team(side: str) -> MockTeam:
    """Create C9 team with real roster."""
    agents = ["jett", "omen", "sova", "killjoy", "raze"]  # Example agents
    players = []
    for i, name in enumerate(C9_ROSTER):
        players.append({
            'id': f'c9_{i}',
            'name': name,
            'agent': agents[i % len(agents)],
        })
    return MockTeam(
        id="c9_team",
        name="Cloud9",
        players=players
    )


def create_opponent_team(side: str, team_name: str = "sentinels") -> MockTeam:
    """Create opponent team with roster from VCT data."""
    # Set the opponent team in the realism service
    service = set_opponent_team(team_name)
    roster = service.roster

    agents = ["phoenix", "cypher", "sage", "breach", "viper"]
    players = []
    for i, name in enumerate(roster[:5]):
        players.append({
            'id': f'opp_{i}',
            'name': name,
            'agent': agents[i % len(agents)],
        })

    # If roster is smaller than 5, fill with generic names
    while len(players) < 5:
        i = len(players)
        players.append({
            'id': f'opp_{i}',
            'name': f'Player{i+1}',
            'agent': agents[i % len(agents)],
        })

    return MockTeam(
        id="opponent_team",
        name=team_name.upper(),
        players=players
    )


async def run_simulation(
    map_name: str,
    c9_side: str,
    round_type: str = "full",
    snapshot_interval_ms: int = 5000,
    max_time_ms: int = 100000,
    opponent_team: str = "sentinels",
) -> Dict[str, Any]:
    """
    Run a single simulation using the main engine.

    Args:
        map_name: Map to simulate on
        c9_side: C9 team side ('attack' or 'defense')
        round_type: Economy round type
        snapshot_interval_ms: How often to capture snapshots
        max_time_ms: Maximum simulation time
        opponent_team: Name of opponent team (e.g., 'sentinels', 'nrg')

    Returns:
        Simulation result dict with snapshots, events, winner, etc.
    """
    # Create mock objects
    mock_db = MockDB()

    # Create teams based on C9 side
    if c9_side == 'attack':
        attack_team = create_c9_team('attack')
        defense_team = create_opponent_team('defense', opponent_team)
    else:
        attack_team = create_opponent_team('attack', opponent_team)
        defense_team = create_c9_team('defense')

    # Register teams with MockDB (engine loads attack first, then defense)
    mock_db.register_team(attack_team)
    mock_db.register_team(defense_team)

    # Create session
    session = MockSession(
        map_name=map_name,
        round_type=round_type,
        attack_team_id=attack_team.id,
        defense_team_id=defense_team.id,
        attack_team=attack_team,
        defense_team=defense_team,
    )

    # Initialize engine
    engine = SimulationEngine(mock_db)

    # Initialize simulation
    state = await engine.initialize(session, round_type=round_type)

    # Collect snapshots
    snapshots = []
    all_events = []
    last_snapshot_ms = 0

    # Capture initial state
    snapshots.append({
        'time_ms': 0,
        'phase': state.phase,
        'positions': [p.model_dump() for p in state.positions],
        'events': [],
        'spike_planted': state.spike_planted,
        'spike_site': state.spike_site,
        'attack_alive': state.attack_alive,
        'defense_alive': state.defense_alive,
    })

    # Run simulation
    while state.current_time_ms < max_time_ms and state.status == 'running':
        # Advance by 10 ticks at a time for speed
        state = await engine.advance(session, ticks=10)

        # CRITICAL: Update session time (engine reads from session.current_time_ms)
        session.current_time_ms = state.current_time_ms

        # Collect events
        for event in state.events:
            event_dict = event.model_dump()
            if event_dict not in all_events:
                all_events.append(event_dict)

        # Capture snapshot at intervals
        if state.current_time_ms - last_snapshot_ms >= snapshot_interval_ms:
            snapshots.append({
                'time_ms': state.current_time_ms,
                'phase': state.phase,
                'positions': [p.model_dump() for p in state.positions],
                'events': [e for e in all_events if e['timestamp_ms'] > last_snapshot_ms],
                'spike_planted': state.spike_planted,
                'spike_site': state.spike_site,
                'attack_alive': state.attack_alive,
                'defense_alive': state.defense_alive,
            })
            last_snapshot_ms = state.current_time_ms

        # Check for round end
        if state.attack_alive == 0 or state.defense_alive == 0:
            break
        if state.spike_planted and state.current_time_ms > 45000:  # Spike exploded
            break

    # Capture final state
    if snapshots[-1]['time_ms'] != state.current_time_ms:
        snapshots.append({
            'time_ms': state.current_time_ms,
            'phase': state.phase,
            'positions': [p.model_dump() for p in state.positions],
            'events': [e for e in all_events if e['timestamp_ms'] > last_snapshot_ms],
            'spike_planted': state.spike_planted,
            'spike_site': state.spike_site,
            'attack_alive': state.attack_alive,
            'defense_alive': state.defense_alive,
        })

    # Determine winner
    if state.attack_alive == 0:
        winner = 'defense'
        win_condition = 'elimination'
    elif state.defense_alive == 0:
        winner = 'attack'
        win_condition = 'elimination'
    elif state.spike_planted:
        winner = 'attack'
        win_condition = 'spike_detonation'
    else:
        winner = 'defense'
        win_condition = 'timeout'

    # Count kills
    kills = [e for e in all_events if e.get('event_type') == 'kill']
    attack_kills = len([k for k in kills if k.get('details', {}).get('killer_side') == 'attack'])
    defense_kills = len([k for k in kills if k.get('details', {}).get('killer_side') == 'defense'])

    return {
        'metadata': {
            'map_name': map_name,
            'c9_side': c9_side,
            'opponent_team': opponent_team,
            'round_type': round_type,
            'timestamp': datetime.now().isoformat(),
            'engine': 'simulation_engine.py',
        },
        'result': {
            'winner': winner,
            'win_condition': win_condition,
            'c9_won': (winner == c9_side),
            'duration_ms': state.current_time_ms,
            'total_kills': len(kills),
            'attack_kills': attack_kills,
            'defense_kills': defense_kills,
            'spike_planted': state.spike_planted,
            'spike_site': state.spike_site,
        },
        'snapshots': snapshots,
        'events': all_events,
    }


async def run_batch(
    map_name: str,
    c9_side: str,
    num_rounds: int,
    round_type: str = "full",
    opponent_team: str = "sentinels",
) -> List[Dict[str, Any]]:
    """Run multiple simulations and return results."""
    results = []
    for i in range(num_rounds):
        print(f"  Running round {i+1}/{num_rounds}...", end=" ", flush=True)
        result = await run_simulation(map_name, c9_side, round_type, opponent_team=opponent_team)
        results.append(result)
        print(f"Winner: {result['result']['winner']}, Kills: {result['result']['total_kills']}")
    return results


def print_summary(results: List[Dict[str, Any]]):
    """Print summary statistics."""
    total = len(results)
    c9_wins = sum(1 for r in results if r['result']['c9_won'])
    total_kills = sum(r['result']['total_kills'] for r in results)

    print(f"\n{'='*60}")
    print(f"SIMULATION SUMMARY ({total} rounds)")
    print(f"{'='*60}")
    print(f"  C9 Win Rate: {c9_wins}/{total} ({100*c9_wins/total:.1f}%)")
    print(f"  Average Kills: {total_kills/total:.1f} per round")

    # Win conditions
    conditions = {}
    for r in results:
        cond = r['result']['win_condition']
        conditions[cond] = conditions.get(cond, 0) + 1
    print(f"  Win Conditions:")
    for cond, count in sorted(conditions.items()):
        print(f"    {cond}: {count} ({100*count/total:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Run simulations using the main engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/simulate.py --map lotus --c9-side attack
  python scripts/simulate.py --map lotus --rounds 10 -o output/batch.json
  python scripts/simulate.py --map ascent --c9-side defense --visualize panels
        """
    )

    parser.add_argument('--map', '-m', default='lotus',
                       help='Map name (default: lotus)')
    parser.add_argument('--c9-side', '-s', choices=['attack', 'defense'], default='attack',
                       help='C9 team side (default: attack)')
    parser.add_argument('--opponent', '--opp', default='sentinels',
                       choices=get_available_teams(),
                       help='Opponent team (default: sentinels)')
    parser.add_argument('--rounds', '-r', type=int, default=1,
                       help='Number of rounds to simulate (default: 1)')
    parser.add_argument('--round-type', '-t', default='full',
                       choices=['pistol', 'eco', 'force', 'half', 'full'],
                       help='Economy round type (default: full)')
    parser.add_argument('--output', '-o', type=str,
                       help='Output JSON file path')
    parser.add_argument('--visualize', '-v', type=str,
                       choices=['panels', 'timeline', 'stats'],
                       help='Visualize after simulation')
    parser.add_argument('--snapshot-interval', type=int, default=5000,
                       help='Snapshot interval in ms (default: 5000)')

    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"SIMULATION: {args.map.upper()} - C9 on {args.c9_side.upper()} vs {args.opponent.upper()}")
    print(f"{'='*60}")

    # Run simulations
    if args.rounds == 1:
        result = asyncio.run(run_simulation(
            map_name=args.map,
            c9_side=args.c9_side,
            round_type=args.round_type,
            snapshot_interval_ms=args.snapshot_interval,
            opponent_team=args.opponent,
        ))
        results = [result]

        print(f"\nResult:")
        print(f"  Winner: {result['result']['winner']} ({result['result']['win_condition']})")
        print(f"  C9 Won: {result['result']['c9_won']}")
        print(f"  Duration: {result['result']['duration_ms']/1000:.1f}s")
        print(f"  Kills: {result['result']['total_kills']}")
        print(f"  Snapshots: {len(result['snapshots'])}")
    else:
        results = asyncio.run(run_batch(
            map_name=args.map,
            c9_side=args.c9_side,
            num_rounds=args.rounds,
            round_type=args.round_type,
            opponent_team=args.opponent,
        ))
        print_summary(results)

    # Save output
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = OUTPUT_DIR / f"sim_{args.map}_{args.c9_side}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        if args.rounds == 1:
            json.dump(results[0], f, indent=2, default=str)
        else:
            json.dump({
                'metadata': {
                    'map_name': args.map,
                    'c9_side': args.c9_side,
                    'round_type': args.round_type,
                    'num_rounds': args.rounds,
                    'timestamp': datetime.now().isoformat(),
                },
                'summary': {
                    'c9_wins': sum(1 for r in results if r['result']['c9_won']),
                    'total_rounds': len(results),
                    'avg_kills': sum(r['result']['total_kills'] for r in results) / len(results),
                },
                'rounds': results,
            }, f, indent=2, default=str)

    print(f"\nSaved: {output_path}")

    # Visualize if requested
    if args.visualize:
        print(f"\nGenerating {args.visualize} visualization...")
        # TODO: Call visualize.py
        print(f"  (visualize.py not yet implemented)")


if __name__ == '__main__':
    main()
