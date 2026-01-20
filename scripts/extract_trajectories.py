#!/usr/bin/env python3
"""
Extract player movement trajectories from VCT JSONL data.

For each round, reconstructs player positions over time from all events.
Outputs trajectory data that can be used for:
- Realistic AI movement paths
- Understanding pro positioning throughout rounds
- Interpolating movement between known positions

Usage:
    python scripts/extract_trajectories.py
"""

import json
import os
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).parent.parent.parent
GRID_DATA_DIR = PROJECT_ROOT / "grid_data"
OUTPUT_DIR = PROJECT_ROOT / "backend" / "app" / "data"


def parse_timestamp(ts: str) -> float:
    """Parse ISO timestamp to unix timestamp."""
    try:
        dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
        return dt.timestamp()
    except:
        return 0


def extract_trajectories_from_file(fpath: Path) -> Dict:
    """Extract all trajectories from a single JSONL file."""

    rounds = []
    current_round = None
    current_map = None
    current_game_id = None
    round_events = []

    with open(fpath, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                timestamp = data.get('occurredAt', '')

                for event in data.get('events', []):
                    etype = event.get('type', '')

                    # Track game/map
                    if etype == 'series-started-game':
                        target = event.get('target', {})
                        state = target.get('state', {})
                        map_data = state.get('map', {})
                        current_map = map_data.get('name', '').lower()
                        current_game_id = state.get('id')

                    # New round started
                    if etype == 'game-started-round':
                        # Save previous round if exists
                        if current_round is not None and round_events:
                            rounds.append({
                                'round_num': current_round,
                                'map': current_map,
                                'game_id': current_game_id,
                                'events': round_events
                            })

                        # Get round number
                        actor = event.get('actor', {})
                        state_delta = actor.get('stateDelta', {})
                        segments = state_delta.get('segments', [])
                        if segments:
                            current_round = segments[0].get('sequenceNumber', 0)

                        round_events = []

                    # Extract positions from any event with seriesState
                    series = event.get('seriesState', {})
                    games = series.get('games', [])

                    if games and current_round is not None:
                        clock = games[0].get('clock', {})
                        clock_seconds = clock.get('currentSeconds', 0)

                        player_positions = {}
                        for team in games[0].get('teams', []):
                            team_name = team.get('name', '')
                            side = team.get('side', '')

                            for player in team.get('players', []):
                                pos = player.get('position')
                                if pos:
                                    player_positions[player.get('name', '')] = {
                                        'x': pos['x'],
                                        'y': pos['y'],
                                        'alive': player.get('alive', True),
                                        'team': team_name,
                                        'side': side
                                    }

                        if player_positions:
                            round_events.append({
                                'timestamp': timestamp,
                                'clock_seconds': clock_seconds,
                                'event_type': etype,
                                'positions': player_positions
                            })

            except json.JSONDecodeError:
                continue
            except Exception as e:
                continue

    # Save last round
    if current_round is not None and round_events:
        rounds.append({
            'round_num': current_round,
            'map': current_map,
            'game_id': current_game_id,
            'events': round_events
        })

    return rounds


def build_trajectories(rounds: List[Dict]) -> Dict:
    """Convert round events into player trajectories."""

    trajectories_by_map = defaultdict(list)

    for round_data in rounds:
        map_name = round_data['map']
        round_num = round_data['round_num']
        events = round_data['events']

        if not events or not map_name:
            continue

        # Build trajectory for each player
        player_trajectories = defaultdict(list)

        for event in events:
            clock = event['clock_seconds']
            for player_name, pos_data in event['positions'].items():
                player_trajectories[player_name].append({
                    'clock': clock,
                    'x': pos_data['x'],
                    'y': pos_data['y'],
                    'alive': pos_data['alive'],
                    'team': pos_data.get('team', ''),
                    'side': pos_data.get('side', '')
                })

        # Deduplicate and sort by clock
        for player_name, trajectory in player_trajectories.items():
            # Sort by clock (descending since clock counts down)
            trajectory.sort(key=lambda p: -p['clock'])

            # Deduplicate consecutive same positions
            deduped = []
            for point in trajectory:
                if not deduped or (point['x'] != deduped[-1]['x'] or point['y'] != deduped[-1]['y']):
                    deduped.append(point)

            player_trajectories[player_name] = deduped

        trajectories_by_map[map_name].append({
            'round_num': round_num,
            'game_id': round_data['game_id'],
            'player_trajectories': dict(player_trajectories)
        })

    return dict(trajectories_by_map)


def compute_statistics(trajectories_by_map: Dict) -> Dict:
    """Compute statistics about extracted trajectories."""

    stats = {
        'total_rounds': 0,
        'total_position_samples': 0,
        'avg_positions_per_round': 0,
        'avg_positions_per_player_per_round': 0,
        'maps': {}
    }

    total_positions = 0
    total_player_positions = 0
    total_players = 0

    for map_name, rounds in trajectories_by_map.items():
        map_positions = 0
        map_player_positions = 0
        map_players = 0

        for round_data in rounds:
            for player, trajectory in round_data['player_trajectories'].items():
                map_positions += len(trajectory)
                map_player_positions += len(trajectory)
                map_players += 1

        stats['maps'][map_name] = {
            'rounds': len(rounds),
            'total_positions': map_positions,
            'avg_positions_per_round': map_positions / len(rounds) if rounds else 0
        }

        stats['total_rounds'] += len(rounds)
        total_positions += map_positions
        total_player_positions += map_player_positions
        total_players += map_players

    stats['total_position_samples'] = total_positions
    stats['avg_positions_per_round'] = total_positions / stats['total_rounds'] if stats['total_rounds'] else 0
    stats['avg_positions_per_player_per_round'] = total_player_positions / total_players if total_players else 0

    return stats


def main():
    print("=" * 60)
    print("EXTRACTING PLAYER TRAJECTORIES FROM VCT DATA")
    print("=" * 60)

    jsonl_files = sorted([f for f in os.listdir(GRID_DATA_DIR) if f.endswith('.jsonl')])
    print(f"\nFound {len(jsonl_files)} JSONL files")

    all_rounds = []

    for i, fname in enumerate(jsonl_files):
        fpath = GRID_DATA_DIR / fname
        print(f"Processing {i+1}/{len(jsonl_files)}: {fname[:30]}...", end=" ")

        rounds = extract_trajectories_from_file(fpath)
        all_rounds.extend(rounds)
        print(f"{len(rounds)} rounds")

    print(f"\nTotal rounds extracted: {len(all_rounds)}")

    # Build trajectories
    print("\nBuilding trajectories...")
    trajectories = build_trajectories(all_rounds)

    # Compute statistics
    stats = compute_statistics(trajectories)

    # Save output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save full trajectories (large file)
    output_file = OUTPUT_DIR / "position_trajectories.json"
    with open(output_file, 'w') as f:
        json.dump({
            'metadata': {
                'description': 'Player movement trajectories from VCT pro matches',
                'source': 'GRID VCT JSONL',
                'statistics': stats
            },
            'trajectories_by_map': trajectories
        }, f)  # No indent to save space

    file_size = output_file.stat().st_size / (1024 * 1024)
    print(f"\nSaved to {output_file} ({file_size:.1f} MB)")

    # Print statistics
    print("\n" + "=" * 60)
    print("TRAJECTORY STATISTICS")
    print("=" * 60)
    print(f"\nTotal rounds: {stats['total_rounds']}")
    print(f"Total position samples: {stats['total_position_samples']:,}")
    print(f"Avg positions per round: {stats['avg_positions_per_round']:.1f}")
    print(f"Avg positions per player per round: {stats['avg_positions_per_player_per_round']:.1f}")

    print("\n=== BY MAP ===")
    for map_name, map_stats in sorted(stats['maps'].items()):
        print(f"{map_name:12} | {map_stats['rounds']:4} rounds | {map_stats['total_positions']:6} positions | {map_stats['avg_positions_per_round']:.1f} avg/round")


if __name__ == "__main__":
    main()
