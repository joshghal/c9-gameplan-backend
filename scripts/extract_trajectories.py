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
    round_kills = []
    round_end_clock = None
    round_winner_team = None

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

                    # Track round end clock and winner
                    if etype in ('game-ended-round', 'team-won-round'):
                        series_end = event.get('seriesState', {})
                        games_end = series_end.get('games', [])
                        if games_end:
                            end_clock = games_end[0].get('clock', {}).get('currentSeconds', 0)
                            if current_round is not None:
                                round_end_clock = end_clock
                    if etype == 'team-won-round' and current_round is not None:
                        round_winner_team = event.get('actor', {}).get('state', {}).get('name', '')

                    # Extract kill events
                    if etype == 'player-killed-player' and current_round is not None:
                        actor = event.get('actor', {})
                        target = event.get('target', {})
                        actor_state = actor.get('state', {})
                        target_state = target.get('state', {})
                        # Weapon from stateDelta round weaponKills
                        delta_round = actor.get('stateDelta', {}).get('round', {})
                        wk = delta_round.get('weaponKills', {})
                        weapon = list(wk.keys())[0] if wk else 'unknown'
                        headshot = delta_round.get('headshots', 0) > 0
                        # Clock
                        kill_series = event.get('seriesState', {})
                        kill_games = kill_series.get('games', [])
                        kill_clock = kill_games[0].get('clock', {}).get('currentSeconds', 0) if kill_games else 0
                        round_kills.append({
                            'clock': kill_clock,
                            'killer': actor_state.get('name', ''),
                            'victim': target_state.get('name', ''),
                            'weapon': weapon,
                            'headshot': headshot,
                        })

                    # New round started
                    if etype == 'game-started-round':
                        # Save previous round if exists
                        if current_round is not None and round_events:
                            entry = {
                                'round_num': current_round,
                                'map': current_map,
                                'game_id': current_game_id,
                                'events': round_events,
                                'kills': round_kills,
                            }
                            if round_end_clock is not None:
                                entry['round_end_clock'] = round_end_clock
                                entry['round_duration_s'] = 100 - round_end_clock
                            if round_winner_team:
                                entry['winner_team'] = round_winner_team
                            rounds.append(entry)

                        # Get round number
                        actor = event.get('actor', {})
                        state_delta = actor.get('stateDelta', {})
                        segments = state_delta.get('segments', [])
                        if segments:
                            current_round = segments[0].get('sequenceNumber', 0)

                        round_events = []
                        round_kills = []
                        round_end_clock = None
                        round_winner_team = None

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
        entry = {
            'round_num': current_round,
            'map': current_map,
            'game_id': current_game_id,
            'events': round_events,
            'kills': round_kills,
        }
        if round_end_clock is not None:
            entry['round_end_clock'] = round_end_clock
            entry['round_duration_s'] = 100 - round_end_clock
        if round_winner_team:
            entry['winner_team'] = round_winner_team
        rounds.append(entry)

    return rounds


def propagate_metadata(rounds: List[Dict]) -> List[Dict]:
    """Propagate team/side metadata within each round.

    GRID events often have empty team/side fields. But kill events and some
    state updates DO have them. For each round, we:
    1. Collect all known (player_name → team, side) mappings
    2. Propagate to every position sample in that round
    """
    for round_data in rounds:
        events = round_data.get('events', [])

        # Pass 1: collect known metadata per player
        player_meta: Dict[str, Dict[str, str]] = {}
        for event in events:
            for pname, pdata in event.get('positions', {}).items():
                team = pdata.get('team', '')
                side = pdata.get('side', '')
                if team and side and pname not in player_meta:
                    player_meta[pname] = {'team': team, 'side': side}

        # Pass 2: propagate to all samples
        if player_meta:
            for event in events:
                for pname, pdata in event.get('positions', {}).items():
                    if (not pdata.get('team') or not pdata.get('side')) and pname in player_meta:
                        pdata['team'] = player_meta[pname]['team']
                        pdata['side'] = player_meta[pname]['side']

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

        # Clean, filter post-round data, deduplicate
        round_end_clock = round_data.get('round_end_clock', 0) or 0

        for player_name, trajectory in player_trajectories.items():
            # Sort by clock descending (= time ascending)
            trajectory.sort(key=lambda p: -p['clock'])

            # Filter 1: Remove post-round samples (clock < round_end_clock).
            # For rounds where timer expired (round_end_clock=0), clock=0
            # is the exact end — strip it too as it often contains spawn resets.
            if round_end_clock > 0:
                trajectory = [p for p in trajectory if p['clock'] >= round_end_clock]
            else:
                trajectory = [p for p in trajectory if p['clock'] > 0]

            # Filter 2: Deduplicate same-clock samples.
            # GRID fires multiple events at the same clock second, each
            # reporting different (often contradictory) player positions.
            # Keep only the first sample per clock value.
            seen_clocks = set()
            clock_deduped = []
            for pt in trajectory:
                if pt['clock'] not in seen_clocks:
                    seen_clocks.add(pt['clock'])
                    clock_deduped.append(pt)
            trajectory = clock_deduped

            # Filter 3: Remove outlier teleports using max-speed heuristic.
            # GRID sometimes reports wrong positions (spectator cam, state
            # glitches) that appear as teleports.  We allow ~500 units/s
            # max movement.  If a point implies faster travel AND the
            # trajectory later returns near the last good point, the
            # intervening points are outliers.
            if len(trajectory) >= 3:
                import math
                MAX_SPEED = 500  # units per clock-second

                def _dist(a, b):
                    return math.sqrt((a['x'] - b['x'])**2 + (a['y'] - b['y'])**2)

                clean = [trajectory[0]]
                i = 1
                while i < len(trajectory):
                    prev = clean[-1]
                    curr = trajectory[i]
                    dt = abs(prev['clock'] - curr['clock']) or 1
                    d = _dist(prev, curr)
                    if d / dt > MAX_SPEED:
                        # Suspect outlier — scan ahead for a point that
                        # returns near prev (within MAX_SPEED * its dt)
                        recovered = False
                        for j in range(i + 1, min(i + 8, len(trajectory))):
                            fut = trajectory[j]
                            dt_fut = abs(prev['clock'] - fut['clock']) or 1
                            if _dist(prev, fut) / dt_fut <= MAX_SPEED:
                                # Skip points i..j-1 (outlier cluster)
                                i = j
                                recovered = True
                                break
                        if recovered:
                            continue
                    clean.append(curr)
                    i += 1
                trajectory = clean

            # Filter 4: Strip clock=0 if it teleports (post-round spawn reset)
            if len(trajectory) >= 2 and trajectory[-1]['clock'] == 0:
                d_last = math.sqrt(
                    (trajectory[-1]['x'] - trajectory[-2]['x'])**2 +
                    (trajectory[-1]['y'] - trajectory[-2]['y'])**2
                )
                dt_last = abs(trajectory[-2]['clock'] - 0) or 1
                if d_last / dt_last > MAX_SPEED:
                    trajectory = trajectory[:-1]

            # Filter 5: Truncate at death — stop including points after
            # the player dies. If they "resurrect" (alive=True after alive=False),
            # that's post-round buy phase data leaking in.
            cleaned = []
            for pt in trajectory:
                cleaned.append(pt)
                if not pt['alive']:
                    break  # death is the last valid point
            trajectory = cleaned

            # Deduplicate consecutive same positions
            deduped = []
            for point in trajectory:
                if not deduped or (point['x'] != deduped[-1]['x'] or point['y'] != deduped[-1]['y']):
                    deduped.append(point)

            player_trajectories[player_name] = deduped

        entry = {
            'round_num': round_num,
            'game_id': round_data['game_id'],
            'player_trajectories': dict(player_trajectories),
            'kills': round_data.get('kills', []),
        }
        if 'round_duration_s' in round_data:
            entry['round_duration_s'] = round_data['round_duration_s']
            entry['round_end_clock'] = round_data['round_end_clock']
        if 'winner_team' in round_data:
            entry['winner_team'] = round_data['winner_team']
        trajectories_by_map[map_name].append(entry)

    return dict(trajectories_by_map)


def compute_statistics(trajectories_by_map: Dict) -> Dict:
    """Compute statistics about extracted trajectories."""

    stats = {
        'total_rounds': 0,
        'total_position_samples': 0,
        'avg_positions_per_round': 0,
        'avg_positions_per_player_per_round': 0,
        'maps': {},
        'metadata_coverage': {}
    }

    total_positions = 0
    total_player_positions = 0
    total_players = 0
    total_with_meta = 0

    for map_name, rounds in trajectories_by_map.items():
        map_positions = 0
        map_player_positions = 0
        map_players = 0
        map_with_meta = 0

        for round_data in rounds:
            for player, trajectory in round_data['player_trajectories'].items():
                map_positions += len(trajectory)
                map_player_positions += len(trajectory)
                map_players += 1
                for pt in trajectory:
                    if pt.get('team') and pt.get('side'):
                        map_with_meta += 1

        stats['maps'][map_name] = {
            'rounds': len(rounds),
            'total_positions': map_positions,
            'avg_positions_per_round': map_positions / len(rounds) if rounds else 0
        }

        stats['total_rounds'] += len(rounds)
        total_positions += map_positions
        total_player_positions += map_player_positions
        total_players += map_players
        total_with_meta += map_with_meta

    stats['total_position_samples'] = total_positions
    stats['avg_positions_per_round'] = total_positions / stats['total_rounds'] if stats['total_rounds'] else 0
    stats['avg_positions_per_player_per_round'] = total_player_positions / total_players if total_players else 0
    stats['metadata_coverage'] = {
        'total_samples': total_positions,
        'with_team_side': total_with_meta,
        'coverage_pct': round(total_with_meta / total_positions * 100, 1) if total_positions else 0
    }

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

    # Propagate team/side metadata within each round
    print("\nPropagating team/side metadata...")
    all_rounds = propagate_metadata(all_rounds)

    # Build trajectories
    print("Building trajectories...")
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

    meta = stats['metadata_coverage']
    print(f"\nMetadata coverage: {meta['with_team_side']:,}/{meta['total_samples']:,} ({meta['coverage_pct']}%)")

    print("\n=== BY MAP ===")
    for map_name, map_stats in sorted(stats['maps'].items()):
        print(f"{map_name:12} | {map_stats['rounds']:4} rounds | {map_stats['total_positions']:6} positions | {map_stats['avg_positions_per_round']:.1f} avg/round")


if __name__ == "__main__":
    main()
