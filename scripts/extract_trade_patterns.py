#!/usr/bin/env python3
"""
Extract trade and damage patterns from VCT JSONL data.

Analyzes engagement sequences to understand:
- Time-to-kill patterns
- Trade timing (how fast teammates respond to deaths)
- Multi-hit engagements
- Damage per engagement
- Distance vs damage relationships

Usage:
    python scripts/extract_trade_patterns.py
"""

import json
import math
import os
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).parent.parent.parent
GRID_DATA_DIR = PROJECT_ROOT / "grid_data"
OUTPUT_DIR = PROJECT_ROOT / "backend" / "app" / "data"


def parse_timestamp(ts: str) -> float:
    """Parse ISO timestamp to seconds."""
    try:
        dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
        return dt.timestamp()
    except:
        return 0


def calculate_distance(x1, y1, x2, y2) -> float:
    """Calculate distance between two points."""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def extract_engagements_from_file(fpath: Path) -> List[Dict]:
    """Extract all damage/kill events from a JSONL file."""

    engagements = []
    current_map = None
    current_round = 0
    current_game_id = None

    with open(fpath, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                timestamp = parse_timestamp(data.get('occurredAt', ''))

                for event in data.get('events', []):
                    etype = event.get('type', '')

                    # Track map
                    if etype == 'series-started-game':
                        target = event.get('target', {})
                        state = target.get('state', {})
                        map_data = state.get('map', {})
                        current_map = map_data.get('name', '').lower()
                        current_game_id = state.get('id')

                    # Track round
                    if etype == 'game-started-round':
                        actor = event.get('actor', {})
                        state_delta = actor.get('stateDelta', {})
                        segments = state_delta.get('segments', [])
                        if segments:
                            current_round = segments[0].get('sequenceNumber', 0)

                    # Extract damage events
                    if etype == 'player-damaged-player':
                        actor = event.get('actor', {})
                        target = event.get('target', {})

                        attacker_state = actor.get('state', {})
                        victim_state = target.get('state', {})
                        victim_delta = target.get('stateDelta', {}).get('game', {})

                        attacker_pos = attacker_state.get('game', {}).get('position')
                        victim_pos = victim_state.get('game', {}).get('position')

                        if attacker_pos and victim_pos:
                            distance = calculate_distance(
                                attacker_pos['x'], attacker_pos['y'],
                                victim_pos['x'], victim_pos['y']
                            )

                            engagements.append({
                                'type': 'damage',
                                'timestamp': timestamp,
                                'map': current_map,
                                'round': current_round,
                                'game_id': current_game_id,
                                'attacker': attacker_state.get('name'),
                                'attacker_team': attacker_state.get('teamId'),
                                'victim': victim_state.get('name'),
                                'victim_team': victim_state.get('teamId'),
                                'damage': victim_delta.get('damageTaken', 0),
                                'distance': distance,
                                'attacker_pos': attacker_pos,
                                'victim_pos': victim_pos,
                            })

                    # Extract kill events
                    if etype == 'player-killed-player':
                        actor = event.get('actor', {})
                        target = event.get('target', {})

                        attacker_state = actor.get('state', {})
                        victim_state = target.get('state', {})

                        attacker_pos = attacker_state.get('game', {}).get('position')
                        victim_pos = victim_state.get('game', {}).get('position')

                        if attacker_pos and victim_pos:
                            distance = calculate_distance(
                                attacker_pos['x'], attacker_pos['y'],
                                victim_pos['x'], victim_pos['y']
                            )

                            engagements.append({
                                'type': 'kill',
                                'timestamp': timestamp,
                                'map': current_map,
                                'round': current_round,
                                'game_id': current_game_id,
                                'attacker': attacker_state.get('name'),
                                'attacker_team': attacker_state.get('teamId'),
                                'victim': victim_state.get('name'),
                                'victim_team': victim_state.get('teamId'),
                                'distance': distance,
                                'attacker_pos': attacker_pos,
                                'victim_pos': victim_pos,
                            })

            except json.JSONDecodeError:
                continue
            except Exception:
                continue

    return engagements


def analyze_trades(engagements: List[Dict]) -> Dict:
    """Analyze trade patterns from engagements."""

    # Group by game and round
    rounds = defaultdict(list)
    for eng in engagements:
        key = (eng['game_id'], eng['round'])
        rounds[key].append(eng)

    trades = []
    trade_times = []

    for (game_id, round_num), events in rounds.items():
        # Sort by timestamp
        kills = sorted([e for e in events if e['type'] == 'kill'], key=lambda x: x['timestamp'])

        # Find trades (kill followed by kill of the killer within 3 seconds)
        for i, kill in enumerate(kills):
            for j in range(i + 1, len(kills)):
                next_kill = kills[j]
                time_diff = next_kill['timestamp'] - kill['timestamp']

                # Trade window: 0-5 seconds
                if time_diff > 5:
                    break

                # Check if the original killer was killed
                if next_kill['victim'] == kill['attacker']:
                    trades.append({
                        'original_kill': kill,
                        'trade_kill': next_kill,
                        'trade_time': time_diff,
                        'map': kill['map'],
                    })
                    trade_times.append(time_diff)
                    break

    return {
        'total_trades': len(trades),
        'avg_trade_time': sum(trade_times) / len(trade_times) if trade_times else 0,
        'trade_time_distribution': {
            'under_1s': sum(1 for t in trade_times if t < 1),
            '1_to_2s': sum(1 for t in trade_times if 1 <= t < 2),
            '2_to_3s': sum(1 for t in trade_times if 2 <= t < 3),
            '3_to_5s': sum(1 for t in trade_times if 3 <= t <= 5),
        },
        'sample_trades': trades[:100]  # Keep sample for analysis
    }


def analyze_time_to_kill(engagements: List[Dict]) -> Dict:
    """Analyze time-to-kill patterns."""

    # Group damage events by victim to reconstruct TTK
    victim_damage = defaultdict(list)

    for eng in engagements:
        if eng['type'] == 'damage':
            key = (eng['game_id'], eng['round'], eng['victim'])
            victim_damage[key].append(eng)

    ttk_data = []

    for (game_id, round_num, victim), damages in victim_damage.items():
        # Sort by timestamp
        damages = sorted(damages, key=lambda x: x['timestamp'])

        if len(damages) >= 2:
            first_hit = damages[0]['timestamp']
            last_hit = damages[-1]['timestamp']
            ttk = last_hit - first_hit

            if 0 < ttk < 10:  # Reasonable TTK range
                total_damage = sum(d['damage'] for d in damages)
                ttk_data.append({
                    'ttk': ttk,
                    'hits': len(damages),
                    'total_damage': total_damage,
                    'avg_distance': sum(d['distance'] for d in damages) / len(damages),
                    'map': damages[0]['map'],
                })

    return {
        'avg_ttk': sum(t['ttk'] for t in ttk_data) / len(ttk_data) if ttk_data else 0,
        'avg_hits_to_kill': sum(t['hits'] for t in ttk_data) / len(ttk_data) if ttk_data else 0,
        'ttk_distribution': {
            'under_0.5s': sum(1 for t in ttk_data if t['ttk'] < 0.5),
            '0.5_to_1s': sum(1 for t in ttk_data if 0.5 <= t['ttk'] < 1),
            '1_to_2s': sum(1 for t in ttk_data if 1 <= t['ttk'] < 2),
            'over_2s': sum(1 for t in ttk_data if t['ttk'] >= 2),
        },
        'samples': len(ttk_data)
    }


def analyze_distance_damage(engagements: List[Dict]) -> Dict:
    """Analyze damage vs distance relationships."""

    damage_events = [e for e in engagements if e['type'] == 'damage' and e['damage']]

    distance_buckets = defaultdict(list)

    for eng in damage_events:
        distance = eng['distance']
        damage = eng['damage']

        # Bucket by distance
        if distance < 500:
            bucket = 'close_0_500'
        elif distance < 1000:
            bucket = 'short_500_1000'
        elif distance < 2000:
            bucket = 'medium_1000_2000'
        elif distance < 3000:
            bucket = 'long_2000_3000'
        else:
            bucket = 'very_long_3000+'

        distance_buckets[bucket].append(damage)

    result = {}
    for bucket, damages in distance_buckets.items():
        result[bucket] = {
            'avg_damage': sum(damages) / len(damages) if damages else 0,
            'samples': len(damages),
            'damage_range': [min(damages), max(damages)] if damages else [0, 0]
        }

    return result


def analyze_by_map(engagements: List[Dict]) -> Dict:
    """Analyze patterns by map."""

    map_stats = defaultdict(lambda: {
        'kills': 0,
        'damage_events': 0,
        'total_damage': 0,
        'distances': []
    })

    for eng in engagements:
        map_name = eng['map']
        if not map_name:
            continue

        if eng['type'] == 'kill':
            map_stats[map_name]['kills'] += 1
            map_stats[map_name]['distances'].append(eng['distance'])
        elif eng['type'] == 'damage':
            map_stats[map_name]['damage_events'] += 1
            map_stats[map_name]['total_damage'] += eng.get('damage', 0)
            map_stats[map_name]['distances'].append(eng['distance'])

    result = {}
    for map_name, stats in map_stats.items():
        distances = stats['distances']
        result[map_name] = {
            'kills': stats['kills'],
            'damage_events': stats['damage_events'],
            'avg_engagement_distance': sum(distances) / len(distances) if distances else 0,
            'avg_damage_per_event': stats['total_damage'] / stats['damage_events'] if stats['damage_events'] else 0
        }

    return result


def main():
    print("=" * 60)
    print("EXTRACTING TRADE AND DAMAGE PATTERNS FROM VCT DATA")
    print("=" * 60)

    jsonl_files = sorted([f for f in os.listdir(GRID_DATA_DIR) if f.endswith('.jsonl')])
    print(f"\nFound {len(jsonl_files)} JSONL files")

    all_engagements = []

    for i, fname in enumerate(jsonl_files):
        fpath = GRID_DATA_DIR / fname
        print(f"Processing {i+1}/{len(jsonl_files)}: {fname[:30]}...", end=" ")

        engagements = extract_engagements_from_file(fpath)
        all_engagements.extend(engagements)
        print(f"{len(engagements)} events")

    print(f"\nTotal engagements: {len(all_engagements)}")

    # Separate by type
    kills = [e for e in all_engagements if e['type'] == 'kill']
    damages = [e for e in all_engagements if e['type'] == 'damage']

    print(f"  Kill events: {len(kills)}")
    print(f"  Damage events: {len(damages)}")

    # Analyze patterns
    print("\nAnalyzing trade patterns...")
    trade_analysis = analyze_trades(all_engagements)

    print("Analyzing time-to-kill...")
    ttk_analysis = analyze_time_to_kill(all_engagements)

    print("Analyzing distance vs damage...")
    distance_analysis = analyze_distance_damage(all_engagements)

    print("Analyzing by map...")
    map_analysis = analyze_by_map(all_engagements)

    # Compile output
    output_data = {
        'metadata': {
            'description': 'Trade and damage patterns from VCT pro matches',
            'total_kills': len(kills),
            'total_damage_events': len(damages),
        },
        'trade_patterns': trade_analysis,
        'time_to_kill': ttk_analysis,
        'distance_damage': distance_analysis,
        'by_map': map_analysis
    }

    # Save output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "trade_patterns.json"

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)

    print(f"\nSaved to {output_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("TRADE PATTERN ANALYSIS")
    print("=" * 60)

    print(f"\n=== TRADE TIMING ===")
    print(f"Total trades detected: {trade_analysis['total_trades']}")
    print(f"Average trade time: {trade_analysis['avg_trade_time']:.2f}s")
    print(f"Trade time distribution:")
    for bucket, count in trade_analysis['trade_time_distribution'].items():
        print(f"  {bucket}: {count}")

    print(f"\n=== TIME TO KILL ===")
    print(f"Average TTK: {ttk_analysis['avg_ttk']:.2f}s")
    print(f"Average hits to kill: {ttk_analysis['avg_hits_to_kill']:.1f}")
    print(f"TTK distribution:")
    for bucket, count in ttk_analysis['ttk_distribution'].items():
        print(f"  {bucket}: {count}")

    print(f"\n=== DISTANCE VS DAMAGE ===")
    for bucket, data in sorted(distance_analysis.items()):
        print(f"{bucket:20} | Avg damage: {data['avg_damage']:5.1f} | Samples: {data['samples']}")

    print(f"\n=== BY MAP ===")
    for map_name, data in sorted(map_analysis.items()):
        print(f"{map_name:12} | Kills: {data['kills']:4} | Avg distance: {data['avg_engagement_distance']:.0f}")


if __name__ == "__main__":
    main()
