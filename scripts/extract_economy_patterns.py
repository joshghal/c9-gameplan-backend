#!/usr/bin/env python3
"""
Extract economy patterns from VCT JSONL data.

Analyzes buy decisions to understand:
- When teams eco/force/full buy
- Loadout choices by economy
- Economy thresholds
- Win rate by economy state

Usage:
    python scripts/extract_economy_patterns.py
"""

import json
import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

PROJECT_ROOT = Path(__file__).parent.parent.parent
GRID_DATA_DIR = PROJECT_ROOT / "grid_data"
OUTPUT_DIR = PROJECT_ROOT / "backend" / "app" / "data"


def classify_buy(loadout_value: int, team_money: int) -> str:
    """Classify buy type based on loadout value."""
    if loadout_value >= 3900:
        return 'full_buy'
    elif loadout_value >= 2000:
        return 'force_buy'
    elif loadout_value >= 1000:
        return 'half_buy'
    else:
        return 'eco'


def extract_economy_from_file(fpath: Path) -> List[Dict]:
    """Extract economy data from a JSONL file."""

    rounds = []
    current_map = None
    current_game_id = None

    with open(fpath, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)

                for event in data.get('events', []):
                    etype = event.get('type', '')

                    # Track map
                    if etype == 'series-started-game':
                        target = event.get('target', {})
                        state = target.get('state', {})
                        map_data = state.get('map', {})
                        current_map = map_data.get('name', '').lower()
                        current_game_id = state.get('id')

                    # Get economy at round start
                    if etype == 'round-ended-freezetime':
                        series = event.get('seriesState', {})
                        games = series.get('games', [])

                        if games:
                            game = games[0]
                            round_data = {
                                'map': current_map,
                                'game_id': current_game_id,
                                'teams': []
                            }

                            for team in game.get('teams', []):
                                team_money = team.get('money', 0)
                                team_loadout = team.get('loadoutValue', 0)
                                side = team.get('side', '')

                                players_data = []
                                for player in team.get('players', []):
                                    players_data.append({
                                        'name': player.get('name'),
                                        'money': player.get('money', 0),
                                        'loadout': player.get('loadoutValue', 0),
                                    })

                                round_data['teams'].append({
                                    'name': team.get('name'),
                                    'side': side,
                                    'money': team_money,
                                    'loadout': team_loadout,
                                    'buy_type': classify_buy(team_loadout, team_money),
                                    'players': players_data
                                })

                            rounds.append(round_data)

                    # Get round outcome
                    if etype == 'team-won-round':
                        actor = event.get('actor', {})
                        winner_name = actor.get('state', {}).get('name')

                        # Attach to previous round
                        if rounds:
                            rounds[-1]['winner'] = winner_name

            except json.JSONDecodeError:
                continue
            except Exception:
                continue

    return rounds


def analyze_buy_patterns(rounds: List[Dict]) -> Dict:
    """Analyze buy patterns and correlate with outcomes."""

    buy_outcomes = defaultdict(lambda: {'wins': 0, 'total': 0})
    loadout_distribution = defaultdict(int)
    eco_thresholds = []

    for round_data in rounds:
        winner = round_data.get('winner')

        for team in round_data.get('teams', []):
            buy_type = team['buy_type']
            won = team['name'] == winner

            buy_outcomes[buy_type]['total'] += 1
            if won:
                buy_outcomes[buy_type]['wins'] += 1

            loadout_distribution[buy_type] += 1

            # Track eco threshold decisions
            if buy_type == 'eco':
                eco_thresholds.append({
                    'money': team['money'],
                    'loadout': team['loadout']
                })

    # Calculate win rates
    buy_win_rates = {}
    for buy_type, data in buy_outcomes.items():
        win_rate = data['wins'] / data['total'] if data['total'] > 0 else 0
        buy_win_rates[buy_type] = {
            'win_rate': win_rate,
            'wins': data['wins'],
            'total': data['total']
        }

    return {
        'buy_win_rates': buy_win_rates,
        'loadout_distribution': dict(loadout_distribution),
        'eco_threshold_avg': sum(e['money'] for e in eco_thresholds) / len(eco_thresholds) if eco_thresholds else 0
    }


def analyze_by_side(rounds: List[Dict]) -> Dict:
    """Analyze economy patterns by attacker/defender side."""

    side_buys = defaultdict(lambda: defaultdict(int))

    for round_data in rounds:
        for team in round_data.get('teams', []):
            side = team['side']
            buy_type = team['buy_type']
            side_buys[side][buy_type] += 1

    return dict(side_buys)


def analyze_by_map(rounds: List[Dict]) -> Dict:
    """Analyze economy patterns by map."""

    map_economy = defaultdict(lambda: {
        'avg_attacker_loadout': [],
        'avg_defender_loadout': [],
        'buy_types': defaultdict(int)
    })

    for round_data in rounds:
        map_name = round_data.get('map')
        if not map_name:
            continue

        for team in round_data.get('teams', []):
            side = team['side']
            loadout = team['loadout']
            buy_type = team['buy_type']

            if side == 'attacker':
                map_economy[map_name]['avg_attacker_loadout'].append(loadout)
            else:
                map_economy[map_name]['avg_defender_loadout'].append(loadout)

            map_economy[map_name]['buy_types'][buy_type] += 1

    result = {}
    for map_name, data in map_economy.items():
        result[map_name] = {
            'avg_attacker_loadout': sum(data['avg_attacker_loadout']) / len(data['avg_attacker_loadout']) if data['avg_attacker_loadout'] else 0,
            'avg_defender_loadout': sum(data['avg_defender_loadout']) / len(data['avg_defender_loadout']) if data['avg_defender_loadout'] else 0,
            'buy_distribution': dict(data['buy_types'])
        }

    return result


def analyze_player_economy(rounds: List[Dict]) -> Dict:
    """Analyze per-player economy patterns."""

    player_loadouts = defaultdict(list)

    for round_data in rounds:
        for team in round_data.get('teams', []):
            for player in team.get('players', []):
                name = player.get('name')
                if name:
                    player_loadouts[name].append(player['loadout'])

    result = {}
    for name, loadouts in player_loadouts.items():
        if len(loadouts) >= 10:  # Minimum sample size
            result[name] = {
                'avg_loadout': sum(loadouts) / len(loadouts),
                'rounds_played': len(loadouts)
            }

    return result


def main():
    print("=" * 60)
    print("EXTRACTING ECONOMY PATTERNS FROM VCT DATA")
    print("=" * 60)

    jsonl_files = sorted([f for f in os.listdir(GRID_DATA_DIR) if f.endswith('.jsonl')])
    print(f"\nFound {len(jsonl_files)} JSONL files")

    all_rounds = []

    for i, fname in enumerate(jsonl_files):
        fpath = GRID_DATA_DIR / fname
        print(f"Processing {i+1}/{len(jsonl_files)}: {fname[:30]}...", end=" ")

        rounds = extract_economy_from_file(fpath)
        all_rounds.extend(rounds)
        print(f"{len(rounds)} rounds")

    print(f"\nTotal rounds: {len(all_rounds)}")

    # Analyze patterns
    print("\nAnalyzing buy patterns...")
    buy_patterns = analyze_buy_patterns(all_rounds)

    print("Analyzing by side...")
    side_patterns = analyze_by_side(all_rounds)

    print("Analyzing by map...")
    map_patterns = analyze_by_map(all_rounds)

    print("Analyzing player economy...")
    player_patterns = analyze_player_economy(all_rounds)

    # Compile output
    output_data = {
        'metadata': {
            'description': 'Economy patterns from VCT pro matches',
            'total_rounds': len(all_rounds),
            'thresholds': {
                'full_buy': 3900,
                'force_buy': 2000,
                'half_buy': 1000,
                'eco': 0
            }
        },
        'buy_patterns': buy_patterns,
        'by_side': side_patterns,
        'by_map': map_patterns,
        'player_economy': player_patterns
    }

    # Save output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "economy_patterns.json"

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved to {output_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("ECONOMY PATTERN ANALYSIS")
    print("=" * 60)

    print(f"\n=== BUY TYPE WIN RATES ===")
    for buy_type, data in sorted(buy_patterns['buy_win_rates'].items()):
        print(f"{buy_type:12} | Win rate: {data['win_rate']*100:5.1f}% | {data['wins']:4}/{data['total']:4}")

    print(f"\n=== BUY DISTRIBUTION ===")
    total = sum(buy_patterns['loadout_distribution'].values())
    for buy_type, count in sorted(buy_patterns['loadout_distribution'].items()):
        print(f"{buy_type:12} | {count:4} rounds ({100*count/total:.1f}%)")

    print(f"\n=== BY SIDE ===")
    for side, buys in side_patterns.items():
        print(f"\n{side}:")
        for buy_type, count in sorted(buys.items()):
            print(f"  {buy_type:12} | {count:4}")

    print(f"\n=== BY MAP (avg loadout) ===")
    for map_name, data in sorted(map_patterns.items()):
        print(f"{map_name:12} | ATK: {data['avg_attacker_loadout']:,.0f} | DEF: {data['avg_defender_loadout']:,.0f}")


if __name__ == "__main__":
    main()
