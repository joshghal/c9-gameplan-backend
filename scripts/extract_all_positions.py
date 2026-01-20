#!/usr/bin/env python3
"""
Extract ALL player positions from GRID JSONL files to learn map boundaries.

Creates dense position plots for each map showing exact walkable areas.
"""

import json
import os
import sys
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / 'app'))
sys.path.insert(0, str(Path(__file__).parent))

# Map transforms from cloud9-webapp
MAP_TRANSFORMS = {
    'ascent': {'xMult': 7e-05, 'yMult': -7e-05, 'xAdd': 0.813895, 'yAdd': 0.573242},
    'bind': {'xMult': 5.9e-05, 'yMult': -5.9e-05, 'xAdd': 0.576941, 'yAdd': 0.967566},
    'haven': {'xMult': 7.5e-05, 'yMult': -7.5e-05, 'xAdd': 1.09345, 'yAdd': 0.642728},
    'split': {'xMult': 7.8e-05, 'yMult': -7.8e-05, 'xAdd': 0.842188, 'yAdd': 0.697578},
    'icebox': {'xMult': 7.2e-05, 'yMult': -7.2e-05, 'xAdd': 0.460214, 'yAdd': 0.304687},
    'breeze': {'xMult': 7e-05, 'yMult': -7e-05, 'xAdd': 0.465123, 'yAdd': 0.833078},
    'fracture': {'xMult': 7.8e-05, 'yMult': -7.8e-05, 'xAdd': 0.556952, 'yAdd': 1.155886},
    'pearl': {'xMult': 7.8e-05, 'yMult': -7.8e-05, 'xAdd': 0.480469, 'yAdd': 0.916016},
    'lotus': {'xMult': 7.2e-05, 'yMult': -7.2e-05, 'xAdd': 0.454789, 'yAdd': 0.917752},
    'sunset': {'xMult': 7.8e-05, 'yMult': -7.8e-05, 'xAdd': 0.5, 'yAdd': 0.515625},
    'abyss': {'xMult': 8.1e-05, 'yMult': -8.1e-05, 'xAdd': 0.5, 'yAdd': 0.5},
}


def game_to_minimap(game_x: float, game_y: float, map_name: str) -> tuple:
    """Convert game coordinates to minimap normalized (0-1) coordinates."""
    t = MAP_TRANSFORMS.get(map_name.lower(), MAP_TRANSFORMS['ascent'])
    minimap_x = game_y * t['xMult'] + t['xAdd']
    minimap_y = game_x * t['yMult'] + t['yAdd']
    return (minimap_x, minimap_y)


def extract_positions_from_file(filepath: Path) -> dict:
    """Extract all player positions from a JSONL file using correct structure."""
    positions_by_map = defaultdict(lambda: {'positions': [], 'game_coords': [], 'bounds': None})

    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f):
            try:
                data = json.loads(line.strip())
                events = data.get('events', [])

                for event in events:
                    # Get map info from actor.state.map or seriesState.games[0].map
                    map_info = None
                    map_name = None

                    # Try actor.state.map
                    actor = event.get('actor', {})
                    actor_state = actor.get('state', {})
                    if 'map' in actor_state:
                        map_info = actor_state['map']
                        map_name = map_info.get('name', '').lower()

                    # Try seriesState.games[0].map
                    if not map_name:
                        series_state = event.get('seriesState', {})
                        games = series_state.get('games', [])
                        if games and 'map' in games[0]:
                            map_info = games[0]['map']
                            map_name = map_info.get('name', '').lower()

                    if not map_name:
                        continue

                    # Store map bounds if available
                    if map_info and 'bounds' in map_info and positions_by_map[map_name]['bounds'] is None:
                        positions_by_map[map_name]['bounds'] = map_info['bounds']

                    # Extract positions from teams
                    teams = actor_state.get('teams', [])
                    if not teams:
                        series_state = event.get('seriesState', {})
                        games = series_state.get('games', [])
                        if games:
                            teams = games[0].get('teams', [])

                    for team in teams:
                        players = team.get('players', [])
                        team_side = team.get('side', '')

                        # Normalize side names (GRID uses "attacker"/"defender", we use "attack"/"defense")
                        if team_side == 'attacker':
                            team_side = 'attack'
                        elif team_side == 'defender':
                            team_side = 'defense'

                        for player in players:
                            pos = player.get('position')
                            if pos and 'x' in pos and 'y' in pos:
                                game_x = pos['x']
                                game_y = pos['y']

                                # Store game coordinates
                                positions_by_map[map_name]['game_coords'].append((game_x, game_y, team_side))

                                # Convert to minimap
                                minimap_x, minimap_y = game_to_minimap(game_x, game_y, map_name)
                                positions_by_map[map_name]['positions'].append((minimap_x, minimap_y, team_side))

            except json.JSONDecodeError:
                continue
            except Exception as e:
                continue

    return dict(positions_by_map)


def main():
    grid_dir = Path(__file__).parent.parent.parent / 'grid_data'
    output_dir = Path(__file__).parent.parent.parent / 'vct_map_data'
    output_dir.mkdir(exist_ok=True)

    # Collect all positions from all files
    all_positions = defaultdict(lambda: {'positions': [], 'game_coords': [], 'bounds': None})

    jsonl_files = list(grid_dir.glob('*.jsonl'))
    print(f"Processing {len(jsonl_files)} JSONL files...")

    for i, filepath in enumerate(jsonl_files):
        if (i + 1) % 5 == 0:
            print(f"  Processed {i + 1}/{len(jsonl_files)} files...")

        file_positions = extract_positions_from_file(filepath)

        for map_name, data in file_positions.items():
            all_positions[map_name]['positions'].extend(data['positions'])
            all_positions[map_name]['game_coords'].extend(data['game_coords'])
            if data['bounds'] and not all_positions[map_name]['bounds']:
                all_positions[map_name]['bounds'] = data['bounds']

    # Print summary and create plots for each map
    print("\n=== Position Extraction Summary ===")

    map_boundaries = {}

    for map_name, data in sorted(all_positions.items()):
        positions = data['positions']
        game_coords = data['game_coords']
        bounds = data['bounds']
        total = len(positions)

        if total < 100:
            print(f"  {map_name}: {total} positions (skipping - too few)")
            continue

        # Separate by side
        attack_pos = [(x, y) for x, y, side in positions if side == 'attack']
        defense_pos = [(x, y) for x, y, side in positions if side == 'defense']

        print(f"\n  {map_name.upper()}: {total:,} positions (ATK: {len(attack_pos):,}, DEF: {len(defense_pos):,})")

        if bounds:
            print(f"    Official bounds: x=[{bounds['min']['x']:.0f}, {bounds['max']['x']:.0f}], y=[{bounds['min']['y']:.0f}, {bounds['max']['y']:.0f}]")

        # Calculate minimap boundaries
        all_minimap = positions
        if all_minimap:
            x_coords = [p[0] for p in all_minimap]
            y_coords = [p[1] for p in all_minimap]

            # Filter to valid range with margin
            valid_x = [x for x in x_coords if -0.1 < x < 1.1]
            valid_y = [y for y in y_coords if -0.1 < y < 1.1]

            if valid_x and valid_y:
                x_min, x_max = np.percentile(valid_x, [1, 99])
                y_min, y_max = np.percentile(valid_y, [1, 99])

                map_boundaries[map_name] = {
                    'x_min': float(x_min),
                    'x_max': float(x_max),
                    'y_min': float(y_min),
                    'y_max': float(y_max),
                    'total_positions': total,
                }

                if bounds:
                    map_boundaries[map_name]['game_bounds'] = bounds

                print(f"    Minimap bounds: x=[{x_min:.3f}, {x_max:.3f}], y=[{y_min:.3f}, {y_max:.3f}]")

        # Calculate game coordinate stats
        if game_coords:
            gx = [p[0] for p in game_coords]
            gy = [p[1] for p in game_coords]
            gx_min, gx_max = np.percentile(gx, [1, 99])
            gy_min, gy_max = np.percentile(gy, [1, 99])
            print(f"    Actual game range: x=[{gx_min:.0f}, {gx_max:.0f}], y=[{gy_min:.0f}, {gy_max:.0f}]")

        # Create dense position plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'VCT Position Density - {map_name.upper()} ({total:,} positions)',
                     fontsize=14, color='white')

        # Plot 1: Attack positions
        if attack_pos:
            axes[0].scatter([p[0] for p in attack_pos], [p[1] for p in attack_pos],
                           c='cyan', alpha=0.2, s=1)
        axes[0].set_title(f'Attackers ({len(attack_pos):,})', color='cyan')
        axes[0].set_xlim(0, 1)
        axes[0].set_ylim(1, 0)
        axes[0].set_facecolor('#1a1a1a')
        axes[0].grid(True, alpha=0.2)

        # Plot 2: Defense positions
        if defense_pos:
            axes[1].scatter([p[0] for p in defense_pos], [p[1] for p in defense_pos],
                           c='red', alpha=0.2, s=1)
        axes[1].set_title(f'Defenders ({len(defense_pos):,})', color='red')
        axes[1].set_xlim(0, 1)
        axes[1].set_ylim(1, 0)
        axes[1].set_facecolor('#1a1a1a')
        axes[1].grid(True, alpha=0.2)

        # Plot 3: Heatmap / density
        valid_pos = [(x, y) for x, y, _ in positions if 0 <= x <= 1 and 0 <= y <= 1]
        if valid_pos:
            x_valid = [p[0] for p in valid_pos]
            y_valid = [p[1] for p in valid_pos]

            # Create 2D histogram
            heatmap, xedges, yedges = np.histogram2d(x_valid, y_valid, bins=100, range=[[0, 1], [0, 1]])
            extent = [xedges[0], xedges[-1], yedges[-1], yedges[0]]

            axes[2].imshow(heatmap.T, extent=extent, origin='upper', cmap='hot', aspect='auto')
            axes[2].set_title('Heatmap (Walkable Areas)', color='white')
        axes[2].set_xlim(0, 1)
        axes[2].set_ylim(1, 0)
        axes[2].set_facecolor('#1a1a1a')

        plt.tight_layout()
        output_path = output_dir / f'vct_dense_{map_name}.png'
        plt.savefig(str(output_path), dpi=150, bbox_inches='tight', facecolor='#1a1a1a')
        plt.close()
        print(f"    Saved: {output_path.name}")

    # Save boundaries to JSON
    boundaries_path = output_dir / 'map_boundaries.json'
    with open(boundaries_path, 'w') as f:
        json.dump(map_boundaries, f, indent=2)
    print(f"\n\nSaved map boundaries to: {boundaries_path}")

    # Print Python dict format for simulation
    print("\n=== MAP WALKABLE BOUNDARIES (for simulation) ===")
    print("MAP_WALKABLE_BOUNDS = {")
    for map_name, bounds in sorted(map_boundaries.items()):
        print(f"    '{map_name}': {{'x': ({bounds['x_min']:.3f}, {bounds['x_max']:.3f}), 'y': ({bounds['y_min']:.3f}, {bounds['y_max']:.3f})}},")
    print("}")


if __name__ == '__main__':
    main()
