#!/usr/bin/env python3
"""
Combine VCT GRID data + Henrik API data to create comprehensive map boundaries.

This script:
1. Loads VCT positions from GRID JSONL files
2. Loads Henrik API positions from snapshots
3. Combines and converts to minimap coordinates
4. Outputs updated map_boundaries.json with more accurate walkable areas
"""

import json
import os
import sys
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# Map transforms from cloud9-webapp (game coords -> minimap 0-1 coords)
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
    'corrode': {'xMult': 7.8e-05, 'yMult': -7.8e-05, 'xAdd': 0.5, 'yAdd': 0.5},  # Estimate
}


def game_to_minimap(game_x: float, game_y: float, map_name: str) -> tuple:
    """Convert game coordinates to minimap normalized (0-1) coordinates."""
    t = MAP_TRANSFORMS.get(map_name.lower(), MAP_TRANSFORMS['ascent'])
    # VALORANT transform: minimap_x = game_y * xMult + xAdd
    minimap_x = game_y * t['xMult'] + t['xAdd']
    minimap_y = game_x * t['yMult'] + t['yAdd']
    return (minimap_x, minimap_y)


def load_vct_positions(grid_dir: Path) -> dict:
    """Load positions from VCT GRID JSONL files."""
    positions_by_map = defaultdict(list)

    jsonl_files = list(grid_dir.glob('*.jsonl'))
    print(f"Loading VCT data from {len(jsonl_files)} JSONL files...")

    for filepath in jsonl_files:
        with open(filepath, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    events = data.get('events', [])

                    for event in events:
                        # Get map name
                        map_name = None
                        actor = event.get('actor', {})
                        actor_state = actor.get('state', {})
                        if 'map' in actor_state:
                            map_name = actor_state['map'].get('name', '').lower()

                        if not map_name:
                            series_state = event.get('seriesState', {})
                            games = series_state.get('games', [])
                            if games and 'map' in games[0]:
                                map_name = games[0]['map'].get('name', '').lower()

                        if not map_name:
                            continue

                        # Extract player positions
                        teams = actor_state.get('teams', [])
                        if not teams:
                            series_state = event.get('seriesState', {})
                            games = series_state.get('games', [])
                            if games:
                                teams = games[0].get('teams', [])

                        for team in teams:
                            for player in team.get('players', []):
                                pos = player.get('position')
                                if pos and 'x' in pos and 'y' in pos:
                                    game_x = pos['x']
                                    game_y = pos['y']
                                    minimap_x, minimap_y = game_to_minimap(game_x, game_y, map_name)
                                    positions_by_map[map_name].append((minimap_x, minimap_y))

                except (json.JSONDecodeError, Exception):
                    continue

    return dict(positions_by_map)


def load_henrik_positions(snapshots_file: Path) -> dict:
    """Load positions from Henrik API snapshots."""
    positions_by_map = defaultdict(list)

    print(f"Loading Henrik API data from {snapshots_file.name}...")

    with open(snapshots_file, 'r') as f:
        data = json.load(f)

    print(f"  Found {len(data):,} kill snapshots")

    for snapshot in data:
        map_name = snapshot.get('map_name', '').lower()
        if not map_name or map_name == 'unknown':
            continue

        # Each snapshot has player_positions for all 10 players
        player_positions = snapshot.get('player_positions', [])
        for pos in player_positions:
            game_x = pos.get('x', 0)
            game_y = pos.get('y', 0)

            if game_x == 0 and game_y == 0:
                continue

            minimap_x, minimap_y = game_to_minimap(game_x, game_y, map_name)
            positions_by_map[map_name].append((minimap_x, minimap_y))

        # Also include victim death location
        victim_x = snapshot.get('victim_x', 0)
        victim_y = snapshot.get('victim_y', 0)
        if victim_x != 0 or victim_y != 0:
            minimap_x, minimap_y = game_to_minimap(victim_x, victim_y, map_name)
            positions_by_map[map_name].append((minimap_x, minimap_y))

    return dict(positions_by_map)


def main():
    project_dir = Path(__file__).parent.parent.parent
    grid_dir = project_dir / 'grid_data'
    henrik_file = project_dir / 'data' / 'raw' / 'all_maps_snapshots.json'
    output_dir = project_dir / 'vct_map_data'
    output_dir.mkdir(exist_ok=True)

    # Load positions from both sources
    vct_positions = load_vct_positions(grid_dir) if grid_dir.exists() else {}
    henrik_positions = load_henrik_positions(henrik_file) if henrik_file.exists() else {}

    # Combine positions
    combined_positions = defaultdict(list)

    for map_name, positions in vct_positions.items():
        combined_positions[map_name].extend(positions)

    for map_name, positions in henrik_positions.items():
        combined_positions[map_name].extend(positions)

    # Print summary
    print("\n" + "=" * 60)
    print("COMBINED POSITION DATA SUMMARY")
    print("=" * 60)
    print(f"{'Map':<12} {'VCT':>10} {'Henrik':>10} {'Total':>12}")
    print("-" * 44)

    total_vct = 0
    total_henrik = 0
    total_combined = 0

    for map_name in sorted(combined_positions.keys()):
        vct_count = len(vct_positions.get(map_name, []))
        henrik_count = len(henrik_positions.get(map_name, []))
        total_count = len(combined_positions[map_name])

        total_vct += vct_count
        total_henrik += henrik_count
        total_combined += total_count

        print(f"{map_name:<12} {vct_count:>10,} {henrik_count:>10,} {total_count:>12,}")

    print("-" * 44)
    print(f"{'TOTAL':<12} {total_vct:>10,} {total_henrik:>10,} {total_combined:>12,}")

    # Calculate boundaries and create plots
    print("\n" + "=" * 60)
    print("CALCULATING MAP BOUNDARIES")
    print("=" * 60)

    map_boundaries = {}

    for map_name, positions in sorted(combined_positions.items()):
        if len(positions) < 100:
            print(f"  {map_name}: {len(positions)} positions (skipping - too few)")
            continue

        # Filter to valid minimap range
        valid_positions = [(x, y) for x, y in positions if -0.1 < x < 1.1 and -0.1 < y < 1.1]

        if not valid_positions:
            continue

        x_coords = [p[0] for p in valid_positions]
        y_coords = [p[1] for p in valid_positions]

        # Use 1st and 99th percentile for robust boundaries
        x_min, x_max = np.percentile(x_coords, [1, 99])
        y_min, y_max = np.percentile(y_coords, [1, 99])

        map_boundaries[map_name] = {
            'x_min': float(x_min),
            'x_max': float(x_max),
            'y_min': float(y_min),
            'y_max': float(y_max),
            'total_positions': len(positions),
            'vct_positions': len(vct_positions.get(map_name, [])),
            'henrik_positions': len(henrik_positions.get(map_name, [])),
        }

        print(f"  {map_name:<12}: x=[{x_min:.3f}, {x_max:.3f}], y=[{y_min:.3f}, {y_max:.3f}] ({len(positions):,} pos)")

        # Create density plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        fig.suptitle(f'{map_name.upper()} - Combined Position Density ({len(positions):,} positions)',
                     fontsize=14, color='white')

        # Scatter plot
        axes[0].scatter(x_coords, y_coords, c='cyan', alpha=0.1, s=1)
        axes[0].set_title('Position Scatter', color='white')
        axes[0].set_xlim(0, 1)
        axes[0].set_ylim(1, 0)  # Inverted Y
        axes[0].set_facecolor('#1a1a1a')
        axes[0].grid(True, alpha=0.2)

        # Draw boundary box
        rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                              fill=False, edgecolor='lime', linewidth=2)
        axes[0].add_patch(rect)

        # Heatmap
        valid_for_heatmap = [(x, y) for x, y in positions if 0 <= x <= 1 and 0 <= y <= 1]
        if valid_for_heatmap:
            x_heat = [p[0] for p in valid_for_heatmap]
            y_heat = [p[1] for p in valid_for_heatmap]
            heatmap, xedges, yedges = np.histogram2d(x_heat, y_heat, bins=100, range=[[0, 1], [0, 1]])
            extent = [xedges[0], xedges[-1], yedges[-1], yedges[0]]
            axes[1].imshow(heatmap.T, extent=extent, origin='upper', cmap='hot', aspect='auto')

        axes[1].set_title('Heatmap (Walkable Areas)', color='white')
        axes[1].set_xlim(0, 1)
        axes[1].set_ylim(1, 0)
        axes[1].set_facecolor('#1a1a1a')

        plt.tight_layout()
        output_path = output_dir / f'combined_dense_{map_name}.png'
        plt.savefig(str(output_path), dpi=150, bbox_inches='tight', facecolor='#1a1a1a')
        plt.close()

    # Save boundaries
    boundaries_path = output_dir / 'map_boundaries_combined.json'
    with open(boundaries_path, 'w') as f:
        json.dump(map_boundaries, f, indent=2)
    print(f"\nSaved combined boundaries to: {boundaries_path}")

    # Print Python dict format for simulation
    print("\n" + "=" * 60)
    print("MAP_WALKABLE_BOUNDS (for realistic_round_sim.py)")
    print("=" * 60)
    print("MAP_WALKABLE_BOUNDS = {")
    for map_name, bounds in sorted(map_boundaries.items()):
        print(f"    '{map_name}': {{'x': ({bounds['x_min']:.3f}, {bounds['x_max']:.3f}), 'y': ({bounds['y_min']:.3f}, {bounds['y_max']:.3f})}},")
    print("}")


if __name__ == '__main__':
    main()
