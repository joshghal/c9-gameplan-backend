#!/usr/bin/env python3
"""
Extract actionable position patterns from Henrik API data for simulation.

Outputs:
1. Zone heatmaps by team side (attack/defense)
2. Engagement hotspots (where kills happen)
3. Hold angles by position (from view_radians)
4. Time-based position progression
"""

import json
import math
from pathlib import Path
from collections import defaultdict
import numpy as np

# Map transforms
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
    'corrode': {'xMult': 7.8e-05, 'yMult': -7.8e-05, 'xAdd': 0.5, 'yAdd': 0.5},
}


def game_to_minimap(game_x: float, game_y: float, map_name: str) -> tuple:
    """Convert game coordinates to minimap normalized (0-1) coordinates."""
    t = MAP_TRANSFORMS.get(map_name.lower(), MAP_TRANSFORMS['ascent'])
    minimap_x = game_y * t['xMult'] + t['xAdd']
    minimap_y = game_x * t['yMult'] + t['yAdd']
    return (minimap_x, minimap_y)


def get_round_phase(time_ms: int) -> str:
    """Determine round phase from time in round."""
    if time_ms < 15000:
        return 'setup'
    elif time_ms < 40000:
        return 'map_control'
    elif time_ms < 60000:
        return 'execute'
    else:
        return 'post_plant'


def discretize_position(x: float, y: float, grid_size: int = 20) -> tuple:
    """Convert position to grid cell for heatmap."""
    cell_x = min(grid_size - 1, max(0, int(x * grid_size)))
    cell_y = min(grid_size - 1, max(0, int(y * grid_size)))
    return (cell_x, cell_y)


def main():
    project_dir = Path(__file__).parent.parent.parent
    henrik_file = project_dir / 'data' / 'raw' / 'all_maps_snapshots.json'
    output_dir = project_dir / 'backend' / 'app' / 'data'
    output_dir.mkdir(exist_ok=True)

    print("Loading Henrik API data...")
    with open(henrik_file) as f:
        snapshots = json.load(f)
    print(f"Loaded {len(snapshots):,} kill snapshots")

    # Data structures for patterns
    GRID_SIZE = 20  # 20x20 grid for heatmaps

    # Per-map data
    map_patterns = defaultdict(lambda: {
        'attack_heatmap': np.zeros((GRID_SIZE, GRID_SIZE)),
        'defense_heatmap': np.zeros((GRID_SIZE, GRID_SIZE)),
        'kill_heatmap': np.zeros((GRID_SIZE, GRID_SIZE)),
        'hold_angles': defaultdict(list),  # cell -> list of view_radians
        'phase_positions': defaultdict(lambda: {'attack': [], 'defense': []}),
        'kill_count': 0,
        'position_count': 0,
    })

    # Process each snapshot
    for snapshot in snapshots:
        map_name = snapshot.get('map_name', '').lower()
        if not map_name or map_name == 'unknown':
            continue

        time_in_round = snapshot.get('kill_time_in_round', 0)
        phase = get_round_phase(time_in_round)

        patterns = map_patterns[map_name]

        # Record kill location
        victim_x = snapshot.get('victim_x', 0)
        victim_y = snapshot.get('victim_y', 0)
        if victim_x != 0 or victim_y != 0:
            mx, my = game_to_minimap(victim_x, victim_y, map_name)
            if 0 <= mx <= 1 and 0 <= my <= 1:
                cell = discretize_position(mx, my, GRID_SIZE)
                patterns['kill_heatmap'][cell[1], cell[0]] += 1
                patterns['kill_count'] += 1

        # Process player positions
        for pos in snapshot.get('player_positions', []):
            game_x = pos.get('x', 0)
            game_y = pos.get('y', 0)
            team = pos.get('player_team', '').lower()
            view_radians = pos.get('view_radians', 0)

            if game_x == 0 and game_y == 0:
                continue

            mx, my = game_to_minimap(game_x, game_y, map_name)
            if not (0 <= mx <= 1 and 0 <= my <= 1):
                continue

            cell = discretize_position(mx, my, GRID_SIZE)
            patterns['position_count'] += 1

            # Determine side (Red is typically defense in VALORANT)
            # But this varies - we'll use position heuristics
            # For now, record both teams separately
            if 'red' in team:
                patterns['defense_heatmap'][cell[1], cell[0]] += 1
                patterns['phase_positions'][phase]['defense'].append((mx, my))
            else:
                patterns['attack_heatmap'][cell[1], cell[0]] += 1
                patterns['phase_positions'][phase]['attack'].append((mx, my))

            # Record hold angle
            if view_radians != 0:
                patterns['hold_angles'][cell].append(view_radians)

    # Process and output patterns
    print("\n" + "=" * 60)
    print("EXTRACTED POSITION PATTERNS")
    print("=" * 60)

    output_data = {
        'metadata': {
            'total_snapshots': len(snapshots),
            'grid_size': GRID_SIZE,
            'source': 'henrik_api',
        },
        'maps': {}
    }

    for map_name in sorted(map_patterns.keys()):
        patterns = map_patterns[map_name]
        print(f"\n{map_name.upper()}:")
        print(f"  Positions: {patterns['position_count']:,}")
        print(f"  Kills: {patterns['kill_count']:,}")

        # Normalize heatmaps
        attack_heat = patterns['attack_heatmap']
        defense_heat = patterns['defense_heatmap']
        kill_heat = patterns['kill_heatmap']

        # Convert to probabilities
        attack_total = attack_heat.sum()
        defense_total = defense_heat.sum()
        kill_total = kill_heat.sum()

        if attack_total > 0:
            attack_heat = attack_heat / attack_total
        if defense_total > 0:
            defense_heat = defense_heat / defense_total
        if kill_total > 0:
            kill_heat = kill_heat / kill_total

        # Find top engagement zones (highest kill density)
        flat_kills = kill_heat.flatten()
        top_indices = np.argsort(flat_kills)[-5:][::-1]  # Top 5 cells
        top_zones = []
        for idx in top_indices:
            if flat_kills[idx] > 0:
                y, x = divmod(idx, GRID_SIZE)
                center_x = (x + 0.5) / GRID_SIZE
                center_y = (y + 0.5) / GRID_SIZE
                top_zones.append({
                    'center': [round(center_x, 3), round(center_y, 3)],
                    'density': round(float(flat_kills[idx]), 4),
                })

        print(f"  Top engagement zones: {len(top_zones)}")

        # Calculate average hold angles per zone
        hold_angle_summary = {}
        for cell, angles in patterns['hold_angles'].items():
            if len(angles) >= 10:  # Need enough samples
                # Convert to degrees and find mean
                angles_deg = [math.degrees(a) % 360 for a in angles]
                # Circular mean for angles
                sin_sum = sum(math.sin(math.radians(a)) for a in angles_deg)
                cos_sum = sum(math.cos(math.radians(a)) for a in angles_deg)
                mean_angle = math.degrees(math.atan2(sin_sum, cos_sum)) % 360

                center_x = (cell[0] + 0.5) / GRID_SIZE
                center_y = (cell[1] + 0.5) / GRID_SIZE
                key = f"{center_x:.2f},{center_y:.2f}"
                hold_angle_summary[key] = {
                    'mean_angle_deg': round(mean_angle, 1),
                    'samples': len(angles),
                }

        print(f"  Hold angle zones: {len(hold_angle_summary)}")

        # Phase position centroids
        phase_centroids = {}
        for phase, sides in patterns['phase_positions'].items():
            phase_centroids[phase] = {}
            for side, positions in sides.items():
                if positions:
                    xs = [p[0] for p in positions]
                    ys = [p[1] for p in positions]
                    phase_centroids[phase][side] = {
                        'centroid': [round(np.mean(xs), 3), round(np.mean(ys), 3)],
                        'std': [round(np.std(xs), 3), round(np.std(ys), 3)],
                        'samples': len(positions),
                    }

        # Store map data
        output_data['maps'][map_name] = {
            'position_count': patterns['position_count'],
            'kill_count': patterns['kill_count'],
            'attack_heatmap': attack_heat.tolist(),
            'defense_heatmap': defense_heat.tolist(),
            'kill_heatmap': kill_heat.tolist(),
            'top_engagement_zones': top_zones,
            'hold_angles': hold_angle_summary,
            'phase_centroids': phase_centroids,
        }

    # Save output
    output_file = output_dir / 'position_patterns.json'
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved position patterns to: {output_file}")

    # Print example usage
    print("\n" + "=" * 60)
    print("EXAMPLE USAGE IN SIMULATION")
    print("=" * 60)
    print("""
# Sample realistic position for attacker on Ascent during execute phase:
from app.services.data_loader import get_data_loader

patterns = get_data_loader()._load_json('position_patterns.json')
ascent = patterns['maps']['ascent']

# Get execute phase centroid for attackers
exec_pos = ascent['phase_centroids']['execute']['attack']
# centroid: [0.45, 0.52], std: [0.12, 0.15]

# Sample position around centroid
import random
x = random.gauss(exec_pos['centroid'][0], exec_pos['std'][0])
y = random.gauss(exec_pos['centroid'][1], exec_pos['std'][1])

# Or use heatmap for weighted random position
heatmap = ascent['attack_heatmap']  # 20x20 probability grid
# Sample cell weighted by density, then random within cell
""")


if __name__ == '__main__':
    main()
