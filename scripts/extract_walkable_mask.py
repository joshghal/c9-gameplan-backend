#!/usr/bin/env python3
"""
Extract walkable/unwalkable masks from position heatmaps.

The heatmap shows where players actually stand - dark areas are walls/buildings.
This script creates a binary mask for each map that can be used to:
1. Validate if a position is walkable
2. Sample only from valid positions
"""

import json
from pathlib import Path
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


def main():
    project_dir = Path(__file__).parent.parent.parent
    henrik_file = project_dir / 'data' / 'raw' / 'all_maps_snapshots.json'
    output_dir = project_dir / 'backend' / 'app' / 'data'

    print("Loading Henrik API data...")
    with open(henrik_file) as f:
        snapshots = json.load(f)
    print(f"Loaded {len(snapshots):,} kill snapshots")

    # Higher resolution grid for better accuracy
    GRID_SIZE = 50  # 50x50 grid (each cell = 2% of map)

    # Build position counts per map
    from collections import defaultdict
    map_grids = defaultdict(lambda: np.zeros((GRID_SIZE, GRID_SIZE)))

    for snapshot in snapshots:
        map_name = snapshot.get('map_name', '').lower()
        if not map_name or map_name == 'unknown':
            continue

        for pos in snapshot.get('player_positions', []):
            game_x = pos.get('x', 0)
            game_y = pos.get('y', 0)
            if game_x == 0 and game_y == 0:
                continue

            mx, my = game_to_minimap(game_x, game_y, map_name)
            if 0 <= mx < 1 and 0 <= my < 1:
                cell_x = int(mx * GRID_SIZE)
                cell_y = int(my * GRID_SIZE)
                map_grids[map_name][cell_y, cell_x] += 1

    # Process each map
    print("\n" + "=" * 60)
    print("WALKABLE MASKS")
    print("=" * 60)

    walkable_data = {
        'metadata': {
            'grid_size': GRID_SIZE,
            'cell_size': 1.0 / GRID_SIZE,
            'description': 'Binary walkable mask derived from player positions. 1=walkable, 0=wall/obstacle'
        },
        'maps': {}
    }

    for map_name in sorted(map_grids.keys()):
        grid = map_grids[map_name]
        total_positions = grid.sum()

        # Create binary mask - cell is walkable if ANY player was ever there
        # Use threshold of at least 1 position (could increase for more confidence)
        walkable_mask = (grid >= 1).astype(int)

        walkable_cells = walkable_mask.sum()
        total_cells = GRID_SIZE * GRID_SIZE
        walkable_pct = walkable_cells / total_cells * 100

        print(f"\n{map_name.upper()}:")
        print(f"  Total positions: {int(total_positions):,}")
        print(f"  Walkable cells: {walkable_cells}/{total_cells} ({walkable_pct:.1f}%)")

        # Store as list of walkable cell coordinates (more compact than full grid)
        walkable_cells_list = []
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                if walkable_mask[y, x] == 1:
                    # Store center of cell in normalized coords
                    center_x = (x + 0.5) / GRID_SIZE
                    center_y = (y + 0.5) / GRID_SIZE
                    walkable_cells_list.append([round(center_x, 3), round(center_y, 3)])

        # Also store the full mask for efficient lookup
        walkable_data['maps'][map_name] = {
            'walkable_cells': walkable_cells_list,
            'walkable_mask': walkable_mask.tolist(),
            'stats': {
                'total_positions': int(total_positions),
                'walkable_cell_count': int(walkable_cells),
                'walkable_percentage': round(walkable_pct, 1),
            }
        }

        # Count unwalkable cells (inner obstacles)
        unwalkable_cells = total_cells - walkable_cells
        print(f"  Unwalkable cells (inner obstacles): {unwalkable_cells} ({100 - walkable_pct:.1f}%)")

    # Save output
    output_file = output_dir / 'walkable_masks.json'
    with open(output_file, 'w') as f:
        json.dump(walkable_data, f)
    print(f"\nSaved walkable masks to: {output_file}")

    # Print usage example
    print("\n" + "=" * 60)
    print("USAGE IN SIMULATION")
    print("=" * 60)
    print("""
def is_position_walkable(x: float, y: float, map_name: str) -> bool:
    '''Check if position is in a walkable cell.'''
    mask = walkable_data['maps'][map_name]['walkable_mask']
    grid_size = 50
    cell_x = min(grid_size - 1, max(0, int(x * grid_size)))
    cell_y = min(grid_size - 1, max(0, int(y * grid_size)))
    return mask[cell_y][cell_x] == 1

def sample_walkable_position(map_name: str) -> tuple:
    '''Sample a random walkable position.'''
    cells = walkable_data['maps'][map_name]['walkable_cells']
    cell = random.choice(cells)
    # Add small random offset within cell
    offset = 0.01  # Half cell size
    x = cell[0] + random.uniform(-offset, offset)
    y = cell[1] + random.uniform(-offset, offset)
    return (x, y)
""")


if __name__ == '__main__':
    main()
