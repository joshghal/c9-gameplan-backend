#!/usr/bin/env python3
"""
Fix V4 walkable mask using player position data.
Cells where players walk but V4 marks as blocked/void -> convert to walkable.
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

GRID_SIZE = 150

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
    'corrode': {'xMult': 7.8e-05, 'yMult': -7.8e-05, 'xAdd': 0.5, 'yAdd': 0.5},
}


def game_to_minimap(game_x, game_y, map_name):
    t = MAP_TRANSFORMS.get(map_name.lower(), MAP_TRANSFORMS['ascent'])
    minimap_x = game_y * t['xMult'] + t['xAdd']
    minimap_y = game_x * t['yMult'] + t['yAdd']
    return (minimap_x, minimap_y)


def load_player_grid(grid_dir, map_name):
    """Load player positions and convert to grid counts."""
    player_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

    jsonl_files = list(grid_dir.glob('*.jsonl'))
    positions = 0

    for filepath in jsonl_files:
        with open(filepath, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    events = data.get('events', [])

                    for event in events:
                        current_map = None
                        actor = event.get('actor', {})
                        actor_state = actor.get('state', {})
                        if 'map' in actor_state:
                            current_map = actor_state['map'].get('name', '').lower()

                        if not current_map:
                            series_state = event.get('seriesState', {})
                            games = series_state.get('games', [])
                            if games and 'map' in games[0]:
                                current_map = games[0]['map'].get('name', '').lower()

                        if current_map != map_name.lower():
                            continue

                        teams = actor_state.get('teams', [])
                        if not teams:
                            series_state = event.get('seriesState', {})
                            games = series_state.get('games', [])
                            if games:
                                teams = games[0].get('teams', [])

                        for team in teams:
                            for player in team.get('players', []):
                                pos = player.get('position', {})
                                if 'x' in pos and 'y' in pos:
                                    mini_x, mini_y = game_to_minimap(pos['x'], pos['y'], map_name)
                                    if 0 <= mini_x <= 1 and 0 <= mini_y <= 1:
                                        gx = min(int(mini_x * GRID_SIZE), GRID_SIZE - 1)
                                        gy = min(int(mini_y * GRID_SIZE), GRID_SIZE - 1)
                                        player_grid[gy, gx] += 1
                                        positions += 1
                except:
                    continue

    return player_grid, positions


def fix_mask(map_name, mask_data, player_grid):
    """Fix mask using player data."""
    walkable = np.array(mask_data['walkable_mask'])
    obstacle = np.array(mask_data['obstacle_mask'])
    void = np.array(mask_data['void_mask'])

    fixes = {'void_to_walkable': 0, 'obstacle_to_walkable': 0}

    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            if player_grid[y, x] >= 1:  # Any player walked here
                if void[y, x]:
                    void[y, x] = 0
                    walkable[y, x] = 1
                    fixes['void_to_walkable'] += 1
                elif obstacle[y, x]:
                    obstacle[y, x] = 0
                    walkable[y, x] = 1
                    fixes['obstacle_to_walkable'] += 1

    # Update stats
    walkable_count = int(walkable.sum())
    obstacle_count = int(obstacle.sum())
    void_count = int(void.sum())
    total = GRID_SIZE * GRID_SIZE

    # Create walkable cells list
    walkable_cells = []
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            if walkable[y, x] == 1:
                center_x = (x + 0.5) / GRID_SIZE
                center_y = (y + 0.5) / GRID_SIZE
                walkable_cells.append([round(center_x, 4), round(center_y, 4)])

    return {
        'walkable_mask': walkable.tolist(),
        'obstacle_mask': obstacle.tolist(),
        'void_mask': void.tolist(),
        'walkable_cells': walkable_cells,
        'stats': {
            'walkable_count': walkable_count,
            'void_count': void_count,
            'obstacle_count': obstacle_count,
            'total_cells': total,
            'walkable_percentage': round(walkable_count/total*100, 1),
        },
        'fixes': fixes
    }


def main():
    project_dir = Path(__file__).parent.parent.parent
    grid_dir = project_dir / 'grid_data'
    mask_file = project_dir / 'backend' / 'app' / 'data' / 'figma_masks' / 'walkable_masks_v4.json'
    output_file = project_dir / 'backend' / 'app' / 'data' / 'figma_masks' / 'walkable_masks_v4_fixed.json'

    print("Loading V4 masks...")
    with open(mask_file, 'r') as f:
        all_masks = json.load(f).get('maps', {})

    maps = ['haven', 'ascent', 'bind', 'split', 'icebox', 'breeze',
            'fracture', 'pearl', 'lotus', 'sunset', 'abyss']

    fixed_masks = {}
    total_fixes = 0

    for map_name in maps:
        print(f"\nProcessing {map_name}...")

        if map_name not in all_masks:
            print(f"  Skipping - no v4 mask")
            continue

        # Load player grid
        player_grid, positions = load_player_grid(grid_dir, map_name)
        print(f"  Loaded {positions:,} player positions")

        if positions < 100:
            print(f"  Skipping - not enough data")
            fixed_masks[map_name] = all_masks[map_name]
            continue

        # Fix mask
        old_walkable = all_masks[map_name]['stats']['walkable_percentage']
        fixed = fix_mask(map_name, all_masks[map_name], player_grid)
        new_walkable = fixed['stats']['walkable_percentage']

        fixes = fixed['fixes']
        map_fixes = fixes['void_to_walkable'] + fixes['obstacle_to_walkable']
        total_fixes += map_fixes

        print(f"  Walkable: {old_walkable}% -> {new_walkable}%")
        print(f"  Fixes: {fixes['void_to_walkable']} void, {fixes['obstacle_to_walkable']} obstacle")

        fixed_masks[map_name] = fixed

    # Save
    with open(output_file, 'w') as f:
        json.dump({'maps': fixed_masks, 'version': 'v4_fixed'}, f)

    print(f"\n{'='*60}")
    print(f"TOTAL FIXES: {total_fixes} cells")
    print(f"Saved to: {output_file}")

    # Also update the main v4 file
    with open(mask_file, 'w') as f:
        json.dump({'maps': fixed_masks, 'version': 'v4_fixed'}, f)
    print(f"Updated: {mask_file}")


if __name__ == '__main__':
    main()
