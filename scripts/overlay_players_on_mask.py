#!/usr/bin/env python3
"""
Overlay player positions on v4 walkable mask.
This helps identify elevation gaps - places where players walk but v4 shows as blocked.
"""

import json
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict

GRID_SIZE = 150

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
    'corrode': {'xMult': 7.8e-05, 'yMult': -7.8e-05, 'xAdd': 0.5, 'yAdd': 0.5},
}


def game_to_minimap(game_x: float, game_y: float, map_name: str) -> tuple:
    """Convert game coordinates to minimap normalized (0-1) coordinates."""
    t = MAP_TRANSFORMS.get(map_name.lower(), MAP_TRANSFORMS['ascent'])
    minimap_x = game_y * t['xMult'] + t['xAdd']
    minimap_y = game_x * t['yMult'] + t['yAdd']
    return (minimap_x, minimap_y)


def load_vct_positions(grid_dir: Path, map_name: str, max_files: int = 33) -> list:
    """Load positions from VCT GRID JSONL files for a specific map."""
    positions = []
    jsonl_files = list(grid_dir.glob('*.jsonl'))[:max_files]  # Limit files

    for filepath in jsonl_files:
        with open(filepath, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    events = data.get('events', [])

                    for event in events:
                        # Get map name from event
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

                        # Extract player positions
                        teams = actor_state.get('teams', [])
                        if not teams:
                            series_state = event.get('seriesState', {})
                            games = series_state.get('games', [])
                            if games:
                                teams = games[0].get('teams', [])

                        for team in teams:
                            players = team.get('players', [])
                            for player in players:
                                pos = player.get('position', {})
                                if 'x' in pos and 'y' in pos:
                                    game_x = pos['x']
                                    game_y = pos['y']
                                    mini_x, mini_y = game_to_minimap(game_x, game_y, map_name)
                                    if 0 <= mini_x <= 1 and 0 <= mini_y <= 1:
                                        positions.append((mini_x, mini_y))
                except:
                    continue

    return positions


def load_v4_mask(mask_file: Path, map_name: str) -> dict:
    """Load v4 mask for a specific map."""
    with open(mask_file, 'r') as f:
        data = json.load(f)
        return data.get('maps', data).get(map_name, {})


def create_overlay(map_name: str, positions: list, mask_data: dict, output_dir: Path):
    """Create overlay visualization."""
    walkable = np.array(mask_data['walkable_mask'])
    obstacle = np.array(mask_data['obstacle_mask'])
    void = np.array(mask_data['void_mask'])

    # Convert positions to numpy
    if not positions:
        print(f"  No positions for {map_name}")
        return

    x_coords = [p[0] for p in positions]
    y_coords = [p[1] for p in positions]

    # Create figure with single combined view
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(f"{map_name.upper()} - Player Positions vs V4 Mask\n"
                 f"{len(positions):,} positions | V4 walkable: {mask_data['stats']['walkable_percentage']:.1f}%",
                 fontsize=14, color='white')
    fig.patch.set_facecolor('#1a1a1a')

    # Panel 1: V4 mask with player scatter overlay
    ax1 = axes[0]
    ax1.set_facecolor('#1a1a1a')

    # Draw v4 mask as background
    mask_overlay = np.zeros((GRID_SIZE, GRID_SIZE, 4))
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            if void[y, x]:
                mask_overlay[y, x] = [0.1, 0.1, 0.1, 0.8]  # Dark gray
            elif obstacle[y, x]:
                mask_overlay[y, x] = [0.8, 0.2, 0.2, 0.6]  # Red - semi-transparent
            elif walkable[y, x]:
                mask_overlay[y, x] = [0.2, 0.6, 0.2, 0.4]  # Green - more transparent

    ax1.imshow(mask_overlay, extent=[0, 1, 1, 0], origin='upper', interpolation='nearest')

    # Scatter player positions on top
    ax1.scatter(x_coords, y_coords, c='cyan', alpha=0.3, s=2, label='Player positions')
    ax1.set_title('V4 Mask + Player Positions (cyan)', color='white')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(1, 0)

    # Panel 2: Identify gaps (player positions in non-walkable areas)
    ax2 = axes[1]
    ax2.set_facecolor('#1a1a1a')

    # Count players per grid cell
    player_counts = np.zeros((GRID_SIZE, GRID_SIZE))
    for x, y in positions:
        gx = min(int(x * GRID_SIZE), GRID_SIZE - 1)
        gy = min(int(y * GRID_SIZE), GRID_SIZE - 1)
        player_counts[gy, gx] += 1

    # Find gaps: cells with ANY players but not marked walkable
    gap_overlay = np.zeros((GRID_SIZE, GRID_SIZE, 4))
    gap_cells = []

    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            if player_counts[y, x] >= 1:  # ANY player position
                if void[y, x]:
                    # GAP IN VOID - players walk here but v4 says void
                    gap_overlay[y, x] = [1, 0.5, 0, 0.8]  # Orange = void gap
                    gap_cells.append((x, y, int(player_counts[y, x]), 'void'))
                elif obstacle[y, x]:
                    # GAP IN OBSTACLE - players walk here but v4 says blocked
                    gap_overlay[y, x] = [1, 1, 0, 0.8]  # Yellow = obstacle gap
                    gap_cells.append((x, y, int(player_counts[y, x]), 'obstacle'))
                elif walkable[y, x]:
                    gap_overlay[y, x] = [0, 1, 0, 0.4]  # Green = correct
            elif void[y, x]:
                gap_overlay[y, x] = [0.1, 0.1, 0.1, 0.6]
            elif obstacle[y, x]:
                gap_overlay[y, x] = [0.5, 0.1, 0.1, 0.4]  # Dim red

    ax2.imshow(gap_overlay, extent=[0, 1, 1, 0], origin='upper', interpolation='nearest')
    ax2.set_title(f'Gap Detection: Yellow = Players walk but V4 blocked ({len(gap_cells)} cells)', color='white')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(1, 0)

    plt.tight_layout()
    out_path = output_dir / f'overlay_{map_name}.png'
    plt.savefig(str(out_path), dpi=150, bbox_inches='tight', facecolor='#1a1a1a')
    plt.close()

    print(f"  Saved: {out_path}")
    print(f"  Gap cells (elevation issues): {len(gap_cells)}")

    # Return gap info
    return gap_cells


def main():
    project_dir = Path(__file__).parent.parent.parent
    grid_dir = project_dir / 'grid_data'  # Correct path
    output_dir = project_dir / 'vct_map_data'
    mask_file = project_dir / 'backend' / 'app' / 'data' / 'figma_masks' / 'walkable_masks_v4.json'

    output_dir.mkdir(exist_ok=True)

    if not grid_dir.exists():
        print(f"Error: VCT grid data not found at {grid_dir}")
        return

    if not mask_file.exists():
        print(f"Error: V4 mask not found at {mask_file}")
        return

    # Load all masks
    with open(mask_file, 'r') as f:
        all_masks = json.load(f).get('maps', {})

    # Test with haven first
    maps = ['haven']

    all_gaps = {}

    for map_name in maps:
        print(f"\nProcessing {map_name}...")

        if map_name not in all_masks:
            print(f"  Skipping - no v4 mask")
            continue

        # Load positions
        positions = load_vct_positions(grid_dir, map_name)
        print(f"  Loaded {len(positions):,} positions")

        if len(positions) < 100:
            print(f"  Skipping - not enough data")
            continue

        # Create overlay
        gaps = create_overlay(map_name, positions, all_masks[map_name], output_dir)
        if gaps:
            all_gaps[map_name] = gaps

    # Summary
    print("\n" + "="*60)
    print("ELEVATION GAP SUMMARY")
    print("-"*60)
    for map_name, gaps in sorted(all_gaps.items(), key=lambda x: -len(x[1])):
        print(f"  {map_name}: {len(gaps)} cells with elevation issues")

    # Save gap data
    gap_file = output_dir / 'elevation_gaps.json'
    with open(gap_file, 'w') as f:
        json.dump(all_gaps, f, indent=2)
    print(f"\nGap data saved to: {gap_file}")


if __name__ == '__main__':
    main()
