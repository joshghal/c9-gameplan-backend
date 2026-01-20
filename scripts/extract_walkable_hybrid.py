#!/usr/bin/env python3
"""
Hybrid walkable mask extraction.

Combines:
1. Map image - detects void (black areas)
2. Player position data - marks cells where players actually stood

Logic:
- Black pixels in map = VOID (unwalkable)
- Cell has player positions = WALKABLE (confirmed by data)
- Gray pixels with NO player data = POTENTIAL OBSTACLE (suspicious)

This catches thin walls and small boxes that players never stood on.
"""

from pathlib import Path
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def extract_hybrid_mask(map_name: str, min_positions: int = 1):
    """
    Create hybrid walkable mask.

    Args:
        map_name: Map name
        min_positions: Minimum player positions in cell to mark as walkable
    """
    project_dir = Path(__file__).parent.parent.parent
    GRID_SIZE = 100  # Higher resolution for thin walls

    # Load map image
    img_path = None
    for path in [
        project_dir / 'map_images' / f'{map_name}.png',
        project_dir / f'{map_name}_map.png',
    ]:
        if path.exists():
            img_path = path
            break

    if img_path is None:
        print(f"ERROR: Map image not found for {map_name}")
        return None

    print(f"\nProcessing {map_name}...")

    # Step 1: Get void mask from image
    img = Image.open(img_path)
    if img.mode == 'RGBA':
        rgba = np.array(img)
        gray = np.array(img.convert('L'))
        gray[rgba[:, :, 3] < 128] = 0  # Transparent = black
    else:
        gray = np.array(img.convert('L'))

    # Resize to grid
    gray_resized = np.array(Image.fromarray(gray).resize((GRID_SIZE, GRID_SIZE), Image.BILINEAR))
    void_mask = gray_resized < 30  # Black = void

    # Step 2: Load player position data from Henrik API
    henrik_file = project_dir / 'data' / 'raw' / 'all_maps_snapshots.json'
    position_counts = np.zeros((GRID_SIZE, GRID_SIZE))

    if henrik_file.exists():
        with open(henrik_file) as f:
            snapshots = json.load(f)

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

        t = MAP_TRANSFORMS.get(map_name.lower())
        if t:
            for snapshot in snapshots:
                if snapshot.get('map_name', '').lower() != map_name.lower():
                    continue

                for pos in snapshot.get('player_positions', []):
                    gx, gy = pos.get('x', 0), pos.get('y', 0)
                    if gx == 0 and gy == 0:
                        continue

                    # Convert to minimap
                    mx = gy * t['xMult'] + t['xAdd']
                    my = gx * t['yMult'] + t['yAdd']

                    if 0 <= mx < 1 and 0 <= my < 1:
                        cx = int(mx * GRID_SIZE)
                        cy = int(my * GRID_SIZE)
                        if 0 <= cx < GRID_SIZE and 0 <= cy < GRID_SIZE:
                            position_counts[cy, cx] += 1

    total_positions = position_counts.sum()
    print(f"  Player positions loaded: {int(total_positions):,}")

    # Step 3: Create hybrid mask
    # Cell is walkable if:
    # - NOT void (from image)
    # - AND has player data (confirmed walkable)
    has_players = position_counts >= min_positions
    walkable = ~void_mask & has_players

    # For cells that are not void but have no player data,
    # they MIGHT be walkable (corners, rarely visited spots) or obstacles
    # We'll mark them as "uncertain" for now
    not_void = ~void_mask
    uncertain = not_void & ~has_players

    walkable_count = walkable.sum()
    uncertain_count = uncertain.sum()
    void_count = void_mask.sum()
    total = GRID_SIZE * GRID_SIZE

    print(f"  Void (black): {void_count} ({void_count/total*100:.1f}%)")
    print(f"  Walkable (confirmed): {walkable_count} ({walkable_count/total*100:.1f}%)")
    print(f"  Uncertain (no data): {uncertain_count} ({uncertain_count/total*100:.1f}%)")

    # For now, treat uncertain as obstacles (conservative)
    # This catches thin walls and boxes that players never stand on
    mask = walkable.astype(int)

    # Create walkable cells list
    walkable_cells = []
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            if mask[y, x] == 1:
                center_x = (x + 0.5) / GRID_SIZE
                center_y = (y + 0.5) / GRID_SIZE
                walkable_cells.append([round(center_x, 3), round(center_y, 3)])

    return {
        'map_name': map_name,
        'grid_size': GRID_SIZE,
        'source': 'hybrid_image_plus_positions',
        'walkable_mask': mask.tolist(),
        'walkable_cells': walkable_cells,
        'position_counts': position_counts.tolist(),
        'void_mask': void_mask.tolist(),
        'stats': {
            'walkable_count': int(walkable_count),
            'void_count': int(void_count),
            'uncertain_count': int(uncertain_count),
            'total_cells': total,
            'walkable_percentage': round(walkable_count/total*100, 1),
            'total_player_positions': int(total_positions),
        }
    }


def visualize_hybrid(map_name: str, mask_data: dict, output_dir: Path):
    """Visualize hybrid mask with three categories."""
    project_dir = Path(__file__).parent.parent.parent
    img_path = project_dir / 'map_images' / f'{map_name}.png'

    if not img_path.exists():
        return

    img = plt.imread(str(img_path))
    mask = np.array(mask_data['walkable_mask'])
    void_mask = np.array(mask_data['void_mask'])
    pos_counts = np.array(mask_data['position_counts'])
    grid_size = mask_data['grid_size']

    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    fig.suptitle(f"{map_name.upper()} - Hybrid Walkable Mask\n"
                 f"Walkable: {mask_data['stats']['walkable_percentage']:.1f}% | "
                 f"Positions: {mask_data['stats']['total_player_positions']:,}",
                 fontsize=14, color='white')

    # Original
    axes[0, 0].imshow(img, extent=[0, 1, 1, 0], origin='upper')
    axes[0, 0].set_title('Original Map', color='white')

    # Position heatmap
    axes[0, 1].imshow(img, extent=[0, 1, 1, 0], origin='upper', alpha=0.5)
    pos_normalized = pos_counts / (pos_counts.max() + 1)
    axes[0, 1].imshow(pos_normalized, extent=[0, 1, 1, 0], origin='upper',
                       cmap='hot', alpha=0.7)
    axes[0, 1].set_title('Player Position Density', color='white')

    # Final mask
    axes[1, 0].imshow(mask, extent=[0, 1, 1, 0], origin='upper', cmap='Greens')
    axes[1, 0].set_title(f'Walkable Mask ({grid_size}x{grid_size})', color='white')

    # Overlay with 3 colors
    axes[1, 1].imshow(img, extent=[0, 1, 1, 0], origin='upper', alpha=0.6)
    overlay = np.zeros((grid_size, grid_size, 4))
    for y in range(grid_size):
        for x in range(grid_size):
            if void_mask[y, x]:
                overlay[y, x] = [0.2, 0.2, 0.2, 0.5]  # Dark gray = void
            elif mask[y, x] == 1:
                overlay[y, x] = [0, 1, 0, 0.4]  # Green = walkable
            else:
                overlay[y, x] = [1, 0, 0, 0.5]  # Red = obstacle (uncertain)

    axes[1, 1].imshow(overlay, extent=[0, 1, 1, 0], origin='upper', interpolation='nearest')
    axes[1, 1].set_title('Green=Walkable, Red=Obstacle, Gray=Void', color='white')

    for ax in axes.flat:
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)
        ax.set_facecolor('#1a1a1a')

    plt.tight_layout()
    output_path = output_dir / f'hybrid_mask_{map_name}.png'
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight', facecolor='#1a1a1a')
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    project_dir = Path(__file__).parent.parent.parent
    output_dir = project_dir / 'vct_map_data'
    masks_dir = project_dir / 'backend' / 'app' / 'data' / 'figma_masks'

    maps = ['ascent', 'bind', 'haven', 'split', 'icebox', 'breeze',
            'fracture', 'pearl', 'lotus', 'sunset', 'abyss', 'corrode']

    all_masks = {}

    for map_name in maps:
        mask_data = extract_hybrid_mask(map_name)
        if mask_data:
            all_masks[map_name] = mask_data
            visualize_hybrid(map_name, mask_data, output_dir)

    # Save combined
    combined_file = masks_dir / 'walkable_masks_hybrid.json'
    with open(combined_file, 'w') as f:
        # Remove large arrays from combined file
        slim_masks = {}
        for name, data in all_masks.items():
            slim_masks[name] = {
                'walkable_mask': data['walkable_mask'],
                'walkable_cells': data['walkable_cells'],
                'stats': data['stats'],
            }
        json.dump({'maps': slim_masks}, f)

    print(f"\n✅ Saved: {combined_file}")


if __name__ == '__main__':
    main()
