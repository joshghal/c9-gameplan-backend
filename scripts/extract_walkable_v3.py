#!/usr/bin/env python3
"""
Walkable mask v3 - Detects thin walls via white outline detection.

The map images have:
- Black = void (outer boundary) ✓
- Gray = walkable floor ✓
- WHITE LINES = walls/separators (need to detect!)
- Slightly darker gray boxes = obstacles

Strategy:
1. Detect void (black)
2. Detect white outline pixels (walls/separators)
3. Everything else = walkable
"""

from pathlib import Path
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def extract_walkable_v3(map_name: str):
    """Extract walkable mask detecting white outlines as walls."""
    project_dir = Path(__file__).parent.parent.parent
    GRID_SIZE = 100

    # Find image
    img_path = project_dir / 'map_images' / f'{map_name}.png'
    if not img_path.exists():
        print(f"ERROR: {map_name} not found")
        return None

    print(f"\nProcessing {map_name}...")

    # Load and process image
    img = Image.open(img_path)
    rgba = np.array(img.convert('RGBA'))
    gray = np.array(img.convert('L'))

    # Handle transparency
    if rgba.shape[2] == 4:
        gray[rgba[:, :, 3] < 128] = 0

    h, w = gray.shape
    print(f"  Original size: {w}x{h}")

    # Detect at full resolution first, then downsample
    # 1. Void = black pixels (< 30)
    void_full = gray < 30

    # 2. White outlines = bright pixels (> 180)
    # These are the thin walls and separators
    outline_full = gray > 180

    # 3. Walkable = not void AND not outline
    walkable_full = ~void_full & ~outline_full

    # Downsample to grid
    # For each grid cell, check if majority is walkable
    cell_h = h // GRID_SIZE
    cell_w = w // GRID_SIZE

    walkable_mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    void_mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    outline_mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

    for gy in range(GRID_SIZE):
        for gx in range(GRID_SIZE):
            # Get cell bounds in full image
            y1, y2 = gy * cell_h, (gy + 1) * cell_h
            x1, x2 = gx * cell_w, (gx + 1) * cell_w

            cell_void = void_full[y1:y2, x1:x2]
            cell_outline = outline_full[y1:y2, x1:x2]
            cell_walkable = walkable_full[y1:y2, x1:x2]

            total_pixels = cell_void.size
            void_pct = cell_void.sum() / total_pixels
            outline_pct = cell_outline.sum() / total_pixels
            walkable_pct = cell_walkable.sum() / total_pixels

            # Cell is void if mostly void
            if void_pct > 0.5:
                void_mask[gy, gx] = 1
            # Cell is wall if has significant outline pixels (thin walls)
            elif outline_pct > 0.15:  # 15% threshold for walls
                outline_mask[gy, gx] = 1
            # Cell is walkable if mostly walkable floor
            elif walkable_pct > 0.5:
                walkable_mask[gy, gx] = 1

    walkable_count = walkable_mask.sum()
    void_count = void_mask.sum()
    outline_count = outline_mask.sum()
    total = GRID_SIZE * GRID_SIZE

    print(f"  Void (black): {void_count} ({void_count/total*100:.1f}%)")
    print(f"  Walls (outline): {outline_count} ({outline_count/total*100:.1f}%)")
    print(f"  Walkable: {walkable_count} ({walkable_count/total*100:.1f}%)")

    # Create walkable cells list
    walkable_cells = []
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            if walkable_mask[y, x] == 1:
                center_x = (x + 0.5) / GRID_SIZE
                center_y = (y + 0.5) / GRID_SIZE
                walkable_cells.append([round(center_x, 3), round(center_y, 3)])

    return {
        'map_name': map_name,
        'grid_size': GRID_SIZE,
        'source': 'v3_outline_detection',
        'walkable_mask': walkable_mask.tolist(),
        'void_mask': void_mask.tolist(),
        'outline_mask': outline_mask.tolist(),
        'walkable_cells': walkable_cells,
        'stats': {
            'walkable_count': int(walkable_count),
            'void_count': int(void_count),
            'outline_count': int(outline_count),
            'total_cells': total,
            'walkable_percentage': round(walkable_count/total*100, 1),
        }
    }


def visualize_v3(map_name: str, data: dict, output_dir: Path):
    """Visualize with 3 categories: void, walls, walkable."""
    project_dir = Path(__file__).parent.parent.parent
    img_path = project_dir / 'map_images' / f'{map_name}.png'

    if not img_path.exists():
        return

    img = plt.imread(str(img_path))
    walkable = np.array(data['walkable_mask'])
    void = np.array(data['void_mask'])
    outline = np.array(data['outline_mask'])
    grid_size = data['grid_size']

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"{map_name.upper()} - V3 Outline Detection\n"
                 f"Walkable: {data['stats']['walkable_percentage']:.1f}% | "
                 f"Walls: {data['stats']['outline_count']} cells",
                 fontsize=14, color='white')

    # Original
    axes[0].imshow(img, extent=[0, 1, 1, 0], origin='upper')
    axes[0].set_title('Original Map', color='white')

    # Mask with 3 categories
    combined = np.zeros((grid_size, grid_size))
    combined[void == 1] = 0  # Void = 0 (black)
    combined[outline == 1] = 1  # Walls = 1 (red)
    combined[walkable == 1] = 2  # Walkable = 2 (green)

    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(['#1a1a1a', '#ff4444', '#44ff44'])
    axes[1].imshow(combined, extent=[0, 1, 1, 0], origin='upper',
                   cmap=cmap, vmin=0, vmax=2)
    axes[1].set_title('Mask: Black=Void, Red=Wall, Green=Walkable', color='white')

    # Overlay on map
    axes[2].imshow(img, extent=[0, 1, 1, 0], origin='upper', alpha=0.6)
    overlay = np.zeros((grid_size, grid_size, 4))
    for y in range(grid_size):
        for x in range(grid_size):
            if void[y, x]:
                overlay[y, x] = [0.1, 0.1, 0.1, 0.6]  # Dark
            elif outline[y, x]:
                overlay[y, x] = [1, 0.2, 0.2, 0.6]  # Red walls
            elif walkable[y, x]:
                overlay[y, x] = [0.2, 1, 0.2, 0.4]  # Green walkable

    axes[2].imshow(overlay, extent=[0, 1, 1, 0], origin='upper', interpolation='nearest')
    axes[2].set_title('Overlay', color='white')

    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)
        ax.set_facecolor('#1a1a1a')

    plt.tight_layout()
    out_path = output_dir / f'v3_mask_{map_name}.png'
    plt.savefig(str(out_path), dpi=150, bbox_inches='tight', facecolor='#1a1a1a')
    plt.close()
    print(f"  Saved: {out_path}")


def main():
    project_dir = Path(__file__).parent.parent.parent
    output_dir = project_dir / 'vct_map_data'
    masks_dir = project_dir / 'backend' / 'app' / 'data' / 'figma_masks'

    maps = ['ascent', 'bind', 'haven', 'split', 'icebox', 'breeze',
            'fracture', 'pearl', 'lotus', 'sunset', 'abyss', 'corrode']

    all_data = {}
    for map_name in maps:
        data = extract_walkable_v3(map_name)
        if data:
            all_data[map_name] = data
            visualize_v3(map_name, data, output_dir)

    # Save
    out_file = masks_dir / 'walkable_masks_v3.json'
    slim = {name: {
        'walkable_mask': d['walkable_mask'],
        'walkable_cells': d['walkable_cells'],
        'stats': d['stats']
    } for name, d in all_data.items()}
    with open(out_file, 'w') as f:
        json.dump({'maps': slim}, f)
    print(f"\n✅ Saved: {out_file}")


if __name__ == '__main__':
    main()
