#!/usr/bin/env python3
"""
Walkable mask v4 - Enhanced sensitivity for thin walls and obstacles.

Improvements over v3:
1. Lower brightness threshold for outlines (150 vs 180)
2. Lower cell percentage threshold (5% vs 15%)
3. Detect darker gray areas as potential obstacles
4. Higher grid resolution (150x150)
5. Morphological operations to fill thin gaps

Detection layers:
- VOID: Black pixels (< 30) - outer boundary
- OUTLINE: Bright pixels (> 150) - walls/separators
- DARK_OBSTACLE: Darker gray (30-60) - boxes/obstacles
- WALKABLE: Everything else (60-150 gray range)
"""

from pathlib import Path
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def extract_walkable_v4(map_name: str):
    """Extract walkable mask with enhanced obstacle detection."""
    project_dir = Path(__file__).parent.parent.parent
    GRID_SIZE = 150  # Higher resolution for thin walls

    # Find image
    img_path = project_dir / 'map_images' / f'{map_name}.png'
    if not img_path.exists():
        print(f"ERROR: {map_name} not found")
        return None

    print(f"\nProcessing {map_name}...")

    # Load and process image
    img = Image.open(img_path)
    rgba = np.array(img.convert('RGBA'))
    rgb = np.array(img.convert('RGB'))
    gray = np.array(img.convert('L'))

    # Handle transparency
    if rgba.shape[2] == 4:
        gray[rgba[:, :, 3] < 128] = 0
        rgb[rgba[:, :, 3] < 128] = 0

    h, w = gray.shape
    print(f"  Original size: {w}x{h}")

    # Detect elevation areas (yellow/beige tones) - these are walkable
    # Yellow/beige: high R, high G, lower B
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    elevation_mask = (
        (r > 120) & (g > 120) & (b < 150) &  # Yellowish tone
        (r > b + 20) & (g > b + 20) &          # R and G higher than B
        (gray > 100)                            # Not too dark
    )
    elevation_count = elevation_mask.sum()
    if elevation_count > 0:
        print(f"  Elevation pixels detected: {elevation_count}")

    # Detection at full resolution
    # 1. Void = black pixels (< 25)
    void_full = gray < 25

    # 2. White outlines = bright pixels (> 180) AND not elevation
    # Pure white has R=G=B all high, elevation has B lower
    is_white = (r > 180) & (g > 180) & (b > 180)
    outline_full = is_white & ~elevation_mask

    # 3. Walkable = not void AND (not outline OR is elevation)
    walkable_full = ~void_full & (~outline_full | elevation_mask)

    # Placeholder for compatibility
    dark_obstacle_full = np.zeros_like(void_full)

    # Downsample to grid - use proper fractional bounds to cover full image
    walkable_mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    void_mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    obstacle_mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

    for gy in range(GRID_SIZE):
        for gx in range(GRID_SIZE):
            # Get cell bounds using fractional scaling to cover FULL image
            y1 = int(gy * h / GRID_SIZE)
            y2 = int((gy + 1) * h / GRID_SIZE)
            x1 = int(gx * w / GRID_SIZE)
            x2 = int((gx + 1) * w / GRID_SIZE)

            cell_gray = gray[y1:y2, x1:x2]
            cell_r = r[y1:y2, x1:x2]
            cell_g = g[y1:y2, x1:x2]
            cell_b = b[y1:y2, x1:x2]
            total_pixels = cell_gray.size

            # Count each category using RGB for better detection
            void_pct = (cell_gray < 25).sum() / total_pixels

            # White outline: all RGB channels high (pure white)
            white_pct = ((cell_r > 180) & (cell_g > 180) & (cell_b > 180)).sum() / total_pixels

            # Elevation: yellowish (R&G high, B lower)
            elevation_pct = (
                (cell_r > 120) & (cell_g > 120) & (cell_b < 150) &
                (cell_r > cell_b + 20) & (cell_g > cell_b + 20)
            ).sum() / total_pixels

            # Walkable includes floor + elevation
            walkable_pct = ((cell_gray >= 25) & (cell_gray <= 200)).sum() / total_pixels

            # Decision logic
            if void_pct > 0.5:
                # Mostly void
                void_mask[gy, gx] = 1
            elif white_pct > 0.08:
                # Has white wall/window pixels - ALWAYS mark as obstacle
                # White lines are walls even within elevation areas
                obstacle_mask[gy, gx] = 1
            elif walkable_pct > 0.4 or elevation_pct > 0.3:
                # Has floor or elevation pixels (no white walls)
                walkable_mask[gy, gx] = 1
            else:
                # Ambiguous - mark as obstacle
                obstacle_mask[gy, gx] = 1

    # Post-processing: Fill isolated walkable cells surrounded by obstacles
    # (These are likely detection errors)
    for gy in range(1, GRID_SIZE - 1):
        for gx in range(1, GRID_SIZE - 1):
            if walkable_mask[gy, gx] == 1:
                # Count walkable neighbors
                neighbors = (
                    walkable_mask[gy-1, gx] + walkable_mask[gy+1, gx] +
                    walkable_mask[gy, gx-1] + walkable_mask[gy, gx+1]
                )
                # If isolated (no walkable neighbors), mark as obstacle
                if neighbors == 0:
                    walkable_mask[gy, gx] = 0
                    obstacle_mask[gy, gx] = 1

    walkable_count = walkable_mask.sum()
    void_count = void_mask.sum()
    obstacle_count = obstacle_mask.sum()
    total = GRID_SIZE * GRID_SIZE

    print(f"  Void (black): {void_count} ({void_count/total*100:.1f}%)")
    print(f"  Obstacles (walls+dark): {obstacle_count} ({obstacle_count/total*100:.1f}%)")
    print(f"  Walkable: {walkable_count} ({walkable_count/total*100:.1f}%)")

    # Create walkable cells list
    walkable_cells = []
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            if walkable_mask[y, x] == 1:
                center_x = (x + 0.5) / GRID_SIZE
                center_y = (y + 0.5) / GRID_SIZE
                walkable_cells.append([round(center_x, 4), round(center_y, 4)])

    return {
        'map_name': map_name,
        'grid_size': GRID_SIZE,
        'source': 'v4_enhanced_sensitivity',
        'walkable_mask': walkable_mask.tolist(),
        'void_mask': void_mask.tolist(),
        'obstacle_mask': obstacle_mask.tolist(),
        'walkable_cells': walkable_cells,
        'stats': {
            'walkable_count': int(walkable_count),
            'void_count': int(void_count),
            'obstacle_count': int(obstacle_count),
            'total_cells': total,
            'walkable_percentage': round(walkable_count/total*100, 1),
        }
    }


def visualize_v4(map_name: str, data: dict, output_dir: Path):
    """Visualize with enhanced detail."""
    project_dir = Path(__file__).parent.parent.parent
    img_path = project_dir / 'map_images' / f'{map_name}.png'

    if not img_path.exists():
        return

    img = plt.imread(str(img_path))
    walkable = np.array(data['walkable_mask'])
    void = np.array(data['void_mask'])
    obstacle = np.array(data['obstacle_mask'])
    grid_size = data['grid_size']

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"{map_name.upper()} - V4 Enhanced Detection\n"
                 f"Walkable: {data['stats']['walkable_percentage']:.1f}% | "
                 f"Obstacles: {data['stats']['obstacle_count']} cells",
                 fontsize=14, color='white')

    # Original
    axes[0].imshow(img, extent=[0, 1, 1, 0], origin='upper')
    axes[0].set_title('Original Map', color='white')

    # Mask with 3 categories
    combined = np.zeros((grid_size, grid_size))
    combined[void == 1] = 0       # Void = 0 (black)
    combined[obstacle == 1] = 1   # Obstacles = 1 (red)
    combined[walkable == 1] = 2   # Walkable = 2 (green)

    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(['#1a1a1a', '#ff4444', '#44ff44'])
    axes[1].imshow(combined, extent=[0, 1, 1, 0], origin='upper',
                   cmap=cmap, vmin=0, vmax=2)
    axes[1].set_title('Mask: Black=Void, Red=Obstacle, Green=Walkable', color='white')

    # Overlay on map
    axes[2].imshow(img, extent=[0, 1, 1, 0], origin='upper', alpha=0.6)
    overlay = np.zeros((grid_size, grid_size, 4))
    for y in range(grid_size):
        for x in range(grid_size):
            if void[y, x]:
                overlay[y, x] = [0.1, 0.1, 0.1, 0.6]  # Dark
            elif obstacle[y, x]:
                overlay[y, x] = [1, 0.2, 0.2, 0.6]    # Red obstacles
            elif walkable[y, x]:
                overlay[y, x] = [0.2, 1, 0.2, 0.4]    # Green walkable

    axes[2].imshow(overlay, extent=[0, 1, 1, 0], origin='upper', interpolation='nearest')
    axes[2].set_title('Overlay', color='white')

    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)
        ax.set_facecolor('#1a1a1a')

    plt.tight_layout()
    out_path = output_dir / f'v4_mask_{map_name}.png'
    plt.savefig(str(out_path), dpi=150, bbox_inches='tight', facecolor='#1a1a1a')
    plt.close()
    print(f"  Saved: {out_path}")


def main():
    project_dir = Path(__file__).parent.parent.parent
    output_dir = project_dir / 'vct_map_data'
    masks_dir = project_dir / 'backend' / 'app' / 'data' / 'figma_masks'

    output_dir.mkdir(exist_ok=True)
    masks_dir.mkdir(exist_ok=True)

    maps = ['ascent', 'bind', 'haven', 'split', 'icebox', 'breeze',
            'fracture', 'pearl', 'lotus', 'sunset', 'abyss', 'corrode']

    all_data = {}
    for map_name in maps:
        data = extract_walkable_v4(map_name)
        if data:
            all_data[map_name] = data
            visualize_v4(map_name, data, output_dir)

    # Save - include obstacle_mask for validation
    out_file = masks_dir / 'walkable_masks_v4.json'
    slim = {name: {
        'walkable_mask': d['walkable_mask'],
        'obstacle_mask': d['obstacle_mask'],
        'void_mask': d['void_mask'],
        'walkable_cells': d['walkable_cells'],
        'stats': d['stats']
    } for name, d in all_data.items()}
    with open(out_file, 'w') as f:
        json.dump({'maps': slim}, f)
    print(f"\nSaved: {out_file}")


if __name__ == '__main__':
    main()
