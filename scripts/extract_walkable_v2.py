#!/usr/bin/env python3
"""
Extract walkable mask v2 - detects inner obstacles using outline detection.

The VALORANT map images use:
- Black = void (easy to detect)
- Gray = walkable floor
- Gray with WHITE OUTLINE = obstacles/walls (the tricky part)

This script:
1. Detects black void areas
2. Detects white/bright outline pixels as obstacle boundaries
3. Uses morphological operations to fill in obstacles
"""

from pathlib import Path
import json
import numpy as np
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt


def extract_walkable_mask_v2(map_name: str):
    """
    Extract walkable mask with inner obstacle detection.
    """
    project_dir = Path(__file__).parent.parent.parent

    # Find map image
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
    print(f"  Image: {img_path}")

    # Load image
    img = Image.open(img_path)
    img_array = np.array(img)

    # Convert to grayscale
    if len(img_array.shape) == 3:
        if img_array.shape[2] == 4:
            # RGBA - use alpha for transparency
            alpha = img_array[:, :, 3]
            gray = np.array(img.convert('L'))
            gray[alpha < 128] = 0
        else:
            gray = np.array(img.convert('L'))
    else:
        gray = img_array

    print(f"  Size: {gray.shape}")

    # Step 1: Detect void (black areas) - brightness < 30
    void_mask = gray < 30

    # Step 2: Detect white outlines (brightness > 200)
    # These are the edges of obstacles
    outline_mask = gray > 200

    # Step 3: Detect medium-dark areas that could be obstacles
    # Obstacles are often darker than the main floor
    # Main floor is usually around 80-120 brightness
    # Obstacles might be 40-70 brightness
    potential_obstacle = (gray > 30) & (gray < 70)

    # Step 4: Combine - start with everything that's not void
    walkable = ~void_mask

    # Step 5: Remove outline pixels (they're walls)
    walkable = walkable & ~outline_mask

    # Step 6: Use edge detection to find obstacle boundaries
    from PIL import ImageFilter
    edges = img.convert('L').filter(ImageFilter.FIND_EDGES)
    edges_array = np.array(edges)
    strong_edges = edges_array > 100  # Strong edges indicate walls

    # Remove areas with strong edges nearby
    # Dilate the edges slightly
    from scipy import ndimage
    try:
        dilated_edges = ndimage.binary_dilation(strong_edges, iterations=2)
        # Areas that are both potential_obstacle AND near edges are obstacles
        detected_obstacles = potential_obstacle & dilated_edges
        walkable = walkable & ~detected_obstacles
    except:
        # If scipy not available, skip this step
        pass

    # Resize to grid
    GRID_SIZE = 50
    walkable_img = Image.fromarray((walkable * 255).astype(np.uint8))
    walkable_resized = walkable_img.resize((GRID_SIZE, GRID_SIZE), Image.BILINEAR)
    mask = (np.array(walkable_resized) > 128).astype(int)

    walkable_count = mask.sum()
    total_cells = GRID_SIZE * GRID_SIZE
    walkable_pct = walkable_count / total_cells * 100

    print(f"  Walkable cells: {walkable_count}/{total_cells} ({walkable_pct:.1f}%)")

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
        'source': 'auto_v2_with_obstacles',
        'walkable_mask': mask.tolist(),
        'walkable_cells': walkable_cells,
        'stats': {
            'walkable_count': int(walkable_count),
            'total_cells': total_cells,
            'walkable_percentage': round(walkable_pct, 1),
        },
        # Also store the high-res masks for debugging
        'debug': {
            'void_pixels': int(void_mask.sum()),
            'outline_pixels': int(outline_mask.sum()),
        }
    }


def visualize_v2(map_name: str, mask_data: dict, output_dir: Path):
    """Visualize the v2 extraction."""
    project_dir = Path(__file__).parent.parent.parent
    img_path = project_dir / 'map_images' / f'{map_name}.png'

    if not img_path.exists():
        return

    img = plt.imread(str(img_path))
    mask = np.array(mask_data['walkable_mask'])
    grid_size = mask_data['grid_size']

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"{map_name.upper()} - V2 Walkable Mask ({mask_data['stats']['walkable_percentage']:.1f}%)",
                 fontsize=14, color='white')

    # Original
    axes[0].imshow(img, extent=[0, 1, 1, 0], origin='upper')
    axes[0].set_title('Original Map', color='white')

    # Mask
    axes[1].imshow(mask, extent=[0, 1, 1, 0], origin='upper', cmap='Greens')
    axes[1].set_title(f'Walkable Mask ({grid_size}x{grid_size})', color='white')

    # Overlay
    axes[2].imshow(img, extent=[0, 1, 1, 0], origin='upper', alpha=0.7)
    overlay = np.zeros((grid_size, grid_size, 4))
    for y in range(grid_size):
        for x in range(grid_size):
            if mask[y, x] == 1:
                overlay[y, x] = [0, 1, 0, 0.3]
            else:
                overlay[y, x] = [1, 0, 0, 0.4]
    axes[2].imshow(overlay, extent=[0, 1, 1, 0], origin='upper', interpolation='nearest')
    axes[2].set_title('Overlay', color='white')

    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)
        ax.set_facecolor('#1a1a1a')

    plt.tight_layout()
    output_path = output_dir / f'auto_mask_v2_{map_name}.png'
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight', facecolor='#1a1a1a')
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    project_dir = Path(__file__).parent.parent.parent
    output_dir = project_dir / 'vct_map_data'

    maps = ['ascent', 'bind', 'haven', 'split', 'icebox', 'breeze',
            'fracture', 'pearl', 'lotus', 'sunset', 'abyss', 'corrode']

    for map_name in maps:
        mask_data = extract_walkable_mask_v2(map_name)
        if mask_data:
            visualize_v2(map_name, mask_data, output_dir)


if __name__ == '__main__':
    main()
