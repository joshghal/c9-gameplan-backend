#!/usr/bin/env python3
"""
Auto-extract walkable mask from map images.

The VALORANT map images use:
- Gray/colored areas = walkable
- Black areas = void/unwalkable

This script detects walkable areas by finding non-black pixels.
"""

from pathlib import Path
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def extract_walkable_mask(map_name: str, threshold: int = 30):
    """
    Extract walkable mask from map image.

    Args:
        map_name: Name of the map
        threshold: Pixel brightness threshold (0-255).
                   Pixels darker than this are considered void.
    """
    project_dir = Path(__file__).parent.parent.parent

    # Try multiple image locations
    img_path = None
    for path in [
        project_dir / 'map_images' / f'{map_name}.png',
        project_dir / f'{map_name}_map.png',
        project_dir / f'{map_name}.png',
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
    print(f"  Size: {img.size}")

    # Convert to grayscale for brightness detection
    if img.mode == 'RGBA':
        # Handle transparency - transparent = void
        rgba = np.array(img)
        alpha = rgba[:, :, 3]
        gray = np.array(img.convert('L'))
        # Where transparent, set to black (void)
        gray[alpha < 128] = 0
    else:
        gray = np.array(img.convert('L'))

    # Create binary mask: pixels brighter than threshold = walkable
    walkable = (gray > threshold).astype(np.uint8)

    # Resize to standard grid
    GRID_SIZE = 50
    img_resized = Image.fromarray(walkable * 255).resize(
        (GRID_SIZE, GRID_SIZE), Image.BILINEAR
    )
    mask = (np.array(img_resized) > 128).astype(int)

    walkable_count = mask.sum()
    total_cells = GRID_SIZE * GRID_SIZE
    walkable_pct = walkable_count / total_cells * 100

    print(f"  Threshold: {threshold}")
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
        'source': 'auto_extracted',
        'threshold': threshold,
        'walkable_mask': mask.tolist(),
        'walkable_cells': walkable_cells,
        'stats': {
            'walkable_count': int(walkable_count),
            'total_cells': total_cells,
            'walkable_percentage': round(walkable_pct, 1),
        }
    }


def visualize_extraction(map_name: str, mask_data: dict, output_dir: Path):
    """Create visualization of extracted mask."""
    project_dir = Path(__file__).parent.parent.parent

    # Load original image
    img_path = project_dir / 'map_images' / f'{map_name}.png'
    if not img_path.exists():
        return

    img = plt.imread(str(img_path))
    mask = np.array(mask_data['walkable_mask'])
    grid_size = mask_data['grid_size']

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"{map_name.upper()} - Auto-Extracted Walkable Mask ({mask_data['stats']['walkable_percentage']:.1f}% walkable)",
                 fontsize=14, color='white')

    # Original map
    axes[0].imshow(img, extent=[0, 1, 1, 0], origin='upper')
    axes[0].set_title('Original Map', color='white')
    axes[0].set_facecolor('#1a1a1a')

    # Binary mask
    axes[1].imshow(mask, extent=[0, 1, 1, 0], origin='upper', cmap='Greens')
    axes[1].set_title(f'Walkable Mask ({grid_size}x{grid_size})', color='white')
    axes[1].set_facecolor('#1a1a1a')

    # Overlay
    axes[2].imshow(img, extent=[0, 1, 1, 0], origin='upper', alpha=0.7)
    overlay = np.zeros((grid_size, grid_size, 4))
    for y in range(grid_size):
        for x in range(grid_size):
            if mask[y, x] == 1:
                overlay[y, x] = [0, 1, 0, 0.3]  # Green = walkable
            else:
                overlay[y, x] = [1, 0, 0, 0.4]  # Red = void
    axes[2].imshow(overlay, extent=[0, 1, 1, 0], origin='upper', interpolation='nearest')
    axes[2].set_title('Map + Walkable Overlay', color='white')
    axes[2].set_facecolor('#1a1a1a')

    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)

    plt.tight_layout()
    output_path = output_dir / f'auto_mask_{map_name}.png'
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight', facecolor='#1a1a1a')
    plt.close()
    print(f"  Visualization: {output_path}")


def main():
    project_dir = Path(__file__).parent.parent.parent
    output_dir = project_dir / 'vct_map_data'
    output_dir.mkdir(exist_ok=True)

    masks_dir = project_dir / 'backend' / 'app' / 'data' / 'figma_masks'
    masks_dir.mkdir(exist_ok=True)

    maps = ['ascent', 'bind', 'haven', 'split', 'icebox', 'breeze',
            'fracture', 'pearl', 'lotus', 'sunset', 'abyss', 'corrode']

    all_masks = {}

    for map_name in maps:
        mask_data = extract_walkable_mask(map_name, threshold=30)
        if mask_data:
            all_masks[map_name] = mask_data
            visualize_extraction(map_name, mask_data, output_dir)

            # Save individual mask
            mask_file = masks_dir / f'{map_name}_auto_mask.json'
            with open(mask_file, 'w') as f:
                json.dump(mask_data, f)

    # Save combined file
    combined_file = masks_dir / 'walkable_masks_auto.json'
    with open(combined_file, 'w') as f:
        json.dump({
            'metadata': {
                'source': 'auto_extracted_from_map_images',
                'grid_size': 50,
                'threshold': 30,
            },
            'maps': all_masks
        }, f)

    print(f"\n✅ Done!")
    print(f"  Combined masks: {combined_file}")
    print(f"  Visualizations: {output_dir}/auto_mask_*.png")


if __name__ == '__main__':
    main()
