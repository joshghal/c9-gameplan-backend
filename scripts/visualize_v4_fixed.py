#!/usr/bin/env python3
"""
Visualize the fixed V4 walkable masks for all maps.
"""

import json
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

GRID_SIZE = 150


def visualize_map(map_name, mask_data, map_img_path, output_dir):
    """Create visualization for a single map."""
    walkable = np.array(mask_data['walkable_mask'])
    obstacle = np.array(mask_data['obstacle_mask'])
    void = np.array(mask_data['void_mask'])
    stats = mask_data['stats']

    # Load map image
    if map_img_path.exists():
        pil_img = Image.open(map_img_path).convert('RGBA')
        background = Image.new('RGBA', pil_img.size, (50, 50, 50, 255))
        composite = Image.alpha_composite(background, pil_img)
        map_img = np.array(composite.convert('RGB')) / 255.0
    else:
        map_img = None

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"{map_name.upper()} - V4 Fixed Mask\n"
                 f"Walkable: {stats['walkable_percentage']:.1f}% | "
                 f"Obstacles: {stats['obstacle_count']} cells",
                 fontsize=14, color='white')
    fig.patch.set_facecolor('#1a1a1a')

    # Panel 1: Original map
    if map_img is not None:
        axes[0].imshow(map_img)
    axes[0].set_title('Original Map', color='white')
    axes[0].set_facecolor('#1a1a1a')

    # Panel 2: Mask only
    mask_viz = np.zeros((GRID_SIZE, GRID_SIZE, 3))
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            if void[y, x]:
                mask_viz[y, x] = [0.1, 0.1, 0.1]  # Dark gray
            elif obstacle[y, x]:
                mask_viz[y, x] = [0.8, 0.2, 0.2]  # Red
            elif walkable[y, x]:
                mask_viz[y, x] = [0.2, 0.8, 0.2]  # Green

    axes[1].imshow(mask_viz, extent=[0, 1, 1, 0], origin='upper', interpolation='nearest')
    axes[1].set_title('Mask: Black=Void, Red=Obstacle, Green=Walkable', color='white')
    axes[1].set_facecolor('#1a1a1a')

    # Panel 3: Overlay on map
    if map_img is not None:
        axes[2].imshow(map_img, alpha=0.5)

    overlay = np.zeros((GRID_SIZE, GRID_SIZE, 4))
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            if void[y, x]:
                overlay[y, x] = [0.1, 0.1, 0.1, 0.6]
            elif obstacle[y, x]:
                overlay[y, x] = [1, 0.2, 0.2, 0.6]  # Red
            elif walkable[y, x]:
                overlay[y, x] = [0.2, 1, 0.2, 0.4]  # Green

    axes[2].imshow(overlay, extent=[0, 1, 1, 0], origin='upper', interpolation='nearest')
    axes[2].set_title('Overlay', color='white')
    axes[2].set_facecolor('#1a1a1a')

    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)

    plt.tight_layout()
    out_path = output_dir / f'v4_mask_{map_name}.png'
    plt.savefig(str(out_path), dpi=150, bbox_inches='tight', facecolor='#1a1a1a')
    plt.close()
    print(f"  Saved: {out_path}")


def main():
    project_dir = Path(__file__).parent.parent.parent
    mask_file = project_dir / 'backend' / 'app' / 'data' / 'figma_masks' / 'walkable_masks_v4.json'
    map_img_dir = project_dir / 'map_images'
    output_dir = project_dir / 'vct_map_data'

    output_dir.mkdir(exist_ok=True)

    print("Loading fixed V4 masks...")
    with open(mask_file, 'r') as f:
        all_masks = json.load(f).get('maps', {})

    maps = ['haven', 'ascent', 'bind', 'split', 'icebox', 'breeze',
            'fracture', 'pearl', 'lotus', 'sunset', 'abyss', 'corrode']

    for map_name in maps:
        print(f"\nProcessing {map_name}...")

        if map_name not in all_masks:
            print(f"  Skipping - no mask data")
            continue

        map_img_path = map_img_dir / f'{map_name}.png'
        visualize_map(map_name, all_masks[map_name], map_img_path, output_dir)

    print(f"\n{'='*60}")
    print("All visualizations saved to vct_map_data/v4_mask_*.png")


if __name__ == '__main__':
    main()
