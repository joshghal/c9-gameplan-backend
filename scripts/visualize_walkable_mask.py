#!/usr/bin/env python3
"""
Visualize walkable masks overlaid on actual map images.
Shows walkable areas (green) vs unwalkable/walls (red overlay).
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def main():
    project_dir = Path(__file__).parent.parent.parent
    mask_file = project_dir / 'backend' / 'app' / 'data' / 'walkable_masks.json'
    maps_dir = project_dir / 'map_images'  # Check multiple locations
    output_dir = project_dir / 'vct_map_data'

    # Load walkable masks
    print("Loading walkable masks...")
    with open(mask_file) as f:
        mask_data = json.load(f)

    grid_size = mask_data['metadata']['grid_size']

    # Find map images
    map_locations = [
        project_dir / 'map_images',  # Downloaded from cloud9-webapp
        project_dir,  # Root has ascent_map.png
        project_dir / 'frontend' / 'public' / 'maps',
    ]

    # Process each map
    for map_name, data in mask_data['maps'].items():
        print(f"\nProcessing {map_name}...")

        # Try to find map image
        map_img = None
        for loc in map_locations:
            for ext in ['.png', '.jpg', '.webp']:
                img_path = loc / f"{map_name}{ext}"
                if img_path.exists():
                    map_img = mpimg.imread(str(img_path))
                    print(f"  Found map image: {img_path.name}")
                    break
                # Also try with _map suffix
                img_path = loc / f"{map_name}_map{ext}"
                if img_path.exists():
                    map_img = mpimg.imread(str(img_path))
                    print(f"  Found map image: {img_path.name}")
                    break
            if map_img is not None:
                break

        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'{map_name.upper()} - Walkable Area Analysis ({data["stats"]["walkable_percentage"]:.1f}% walkable)',
                     fontsize=14, color='white')

        # Get walkable mask
        mask = np.array(data['walkable_mask'])

        # Plot 1: Map with walkable overlay
        ax = axes[0]
        if map_img is not None:
            ax.imshow(map_img, extent=[0, 1, 1, 0], origin='upper', alpha=0.7)

        # Create colored overlay
        overlay = np.zeros((grid_size, grid_size, 4))
        for y in range(grid_size):
            for x in range(grid_size):
                if mask[y, x] == 1:
                    overlay[y, x] = [0, 1, 0, 0.3]  # Green for walkable
                else:
                    overlay[y, x] = [1, 0, 0, 0.4]  # Red for unwalkable

        ax.imshow(overlay, extent=[0, 1, 1, 0], origin='upper', interpolation='nearest')
        ax.set_title('Map + Walkable Overlay\n(Green=walkable, Red=walls)', color='white')
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)
        ax.set_facecolor('#1a1a1a')

        # Plot 2: Binary walkable mask
        ax = axes[1]
        ax.imshow(mask, extent=[0, 1, 1, 0], origin='upper', cmap='Greens', interpolation='nearest')
        ax.set_title(f'Walkable Mask ({grid_size}x{grid_size} grid)\n{data["stats"]["walkable_cell_count"]} walkable cells', color='white')
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)
        ax.set_facecolor('#1a1a1a')

        # Plot 3: Walkable cells as scatter
        ax = axes[2]
        if map_img is not None:
            ax.imshow(map_img, extent=[0, 1, 1, 0], origin='upper', alpha=0.5)

        cells = data['walkable_cells']
        xs = [c[0] for c in cells]
        ys = [c[1] for c in cells]
        ax.scatter(xs, ys, c='lime', s=8, alpha=0.7, marker='s')
        ax.set_title('Walkable Cell Centers\n(for position sampling)', color='white')
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)
        ax.set_facecolor('#1a1a1a')

        plt.tight_layout()

        # Save
        output_path = output_dir / f'walkable_mask_{map_name}.png'
        plt.savefig(str(output_path), dpi=150, bbox_inches='tight', facecolor='#1a1a1a')
        plt.close()
        print(f"  Saved: {output_path.name}")

    print("\n✅ Done! Check vct_map_data/walkable_mask_*.png")


if __name__ == '__main__':
    main()
