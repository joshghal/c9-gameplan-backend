#!/usr/bin/env python3
"""
Generate map images with square grid overlay for Figma drawing assistance.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


def main():
    project_dir = Path(__file__).parent.parent.parent
    maps_dir = project_dir / 'map_images'
    output_dir = project_dir / 'backend' / 'app' / 'data' / 'figma_masks'
    output_dir.mkdir(exist_ok=True)

    GRID_SIZE = 50  # 50x50 grid = 2% per cell (better accuracy)

    maps = ['ascent', 'bind', 'haven', 'split', 'icebox', 'breeze',
            'fracture', 'pearl', 'lotus', 'sunset', 'abyss', 'corrode']

    for map_name in maps:
        map_path = maps_dir / f"{map_name}.png"
        if not map_path.exists():
            print(f"Skipping {map_name} - image not found")
            continue

        print(f"Generating grid for {map_name}...")

        # Load map image
        img = mpimg.imread(str(map_path))

        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))

        # Show map
        ax.imshow(img, extent=[0, 1, 1, 0], origin='upper')

        # Draw grid lines
        for i in range(GRID_SIZE + 1):
            pos = i / GRID_SIZE
            # Thicker lines every 10 cells for reference
            lw = 1.0 if i % 10 == 0 else 0.3
            alpha = 0.8 if i % 10 == 0 else 0.4
            # Vertical lines
            ax.axvline(x=pos, color='white', linewidth=lw, alpha=alpha)
            # Horizontal lines
            ax.axhline(y=pos, color='white', linewidth=lw, alpha=alpha)

        # Add coordinate labels every 10 cells
        for i in range(0, GRID_SIZE + 1, 10):
            pos = i / GRID_SIZE
            # X labels at bottom
            ax.text(pos, 1.02, f'{i}', ha='center', va='bottom',
                    fontsize=8, color='white')
            # Y labels on left
            ax.text(-0.02, pos, f'{i}', ha='right', va='center',
                    fontsize=8, color='white')

        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(1.05, -0.05)
        ax.set_title(f'{map_name.upper()} - {GRID_SIZE}x{GRID_SIZE} Grid',
                     color='white', fontsize=14, pad=10)
        ax.axis('off')

        plt.tight_layout()
        output_path = output_dir / f'{map_name}_grid.png'
        plt.savefig(str(output_path), dpi=150, bbox_inches='tight',
                    facecolor='#1a1a1a', pad_inches=0.1)
        plt.close()
        print(f"  Saved: {output_path}")

    print(f"\n✅ Done! Grid maps saved to: {output_dir}")


if __name__ == '__main__':
    main()
