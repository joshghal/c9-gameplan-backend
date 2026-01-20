#!/usr/bin/env python3
"""
Import walkable masks from Figma-exported images.

HOW TO USE:
1. In Figma, create a 1:1 square frame (e.g., 1000x1000px)
2. Place the map image as background
3. Draw WHITE shapes over WALKABLE areas
4. Draw BLACK shapes over OBSTACLES (or leave transparent)
5. Export as PNG with just the shapes (hide map background)
6. Save to: backend/app/data/figma_masks/{map_name}_walkable.png
7. Run: python import_figma_mask.py ascent

The script will:
- Parse the image (white=walkable, black/transparent=obstacle)
- Create a 100x100 binary mask
- Save to JSON for use in simulation
- Generate a visualization overlay

FIGMA EXPORT SETTINGS:
- Format: PNG
- Size: 1000x1000 (or any square)
- Background: Transparent or Black
- Walkable areas: White (#FFFFFF)
"""

import argparse
import json
from pathlib import Path
import numpy as np

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("WARNING: PIL not installed. Run: pip install Pillow")


def import_figma_mask(map_name: str, input_path: Path = None):
    """Import a Figma-drawn walkable mask."""
    project_dir = Path(__file__).parent.parent.parent
    data_dir = project_dir / 'backend' / 'app' / 'data'
    masks_dir = data_dir / 'figma_masks'
    masks_dir.mkdir(exist_ok=True)

    # Find input file
    if input_path is None:
        input_path = masks_dir / f"{map_name}_walkable.png"

    if not input_path.exists():
        print(f"ERROR: Mask file not found: {input_path}")
        print(f"\nTo create a mask:")
        print(f"1. In Figma, draw WHITE shapes over walkable areas")
        print(f"2. Export as PNG to: {input_path}")
        return None

    print(f"Importing Figma mask for {map_name}...")
    print(f"  Input: {input_path}")

    # Load image
    img = Image.open(input_path)
    print(f"  Original size: {img.size}")

    # Convert to grayscale
    if img.mode == 'RGBA':
        # Handle transparency: transparent = obstacle (black)
        alpha = np.array(img.split()[3])
        gray = np.array(img.convert('L'))
        # Where transparent, set to black (obstacle)
        gray[alpha < 128] = 0
    else:
        gray = np.array(img.convert('L'))

    # Resize to standard grid
    GRID_SIZE = 100  # Higher resolution for better accuracy
    img_resized = Image.fromarray(gray).resize((GRID_SIZE, GRID_SIZE), Image.BILINEAR)
    mask_array = np.array(img_resized)

    # Threshold: >128 = walkable (1), <=128 = obstacle (0)
    binary_mask = (mask_array > 128).astype(int)

    walkable_count = binary_mask.sum()
    total_cells = GRID_SIZE * GRID_SIZE
    walkable_pct = walkable_count / total_cells * 100

    print(f"  Grid size: {GRID_SIZE}x{GRID_SIZE}")
    print(f"  Walkable cells: {walkable_count}/{total_cells} ({walkable_pct:.1f}%)")

    # Create walkable cells list
    walkable_cells = []
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            if binary_mask[y, x] == 1:
                center_x = (x + 0.5) / GRID_SIZE
                center_y = (y + 0.5) / GRID_SIZE
                walkable_cells.append([round(center_x, 3), round(center_y, 3)])

    # Save as JSON
    output_data = {
        'map_name': map_name,
        'grid_size': GRID_SIZE,
        'source': 'figma',
        'walkable_mask': binary_mask.tolist(),
        'walkable_cells': walkable_cells,
        'stats': {
            'walkable_count': int(walkable_count),
            'total_cells': total_cells,
            'walkable_percentage': round(walkable_pct, 1),
        }
    }

    output_file = masks_dir / f"{map_name}_mask.json"
    with open(output_file, 'w') as f:
        json.dump(output_data, f)
    print(f"  Saved: {output_file}")

    # Also copy the processed mask as PNG for reference
    processed_img = Image.fromarray((binary_mask * 255).astype(np.uint8))
    processed_path = masks_dir / f"{map_name}_processed.png"
    processed_img.save(processed_path)
    print(f"  Processed mask: {processed_path}")

    # Generate visualization with map overlay if available
    try:
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg

        map_img_path = project_dir / f"{map_name}_map.png"

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'{map_name.upper()} - Figma Walkable Mask ({walkable_pct:.1f}% walkable)',
                     fontsize=14, color='white')

        # Plot 1: Original mask
        axes[0].imshow(mask_array, cmap='gray', extent=[0, 1, 1, 0], origin='upper')
        axes[0].set_title('Figma Export (Grayscale)', color='white')
        axes[0].set_facecolor('#1a1a1a')

        # Plot 2: Binary mask
        axes[1].imshow(binary_mask, cmap='Greens', extent=[0, 1, 1, 0], origin='upper')
        axes[1].set_title(f'Binary Mask ({GRID_SIZE}x{GRID_SIZE})', color='white')
        axes[1].set_facecolor('#1a1a1a')

        # Plot 3: Overlay on map
        if map_img_path.exists():
            map_img = mpimg.imread(str(map_img_path))
            axes[2].imshow(map_img, extent=[0, 1, 1, 0], origin='upper', alpha=0.6)

        # Overlay walkable areas
        overlay = np.zeros((GRID_SIZE, GRID_SIZE, 4))
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                if binary_mask[y, x] == 1:
                    overlay[y, x] = [0, 1, 0, 0.4]  # Green
                else:
                    overlay[y, x] = [1, 0, 0, 0.3]  # Red

        axes[2].imshow(overlay, extent=[0, 1, 1, 0], origin='upper', interpolation='nearest')
        axes[2].set_title('Map + Walkable Overlay', color='white')
        axes[2].set_facecolor('#1a1a1a')

        for ax in axes:
            ax.set_xlim(0, 1)
            ax.set_ylim(1, 0)

        plt.tight_layout()
        viz_path = project_dir / 'vct_map_data' / f'figma_mask_{map_name}.png'
        plt.savefig(str(viz_path), dpi=150, bbox_inches='tight', facecolor='#1a1a1a')
        plt.close()
        print(f"  Visualization: {viz_path}")

    except Exception as e:
        print(f"  Could not create visualization: {e}")

    return output_data


def create_template(map_name: str):
    """Create a template PNG for drawing in Figma."""
    project_dir = Path(__file__).parent.parent.parent
    masks_dir = project_dir / 'backend' / 'app' / 'data' / 'figma_masks'
    masks_dir.mkdir(exist_ok=True)

    # Try to load map image from multiple locations
    map_img_path = None
    for path in [
        project_dir / 'map_images' / f"{map_name}.png",
        project_dir / f"{map_name}_map.png",
        project_dir / f"{map_name}.png",
    ]:
        if path.exists():
            map_img_path = path
            break

    if map_img_path is not None:
        img = Image.open(map_img_path)
        # Make it square
        size = max(img.size)
        square = Image.new('RGB', (size, size), (0, 0, 0))
        offset = ((size - img.size[0]) // 2, (size - img.size[1]) // 2)
        square.paste(img, offset)
        # Resize to 1000x1000
        square = square.resize((1000, 1000), Image.LANCZOS)
        template_path = masks_dir / f"{map_name}_template.png"
        square.save(template_path)
        print(f"Created template: {template_path}")
        print(f"\nInstructions:")
        print(f"1. Import this image into Figma as background")
        print(f"2. Create a new layer above it")
        print(f"3. Draw WHITE shapes over walkable areas")
        print(f"4. Hide the background, export the shapes layer as PNG")
        print(f"5. Save as: {masks_dir}/{map_name}_walkable.png")
    else:
        # Create blank template
        blank = Image.new('RGB', (1000, 1000), (30, 30, 30))
        template_path = masks_dir / f"{map_name}_template.png"
        blank.save(template_path)
        print(f"Created blank template: {template_path}")
        print(f"(Map image not found at {map_img_path})")


def main():
    parser = argparse.ArgumentParser(description="Import Figma walkable masks")
    parser.add_argument("map_name", help="Map name (e.g., ascent, bind)")
    parser.add_argument("--input", "-i", help="Input PNG file path")
    parser.add_argument("--template", "-t", action="store_true",
                        help="Create a template for Figma drawing")

    args = parser.parse_args()

    if not HAS_PIL:
        print("ERROR: Pillow required. Run: pip install Pillow")
        return

    if args.template:
        create_template(args.map_name)
    else:
        input_path = Path(args.input) if args.input else None
        import_figma_mask(args.map_name, input_path)


if __name__ == '__main__':
    main()
