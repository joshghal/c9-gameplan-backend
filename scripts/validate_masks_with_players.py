#!/usr/bin/env python3
"""
Validate v4 walkable masks against actual player position data.
Identifies areas where players cross obstacle cells (potential elevation/jump spots).
"""

import json
import numpy as np
from PIL import Image
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
MAP_DATA_DIR = PROJECT_ROOT / "vct_map_data"
MASK_FILE = PROJECT_ROOT / "backend" / "app" / "data" / "figma_masks" / "walkable_masks_v4.json"

GRID_SIZE = 150

MAPS = [
    "abyss", "ascent", "bind", "breeze", "corrode",
    "fracture", "haven", "icebox", "lotus", "pearl", "split", "sunset"
]

def load_masks():
    """Load v4 walkable masks."""
    with open(MASK_FILE, 'r') as f:
        data = json.load(f)
        return data.get('maps', data)  # Handle both structures

def analyze_player_crossings(map_name, masks):
    """Analyze where player positions cross obstacle cells."""
    combined_path = MAP_DATA_DIR / f"combined_dense_{map_name}.png"

    if not combined_path.exists():
        print(f"  Skipping {map_name} - no combined_dense image")
        return None

    # Load combined dense image (has blue player trails)
    img = Image.open(combined_path)
    img_array = np.array(img.convert('RGB'))
    h, w = img_array.shape[:2]

    # Get mask data
    map_mask = masks.get(map_name)
    if not map_mask:
        print(f"  Skipping {map_name} - no mask data")
        return None

    walkable = np.array(map_mask['walkable_mask'])
    obstacle = np.array(map_mask['obstacle_mask'])

    # Detect blue pixels (player trails)
    r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]

    # Blue pixels: B > 150, B > R + 50, B > G + 30
    blue_mask = (b > 150) & (b > r + 50) & (b > g + 30)

    # Ignore top 15% of image (contains text labels, not player data)
    top_cutoff = int(h * 0.15)
    blue_mask[:top_cutoff, :] = False

    # Count blue pixels in each grid cell
    crossing_cells = []

    for gy in range(GRID_SIZE):
        for gx in range(GRID_SIZE):
            y1 = int(gy * h / GRID_SIZE)
            y2 = int((gy + 1) * h / GRID_SIZE)
            x1 = int(gx * w / GRID_SIZE)
            x2 = int((gx + 1) * w / GRID_SIZE)

            cell_blue = blue_mask[y1:y2, x1:x2]
            blue_count = cell_blue.sum()

            # If cell is marked as obstacle but has significant player presence
            if obstacle[gy, gx] == 1 and blue_count > 10:
                crossing_cells.append({
                    'grid_x': gx,
                    'grid_y': gy,
                    'blue_pixels': int(blue_count),
                    'pixel_x': (x1 + x2) // 2,
                    'pixel_y': (y1 + y2) // 2
                })

    return {
        'total_blue_pixels': int(blue_mask.sum()),
        'obstacle_crossings': len(crossing_cells),
        'crossings': sorted(crossing_cells, key=lambda x: -x['blue_pixels'])[:20]  # Top 20
    }

def main():
    print("Loading v4 masks...")
    masks = load_masks()

    print("\nValidating masks against player movement data:")
    print("=" * 60)

    results = {}

    for map_name in MAPS:
        print(f"\nAnalyzing {map_name}...")
        result = analyze_player_crossings(map_name, masks)

        if result:
            results[map_name] = result
            print(f"  Total blue (player) pixels: {result['total_blue_pixels']:,}")
            print(f"  Obstacle cells with player crossings: {result['obstacle_crossings']}")

            if result['crossings']:
                print(f"  Top crossings (potential elevation/jump spots):")
                for i, cross in enumerate(result['crossings'][:5]):
                    print(f"    {i+1}. Grid ({cross['grid_x']}, {cross['grid_y']}) - {cross['blue_pixels']} blue pixels")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY - Maps with most obstacle crossings (potential elevation issues):")
    print("-" * 60)

    sorted_maps = sorted(
        [(m, r['obstacle_crossings']) for m, r in results.items()],
        key=lambda x: -x[1]
    )

    for map_name, crossings in sorted_maps:
        if crossings > 0:
            print(f"  {map_name}: {crossings} cells where players cross obstacles")

    # Save detailed results
    output_file = MAP_DATA_DIR / "mask_validation_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to: {output_file}")

if __name__ == "__main__":
    main()
