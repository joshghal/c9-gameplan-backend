#!/usr/bin/env python3
"""
Generate Walkable Masks from Map Images + VCT Data

Combines two approaches:
1. Map geometry from map image colors (accurate boundaries)
2. VCT position data for activity weighting (where players actually go)

Usage:
    cd backend
    source venv/bin/activate
    python scripts/generate_map_masks.py
    python scripts/generate_map_masks.py --preview
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from PIL import Image
    from scipy import ndimage
    from scipy.ndimage import gaussian_filter
    HAS_DEPS = True
except ImportError as e:
    HAS_DEPS = False
    print(f"Missing dependency: {e}")
    sys.exit(1)

# Paths
MAP_IMAGES_DIR = Path(__file__).parent.parent.parent / "map_images"
VCT_DATA_DIR = Path(__file__).parent.parent.parent / "vct_map_data"
MASK_JSON_PATH = Path(__file__).parent.parent / "app" / "data" / "figma_masks" / "walkable_masks_v4.json"
OUTPUT_DIR = Path(__file__).parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# All maps
MAPS = ['abyss', 'ascent', 'bind', 'breeze', 'fracture',
        'haven', 'icebox', 'lotus', 'pearl', 'split', 'sunset']

# Grid size for final masks
GRID_SIZE = 150

# Map image color thresholds
# Walkable areas are typically gray (not white background, not too dark)
MIN_BRIGHTNESS = 20    # Exclude very dark pixels
MAX_BRIGHTNESS = 220   # Exclude white/very bright pixels


def load_map_image(map_name: str) -> Optional[np.ndarray]:
    """Load map image."""
    path = MAP_IMAGES_DIR / f"{map_name}.png"
    if path.exists():
        return np.array(Image.open(path).convert('RGB'))
    print(f"  Warning: Map image not found: {path}")
    return None


def load_vct_dense(map_name: str) -> Optional[np.ndarray]:
    """Load VCT dense position heatmap (right side of combined_dense)."""
    path = VCT_DATA_DIR / f"combined_dense_{map_name}.png"
    if path.exists():
        img = Image.open(path)
        # Get right half (heatmap)
        w = img.width // 2
        heatmap = img.crop((w, 0, img.width, img.height))
        return np.array(heatmap)
    return None


def extract_walkable_from_map(map_arr: np.ndarray) -> np.ndarray:
    """Extract walkable areas from map image colors.

    Walkable areas are the non-white, non-black regions of the map.
    """
    # Convert to grayscale
    if len(map_arr.shape) == 3:
        gray = np.mean(map_arr, axis=2)
    else:
        gray = map_arr.astype(float)

    # Walkable = medium brightness (not white background, not too dark)
    walkable = (gray > MIN_BRIGHTNESS) & (gray < MAX_BRIGHTNESS)

    return walkable.astype(np.uint8)


def extract_vct_activity(vct_arr: np.ndarray, map_shape: Tuple[int, int]) -> np.ndarray:
    """Extract VCT activity as a weight map.

    Returns normalized activity map (0-1) resized to map dimensions.
    """
    # Get brightness from heatmap (red channel is typically highest in heatmaps)
    if len(vct_arr.shape) == 3:
        # Red channel for heatmap intensity
        activity = vct_arr[:, :, 0].astype(float)
    else:
        activity = vct_arr.astype(float)

    # Normalize to 0-1
    if activity.max() > 0:
        activity = activity / activity.max()

    # Resize to match map
    activity_img = Image.fromarray((activity * 255).astype(np.uint8))
    activity_resized = activity_img.resize((map_shape[1], map_shape[0]), Image.BILINEAR)

    return np.array(activity_resized).astype(float) / 255.0


def combine_map_and_vct(map_walkable: np.ndarray, vct_activity: np.ndarray) -> np.ndarray:
    """Combine map geometry with VCT activity weighting.

    The map provides accurate boundaries, VCT provides activity weighting.
    Final walkable = map_walkable (binary geometry)
    Activity weighting can be used for other purposes (engagement probability)
    """
    # For the walkable mask, we use map geometry as ground truth
    # VCT activity is informational but doesn't change walkability

    # However, we can use VCT to validate - areas with high VCT activity
    # that aren't marked walkable might indicate issues

    # For now, just use map geometry
    return map_walkable


def cleanup_mask(mask: np.ndarray) -> np.ndarray:
    """Apply morphological cleanup to remove noise."""
    struct = ndimage.generate_binary_structure(2, 2)

    # Remove small isolated regions
    cleaned = ndimage.binary_opening(mask, struct, iterations=2)

    # Fill small holes
    cleaned = ndimage.binary_closing(cleaned, struct, iterations=2)

    return cleaned.astype(np.uint8)


def resize_mask(mask: np.ndarray, target_size: int = GRID_SIZE) -> np.ndarray:
    """Resize mask to target grid size."""
    img = Image.fromarray(mask * 255)
    img = img.resize((target_size, target_size), Image.NEAREST)
    return (np.array(img) > 128).astype(np.uint8)


def generate_mask_for_map(map_name: str, preview: bool = False) -> Optional[Dict]:
    """Generate walkable mask for a single map."""
    print(f"Processing {map_name}...")

    # Load map image
    map_arr = load_map_image(map_name)
    if map_arr is None:
        return None

    print(f"  Map image: {map_arr.shape}")

    # Extract walkable from map
    map_walkable = extract_walkable_from_map(map_arr)
    map_pct = map_walkable.sum() / map_walkable.size * 100
    print(f"  Map walkable: {map_pct:.1f}%")

    # Load VCT activity (optional)
    vct_arr = load_vct_dense(map_name)
    if vct_arr is not None:
        print(f"  VCT heatmap: {vct_arr.shape}")
        vct_activity = extract_vct_activity(vct_arr, map_arr.shape[:2])
        vct_coverage = (vct_activity > 0.1).sum() / vct_activity.size * 100
        print(f"  VCT coverage: {vct_coverage:.1f}%")
    else:
        vct_activity = None
        print("  VCT data: Not found")

    # Combine (for now, just use map geometry)
    combined = combine_map_and_vct(map_walkable, vct_activity) if vct_activity is not None else map_walkable

    # Cleanup
    cleaned = cleanup_mask(combined)
    cleaned_pct = cleaned.sum() / cleaned.size * 100
    print(f"  After cleanup: {cleaned_pct:.1f}%")

    # Resize to grid
    final_mask = resize_mask(cleaned, GRID_SIZE)
    final_pct = final_mask.sum() / final_mask.size * 100
    print(f"  Final ({GRID_SIZE}x{GRID_SIZE}): {final_pct:.1f}%")

    # Save preview
    if preview:
        # Create overlay visualization
        map_rgba = Image.open(MAP_IMAGES_DIR / f"{map_name}.png").convert('RGBA')
        overlay = np.zeros((map_arr.shape[0], map_arr.shape[1], 4), dtype=np.uint8)
        overlay[cleaned == 1] = [0, 255, 0, 80]

        overlay_img = Image.fromarray(overlay, 'RGBA')
        result = Image.alpha_composite(map_rgba, overlay_img)

        preview_path = OUTPUT_DIR / f"map_mask_preview_{map_name}.png"
        result.save(preview_path)
        print(f"  Preview: {preview_path}")

    # Create obstacle mask
    obstacle_mask = 1 - final_mask
    void_mask = np.zeros_like(final_mask)

    return {
        'walkable_mask': final_mask.tolist(),
        'obstacle_mask': obstacle_mask.tolist(),
        'void_mask': void_mask.tolist(),
        'walkable_cells': int(final_mask.sum()),
        'stats': {
            'walkable_percent': round(final_pct, 1),
            'grid_size': GRID_SIZE,
            'source': 'map_image',
        },
        'fixes': []
    }


def main():
    parser = argparse.ArgumentParser(description="Generate walkable masks from map images")
    parser.add_argument('--preview', action='store_true', help='Generate preview images')
    parser.add_argument('--dry-run', action='store_true', help="Don't save to JSON")
    parser.add_argument('--maps', nargs='+', default=MAPS, help='Specific maps to process')
    args = parser.parse_args()

    print("=" * 60)
    print("GENERATE WALKABLE MASKS FROM MAP IMAGES")
    print("=" * 60)
    print(f"Map images: {MAP_IMAGES_DIR}")
    print(f"VCT data: {VCT_DATA_DIR}")
    print(f"Output: {MASK_JSON_PATH}")
    print()

    # Load existing JSON
    if MASK_JSON_PATH.exists():
        with open(MASK_JSON_PATH, 'r') as f:
            json_data = json.load(f)
    else:
        json_data = {'maps': {}}

    # Process maps
    results = {}
    for map_name in args.maps:
        mask_data = generate_mask_for_map(map_name, preview=args.preview)
        if mask_data:
            results[map_name] = mask_data
        print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Map':<12} {'Old %':<10} {'New %':<10} {'Status'}")
    print("-" * 44)

    for map_name in args.maps:
        old_pct = "N/A"
        if map_name in json_data.get('maps', {}):
            old_mask = np.array(json_data['maps'][map_name].get('walkable_mask', []))
            if old_mask.size > 0:
                old_pct = f"{old_mask.sum() / old_mask.size * 100:.1f}%"

        if map_name in results:
            new_pct = f"{results[map_name]['stats']['walkable_percent']:.1f}%"
            status = "Updated"
        else:
            new_pct = "N/A"
            status = "Failed"

        print(f"{map_name:<12} {old_pct:<10} {new_pct:<10} {status}")

    # Save
    if not args.dry_run and results:
        json_data['maps'].update(results)
        json_data['metadata'] = {
            'generated_from': 'map_images',
            'grid_size': GRID_SIZE,
        }

        MASK_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(MASK_JSON_PATH, 'w') as f:
            json.dump(json_data, f)

        print()
        print(f"Saved: {MASK_JSON_PATH}")


if __name__ == '__main__':
    main()
