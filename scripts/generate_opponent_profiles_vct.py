"""
Generate Opponent Profiles from VCT Data

Creates per-player profiles similar to C9's:
- opponent_opening_setups.json - Position clusters per player/map/side
- opponent_distance_preferences.json - Engagement distances per player
- opponent_movement_models.json - KDE heatmaps per player/map/side

Uses extracted_player_positions.json from GRID VCT data.
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

# Paths
DATA_DIR = Path(__file__).parent.parent / "app" / "data"
POSITIONS_FILE = DATA_DIR / "extracted_player_positions.json"

# C9 roster to exclude
C9_ROSTER = {"oxy", "v1c", "xeppaa", "net", "mitch"}

# Minimum samples for reliable statistics
MIN_SAMPLES = 30


def load_positions() -> Dict:
    """Load extracted player positions."""
    with open(POSITIONS_FILE) as f:
        return json.load(f)


def filter_opponents(data: Dict) -> Dict:
    """Filter to non-C9 players only."""
    return {
        player: maps_data
        for player, maps_data in data.items()
        if player.lower() not in C9_ROSTER
    }


def compute_cluster_stats(positions: List[dict]) -> dict:
    """Compute position cluster statistics."""
    if not positions:
        return None

    # Extract coordinates (use raw x,y for consistency with C9 format)
    xs = [p['x'] for p in positions]
    ys = [p['y'] for p in positions]

    return {
        'x': np.mean(xs),
        'y': np.mean(ys),
        'weight': 1.0,
        'std_x': np.std(xs) if len(xs) > 1 else 500,
        'std_y': np.std(ys) if len(ys) > 1 else 500,
    }


def generate_opening_setups(data: Dict) -> Dict:
    """Generate opening setups similar to C9's format."""
    setups = {}

    for player, maps_data in data.items():
        player_setups = {}

        for map_name, sides_data in maps_data.items():
            if not map_name:  # Skip empty map names
                continue

            map_setups = {}

            for side, positions in sides_data.items():
                if len(positions) < MIN_SAMPLES:
                    continue

                cluster = compute_cluster_stats(positions)
                if cluster:
                    map_setups[side] = {
                        'player': player,
                        'map_name': map_name,
                        'side': side,
                        'n_samples': len(positions),
                        'positions': [cluster],
                        'confidence': min(0.95, 0.5 + len(positions) / 200),
                        'method': 'simple'
                    }

            if map_setups:
                player_setups[map_name] = map_setups

        if player_setups:
            setups[player] = player_setups

    return {
        'metadata': {
            'description': 'Opponent opening positions extracted from VCT data',
            'methodology': 'Mean/std clustering from GRID position data',
            'min_samples': MIN_SAMPLES,
            'n_players': len(setups),
        },
        'setups': setups
    }


def generate_distance_preferences(data: Dict) -> Dict:
    """Generate distance preferences (estimated from position variance)."""
    preferences = {}

    for player, maps_data in data.items():
        all_positions = []

        for map_name, sides_data in maps_data.items():
            if not map_name:
                continue
            for side, positions in sides_data.items():
                all_positions.extend(positions)

        if len(all_positions) < MIN_SAMPLES:
            continue

        # Estimate engagement distance from position spread
        # Players with tighter positions prefer closer engagements
        xs = [p['x'] for p in all_positions]
        ys = [p['y'] for p in all_positions]

        # Use position variance as proxy for engagement distance preference
        spread = np.sqrt(np.var(xs) + np.var(ys))

        # Map spread to distance (typical range 1400-2200 game units)
        mean_distance = 1400 + min(800, spread / 5)
        std_distance = mean_distance * 0.15

        preferences[player] = {
            'player': player,
            'n_samples': len(all_positions),
            'mean_distance': round(mean_distance, 1),
            'std_distance': round(std_distance, 1),
            'ci_lower': round(mean_distance - 1.96 * std_distance, 1),
            'ci_upper': round(mean_distance + 1.96 * std_distance, 1),
        }

    return {
        'metadata': {
            'description': 'Opponent engagement distance preferences from VCT data',
            'methodology': 'Estimated from position variance',
            'n_players': len(preferences),
        },
        'preferences': preferences
    }


def generate_movement_models(data: Dict, grid_resolution: int = 50) -> Dict:
    """Generate KDE-style movement heatmaps per player/map/side."""
    models = {}

    for player, maps_data in data.items():
        player_models = {}

        for map_name, sides_data in maps_data.items():
            if not map_name:
                continue

            map_models = {}

            for side, positions in sides_data.items():
                if len(positions) < MIN_SAMPLES:
                    continue

                # Build heatmap from minimap coordinates
                heatmap = np.zeros((grid_resolution, grid_resolution))

                for pos in positions:
                    mx = pos.get('mx', 0.5)
                    my = pos.get('my', 0.5)

                    # Clamp to valid range
                    mx = max(0, min(0.99, mx))
                    my = max(0, min(0.99, my))

                    gx = int(mx * grid_resolution)
                    gy = int(my * grid_resolution)

                    heatmap[gy, gx] += 1

                # Normalize
                if heatmap.sum() > 0:
                    heatmap = heatmap / heatmap.sum()

                map_models[side] = {
                    'all': {
                        'heatmap': heatmap.tolist(),
                        'grid_resolution': grid_resolution,
                        'n_samples': len(positions),
                        'confidence': min(0.95, 0.5 + len(positions) / 200),
                    }
                }

            if map_models:
                player_models[map_name] = map_models

        if player_models:
            models[player] = player_models

    return {
        'metadata': {
            'description': 'Opponent movement models (KDE heatmaps) from VCT data',
            'methodology': 'Position histogram normalized to probability',
            'grid_resolution': grid_resolution,
            'n_players': len(models),
        },
        'models': models
    }


def main():
    print("Loading extracted positions...")
    data = load_positions()
    print(f"Loaded {len(data)} players")

    print("Filtering to opponents...")
    opponent_data = filter_opponents(data)
    print(f"Found {len(opponent_data)} opponent players")

    # Generate opening setups
    print("\nGenerating opening setups...")
    setups = generate_opening_setups(opponent_data)
    setups_path = DATA_DIR / "opponent_opening_setups.json"
    with open(setups_path, 'w') as f:
        json.dump(setups, f, indent=2)
    print(f"Saved {len(setups['setups'])} players to {setups_path}")

    # Generate distance preferences
    print("\nGenerating distance preferences...")
    prefs = generate_distance_preferences(opponent_data)
    prefs_path = DATA_DIR / "opponent_distance_preferences.json"
    with open(prefs_path, 'w') as f:
        json.dump(prefs, f, indent=2)
    print(f"Saved {len(prefs['preferences'])} players to {prefs_path}")

    # Generate movement models
    print("\nGenerating movement models...")
    models = generate_movement_models(opponent_data)
    models_path = DATA_DIR / "opponent_movement_models.json"
    with open(models_path, 'w') as f:
        json.dump(models, f, indent=2)
    print(f"Saved {len(models['models'])} players to {models_path}")

    # Summary
    print("\n" + "="*60)
    print("OPPONENT PROFILE GENERATION COMPLETE")
    print("="*60)
    print(f"Opening setups:      {len(setups['setups'])} players")
    print(f"Distance preferences: {len(prefs['preferences'])} players")
    print(f"Movement models:      {len(models['models'])} players")

    # Show sample players
    print("\nSample players with data:")
    for player in list(setups['setups'].keys())[:10]:
        maps = list(setups['setups'][player].keys())
        print(f"  {player}: {', '.join(maps)}")


if __name__ == "__main__":
    main()
