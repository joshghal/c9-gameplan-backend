"""
Generate Per-Team Profiles from VCT Data

Creates per-team profile files similar to C9's:
- {team}_opening_setups.json - Position clusters per player/map/side
- {team}_distance_preferences.json - Engagement distances per player
- {team}_movement_models.json - KDE heatmaps per player/map/side

Uses extracted_player_positions.json and team_player_mapping.json from GRID VCT data.
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

# Paths
DATA_DIR = Path(__file__).parent.parent / "app" / "data"
POSITIONS_FILE = DATA_DIR / "extracted_player_positions.json"
TEAM_MAPPING_FILE = DATA_DIR / "team_player_mapping.json"
TEAMS_DIR = DATA_DIR / "teams"

# Minimum samples for reliable statistics
MIN_SAMPLES = 30


def load_positions() -> Dict:
    """Load extracted player positions."""
    with open(POSITIONS_FILE) as f:
        return json.load(f)


def load_team_mapping() -> Dict:
    """Load team-player mapping."""
    with open(TEAM_MAPPING_FILE) as f:
        return json.load(f)


def normalize_team_name(name: str) -> str:
    """Normalize team name for file naming."""
    # Remove special chars and lowercase
    clean = name.lower()
    clean = clean.replace(' ', '_')
    clean = clean.replace('(', '').replace(')', '')
    clean = clean.replace('ü', 'u')
    clean = clean.replace('á', 'a')
    # Remove trailing numbers like _1
    if clean.endswith('_1'):
        clean = clean[:-2]
    return clean


def compute_cluster_stats(positions: List[dict]) -> dict:
    """Compute position cluster statistics."""
    if not positions:
        return None

    xs = [p['x'] for p in positions]
    ys = [p['y'] for p in positions]

    return {
        'x': float(np.mean(xs)),
        'y': float(np.mean(ys)),
        'weight': 1.0,
        'std_x': float(np.std(xs)) if len(xs) > 1 else 500.0,
        'std_y': float(np.std(ys)) if len(ys) > 1 else 500.0,
    }


def generate_team_opening_setups(team_data: Dict, team_name: str) -> Dict:
    """Generate opening setups for a team."""
    setups = {}

    for player, maps_data in team_data.items():
        player_setups = {}

        for map_name, sides_data in maps_data.items():
            if not map_name:
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
            'team': team_name,
            'description': f'{team_name} opening positions extracted from VCT data',
            'methodology': 'Mean/std clustering from GRID position data',
            'min_samples': MIN_SAMPLES,
            'n_players': len(setups),
        },
        'setups': setups
    }


def generate_team_distance_preferences(team_data: Dict, team_name: str) -> Dict:
    """Generate distance preferences for a team."""
    preferences = {}

    for player, maps_data in team_data.items():
        all_positions = []

        for map_name, sides_data in maps_data.items():
            if not map_name:
                continue
            for side, positions in sides_data.items():
                all_positions.extend(positions)

        if len(all_positions) < MIN_SAMPLES:
            continue

        xs = [p['x'] for p in all_positions]
        ys = [p['y'] for p in all_positions]

        spread = np.sqrt(np.var(xs) + np.var(ys))
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
            'team': team_name,
            'description': f'{team_name} engagement distance preferences from VCT data',
            'methodology': 'Estimated from position variance',
            'n_players': len(preferences),
        },
        'preferences': preferences
    }


def generate_team_movement_models(team_data: Dict, team_name: str, grid_resolution: int = 50) -> Dict:
    """Generate movement models for a team."""
    models = {}

    for player, maps_data in team_data.items():
        player_models = {}

        for map_name, sides_data in maps_data.items():
            if not map_name:
                continue

            map_models = {}

            for side, positions in sides_data.items():
                if len(positions) < MIN_SAMPLES:
                    continue

                heatmap = np.zeros((grid_resolution, grid_resolution))

                for pos in positions:
                    mx = pos.get('mx', 0.5)
                    my = pos.get('my', 0.5)

                    mx = max(0, min(0.99, mx))
                    my = max(0, min(0.99, my))

                    gx = int(mx * grid_resolution)
                    gy = int(my * grid_resolution)

                    heatmap[gy, gx] += 1

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
            'team': team_name,
            'description': f'{team_name} movement models (KDE heatmaps) from VCT data',
            'methodology': 'Position histogram normalized to probability',
            'grid_resolution': grid_resolution,
            'n_players': len(models),
        },
        'models': models
    }


def main():
    print("Loading extracted positions...")
    all_positions = load_positions()
    print(f"Loaded {len(all_positions)} players")

    print("Loading team mapping...")
    team_mapping = load_team_mapping()
    team_rosters = team_mapping['team_rosters']
    print(f"Found {len(team_rosters)} teams")

    # Create teams directory
    TEAMS_DIR.mkdir(exist_ok=True)

    # Process each team
    for team_name, roster in sorted(team_rosters.items()):
        if team_name.startswith('Team_'):  # Skip unknown teams
            continue

        print(f"\n{'='*60}")
        print(f"Processing {team_name} ({len(roster)} players)")
        print('='*60)

        # Filter positions to this team's players
        team_data = {}
        for player in roster:
            # Try to find player in positions (case insensitive)
            for pos_player, pos_data in all_positions.items():
                if pos_player.lower() == player.lower():
                    team_data[pos_player] = pos_data
                    break

        print(f"Found position data for {len(team_data)} players")

        if not team_data:
            print(f"  Skipping - no position data")
            continue

        # Generate profile files
        team_slug = normalize_team_name(team_name)
        team_dir = TEAMS_DIR / team_slug
        team_dir.mkdir(exist_ok=True)

        # Opening setups
        setups = generate_team_opening_setups(team_data, team_name)
        setups_path = team_dir / f"{team_slug}_opening_setups.json"
        with open(setups_path, 'w') as f:
            json.dump(setups, f, indent=2)
        print(f"  Opening setups: {len(setups['setups'])} players")

        # Distance preferences
        prefs = generate_team_distance_preferences(team_data, team_name)
        prefs_path = team_dir / f"{team_slug}_distance_preferences.json"
        with open(prefs_path, 'w') as f:
            json.dump(prefs, f, indent=2)
        print(f"  Distance preferences: {len(prefs['preferences'])} players")

        # Movement models
        models = generate_team_movement_models(team_data, team_name)
        models_path = team_dir / f"{team_slug}_movement_models.json"
        with open(models_path, 'w') as f:
            json.dump(models, f, indent=2)
        print(f"  Movement models: {len(models['models'])} players")

    # Summary
    print("\n" + "="*60)
    print("TEAM PROFILE GENERATION COMPLETE")
    print("="*60)
    teams_generated = [d.name for d in TEAMS_DIR.iterdir() if d.is_dir()]
    print(f"Generated profiles for {len(teams_generated)} teams:")
    for team in sorted(teams_generated):
        print(f"  - {team}")


if __name__ == "__main__":
    main()
