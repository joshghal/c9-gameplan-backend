#!/usr/bin/env python3
"""
Extract hold angles from VCT damage/kill events.

For each engagement, we know:
- Attacker position
- Victim position
- Therefore: angle attacker was facing

Aggregate by zone to get common hold angles for each map position.
Falls back to Henrik data if insufficient VCT samples for a zone.

Usage:
    python scripts/extract_hold_angles.py
"""

import json
import math
import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
GRID_DATA_DIR = PROJECT_ROOT / "grid_data"
HENRIK_DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "backend" / "app" / "data"

# Minimum samples needed per zone before falling back to Henrik
MIN_SAMPLES_THRESHOLD = 10


def load_henrik_zones() -> Dict[str, Dict[str, List[float]]]:
    """Load zone definitions from Henrik data."""
    zone_file = HENRIK_DATA_DIR / "processed" / "movement_graph.json"
    if not zone_file.exists():
        print(f"Warning: Henrik zone file not found at {zone_file}")
        return {}

    with open(zone_file) as f:
        data = json.load(f)

    zones = {}
    for map_name, map_data in data.get('maps', {}).items():
        zones[map_name] = {}
        for zone_name, zone_def in map_data.get('zone_definitions', {}).items():
            bounds = zone_def.get('bounds')
            if bounds:
                zones[map_name][zone_name] = bounds  # [x_min, x_max, y_min, y_max]

    return zones


def load_henrik_view_angles(zones_by_map: Dict) -> Dict[str, Dict[str, Dict]]:
    """Load view angle data from Henrik as fallback."""
    snapshots_file = HENRIK_DATA_DIR / "raw" / "all_maps_snapshots.json"
    if not snapshots_file.exists():
        print(f"Warning: Henrik snapshots not found at {snapshots_file}")
        return {}

    with open(snapshots_file) as f:
        data = json.load(f)

    # Aggregate view angles by map and zone
    henrik_angles = defaultdict(lambda: defaultdict(list))

    # Data is a list of kill snapshots
    for snapshot in data:
        map_name = snapshot.get('map_name', '').lower()
        if map_name not in zones_by_map:
            continue

        for player in snapshot.get('player_positions', []):
            view_radians = player.get('view_radians')
            x = player.get('x')
            y = player.get('y')

            if view_radians is not None and x is not None and y is not None:
                # Find zone for this position
                zone = get_zone_for_position(x, y, zones_by_map.get(map_name, {}))
                if zone:
                    henrik_angles[map_name][zone].append(view_radians)

    # Convert to statistics
    result = {}
    for map_name, zones in henrik_angles.items():
        result[map_name] = {}
        for zone, angles in zones.items():
            if angles:
                # Calculate circular mean
                sin_sum = sum(math.sin(a) for a in angles)
                cos_sum = sum(math.cos(a) for a in angles)
                mean_angle = math.atan2(sin_sum, cos_sum)

                result[map_name][zone] = {
                    'mean': mean_angle,
                    'samples': len(angles),
                }

    return result


def get_zone_for_position(x: float, y: float, zones: Dict[str, List[float]]) -> Optional[str]:
    """Find which zone a position belongs to."""
    for zone_name, bounds in zones.items():
        x_min, x_max, y_min, y_max = bounds
        if x_min <= x <= x_max and y_min <= y <= y_max:
            return zone_name
    return None


def calculate_angle(from_x: float, from_y: float, to_x: float, to_y: float) -> float:
    """Calculate angle from one point to another in radians."""
    return math.atan2(to_y - from_y, to_x - from_x)


def extract_vct_angles(zones_by_map: Dict) -> Dict[str, Dict[str, Dict]]:
    """Extract hold angles from VCT damage and kill events."""

    # Collect angles: map -> zone -> list of angles
    angle_data = defaultdict(lambda: defaultdict(list))

    # Track statistics
    stats = {
        'files_processed': 0,
        'damage_events': 0,
        'kill_events': 0,
        'angles_extracted': 0,
        'positions_without_zone': 0
    }

    jsonl_files = [f for f in os.listdir(GRID_DATA_DIR) if f.endswith('.jsonl')]

    for fname in jsonl_files:
        fpath = GRID_DATA_DIR / fname
        stats['files_processed'] += 1

        current_map = None

        with open(fpath, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)

                    for event in data.get('events', []):
                        etype = event.get('type', '')

                        # Get map name from game-started-round events
                        if etype == 'game-started-round':
                            actor = event.get('actor', {})
                            state = actor.get('state', {})
                            map_data = state.get('map', {})
                            if map_data.get('name'):
                                current_map = map_data['name'].lower()

                        # Extract angles from damage and kill events
                        if etype in ('player-damaged-player', 'player-killed-player'):
                            if etype == 'player-damaged-player':
                                stats['damage_events'] += 1
                            else:
                                stats['kill_events'] += 1

                            if not current_map or current_map not in zones_by_map:
                                continue

                            # Get attacker and victim positions
                            actor = event.get('actor', {})
                            target = event.get('target', {})

                            attacker_pos = actor.get('state', {}).get('game', {}).get('position')
                            victim_pos = target.get('state', {}).get('game', {}).get('position')

                            if not attacker_pos or not victim_pos:
                                continue

                            ax, ay = attacker_pos['x'], attacker_pos['y']
                            vx, vy = victim_pos['x'], victim_pos['y']

                            # Get zone for attacker position
                            zone = get_zone_for_position(ax, ay, zones_by_map[current_map])

                            if not zone:
                                stats['positions_without_zone'] += 1
                                continue

                            # Calculate angle attacker was facing
                            angle = calculate_angle(ax, ay, vx, vy)

                            # Calculate distance (for weighting close vs far engagements)
                            distance = math.sqrt((vx - ax) ** 2 + (vy - ay) ** 2)

                            angle_data[current_map][zone].append({
                                'angle': angle,
                                'distance': distance
                            })
                            stats['angles_extracted'] += 1

                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    continue

    print(f"\n=== VCT EXTRACTION STATS ===")
    print(f"Files processed: {stats['files_processed']}")
    print(f"Damage events: {stats['damage_events']}")
    print(f"Kill events: {stats['kill_events']}")
    print(f"Angles extracted: {stats['angles_extracted']}")
    print(f"Positions without zone match: {stats['positions_without_zone']}")

    # Convert to statistics
    result = {}
    for map_name, zones in angle_data.items():
        result[map_name] = {}
        for zone, angle_list in zones.items():
            angles = [a['angle'] for a in angle_list]
            distances = [a['distance'] for a in angle_list]

            if angles:
                # Calculate circular mean for angles
                sin_sum = sum(math.sin(a) for a in angles)
                cos_sum = sum(math.cos(a) for a in angles)
                mean_angle = math.atan2(sin_sum, cos_sum)

                # Calculate circular standard deviation
                r = math.sqrt(sin_sum**2 + cos_sum**2) / len(angles)
                std_angle = math.sqrt(-2 * math.log(r)) if r > 0 else math.pi

                result[map_name][zone] = {
                    'mean_angle': mean_angle,
                    'std_angle': min(std_angle, math.pi),  # Cap at pi
                    'samples': len(angles),
                    'avg_distance': sum(distances) / len(distances),
                    'source': 'vct'
                }

    return result


def merge_with_fallback(vct_angles: Dict, henrik_angles: Dict) -> Dict:
    """Merge VCT angles with Henrik fallback for zones with insufficient data."""

    result = defaultdict(dict)
    stats = {
        'vct_zones': 0,
        'henrik_fallback_zones': 0,
        'no_data_zones': 0
    }

    # Get all maps from both sources
    all_maps = set(vct_angles.keys()) | set(henrik_angles.keys())

    for map_name in all_maps:
        vct_zones = vct_angles.get(map_name, {})
        henrik_zones = henrik_angles.get(map_name, {})

        # Get all zones for this map
        all_zones = set(vct_zones.keys()) | set(henrik_zones.keys())

        for zone in all_zones:
            vct_data = vct_zones.get(zone)
            henrik_data = henrik_zones.get(zone)

            # Prefer VCT if we have enough samples
            if vct_data and vct_data.get('samples', 0) >= MIN_SAMPLES_THRESHOLD:
                result[map_name][zone] = vct_data
                stats['vct_zones'] += 1
            # Fallback to Henrik
            elif henrik_data:
                result[map_name][zone] = {
                    'mean_angle': henrik_data.get('mean', 0),
                    'std_angle': math.pi / 4,  # Default spread
                    'samples': henrik_data.get('samples', 0),
                    'source': 'henrik_fallback'
                }
                stats['henrik_fallback_zones'] += 1
            # Use VCT even with low samples if no Henrik
            elif vct_data:
                vct_data['source'] = 'vct_low_samples'
                result[map_name][zone] = vct_data
                stats['vct_zones'] += 1
            else:
                stats['no_data_zones'] += 1

    print(f"\n=== MERGE STATS ===")
    print(f"Zones using VCT data: {stats['vct_zones']}")
    print(f"Zones using Henrik fallback: {stats['henrik_fallback_zones']}")
    print(f"Zones with no data: {stats['no_data_zones']}")

    return dict(result)


def main():
    print("="*60)
    print("EXTRACTING HOLD ANGLES FROM VCT DATA")
    print("="*60)

    # Load Henrik zone definitions for position labeling
    print("\n1. Loading Henrik zone definitions...")
    zones_by_map = load_henrik_zones()
    print(f"   Loaded zones for {len(zones_by_map)} maps")

    # Load Henrik view angles as fallback
    print("\n2. Loading Henrik view angles (fallback)...")
    henrik_angles = load_henrik_view_angles(zones_by_map)
    print(f"   Loaded Henrik angles for {len(henrik_angles)} maps")

    # Extract angles from VCT data
    print("\n3. Extracting view angles from VCT events...")
    vct_angles = extract_vct_angles(zones_by_map)
    print(f"   Extracted VCT angles for {len(vct_angles)} maps")

    # Merge with fallback
    print("\n4. Merging VCT with Henrik fallback...")
    final_angles = merge_with_fallback(vct_angles, henrik_angles)

    # Save output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "hold_angles.json"

    output_data = {
        'metadata': {
            'description': 'Hold angles per zone, derived from VCT pro engagement data',
            'fallback': 'Henrik ranked data used when VCT samples < 10',
            'min_samples_threshold': MIN_SAMPLES_THRESHOLD,
        },
        'angles_by_map': final_angles
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n5. Saved to {output_file}")

    # Print sample output
    print("\n" + "="*60)
    print("SAMPLE OUTPUT")
    print("="*60)

    for map_name in list(final_angles.keys())[:2]:
        print(f"\n{map_name}:")
        zones = final_angles[map_name]
        for zone_name in list(zones.keys())[:3]:
            zone_data = zones[zone_name]
            angle_deg = math.degrees(zone_data['mean_angle'])
            source = zone_data.get('source', 'unknown')
            samples = zone_data.get('samples', 0)
            print(f"  {zone_name}: {angle_deg:.1f}° ({samples} samples, {source})")


if __name__ == "__main__":
    main()
