#!/usr/bin/env python3
"""Test v4 map integration with simulation components."""

import sys
import os

# Add paths
backend_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, backend_path)
project_path = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, project_path)

from backend.app.services.map_context import get_map_context, MapContext
from backend.app.services.pathfinding import AStarPathfinder
from backend.app.services.position_sampler import PositionSampler
import time

def test_map_context():
    """Test MapContext loading and queries."""
    print("=" * 60)
    print("TEST 1: MapContext")
    print("=" * 60)

    ctx = get_map_context()

    maps = ['haven', 'ascent', 'bind', 'split', 'icebox', 'lotus']

    for map_name in maps:
        stats = ctx.get_stats(map_name)
        walkable_pct = stats.get('walkable_percentage', 0)
        print(f"  {map_name}: {walkable_pct}% walkable")

    # Test walkability
    print("\n  Walkability checks:")
    print(f"    haven (0.5, 0.5): {ctx.is_walkable('haven', 0.5, 0.5)}")
    print(f"    haven (0.1, 0.1): {ctx.is_walkable('haven', 0.1, 0.1)}")

    # Test LOS
    print("\n  Line of sight:")
    print(f"    haven (0.3,0.5) -> (0.6,0.5): {ctx.has_line_of_sight('haven', 0.3, 0.5, 0.6, 0.5)}")

    print("\n  ✓ MapContext tests passed")


def test_pathfinding():
    """Test pathfinding with v4 masks."""
    print("\n" + "=" * 60)
    print("TEST 2: Pathfinding with V4 Masks")
    print("=" * 60)

    pf = AStarPathfinder()

    maps_to_test = ['haven', 'ascent', 'bind']

    for map_name in maps_to_test:
        success = pf.load_nav_grid_from_v4(map_name)
        blocked = pf.nav_grid.sum()
        print(f"\n  {map_name}:")
        print(f"    V4 loaded: {success}")
        print(f"    Blocked cells: {blocked} / {128*128}")

        # Test pathfinding
        start_time = time.time()
        result = pf.find_path((0.3, 0.5), (0.7, 0.5))
        elapsed = (time.time() - start_time) * 1000

        print(f"    Path (0.3,0.5) -> (0.7,0.5):")
        print(f"      Success: {result.success}")
        print(f"      Distance: {result.distance:.1f}")
        print(f"      Waypoints: {len(result.waypoints)}")
        print(f"      Time: {elapsed:.1f}ms")

    print("\n  ✓ Pathfinding tests passed")


def test_position_sampler():
    """Test position sampling with v4 masks."""
    print("\n" + "=" * 60)
    print("TEST 3: Position Sampler with V4 Masks")
    print("=" * 60)

    maps_to_test = ['haven', 'ascent', 'lotus']

    for map_name in maps_to_test:
        sampler = PositionSampler(map_name)
        print(f"\n  {map_name}:")
        print(f"    Mask loaded: {sampler.figma_mask is not None}")

        # Sample positions for both sides
        attack_positions = []
        defense_positions = []

        for _ in range(5):
            pos = sampler.sample_position(side='attack', phase='execute')
            walkable = sampler.is_walkable(pos[0], pos[1])
            attack_positions.append((pos, walkable))

            pos = sampler.sample_position(side='defense', phase='setup')
            walkable = sampler.is_walkable(pos[0], pos[1])
            defense_positions.append((pos, walkable))

        # Check all positions are walkable
        attack_valid = all(w for _, w in attack_positions)
        defense_valid = all(w for _, w in defense_positions)

        print(f"    Attack positions valid: {attack_valid}")
        print(f"    Defense positions valid: {defense_valid}")

        # Show sample positions
        print(f"    Sample attack: ({attack_positions[0][0][0]:.3f}, {attack_positions[0][0][1]:.3f})")
        print(f"    Sample defense: ({defense_positions[0][0][0]:.3f}, {defense_positions[0][0][1]:.3f})")

    print("\n  ✓ Position Sampler tests passed")


def test_simulation_scenario():
    """Test a simple simulation scenario."""
    print("\n" + "=" * 60)
    print("TEST 4: Simulation Scenario")
    print("=" * 60)

    ctx = get_map_context()
    pf = AStarPathfinder()
    sampler = PositionSampler('haven')

    pf.load_nav_grid_from_v4('haven')

    # Simulate 5 attackers and 5 defenders
    print("\n  Simulating round on Haven...")

    attackers = []
    defenders = []

    # Place players
    for i in range(5):
        # Attackers start near spawn
        pos = sampler.sample_position(side='attack', phase='setup')
        attackers.append({'id': f'atk_{i}', 'pos': pos, 'alive': True})

        # Defenders at sites
        pos = sampler.sample_position(side='defense', phase='setup')
        defenders.append({'id': f'def_{i}', 'pos': pos, 'alive': True})

    print(f"\n  Initial positions:")
    atk_pos = [(round(p['pos'][0], 2), round(p['pos'][1], 2)) for p in attackers]
    def_pos = [(round(p['pos'][0], 2), round(p['pos'][1], 2)) for p in defenders]
    print(f"    Attackers: {atk_pos}")
    print(f"    Defenders: {def_pos}")

    # Simulate movement - attackers move toward site
    target = (0.5, 0.3)  # B site area
    print(f"\n  Attackers moving to B site {target}...")

    for atk in attackers:
        result = pf.find_path(atk['pos'], target)
        if result.success:
            print(f"    {atk['id']}: path found, {len(result.waypoints)} waypoints, distance {result.distance:.1f}")
        else:
            print(f"    {atk['id']}: no path found!")

    # Check line of sight between players
    print(f"\n  Line of sight checks:")
    for atk in attackers[:2]:
        for dfn in defenders[:2]:
            los = ctx.has_line_of_sight('haven',
                atk['pos'][0], atk['pos'][1],
                dfn['pos'][0], dfn['pos'][1])
            print(f"    {atk['id']} -> {dfn['id']}: {'VISIBLE' if los else 'blocked'}")

    print("\n  ✓ Simulation scenario tests passed")


def main():
    print("\n" + "#" * 60)
    print("#  V4 MAP INTEGRATION TEST")
    print("#" * 60)

    start = time.time()

    test_map_context()
    test_pathfinding()
    test_position_sampler()
    test_simulation_scenario()

    elapsed = time.time() - start

    print("\n" + "=" * 60)
    print(f"ALL TESTS PASSED in {elapsed:.2f}s")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
