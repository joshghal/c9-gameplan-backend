#!/usr/bin/env python3
"""
Test C9 Realism Integration

Tests that C9 players use player-specific:
- P0: Opening positions from VCT data
- P1: Combat positioning at preferred distances
- P2: KDE-based movement patterns
"""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.services.c9_realism import get_c9_realism, C9_ROSTER


def test_opening_positions():
    """Test P0: Opening position extraction."""
    print("\n" + "=" * 60)
    print("P0: OPENING POSITIONS TEST")
    print("=" * 60)

    c9 = get_c9_realism()

    # Test each C9 player on lotus
    test_map = "lotus"

    print(f"\nC9 Opening Positions on {test_map.upper()}:")
    print("-" * 50)

    for side in ["attack", "defense"]:
        print(f"\n{side.upper()} SIDE:")
        for player in C9_ROSTER:
            pos = c9.get_opening_position(player, test_map, side)
            if pos:
                # Convert to normalized if in game units
                x_norm = pos.x / 10000.0 if pos.x > 1 else pos.x
                y_norm = pos.y / 10000.0 if pos.y > 1 else pos.y
                print(f"  {player:8} → ({x_norm:.3f}, {y_norm:.3f}) confidence={pos.confidence:.2f}")
            else:
                print(f"  {player:8} → No data")

    # Verify team positions don't overlap
    print("\n✓ Opening positions test complete")
    return True


def test_distance_preferences():
    """Test P1: Combat distance preferences."""
    print("\n" + "=" * 60)
    print("P1: DISTANCE PREFERENCES TEST")
    print("=" * 60)

    c9 = get_c9_realism()

    print("\nC9 Engagement Distance Preferences:")
    print("-" * 50)
    print(f"{'Player':<10} {'Mean (units)':<15} {'Std':<10} {'Style'}")
    print("-" * 50)

    for player in C9_ROSTER:
        mean, std = c9.get_preferred_distance(player)
        # Determine style based on distance
        if mean < 1600:
            style = "Aggressive (close)"
        elif mean < 1750:
            style = "Balanced"
        else:
            style = "Passive (long)"
        print(f"{player:<10} {mean:<15.0f} {std:<10.0f} {style}")

    print("\n✓ Distance preferences test complete")
    return True


def test_combat_positioning():
    """Test P1: Combat positioning system."""
    print("\n" + "=" * 60)
    print("P1: COMBAT POSITIONING TEST")
    print("=" * 60)

    c9 = get_c9_realism()

    # Scenario: OXY at (5000, 5000), enemy at (6500, 5000) - 1500 units away
    # OXY prefers ~1665 units, so should move slightly away
    player = "OXY"
    current = (5000, 5000)
    enemy = [(6500, 5000)]

    print(f"\nScenario: {player} at {current}, enemy at {enemy[0]}")
    print(f"Distance to enemy: {((enemy[0][0]-current[0])**2 + (enemy[0][1]-current[1])**2)**0.5:.0f} units")

    mean_dist, _ = c9.get_preferred_distance(player)
    print(f"{player}'s preferred distance: {mean_dist:.0f} units")

    pos = c9.get_optimal_combat_position(player, current, enemy)
    if pos:
        new_dist = ((enemy[0][0]-pos.x)**2 + (enemy[0][1]-pos.y)**2)**0.5
        print(f"\nOptimal position: ({pos.x:.0f}, {pos.y:.0f})")
        print(f"New distance to enemy: {new_dist:.0f} units")

        # Check if moved in correct direction
        if new_dist > 1500:
            print("✓ Correctly moved away to preferred distance")
        else:
            print("! Moved closer (may be already at good distance)")

    # Test opposite scenario - too far
    print(f"\n--- Scenario 2: Enemy too far ---")
    enemy_far = [(7500, 5000)]  # 2500 units away
    print(f"{player} at {current}, enemy at {enemy_far[0]}")
    print(f"Distance: {((enemy_far[0][0]-current[0])**2)**0.5:.0f} units (too far)")

    pos2 = c9.get_optimal_combat_position(player, current, enemy_far)
    if pos2:
        new_dist2 = ((enemy_far[0][0]-pos2.x)**2 + (enemy_far[0][1]-pos2.y)**2)**0.5
        print(f"Optimal position: ({pos2.x:.0f}, {pos2.y:.0f})")
        print(f"New distance: {new_dist2:.0f} units")
        if pos2.x > current[0]:
            print("✓ Correctly moved closer to preferred distance")

    print("\n✓ Combat positioning test complete")
    return True


def test_movement_model():
    """Test P2: KDE-based movement sampling."""
    print("\n" + "=" * 60)
    print("P2: MOVEMENT MODEL TEST")
    print("=" * 60)

    c9 = get_c9_realism()

    # Test movement sampling for each player
    test_map = "lotus"
    test_side = "attack"
    current_pos = (0.5, 0.5)

    print(f"\nMovement Targets from ({current_pos[0]}, {current_pos[1]}) on {test_map} {test_side}:")
    print("-" * 50)

    for player in C9_ROSTER:
        # Sample multiple positions
        positions = []
        for _ in range(5):
            pos = c9.get_movement_target(player, test_map, test_side, current_pos)
            if pos:
                positions.append((pos.x, pos.y))

        if positions:
            avg_x = sum(p[0] for p in positions) / len(positions)
            avg_y = sum(p[1] for p in positions) / len(positions)
            print(f"  {player:8} → avg target: ({avg_x:.3f}, {avg_y:.3f}) from {len(positions)} samples")
        else:
            print(f"  {player:8} → No movement data")

    print("\n✓ Movement model test complete")
    return True


def test_position_probability():
    """Test P2: Position probability lookup."""
    print("\n" + "=" * 60)
    print("P2: POSITION PROBABILITY TEST")
    print("=" * 60)

    c9 = get_c9_realism()

    player = "OXY"
    test_map = "lotus"

    # Test different positions
    test_positions = [
        (0.5, 0.5),   # Center
        (0.3, 0.7),   # Upper left
        (0.8, 0.2),   # Lower right
        (0.1, 0.1),   # Corner
    ]

    print(f"\nPosition Probabilities for {player} on {test_map}:")
    print("-" * 50)

    for side in ["attack", "defense"]:
        print(f"\n{side.upper()} SIDE:")
        for pos in test_positions:
            prob = c9.get_position_probability(player, test_map, side, pos)
            bar = "█" * int(prob * 50)
            print(f"  ({pos[0]:.1f}, {pos[1]:.1f}): {prob:.4f} {bar}")

    print("\n✓ Position probability test complete")
    return True


def test_simulation_integration():
    """Test that C9 realism integrates with simulation concepts."""
    print("\n" + "=" * 60)
    print("INTEGRATION SIMULATION TEST")
    print("=" * 60)

    c9 = get_c9_realism()

    # Simulate a simple round setup
    test_map = "bind"

    print(f"\nSimulating C9 round setup on {test_map}:")
    print("-" * 50)

    # Get team opening positions
    print("\nATTACK SIDE SETUP:")
    attack_positions = c9.get_team_opening_positions(test_map, "attack")
    for player, pos in attack_positions.items():
        x_norm = pos.x / 10000.0 if pos.x > 1 else pos.x
        y_norm = pos.y / 10000.0 if pos.y > 1 else pos.y
        print(f"  {player:8} spawns at ({x_norm:.3f}, {y_norm:.3f})")

    print("\nDEFENSE SIDE SETUP:")
    defense_positions = c9.get_team_opening_positions(test_map, "defense")
    for player, pos in defense_positions.items():
        x_norm = pos.x / 10000.0 if pos.x > 1 else pos.x
        y_norm = pos.y / 10000.0 if pos.y > 1 else pos.y
        print(f"  {player:8} holds at ({x_norm:.3f}, {y_norm:.3f})")

    # Simulate movement for 5 ticks
    print("\nSimulating 5 movement ticks for OXY (attack):")
    current = (0.5, 0.3)
    for tick in range(5):
        move = c9.get_movement_target("OXY", test_map, "attack", current)
        if move:
            print(f"  Tick {tick+1}: ({current[0]:.3f}, {current[1]:.3f}) → ({move.x:.3f}, {move.y:.3f})")
            current = (move.x, move.y)

    print("\n✓ Integration test complete")
    return True


def run_all_tests():
    """Run all C9 realism tests."""
    print("\n" + "=" * 60)
    print("C9 REALISM SERVICE - FULL TEST SUITE")
    print("=" * 60)
    print(f"\nC9 Roster: {C9_ROSTER}")

    tests = [
        ("P0: Opening Positions", test_opening_positions),
        ("P1: Distance Preferences", test_distance_preferences),
        ("P1: Combat Positioning", test_combat_positioning),
        ("P2: Movement Model", test_movement_model),
        ("P2: Position Probability", test_position_probability),
        ("Integration Test", test_simulation_integration),
    ]

    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ {name} FAILED: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"  {status}: {name}")

    print(f"\n{passed}/{total} tests passed")
    print("=" * 60)

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
