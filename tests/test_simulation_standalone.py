#!/usr/bin/env python3
"""Standalone simulation test - runs without database.

This script tests the simulation systems directly without needing
the full FastAPI application or database connection.

Usage:
    cd backend
    python3 tests/test_simulation_standalone.py
"""

import sys
import os
import random

# Add the backend directory to path
backend_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, backend_path)

# We need to import the service modules directly, bypassing __init__.py
# which tries to import simulation_engine with database dependencies

import importlib.util
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum

def import_module_direct(module_name, file_path):
    """Import a module directly from file, ignoring package structure."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)

    # We need to handle relative imports by pre-loading dependencies
    return spec, module

# Get paths
services_dir = os.path.join(backend_path, 'app', 'services')

# ============================================================================
# LOAD WEAPON SYSTEM
# ============================================================================
weapon_system_path = os.path.join(services_dir, 'weapon_system.py')
with open(weapon_system_path) as f:
    weapon_code = f.read()

# Execute weapon_system in isolation
weapon_ns = {'__name__': 'weapon_system', 'dataclass': dataclass, 'Enum': Enum}
exec(compile(weapon_code, weapon_system_path, 'exec'), weapon_ns)

WeaponCategory = weapon_ns['WeaponCategory']
WeaponStats = weapon_ns['WeaponStats']
ArmorStats = weapon_ns['ArmorStats']
WeaponDatabase = weapon_ns['WeaponDatabase']
meters_from_normalized = weapon_ns['meters_from_normalized']

# ============================================================================
# LOAD ECONOMY ENGINE
# ============================================================================
economy_path = os.path.join(services_dir, 'economy_engine.py')
with open(economy_path) as f:
    economy_code = f.read()

# Replace the relative import with our loaded module
economy_code = economy_code.replace(
    'from .weapon_system import WeaponDatabase, WeaponStats, ArmorStats, WeaponCategory',
    '# Imported externally'
)

economy_ns = {
    '__name__': 'economy_engine',
    'dataclass': dataclass,
    'field': __import__('dataclasses').field,
    'Enum': Enum,
    'Dict': Dict,
    'List': List,
    'Optional': Optional,
    'Tuple': Tuple,
    'random': random,
    'WeaponDatabase': WeaponDatabase,
    'WeaponStats': WeaponStats,
    'ArmorStats': ArmorStats,
    'WeaponCategory': WeaponCategory,
}
exec(compile(economy_code, economy_path, 'exec'), economy_ns)

BuyType = economy_ns['BuyType']
Loadout = economy_ns['Loadout']
TeamEconomy = economy_ns['TeamEconomy']
EconomyEngine = economy_ns['EconomyEngine']

# ============================================================================
# LOAD ROUND STATE
# ============================================================================
round_state_path = os.path.join(services_dir, 'round_state.py')
with open(round_state_path) as f:
    round_state_code = f.read()

round_state_ns = {
    '__name__': 'round_state',
    'dataclass': dataclass,
    'field': __import__('dataclasses').field,
    'Dict': Dict,
    'List': List,
    'Optional': Optional,
    'Tuple': Tuple,
    'datetime': __import__('datetime'),
}
exec(compile(round_state_code, round_state_path, 'exec'), round_state_ns)

RoundState = round_state_ns['RoundState']
WinProbabilityCalculator = round_state_ns['WinProbabilityCalculator']
KillEvent = round_state_ns['KillEvent']

# ============================================================================
# LOAD BEHAVIOR ADAPTATION
# ============================================================================
behavior_path = os.path.join(services_dir, 'behavior_adaptation.py')
with open(behavior_path) as f:
    behavior_code = f.read()

behavior_code = behavior_code.replace(
    'from .round_state import RoundState',
    '# Imported externally'
)

import math
behavior_ns = {
    '__name__': 'behavior_adaptation',
    'dataclass': dataclass,
    'Dict': Dict,
    'List': List,
    'Optional': Optional,
    'Tuple': Tuple,
    'math': math,
    'RoundState': RoundState,
}
exec(compile(behavior_code, behavior_path, 'exec'), behavior_ns)

BehaviorAdapter = behavior_ns['BehaviorAdapter']
BehaviorModifiers = behavior_ns['BehaviorModifiers']
PlayerTendencies = behavior_ns['PlayerTendencies']

# ============================================================================
# LOAD STRATEGY COORDINATOR
# ============================================================================
strategy_path = os.path.join(services_dir, 'strategy_coordinator.py')
with open(strategy_path) as f:
    strategy_code = f.read()

from typing import Set
strategy_ns = {
    '__name__': 'strategy_coordinator',
    'dataclass': dataclass,
    'field': __import__('dataclasses').field,
    'Dict': Dict,
    'List': List,
    'Optional': Optional,
    'Tuple': Tuple,
    'Set': Set,
    'Enum': Enum,
    'random': random,
    'math': math,
}
exec(compile(strategy_code, strategy_path, 'exec'), strategy_ns)

Role = strategy_ns['Role']
Strategy = strategy_ns['Strategy']
StrategyDatabase = strategy_ns['StrategyDatabase']
StrategyCoordinator = strategy_ns['StrategyCoordinator']

# ============================================================================
# LOAD ABILITY SYSTEM
# ============================================================================
ability_path = os.path.join(services_dir, 'ability_system.py')
with open(ability_path) as f:
    ability_code = f.read()

ability_ns = {
    '__name__': 'ability_system',
    'dataclass': dataclass,
    'field': __import__('dataclasses').field,
    'Dict': Dict,
    'List': List,
    'Optional': Optional,
    'Tuple': Tuple,
    'Set': Set,
    'Enum': Enum,
    'random': random,
    'math': math,
}
exec(compile(ability_code, ability_path, 'exec'), ability_ns)

AbilityCategory = ability_ns['AbilityCategory']
AbilityStats = ability_ns['AbilityStats']
AbilityDatabase = ability_ns['AbilityDatabase']
AbilitySystem = ability_ns['AbilitySystem']

# ============================================================================
# TEST FUNCTIONS
# ============================================================================

def test_weapon_system():
    """Test weapon damage and kill probability."""
    print("\n" + "="*60)
    print("WEAPON SYSTEM TEST")
    print("="*60)

    # Test damage calculation
    vandal = WeaponDatabase.WEAPONS['vandal']
    phantom = WeaponDatabase.WEAPONS['phantom']
    heavy_armor = WeaponDatabase.ARMOR['heavy']

    print(f"\nVandal vs Heavy Armor (20m):")
    for region in ['head', 'body', 'leg']:
        health_dmg, shield_dmg, _ = WeaponDatabase.calculate_damage(
            vandal, 20.0, region, heavy_armor, 100, 50
        )
        total = health_dmg + shield_dmg
        print(f"  {region}: {total} damage ({health_dmg} HP, {shield_dmg} shield)")

    # Test kill probability
    print(f"\nKill Probability Tests:")

    scenarios = [
        ("Vandal vs Vandal (20m, equal skill)", vandal, vandal, 20, 0.25, 0.25),
        ("Vandal vs Classic (20m)", vandal, WeaponDatabase.WEAPONS['classic'], 20, 0.25, 0.25),
        ("Operator vs Vandal (40m)", WeaponDatabase.WEAPONS['operator'], vandal, 40, 0.8, 0.25),
        ("Sheriff vs Phantom (15m)", WeaponDatabase.WEAPONS['sheriff'], phantom, 15, 0.35, 0.25),
    ]

    for name, atk_wpn, def_wpn, dist, atk_hs, def_hs in scenarios:
        atk_prob, def_prob = WeaponDatabase.calculate_kill_probability(
            atk_wpn, def_wpn, float(dist), atk_hs, def_hs,
            heavy_armor, heavy_armor, 100, 100, 50, 50
        )
        print(f"  {name}")
        print(f"    Attacker ({atk_wpn.name}): {atk_prob:.1%}")
        print(f"    Defender ({def_wpn.name}): {def_prob:.1%}")


def test_economy_system():
    """Test economy and loadout generation."""
    print("\n" + "="*60)
    print("ECONOMY SYSTEM TEST")
    print("="*60)

    buy_types = ['pistol', 'eco', 'force', 'half', 'full']

    for bt in buy_types:
        if bt == 'pistol':
            economy = TeamEconomy(credits=[800] * 5)
            forced = BuyType.PISTOL
        elif bt == 'eco':
            economy = TeamEconomy(credits=[2000] * 5, loss_streak=1)
            forced = BuyType.ECO
        elif bt == 'force':
            economy = TeamEconomy(credits=[3000] * 5, loss_streak=2)
            forced = BuyType.FORCE
        elif bt == 'half':
            economy = TeamEconomy(credits=[4000] * 5)
            forced = BuyType.HALF
        else:
            economy = TeamEconomy(credits=[5000] * 5)
            forced = BuyType.FULL

        loadouts = EconomyEngine.generate_team_loadout(
            economy, round_num=5, side='attack', forced_buy_type=forced
        )

        print(f"\n{bt.upper()} BUY (avg {economy.average_credits} credits):")
        total_value = 0
        for i, loadout in enumerate(loadouts):
            print(f"  Player {i+1}: {loadout.weapon.name} + {loadout.armor.name} (${loadout.total_value})")
            total_value += loadout.total_value
        print(f"  Team loadout value: ${total_value}")


def test_round_state_and_win_probability():
    """Test round state tracking and win probability."""
    print("\n" + "="*60)
    print("ROUND STATE & WIN PROBABILITY TEST")
    print("="*60)

    # Simulate a round with kills
    round_state = RoundState(attack_buy_type='full', defense_buy_type='full')

    print("\nInitial state (5v5):")
    prob = WinProbabilityCalculator.calculate_win_probability(round_state, 0)
    print(f"  Attack: {prob['attack']:.1%}, Defense: {prob['defense']:.1%}")

    # First blood for attack
    round_state.record_kill(
        time_ms=15000,
        killer_id='atk_1',
        killer_team='attack',
        victim_id='def_1',
        victim_team='defense',
        position=(0.5, 0.5),
        weapon='Vandal',
        is_headshot=True
    )

    print("\nAfter first blood (attack gets FB, 5v4):")
    prob = WinProbabilityCalculator.calculate_win_probability(round_state, 15000)
    print(f"  Attack: {prob['attack']:.1%}, Defense: {prob['defense']:.1%}")

    # Trade kill
    round_state.record_kill(
        time_ms=18000,
        killer_id='def_2',
        killer_team='defense',
        victim_id='atk_1',
        victim_team='attack',
        position=(0.5, 0.5),
        weapon='Phantom'
    )

    print("\nAfter trade (4v4):")
    prob = WinProbabilityCalculator.calculate_win_probability(round_state, 18000)
    print(f"  Attack: {prob['attack']:.1%}, Defense: {prob['defense']:.1%}")
    print(f"  Trade detected: {round_state.kills[-1].is_trade}")

    # Spike planted
    round_state.plant_spike(35000, 'A')

    print("\nAfter spike plant (4v4, post-plant):")
    prob = WinProbabilityCalculator.calculate_win_probability(round_state, 35000)
    print(f"  Attack: {prob['attack']:.1%}, Defense: {prob['defense']:.1%}")

    # More kills favor attack
    round_state.record_kill(40000, 'atk_2', 'attack', 'def_3', 'defense', (0.3, 0.3), 'Vandal')
    round_state.record_kill(42000, 'atk_3', 'attack', 'def_4', 'defense', (0.3, 0.3), 'Phantom')

    print("\nLate round (4v2, post-plant):")
    prob = WinProbabilityCalculator.calculate_win_probability(round_state, 42000)
    print(f"  Attack: {prob['attack']:.1%}, Defense: {prob['defense']:.1%}")


def test_behavior_adaptation():
    """Test behavior adaptation based on round state."""
    print("\n" + "="*60)
    print("BEHAVIOR ADAPTATION TEST")
    print("="*60)

    tendencies = PlayerTendencies(
        base_aggression=0.5,
        clutch_factor=0.6,
        trade_awareness=0.7
    )

    # Scenario 1: Even round
    round_state = RoundState()
    modifiers = BehaviorAdapter.calculate_behavior_modifiers(
        'player_1', 'attack', round_state, 20000, tendencies
    )
    print(f"\nEven round (5v5):")
    print(f"  Aggression: {modifiers.aggression:.2f}")
    print(f"  Movement speed: {modifiers.movement_speed:.2f}")
    print(f"  Peek willingness: {modifiers.peek_willingness:.2f}")

    # Scenario 2: Man advantage
    round_state.attack_alive = 5
    round_state.defense_alive = 3
    modifiers = BehaviorAdapter.calculate_behavior_modifiers(
        'player_1', 'attack', round_state, 30000, tendencies
    )
    print(f"\nMan advantage (5v3, attack):")
    print(f"  Aggression: {modifiers.aggression:.2f} (should be higher)")
    print(f"  Movement speed: {modifiers.movement_speed:.2f}")

    # Scenario 3: Man disadvantage
    modifiers = BehaviorAdapter.calculate_behavior_modifiers(
        'player_1', 'defense', round_state, 30000, tendencies
    )
    print(f"\nMan disadvantage (5v3, defense):")
    print(f"  Aggression: {modifiers.aggression:.2f} (should be lower)")

    # Scenario 4: Post-plant attack (should hold)
    round_state.spike_planted = True
    round_state.spike_plant_time_ms = 35000
    modifiers = BehaviorAdapter.calculate_behavior_modifiers(
        'player_1', 'attack', round_state, 40000, tendencies
    )
    print(f"\nPost-plant (attack side):")
    print(f"  Aggression: {modifiers.aggression:.2f} (should be passive)")
    print(f"  Movement speed: {modifiers.movement_speed:.2f} (should be slow)")

    # Scenario 5: Post-plant defense (should push)
    modifiers = BehaviorAdapter.calculate_behavior_modifiers(
        'player_1', 'defense', round_state, 40000, tendencies
    )
    print(f"\nPost-plant (defense side):")
    print(f"  Aggression: {modifiers.aggression:.2f} (should be aggressive)")
    print(f"  Rotation urgency: {modifiers.rotation_urgency:.2f}")


def test_strategy_coordinator():
    """Test strategy selection and role assignment."""
    print("\n" + "="*60)
    print("STRATEGY COORDINATOR TEST")
    print("="*60)

    coordinator = StrategyCoordinator()

    # Test strategy selection
    print("\nAttack strategy selection (5 rounds):")
    for i in range(5):
        strategy = coordinator.select_strategy(
            team_id='c9',
            map_name='ascent',
            side='attack',
            round_type='full',
            team_credits=25000,
            round_number=i
        )
        print(f"  Round {i+1}: {strategy.name}")

    # Test role assignment
    print("\nRole assignment for A Execute:")
    strategy = StrategyDatabase.ATTACK_STRATEGIES['a_execute']
    players = [
        {'player_id': 'p1', 'agent': 'jett'},
        {'player_id': 'p2', 'agent': 'sova'},
        {'player_id': 'p3', 'agent': 'omen'},
        {'player_id': 'p4', 'agent': 'killjoy'},
        {'player_id': 'p5', 'agent': 'skye'},
    ]

    coordinator.current_strategy = strategy
    assignments = coordinator.assign_roles(players, strategy)

    for player_id, assignment in assignments.items():
        agent = next(p['agent'] for p in players if p['player_id'] == player_id)
        print(f"  {agent}: {assignment.role.value}")

    # Test waypoint generation (verify variance)
    print("\nWaypoint variance test (same role, 3 iterations):")
    for i in range(3):
        assignment = coordinator.assign_roles(players[:1], strategy)['p1']
        if assignment.waypoints:
            wp = assignment.waypoints[0]
            print(f"  Iteration {i+1}: position=({wp.position[0]:.3f}, {wp.position[1]:.3f}), time={wp.time_window_ms}")


def test_ability_system():
    """Test ability database and effects."""
    print("\n" + "="*60)
    print("ABILITY SYSTEM TEST")
    print("="*60)

    # Test ability database
    print("\nAgent abilities:")
    agents_to_test = ['jett', 'sova', 'omen', 'sage']
    for agent in agents_to_test:
        abilities = AbilityDatabase.get_agent_abilities(agent)
        print(f"\n  {agent.capitalize()}:")
        for ability_id, ability in abilities.items():
            print(f"    - {ability.name} ({ability.category.value})")

    # Test ability system
    print("\nAbility system initialization:")
    ability_system = AbilitySystem()
    ability_system.initialize_player('player_1', 'omen')

    state = ability_system.player_states.get('player_1')
    print(f"  Agent: {state.agent}")
    print(f"  Available abilities: {list(state.available_abilities.keys())}")

    # Test smoke detection
    print("\nSmoke vision blocking test:")

    # Use a smoke ability
    omen_smoke = AbilityDatabase.get_ability('omen', 'dark_cover')
    ability_system.use_ability(
        player_id='player_1',
        ability=omen_smoke,
        position=(0.5, 0.5),
        time_ms=5000,
        team='attack'
    )

    # Check if position is smoked
    is_smoked = ability_system.is_position_smoked((0.5, 0.5), 6000)
    print(f"  Position (0.5, 0.5) at 6000ms: {'SMOKED' if is_smoked else 'clear'}")

    is_smoked = ability_system.is_position_smoked((0.8, 0.8), 6000)
    print(f"  Position (0.8, 0.8) at 6000ms: {'SMOKED' if is_smoked else 'clear'}")


def run_mini_simulation():
    """Run a mini simulation to test combat resolution."""
    print("\n" + "="*60)
    print("MINI SIMULATION TEST")
    print("="*60)

    # Setup
    round_state = RoundState(attack_buy_type='full', defense_buy_type='full')

    # Create players
    attackers = [
        {'id': f'atk_{i}', 'agent': agent, 'weapon': 'vandal', 'armor': 'heavy'}
        for i, agent in enumerate(['jett', 'sova', 'omen', 'killjoy', 'skye'])
    ]
    defenders = [
        {'id': f'def_{i}', 'agent': agent, 'weapon': 'vandal', 'armor': 'heavy'}
        for i, agent in enumerate(['chamber', 'fade', 'viper', 'cypher', 'kayo'])
    ]

    print("\nInitial setup:")
    print(f"  Attackers: {[p['agent'] for p in attackers]}")
    print(f"  Defenders: {[p['agent'] for p in defenders]}")

    # Simulate some engagements
    print("\nSimulating engagements...")
    time_ms = 15000

    vandal = WeaponDatabase.WEAPONS['vandal']
    heavy = WeaponDatabase.ARMOR['heavy']

    kills = []
    alive_atk = list(attackers)
    alive_def = list(defenders)

    while len(alive_atk) > 0 and len(alive_def) > 0 and time_ms < 100000:
        # Random engagement
        if random.random() < 0.1:  # 10% chance per tick
            atk = random.choice(alive_atk)
            defender = random.choice(alive_def)

            # Calculate kill probability
            atk_prob, def_prob = WeaponDatabase.calculate_kill_probability(
                vandal, vandal, 20.0, 0.25, 0.25,
                heavy, heavy, 100, 100, 50, 50
            )

            # Roll for outcome
            if random.random() < atk_prob / (atk_prob + def_prob):
                # Attacker wins
                kills.append({
                    'time': time_ms,
                    'killer': atk['id'],
                    'victim': defender['id']
                })
                alive_def.remove(defender)
                round_state.record_kill(
                    time_ms, atk['id'], 'attack',
                    defender['id'], 'defense',
                    (0.5, 0.5), 'Vandal'
                )
            else:
                # Defender wins
                kills.append({
                    'time': time_ms,
                    'killer': defender['id'],
                    'victim': atk['id']
                })
                alive_atk.remove(atk)
                round_state.record_kill(
                    time_ms, defender['id'], 'defense',
                    atk['id'], 'attack',
                    (0.5, 0.5), 'Vandal'
                )

            # Show win probability
            prob = WinProbabilityCalculator.calculate_win_probability(round_state, time_ms)
            print(f"  {time_ms}ms: {kills[-1]['killer']} killed {kills[-1]['victim']}")
            print(f"           ({len(alive_atk)}v{len(alive_def)}) Win prob: Atk {prob['attack']:.0%} / Def {prob['defense']:.0%}")

        time_ms += 128

    # Determine winner
    winner = 'attack' if len(alive_def) == 0 else 'defense'
    print(f"\nResult: {winner.upper()} wins!")
    print(f"  Total kills: {len(kills)}")
    print(f"  First blood: {round_state.first_blood_team}")
    print(f"  Trades: {sum(1 for k in round_state.kills if k.is_trade)}")


def main():
    """Run all tests."""
    print("\n" + "#"*60)
    print("# C9 TACTICAL VISION - SIMULATION SYSTEM TESTS")
    print("#"*60)

    test_weapon_system()
    test_economy_system()
    test_round_state_and_win_probability()
    test_behavior_adaptation()
    test_strategy_coordinator()
    test_ability_system()
    run_mini_simulation()

    print("\n" + "="*60)
    print("ALL TESTS COMPLETED")
    print("="*60)


if __name__ == "__main__":
    main()
