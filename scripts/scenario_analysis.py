#!/usr/bin/env python3
"""
Comprehensive Scenario Analysis for C9 Tactical Vision

Tests the simulation model across ALL game scenarios:
1. Attack Execute (fast/slow)
2. Post-Plant Defense
3. Retake
4. Mid Control Battle
5. Defender Aggression
6. Eco Round
7. Man Advantage (4v5, 3v5, etc.)
8. Clutch Situations (1v1, 1v2, etc.)

Identifies strengths and weaknesses of the model.
"""

import asyncio
import json
import sys
import math
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from enum import Enum

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.simulation_engine import SimulationEngine, SimulatedPlayer
from app.services.weapon_system import WeaponDatabase
from app.services.ability_system import AbilityCategory
from app.services.pathfinding import AStarPathfinder


class Scenario(Enum):
    ATTACK_EXECUTE_FAST = "attack_execute_fast"
    ATTACK_EXECUTE_SLOW = "attack_execute_slow"
    POST_PLANT_DEFENSE = "post_plant_defense"
    RETAKE = "retake"
    MID_CONTROL = "mid_control"
    DEFENDER_AGGRESSION = "defender_aggression"
    ECO_VS_FULL = "eco_vs_full"
    FULL_VS_ECO = "full_vs_eco"
    CLUTCH_1V1 = "clutch_1v1"
    CLUTCH_1V2 = "clutch_1v2"
    CLUTCH_2V1 = "clutch_2v1"
    MAN_ADVANTAGE_4V5 = "man_advantage_4v5"
    MAN_ADVANTAGE_5V4 = "man_advantage_5v4"
    MAN_ADVANTAGE_3V5 = "man_advantage_3v5"


# VCT Reference data for each scenario
VCT_REFERENCE = {
    Scenario.ATTACK_EXECUTE_FAST: {
        'attack_win_rate': 0.52,  # Fast executes favor attackers
        'plant_rate': 0.75,
        'avg_duration_ms': 45000,
    },
    Scenario.ATTACK_EXECUTE_SLOW: {
        'attack_win_rate': 0.45,  # Slow defaults favor defense
        'plant_rate': 0.60,
        'avg_duration_ms': 70000,
    },
    Scenario.POST_PLANT_DEFENSE: {
        'attack_win_rate': 0.65,  # Post-plant heavily favors attackers
        'description': 'Attackers defending planted spike',
    },
    Scenario.RETAKE: {
        'attack_win_rate': 0.65,  # Same as above (defense retaking)
        'defense_win_rate': 0.35,
        'description': 'Defenders retaking site',
    },
    Scenario.MID_CONTROL: {
        'attack_win_rate': 0.48,  # Roughly even
        'description': 'Fighting for mid control',
    },
    Scenario.DEFENDER_AGGRESSION: {
        'attack_win_rate': 0.55,  # Aggressive defense is risky
        'description': 'Defenders pushing for picks',
    },
    Scenario.ECO_VS_FULL: {
        'attack_win_rate': 0.15,  # Eco rarely wins
        'description': 'Eco attackers vs full buy defense',
    },
    Scenario.FULL_VS_ECO: {
        'attack_win_rate': 0.85,  # Full buy dominates
        'description': 'Full buy attackers vs eco defense',
    },
    Scenario.CLUTCH_1V1: {
        'attack_win_rate': 0.50,  # Even in 1v1
        'description': '1v1 clutch situation',
    },
    Scenario.CLUTCH_1V2: {
        'attack_win_rate': 0.25,  # Hard for solo player
        'description': '1 attacker vs 2 defenders',
    },
    Scenario.CLUTCH_2V1: {
        'attack_win_rate': 0.75,  # 2v1 favors the duo
        'description': '2 attackers vs 1 defender',
    },
    Scenario.MAN_ADVANTAGE_4V5: {
        'attack_win_rate': 0.35,  # Down a player
        'description': '4 attackers vs 5 defenders',
    },
    Scenario.MAN_ADVANTAGE_5V4: {
        'attack_win_rate': 0.60,  # Up a player
        'description': '5 attackers vs 4 defenders',
    },
    Scenario.MAN_ADVANTAGE_3V5: {
        'attack_win_rate': 0.15,  # Very hard
        'description': '3 attackers vs 5 defenders',
    },
}


@dataclass
class ScenarioResult:
    scenario: Scenario
    total_rounds: int
    attack_wins: int
    defense_wins: int
    plants: int
    avg_duration_ms: float
    avg_kills: float
    avg_trades: float
    first_blood_attack_rate: float

    @property
    def attack_win_rate(self) -> float:
        return self.attack_wins / max(1, self.total_rounds)

    @property
    def plant_rate(self) -> float:
        return self.plants / max(1, self.total_rounds)


class MockDB:
    async def execute(self, *args, **kwargs): return None
    async def commit(self): pass
    async def rollback(self): pass


class ScenarioAnalyzer:
    """Analyzes simulation performance across different scenarios."""

    MAPS = ['ascent', 'bind', 'haven', 'split', 'icebox', 'breeze', 'pearl']
    AGENTS_ATTACK = ['jett', 'raze', 'omen', 'sova', 'killjoy']
    AGENTS_DEFENSE = ['chamber', 'cypher', 'viper', 'fade', 'sage']

    def __init__(self):
        self.pathfinder = AStarPathfinder()
        self.results: Dict[Scenario, ScenarioResult] = {}

    async def run_all_scenarios(self, rounds_per_scenario: int = 50) -> Dict[Scenario, ScenarioResult]:
        """Run all scenarios and collect results."""
        print(f"\n{'='*70}")
        print("COMPREHENSIVE SCENARIO ANALYSIS")
        print(f"{'='*70}")
        print(f"Testing {len(Scenario)} scenarios with {rounds_per_scenario} rounds each")
        print(f"Total simulations: {len(Scenario) * rounds_per_scenario}")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        for scenario in Scenario:
            print(f"Testing {scenario.value}...", end=" ", flush=True)
            result = await self._run_scenario(scenario, rounds_per_scenario)
            self.results[scenario] = result

            ref = VCT_REFERENCE.get(scenario, {})
            target_win = ref.get('attack_win_rate', 0.5) * 100
            actual_win = result.attack_win_rate * 100
            diff = actual_win - target_win
            status = "✓" if abs(diff) < 15 else "✗"

            print(f"{status} Attack Win: {actual_win:.0f}% (target: {target_win:.0f}%, diff: {diff:+.0f}%)")

        return self.results

    async def _run_scenario(self, scenario: Scenario, num_rounds: int) -> ScenarioResult:
        """Run a specific scenario multiple times."""
        attack_wins = 0
        defense_wins = 0
        plants = 0
        total_duration = 0
        total_kills = 0
        total_trades = 0
        fb_attack = 0
        fb_total = 0

        rounds_per_map = max(1, num_rounds // len(self.MAPS))

        for map_name in self.MAPS:
            self.pathfinder.load_nav_grid(map_name)

            for _ in range(rounds_per_map):
                result = await self._simulate_scenario(scenario, map_name)
                if result:
                    if result['winner'] == 'attack':
                        attack_wins += 1
                    else:
                        defense_wins += 1
                    if result['planted']:
                        plants += 1
                    total_duration += result['duration_ms']
                    total_kills += result['kills']
                    total_trades += result['trades']
                    if result['first_blood_side']:
                        fb_total += 1
                        if result['first_blood_side'] == 'attack':
                            fb_attack += 1

        total = rounds_per_map * len(self.MAPS)

        return ScenarioResult(
            scenario=scenario,
            total_rounds=total,
            attack_wins=attack_wins,
            defense_wins=defense_wins,
            plants=plants,
            avg_duration_ms=total_duration / max(1, total),
            avg_kills=total_kills / max(1, total),
            avg_trades=total_trades / max(1, total),
            first_blood_attack_rate=fb_attack / max(1, fb_total),
        )

    async def _simulate_scenario(self, scenario: Scenario, map_name: str) -> Optional[Dict]:
        """Simulate a single round for a specific scenario."""
        try:
            db = MockDB()
            engine = SimulationEngine(db)
            engine.map_name = map_name
            map_data = engine.MAP_DATA.get(map_name, engine.DEFAULT_MAP_DATA)

            # Setup based on scenario
            config = self._get_scenario_config(scenario, map_data)

            # Initialize players
            self._setup_players(engine, map_data, config)

            # Run simulation with scenario-specific behavior
            return await self._run_scenario_loop(engine, map_data, scenario, config)

        except Exception as e:
            return None

    def _get_scenario_config(self, scenario: Scenario, map_data: Dict) -> Dict:
        """Get configuration for a specific scenario."""
        config = {
            'num_attackers': 5,
            'num_defenders': 5,
            'attacker_weapons': ['vandal'] * 5,
            'defender_weapons': ['vandal'] * 5,
            'attacker_shields': [50] * 5,
            'defender_shields': [50] * 5,
            'spike_planted': False,
            'spike_site': None,
            'start_time_ms': 0,
            'max_time_ms': 100000,
            'attacker_behavior': 'push',  # push, hold, aggressive
            'defender_behavior': 'hold',  # hold, retake, aggressive
            'attacker_speed': 'normal',   # fast, normal, slow
        }

        if scenario == Scenario.ATTACK_EXECUTE_FAST:
            config['attacker_speed'] = 'fast'
            config['attacker_behavior'] = 'push'

        elif scenario == Scenario.ATTACK_EXECUTE_SLOW:
            config['attacker_speed'] = 'slow'
            config['attacker_behavior'] = 'push'
            config['start_time_ms'] = 30000  # Start later

        elif scenario == Scenario.POST_PLANT_DEFENSE:
            config['spike_planted'] = True
            config['spike_site'] = random.choice(list(map_data['sites'].keys()))
            config['attacker_behavior'] = 'hold'
            config['defender_behavior'] = 'retake'
            config['start_time_ms'] = 60000

        elif scenario == Scenario.RETAKE:
            config['spike_planted'] = True
            config['spike_site'] = random.choice(list(map_data['sites'].keys()))
            config['attacker_behavior'] = 'hold'
            config['defender_behavior'] = 'retake'
            config['start_time_ms'] = 65000

        elif scenario == Scenario.MID_CONTROL:
            config['attacker_behavior'] = 'mid_control'
            config['defender_behavior'] = 'mid_control'

        elif scenario == Scenario.DEFENDER_AGGRESSION:
            config['defender_behavior'] = 'aggressive'

        elif scenario == Scenario.ECO_VS_FULL:
            config['attacker_weapons'] = ['classic'] * 5
            config['attacker_shields'] = [0] * 5

        elif scenario == Scenario.FULL_VS_ECO:
            config['defender_weapons'] = ['classic'] * 5
            config['defender_shields'] = [0] * 5

        elif scenario == Scenario.CLUTCH_1V1:
            config['num_attackers'] = 1
            config['num_defenders'] = 1
            config['start_time_ms'] = 70000

        elif scenario == Scenario.CLUTCH_1V2:
            config['num_attackers'] = 1
            config['num_defenders'] = 2
            config['start_time_ms'] = 70000

        elif scenario == Scenario.CLUTCH_2V1:
            config['num_attackers'] = 2
            config['num_defenders'] = 1
            config['start_time_ms'] = 70000

        elif scenario == Scenario.MAN_ADVANTAGE_4V5:
            config['num_attackers'] = 4

        elif scenario == Scenario.MAN_ADVANTAGE_5V4:
            config['num_defenders'] = 4

        elif scenario == Scenario.MAN_ADVANTAGE_3V5:
            config['num_attackers'] = 3

        return config

    def _setup_players(self, engine: SimulationEngine, map_data: Dict, config: Dict):
        """Setup players based on scenario configuration."""
        attack_spawns = map_data['spawns']['attack']
        defense_spawns = map_data['spawns']['defense']

        # Attackers
        for i in range(config['num_attackers']):
            player_id = f"attack_{i}"
            spawn = attack_spawns[i % len(attack_spawns)]

            # For post-plant, position attackers on site
            if config['spike_planted'] and config['spike_site']:
                site_data = map_data['sites'][config['spike_site']]
                spawn = (
                    site_data['center'][0] + random.uniform(-0.05, 0.05),
                    site_data['center'][1] + random.uniform(-0.05, 0.05)
                )

            player = SimulatedPlayer(
                player_id=player_id,
                team_id="attack_team",
                side="attack",
                x=spawn[0],
                y=spawn[1],
                agent=self.AGENTS_ATTACK[i % len(self.AGENTS_ATTACK)],
                has_spike=(i == 0 and not config['spike_planted'])
            )
            player.weapon = WeaponDatabase.get_weapon(config['attacker_weapons'][i % len(config['attacker_weapons'])])
            player.shield = config['attacker_shields'][i % len(config['attacker_shields'])]
            engine.players[player_id] = player
            engine.info_manager.initialize_player(player_id, "attack")
            engine.ability_system.initialize_player(player_id, player.agent)

        # Defenders
        for i in range(config['num_defenders']):
            player_id = f"defense_{i}"
            spawn = defense_spawns[i % len(defense_spawns)]

            # For retake, position defenders away from site
            if config['spike_planted'] and config['defender_behavior'] == 'retake':
                # Position at spawn or rotating
                spawn = defense_spawns[i % len(defense_spawns)]

            player = SimulatedPlayer(
                player_id=player_id,
                team_id="defense_team",
                side="defense",
                x=spawn[0],
                y=spawn[1],
                agent=self.AGENTS_DEFENSE[i % len(self.AGENTS_DEFENSE)]
            )
            player.weapon = WeaponDatabase.get_weapon(config['defender_weapons'][i % len(config['defender_weapons'])])
            player.shield = config['defender_shields'][i % len(config['defender_shields'])]
            engine.players[player_id] = player
            engine.info_manager.initialize_player(player_id, "defense")
            engine.ability_system.initialize_player(player_id, player.agent)

        # Setup spike if planted
        if config['spike_planted']:
            engine.spike_planted = True
            engine.spike_site = config['spike_site']
            engine.spike_plant_time = config['start_time_ms'] - 5000

    async def _run_scenario_loop(self, engine: SimulationEngine, map_data: Dict,
                                  scenario: Scenario, config: Dict) -> Dict:
        """Run the simulation loop with scenario-specific behavior."""
        time_ms = config['start_time_ms']
        max_time = config['max_time_ms']
        tick_ms = 128

        kills = []
        trades = 0
        first_blood_side = None

        # Track player facings
        player_facings = {}
        for p in engine.players.values():
            player_facings[p.player_id] = -math.pi/2 if p.side == 'attack' else math.pi/2

        while time_ms < max_time:
            attack_alive = [p for p in engine.players.values() if p.side == 'attack' and p.is_alive]
            defense_alive = [p for p in engine.players.values() if p.side == 'defense' and p.is_alive]

            # Win conditions
            if len(attack_alive) == 0:
                break
            if len(defense_alive) == 0:
                break
            if engine.spike_planted and engine.spike_plant_time:
                if time_ms - engine.spike_plant_time >= 45000:
                    break  # Spike detonated
            if not engine.spike_planted and time_ms >= 100000:
                break  # Time ran out

            # Get smoke positions
            smoke_positions = engine.ability_system.get_active_smokes(time_ms)

            # Move players based on scenario behavior
            for player in engine.players.values():
                if not player.is_alive:
                    continue

                target = self._get_movement_target(
                    player, engine, map_data, config, time_ms
                )

                dx = target[0] - player.x
                dy = target[1] - player.y
                dist = math.sqrt(dx*dx + dy*dy)

                # Speed based on scenario
                if config['attacker_speed'] == 'fast' and player.side == 'attack':
                    speed = 0.020
                elif config['attacker_speed'] == 'slow' and player.side == 'attack':
                    speed = 0.010
                else:
                    speed = 0.015

                if dist > 0.01:
                    player.prev_x = player.x
                    player.prev_y = player.y
                    player.x += (dx / dist) * min(speed, dist)
                    player.y += (dy / dist) * min(speed, dist)
                    player.x = max(0.05, min(0.95, player.x))
                    player.y = max(0.05, min(0.95, player.y))
                    player.moved_this_tick = True
                    player_facings[player.player_id] = math.atan2(dy, dx)
                else:
                    player.moved_this_tick = False

            # Combat
            kill_this_tick = None
            for attacker in engine.players.values():
                if not attacker.is_alive:
                    continue
                for defender in engine.players.values():
                    if not defender.is_alive or defender.side == attacker.side:
                        continue

                    dist = math.sqrt((attacker.x - defender.x)**2 + (attacker.y - defender.y)**2)
                    if dist > 0.3:
                        continue

                    if not self.pathfinder.has_line_of_sight(
                        (attacker.x, attacker.y), (defender.x, defender.y)
                    ):
                        continue

                    # Smoke check
                    through_smoke = False
                    for smoke_pos, smoke_radius in smoke_positions:
                        mid_x = (attacker.x + defender.x) / 2
                        mid_y = (attacker.y + defender.y) / 2
                        if math.sqrt((mid_x - smoke_pos[0])**2 + (mid_y - smoke_pos[1])**2) < smoke_radius + 0.03:
                            through_smoke = True
                            break

                    if through_smoke and random.random() < 0.80:
                        continue

                    # Combat chance
                    combat_chance = 0.01 + 0.09 * (1 - dist / 0.3)

                    # Eco weapons are less effective
                    if attacker.weapon.name == 'Classic':
                        combat_chance *= 0.5

                    if random.random() < combat_chance:
                        # Determine winner
                        adv = 0.45  # Base

                        # Peeker's advantage
                        if attacker.moved_this_tick and not defender.moved_this_tick:
                            adv += 0.12

                        # Weapon advantage
                        if attacker.weapon.name != 'Classic' and defender.weapon.name == 'Classic':
                            adv += 0.20
                        elif attacker.weapon.name == 'Classic' and defender.weapon.name != 'Classic':
                            adv -= 0.20

                        # Shield advantage
                        if attacker.shield > defender.shield:
                            adv += 0.05
                        elif defender.shield > attacker.shield:
                            adv -= 0.05

                        # Post-plant: defenders have urgency disadvantage
                        if engine.spike_planted and defender.side == 'defense':
                            adv += 0.05  # Attackers can play time

                        # Defender on site bonus
                        if defender.side == 'defense' and not engine.spike_planted:
                            for site_data in map_data['sites'].values():
                                site_x, site_y = site_data['center']
                                if math.sqrt((defender.x - site_x)**2 + (defender.y - site_y)**2) < site_data['radius'] + 0.1:
                                    adv -= 0.05
                                    break

                        if random.random() < adv:
                            victim = defender
                            killer = attacker
                        else:
                            victim = attacker
                            killer = defender

                        victim.is_alive = False
                        kill_data = {'time_ms': time_ms, 'killer_side': killer.side}

                        if first_blood_side is None:
                            first_blood_side = killer.side

                        # Trade check
                        if len(kills) >= 1:
                            prev = kills[-1]
                            if time_ms - prev['time_ms'] <= 4000 and kill_data['killer_side'] != prev['killer_side']:
                                trades += 1

                        kills.append(kill_data)
                        kill_this_tick = kill_data
                        break

                if kill_this_tick:
                    break

            # Spike plant (only for non-planted scenarios)
            if not engine.spike_planted:
                carrier = next((p for p in engine.players.values() if p.has_spike and p.is_alive), None)
                if carrier:
                    for site_name, site_data in map_data['sites'].items():
                        site_x, site_y = site_data['center']
                        dist = math.sqrt((carrier.x - site_x)**2 + (carrier.y - site_y)**2)
                        if dist < site_data['radius'] + 0.08:
                            smoke_coverage = sum(1 for sp, sr in smoke_positions
                                               if math.sqrt((site_x - sp[0])**2 + (site_y - sp[1])**2) < sr + 0.1)

                            defenders_watching = sum(1 for d in engine.players.values()
                                                   if d.side == 'defense' and d.is_alive and
                                                   self.pathfinder.has_line_of_sight((carrier.x, carrier.y), (d.x, d.y)))

                            plant_chance = 0.03
                            plant_chance += 0.05 * smoke_coverage
                            if defenders_watching == 0:
                                plant_chance += 0.10
                            elif defenders_watching >= 2:
                                plant_chance *= 0.3

                            if random.random() < plant_chance:
                                engine.spike_planted = True
                                engine.spike_plant_time = time_ms
                                engine.spike_site = site_name
                                break

            # Ability usage
            for player in engine.players.values():
                if not player.is_alive:
                    continue

                ability_chance = 0.01
                if player.side == 'attack':
                    for site_data in map_data['sites'].values():
                        site_x, site_y = site_data['center']
                        if math.sqrt((player.x - site_x)**2 + (player.y - site_y)**2) < 0.25:
                            ability_chance = 0.04
                            break

                if random.random() < ability_chance:
                    result = engine.ability_system.should_use_ability(
                        player.player_id, time_ms, 'mid_round', engine.round_state,
                        (player.x, player.y), player.side
                    )
                    if result:
                        ability, target = result
                        engine.ability_system.use_ability(player.player_id, ability, target, time_ms, player.side)

            time_ms += tick_ms

        # Determine winner
        attack_alive = len([p for p in engine.players.values() if p.side == 'attack' and p.is_alive])
        defense_alive = len([p for p in engine.players.values() if p.side == 'defense' and p.is_alive])

        if attack_alive == 0:
            winner = 'defense'
        elif defense_alive == 0:
            winner = 'attack'
        elif engine.spike_planted:
            # Check if spike detonated
            if engine.spike_plant_time and time_ms - engine.spike_plant_time >= 45000:
                winner = 'attack'
            else:
                winner = 'defense'  # Defused or time
        else:
            winner = 'defense'  # Time ran out

        return {
            'winner': winner,
            'kills': len(kills),
            'trades': trades,
            'planted': engine.spike_planted,
            'duration_ms': time_ms - config['start_time_ms'],
            'first_blood_side': first_blood_side,
        }

    def _get_movement_target(self, player: SimulatedPlayer, engine: SimulationEngine,
                             map_data: Dict, config: Dict, time_ms: int) -> Tuple[float, float]:
        """Get movement target based on scenario behavior."""

        if player.side == 'attack':
            behavior = config['attacker_behavior']

            if behavior == 'push':
                # Move toward sites
                target_site = random.choice(list(map_data['sites'].keys()))
                site_data = map_data['sites'][target_site]
                return (
                    site_data['center'][0] + random.uniform(-0.1, 0.1),
                    site_data['center'][1] + random.uniform(-0.1, 0.1)
                )

            elif behavior == 'hold':
                # Hold current position (post-plant)
                return (
                    player.x + random.uniform(-0.02, 0.02),
                    player.y + random.uniform(-0.02, 0.02)
                )

            elif behavior == 'mid_control':
                # Fight for mid
                return (0.5 + random.uniform(-0.1, 0.1), 0.5 + random.uniform(-0.1, 0.1))

        else:  # Defense
            behavior = config['defender_behavior']

            if behavior == 'hold':
                # Small adjustments
                return (
                    player.x + random.uniform(-0.02, 0.02),
                    player.y + random.uniform(-0.02, 0.02)
                )

            elif behavior == 'retake':
                # Move toward planted site
                if engine.spike_planted and engine.spike_site:
                    site_data = map_data['sites'][engine.spike_site]
                    return (
                        site_data['center'][0] + random.uniform(-0.1, 0.1),
                        site_data['center'][1] + random.uniform(-0.1, 0.1)
                    )
                return (player.x, player.y)

            elif behavior == 'aggressive':
                # Push toward attackers
                return (
                    player.x + random.uniform(-0.05, 0.05),
                    player.y + 0.05  # Move toward attack spawn
                )

            elif behavior == 'mid_control':
                # Fight for mid
                return (0.5 + random.uniform(-0.1, 0.1), 0.5 + random.uniform(-0.1, 0.1))

        return (player.x, player.y)

    def analyze_results(self) -> Dict:
        """Analyze results and identify strengths/weaknesses."""
        analysis = {
            'strengths': [],
            'weaknesses': [],
            'suggestions': [],
            'by_scenario': {},
        }

        for scenario, result in self.results.items():
            ref = VCT_REFERENCE.get(scenario, {})
            target_win = ref.get('attack_win_rate', 0.5)
            actual_win = result.attack_win_rate
            diff = actual_win - target_win

            scenario_analysis = {
                'actual_attack_win': actual_win,
                'target_attack_win': target_win,
                'difference': diff,
                'plant_rate': result.plant_rate,
                'avg_duration_ms': result.avg_duration_ms,
                'avg_kills': result.avg_kills,
                'status': 'good' if abs(diff) < 0.15 else ('attack_favored' if diff > 0 else 'defense_favored'),
            }
            analysis['by_scenario'][scenario.value] = scenario_analysis

            # Categorize
            if abs(diff) < 0.10:
                analysis['strengths'].append(f"{scenario.value}: Accurate ({actual_win*100:.0f}% vs {target_win*100:.0f}% target)")
            elif abs(diff) < 0.20:
                if diff > 0:
                    analysis['weaknesses'].append(f"{scenario.value}: Attack too strong ({actual_win*100:.0f}% vs {target_win*100:.0f}% target)")
                else:
                    analysis['weaknesses'].append(f"{scenario.value}: Defense too strong ({actual_win*100:.0f}% vs {target_win*100:.0f}% target)")
            else:
                if diff > 0:
                    analysis['weaknesses'].append(f"{scenario.value}: MAJOR - Attack way too strong ({actual_win*100:.0f}% vs {target_win*100:.0f}% target)")
                else:
                    analysis['weaknesses'].append(f"{scenario.value}: MAJOR - Defense way too strong ({actual_win*100:.0f}% vs {target_win*100:.0f}% target)")

        # Generate suggestions based on patterns
        self._generate_suggestions(analysis)

        return analysis

    def _generate_suggestions(self, analysis: Dict):
        """Generate improvement suggestions based on analysis."""
        scenarios = analysis['by_scenario']

        # Check eco scenarios
        eco_vs_full = scenarios.get('eco_vs_full', {})
        full_vs_eco = scenarios.get('full_vs_eco', {})

        if eco_vs_full.get('actual_attack_win', 0) > 0.25:
            analysis['suggestions'].append(
                "ECO FIX: Eco rounds winning too often. Increase weapon damage differential or accuracy penalty for pistols."
            )

        if full_vs_eco.get('actual_attack_win', 0) < 0.75:
            analysis['suggestions'].append(
                "FULL BUY FIX: Full buy not dominating eco enough. Increase rifle effectiveness or eco accuracy penalty."
            )

        # Check post-plant
        post_plant = scenarios.get('post_plant_defense', {})
        if post_plant.get('actual_attack_win', 0) < 0.55:
            analysis['suggestions'].append(
                "POST-PLANT FIX: Attackers should win more post-plants. Add time pressure penalty for retaking defenders."
            )

        # Check clutches
        clutch_1v2 = scenarios.get('clutch_1v2', {})
        if clutch_1v2.get('actual_attack_win', 0) > 0.35:
            analysis['suggestions'].append(
                "CLUTCH FIX: 1v2 should be harder for solo player. Add coordination bonus for the duo."
            )

        # Check man advantage
        adv_4v5 = scenarios.get('man_advantage_4v5', {})
        adv_5v4 = scenarios.get('man_advantage_5v4', {})

        if adv_4v5.get('actual_attack_win', 0) > 0.45:
            analysis['suggestions'].append(
                "MAN DISADVANTAGE FIX: 4v5 attackers winning too much. Numbers should matter more."
            )

        if adv_5v4.get('actual_attack_win', 0) < 0.55:
            analysis['suggestions'].append(
                "MAN ADVANTAGE FIX: 5v4 attackers should win more. Numbers advantage not impactful enough."
            )

        # Check mid control
        mid = scenarios.get('mid_control', {})
        if abs(mid.get('difference', 0)) > 0.15:
            analysis['suggestions'].append(
                "MID CONTROL FIX: Mid battles should be roughly even. Check positioning and movement patterns."
            )

    def print_report(self, analysis: Dict):
        """Print comprehensive analysis report."""
        print(f"\n{'='*70}")
        print("SCENARIO ANALYSIS REPORT")
        print(f"{'='*70}")

        print(f"\n{'─'*70}")
        print("DETAILED RESULTS BY SCENARIO")
        print(f"{'─'*70}")

        # Group by category
        categories = {
            'Standard Attack': [Scenario.ATTACK_EXECUTE_FAST, Scenario.ATTACK_EXECUTE_SLOW],
            'Post-Plant': [Scenario.POST_PLANT_DEFENSE, Scenario.RETAKE],
            'Map Control': [Scenario.MID_CONTROL, Scenario.DEFENDER_AGGRESSION],
            'Economy': [Scenario.ECO_VS_FULL, Scenario.FULL_VS_ECO],
            'Clutch': [Scenario.CLUTCH_1V1, Scenario.CLUTCH_1V2, Scenario.CLUTCH_2V1],
            'Man Advantage': [Scenario.MAN_ADVANTAGE_4V5, Scenario.MAN_ADVANTAGE_5V4, Scenario.MAN_ADVANTAGE_3V5],
        }

        for category, scenarios in categories.items():
            print(f"\n{category}:")
            for scenario in scenarios:
                if scenario in self.results:
                    result = self.results[scenario]
                    ref = VCT_REFERENCE.get(scenario, {})
                    target = ref.get('attack_win_rate', 0.5) * 100
                    actual = result.attack_win_rate * 100
                    diff = actual - target

                    status = "✓" if abs(diff) < 15 else "✗"
                    bar_len = int(actual / 5)
                    bar = "█" * bar_len + "░" * (20 - bar_len)

                    print(f"  {status} {scenario.value:25s} │ {bar} │ {actual:5.1f}% (target: {target:.0f}%, diff: {diff:+.1f}%)")

        print(f"\n{'─'*70}")
        print("STRENGTHS (Within 10% of target)")
        print(f"{'─'*70}")
        for s in analysis['strengths']:
            print(f"  ✓ {s}")
        if not analysis['strengths']:
            print("  (none)")

        print(f"\n{'─'*70}")
        print("WEAKNESSES (More than 10% off target)")
        print(f"{'─'*70}")
        for w in analysis['weaknesses']:
            print(f"  ✗ {w}")
        if not analysis['weaknesses']:
            print("  (none)")

        print(f"\n{'─'*70}")
        print("IMPROVEMENT SUGGESTIONS")
        print(f"{'─'*70}")
        for i, s in enumerate(analysis['suggestions'], 1):
            print(f"  {i}. {s}")
        if not analysis['suggestions']:
            print("  (none - model is well balanced)")

        print(f"\n{'='*70}")


async def main():
    analyzer = ScenarioAnalyzer()

    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 35

    await analyzer.run_all_scenarios(rounds_per_scenario=rounds)

    analysis = analyzer.analyze_results()
    analyzer.print_report(analysis)

    # Save results
    output_file = Path(__file__).parent / 'scenario_analysis_results.json'

    # Convert enum keys to strings for JSON
    json_results = {}
    for scenario, result in analyzer.results.items():
        json_results[scenario.value] = {
            'total_rounds': result.total_rounds,
            'attack_wins': result.attack_wins,
            'defense_wins': result.defense_wins,
            'attack_win_rate': result.attack_win_rate,
            'plant_rate': result.plant_rate,
            'avg_duration_ms': result.avg_duration_ms,
            'avg_kills': result.avg_kills,
        }

    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': json_results,
            'analysis': {
                'strengths': analysis['strengths'],
                'weaknesses': analysis['weaknesses'],
                'suggestions': analysis['suggestions'],
            },
        }, f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    asyncio.run(main())
