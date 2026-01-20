#!/usr/bin/env python3
"""
Parameter Sweep for Simulation Realism Optimization

Tests different parameter combinations to find optimal settings.
"""

import asyncio
import itertools
import json
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple
from datetime import datetime
import random
import math

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.simulation_engine import SimulationEngine, SimulatedPlayer
from app.services.weapon_system import WeaponDatabase
from app.services.ability_system import AbilityCategory
from app.services.pathfinding import AStarPathfinder


# VCT Reference targets
VCT_TARGETS = {
    'attack_win_rate': 0.48,
    'trade_rate': 0.252,
    'plant_rate': 0.65,
    'round_duration_ms': 75000,
    'first_blood_time_ms': 25000,
}


@dataclass
class SimParams:
    """Simulation parameters to test."""
    # Combat parameters
    attacker_base: float = 0.45      # Base win chance for attacker
    peeker_bonus: float = 0.12       # Bonus for moving player
    info_bonus: float = 0.10         # Bonus for having info
    defender_site_bonus: float = 0.05  # Bonus for defender on site

    # Smoke parameters
    smoke_block_chance: float = 0.80  # Chance smoke blocks engagement

    # Plant parameters
    base_plant_chance: float = 0.03
    smoke_plant_bonus: float = 0.05
    clear_site_bonus: float = 0.10
    watched_penalty: float = 0.30     # Multiplier when defenders watching

    # Ability parameters
    base_ability_chance: float = 0.01
    near_site_ability_chance: float = 0.04
    controller_ability_chance: float = 0.02
    initiator_ability_chance: float = 0.015

    def to_dict(self):
        return {
            'attacker_base': self.attacker_base,
            'peeker_bonus': self.peeker_bonus,
            'info_bonus': self.info_bonus,
            'defender_site_bonus': self.defender_site_bonus,
            'smoke_block_chance': self.smoke_block_chance,
            'base_plant_chance': self.base_plant_chance,
            'smoke_plant_bonus': self.smoke_plant_bonus,
            'clear_site_bonus': self.clear_site_bonus,
            'watched_penalty': self.watched_penalty,
            'base_ability_chance': self.base_ability_chance,
            'near_site_ability_chance': self.near_site_ability_chance,
        }


class MockDB:
    async def execute(self, *args, **kwargs): return None
    async def commit(self): pass
    async def rollback(self): pass


class ParameterSweep:
    """Run parameter sweep to find optimal simulation settings."""

    MAPS = ['ascent', 'bind', 'haven', 'split', 'icebox', 'breeze', 'pearl', 'lotus', 'sunset']
    AGENTS_ATTACK = ['jett', 'raze', 'omen', 'sova', 'killjoy']
    AGENTS_DEFENSE = ['chamber', 'cypher', 'viper', 'fade', 'sage']

    def __init__(self):
        self.pathfinder = AStarPathfinder()
        self.results: List[Dict] = []

    async def run_sweep(self, rounds_per_config: int = 30) -> List[Dict]:
        """Run parameter sweep with different combinations."""

        # Define parameter ranges to test
        param_ranges = {
            'attacker_base': [0.40, 0.45, 0.50],
            'peeker_bonus': [0.08, 0.12, 0.16],
            'info_bonus': [0.05, 0.10, 0.15],
            'defender_site_bonus': [0.00, 0.05, 0.10],
            'smoke_block_chance': [0.70, 0.80, 0.90],
            'base_plant_chance': [0.02, 0.03, 0.04],
        }

        # Generate all combinations (limited to avoid explosion)
        # We'll test key parameters in groups

        # Group 1: Combat parameters
        combat_combos = list(itertools.product(
            param_ranges['attacker_base'],
            param_ranges['peeker_bonus'],
            param_ranges['defender_site_bonus'],
        ))

        # Group 2: Smoke/Plant parameters
        smoke_combos = list(itertools.product(
            param_ranges['smoke_block_chance'],
            param_ranges['base_plant_chance'],
        ))

        total_configs = len(combat_combos) * len(smoke_combos)
        print(f"\n{'='*70}")
        print(f"PARAMETER SWEEP - Testing {total_configs} configurations")
        print(f"{'='*70}")
        print(f"Rounds per config: {rounds_per_config}")
        print(f"Total simulations: {total_configs * rounds_per_config}")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        config_num = 0
        best_score = 0
        best_params = None

        for combat in combat_combos:
            for smoke in smoke_combos:
                config_num += 1

                params = SimParams(
                    attacker_base=combat[0],
                    peeker_bonus=combat[1],
                    defender_site_bonus=combat[2],
                    smoke_block_chance=smoke[0],
                    base_plant_chance=smoke[1],
                )

                print(f"[{config_num}/{total_configs}] Testing: atk={params.attacker_base}, peek={params.peeker_bonus}, def_site={params.defender_site_bonus}, smoke={params.smoke_block_chance}, plant={params.base_plant_chance}", end=" ")

                metrics = await self._run_config(params, rounds_per_config)
                score = self._calculate_score(metrics)

                result = {
                    'config_num': config_num,
                    'params': params.to_dict(),
                    'metrics': metrics,
                    'score': score,
                }
                self.results.append(result)

                if score > best_score:
                    best_score = score
                    best_params = params

                print(f"→ Score: {score:.1f}% (atk_win={metrics['attack_win_rate']*100:.0f}%, plant={metrics['plant_rate']*100:.0f}%)")

        print(f"\n{'='*70}")
        print(f"SWEEP COMPLETE")
        print(f"{'='*70}")
        print(f"Best Score: {best_score:.1f}%")
        print(f"Best Parameters:")
        for k, v in best_params.to_dict().items():
            print(f"  {k}: {v}")

        return self.results

    async def _run_config(self, params: SimParams, num_rounds: int) -> Dict:
        """Run simulation with specific parameters."""
        attack_wins = 0
        total_kills = 0
        trade_kills = 0
        plants = 0
        total_duration = 0
        total_first_blood = 0
        rounds_with_fb = 0

        rounds_per_map = max(1, num_rounds // len(self.MAPS))

        for map_name in self.MAPS:
            self.pathfinder.load_nav_grid(map_name)

            for _ in range(rounds_per_map):
                result = await self._simulate_round(map_name, params)
                if result:
                    if result['winner'] == 'attack':
                        attack_wins += 1
                    total_kills += result['kills']
                    trade_kills += result['trades']
                    if result['planted']:
                        plants += 1
                    total_duration += result['duration_ms']
                    if result['first_blood_ms'] > 0:
                        total_first_blood += result['first_blood_ms']
                        rounds_with_fb += 1

        total_rounds = rounds_per_map * len(self.MAPS)

        return {
            'attack_win_rate': attack_wins / max(1, total_rounds),
            'trade_rate': trade_kills / max(1, total_kills),
            'plant_rate': plants / max(1, total_rounds),
            'avg_duration_ms': total_duration / max(1, total_rounds),
            'avg_first_blood_ms': total_first_blood / max(1, rounds_with_fb),
            'avg_kills': total_kills / max(1, total_rounds),
        }

    async def _simulate_round(self, map_name: str, params: SimParams) -> Dict:
        """Simulate a single round with given parameters."""
        try:
            db = MockDB()
            engine = SimulationEngine(db)
            engine.map_name = map_name
            map_data = engine.MAP_DATA.get(map_name, engine.DEFAULT_MAP_DATA)

            # Initialize players
            attack_spawns = map_data['spawns']['attack']
            for i, agent in enumerate(self.AGENTS_ATTACK):
                player_id = f"attack_{i}"
                spawn = attack_spawns[i % len(attack_spawns)]
                player = SimulatedPlayer(
                    player_id=player_id, team_id="attack_team", side="attack",
                    x=spawn[0], y=spawn[1], agent=agent, has_spike=(i == 0)
                )
                player.weapon = WeaponDatabase.get_weapon('vandal')
                player.shield = 50
                engine.players[player_id] = player
                engine.info_manager.initialize_player(player_id, "attack")
                engine.ability_system.initialize_player(player_id, agent)

            defense_spawns = map_data['spawns']['defense']
            for i, agent in enumerate(self.AGENTS_DEFENSE):
                player_id = f"defense_{i}"
                spawn = defense_spawns[i % len(defense_spawns)]
                player = SimulatedPlayer(
                    player_id=player_id, team_id="defense_team", side="defense",
                    x=spawn[0], y=spawn[1], agent=agent
                )
                player.weapon = WeaponDatabase.get_weapon('vandal')
                player.shield = 50
                engine.players[player_id] = player
                engine.info_manager.initialize_player(player_id, "defense")
                engine.ability_system.initialize_player(player_id, agent)

            # Run simulation
            return await self._run_round_loop(engine, map_data, params)

        except Exception as e:
            return None

    async def _run_round_loop(self, engine, map_data, params: SimParams) -> Dict:
        """Main round simulation loop."""
        time_ms = 0
        max_time = 100000
        tick_ms = 128

        kills = []
        trades = 0
        first_blood_ms = 0

        # Track facings
        player_facings = {}
        for p in engine.players.values():
            player_facings[p.player_id] = -math.pi/2 if p.side == 'attack' else math.pi/2

        while time_ms < max_time:
            attack_alive = [p for p in engine.players.values() if p.side == 'attack' and p.is_alive]
            defense_alive = [p for p in engine.players.values() if p.side == 'defense' and p.is_alive]

            if len(attack_alive) == 0 or len(defense_alive) == 0:
                break
            if engine.spike_planted and engine.spike_plant_time:
                if time_ms - engine.spike_plant_time >= 45000:
                    break

            # Get smoke positions
            smoke_positions = engine.ability_system.get_active_smokes(time_ms)

            # Move players
            for player in engine.players.values():
                if not player.is_alive:
                    continue

                if player.side == 'attack':
                    target_y = 0.3 + random.uniform(-0.1, 0.1)
                    target_x = 0.3 if random.random() < 0.5 else 0.7
                else:
                    target_x = player.x + random.uniform(-0.02, 0.02)
                    target_y = player.y + random.uniform(-0.02, 0.02)

                dx = target_x - player.x
                dy = target_y - player.y
                dist = math.sqrt(dx*dx + dy*dy)

                if dist > 0.01:
                    speed = 0.015
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
            for attacker in engine.players.values():
                if not attacker.is_alive:
                    continue
                for defender in engine.players.values():
                    if not defender.is_alive or defender.side == attacker.side:
                        continue

                    dist = math.sqrt((attacker.x - defender.x)**2 + (attacker.y - defender.y)**2)
                    if dist > 0.3:
                        continue

                    if not self.pathfinder.has_line_of_sight((attacker.x, attacker.y), (defender.x, defender.y)):
                        continue

                    # Smoke check
                    through_smoke = False
                    for smoke_pos, smoke_radius in smoke_positions:
                        mid_x = (attacker.x + defender.x) / 2
                        mid_y = (attacker.y + defender.y) / 2
                        if math.sqrt((mid_x - smoke_pos[0])**2 + (mid_y - smoke_pos[1])**2) < smoke_radius + 0.03:
                            through_smoke = True
                            break

                    if through_smoke and random.random() < params.smoke_block_chance:
                        continue

                    combat_chance = 0.01 + 0.09 * (1 - dist / 0.3)
                    if random.random() < combat_chance:
                        # Determine winner
                        adv = params.attacker_base
                        if attacker.moved_this_tick and not defender.moved_this_tick:
                            adv += params.peeker_bonus

                        # Defender on site bonus
                        if defender.side == 'defense':
                            for site_data in map_data['sites'].values():
                                site_x, site_y = site_data['center']
                                if math.sqrt((defender.x - site_x)**2 + (defender.y - site_y)**2) < site_data['radius'] + 0.1:
                                    adv -= params.defender_site_bonus
                                    break

                        if random.random() < adv:
                            victim = defender
                            killer = attacker
                        else:
                            victim = attacker
                            killer = defender

                        victim.is_alive = False
                        kill_data = {'time_ms': time_ms, 'killer_side': killer.side}

                        if first_blood_ms == 0:
                            first_blood_ms = time_ms

                        # Check for trade
                        if len(kills) >= 1:
                            prev = kills[-1]
                            if time_ms - prev['time_ms'] <= 4000 and kill_data['killer_side'] != prev['killer_side']:
                                trades += 1

                        kills.append(kill_data)
                        break
                if kills and kills[-1]['time_ms'] == time_ms:
                    break

            # Spike plant
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

                            plant_chance = params.base_plant_chance
                            plant_chance += params.smoke_plant_bonus * smoke_coverage
                            if defenders_watching == 0:
                                plant_chance += params.clear_site_bonus
                            elif defenders_watching >= 2:
                                plant_chance *= params.watched_penalty

                            if random.random() < plant_chance:
                                engine.spike_planted = True
                                engine.spike_plant_time = time_ms
                                break

            # Ability usage
            for player in engine.players.values():
                if not player.is_alive:
                    continue

                ability_chance = params.base_ability_chance

                if player.side == 'attack':
                    for site_data in map_data['sites'].values():
                        site_x, site_y = site_data['center']
                        if math.sqrt((player.x - site_x)**2 + (player.y - site_y)**2) < 0.25:
                            ability_chance = params.near_site_ability_chance
                            break

                    if player.agent.lower() in ['omen', 'brimstone', 'viper', 'astra']:
                        if time_ms < 30000:
                            ability_chance = max(ability_chance, params.controller_ability_chance)

                if player.agent.lower() in ['breach', 'skye', 'kayo', 'fade', 'sova']:
                    if time_ms > 15000:
                        ability_chance = max(ability_chance, params.initiator_ability_chance)

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
            winner = 'attack'
        else:
            winner = 'defense'

        return {
            'winner': winner,
            'kills': len(kills),
            'trades': trades,
            'planted': engine.spike_planted,
            'duration_ms': time_ms,
            'first_blood_ms': first_blood_ms,
        }

    def _calculate_score(self, metrics: Dict) -> float:
        """Calculate realism score from metrics."""
        scores = []

        # Attack win rate (most important) - target 48%
        atk_diff = abs(metrics['attack_win_rate'] - VCT_TARGETS['attack_win_rate'])
        atk_score = max(0, 1 - atk_diff * 4)  # 25% diff = 0 score
        scores.append(('attack_win', atk_score, 30))

        # Trade rate - target 25.2%
        trade_diff = abs(metrics['trade_rate'] - VCT_TARGETS['trade_rate'])
        trade_score = max(0, 1 - trade_diff * 5)
        scores.append(('trade', trade_score, 15))

        # Plant rate - target 65%
        plant_diff = abs(metrics['plant_rate'] - VCT_TARGETS['plant_rate'])
        plant_score = max(0, 1 - plant_diff * 3)
        scores.append(('plant', plant_score, 25))

        # Round duration - target 75s
        dur_diff = abs(metrics['avg_duration_ms'] - VCT_TARGETS['round_duration_ms']) / VCT_TARGETS['round_duration_ms']
        dur_score = max(0, 1 - dur_diff * 2)
        scores.append(('duration', dur_score, 15))

        # First blood time - target 25s
        fb_diff = abs(metrics['avg_first_blood_ms'] - VCT_TARGETS['first_blood_time_ms']) / VCT_TARGETS['first_blood_time_ms']
        fb_score = max(0, 1 - fb_diff)
        scores.append(('first_blood', fb_score, 15))

        total_weight = sum(w for _, _, w in scores)
        weighted_score = sum(s * w for _, s, w in scores) / total_weight

        return weighted_score * 100

    def analyze_results(self) -> Dict:
        """Analyze sweep results and find patterns."""
        if not self.results:
            return {}

        # Sort by score
        sorted_results = sorted(self.results, key=lambda x: x['score'], reverse=True)

        # Top 5 configurations
        top5 = sorted_results[:5]
        bottom5 = sorted_results[-5:]

        # Parameter correlation analysis
        param_correlations = {}
        for param in ['attacker_base', 'peeker_bonus', 'defender_site_bonus', 'smoke_block_chance', 'base_plant_chance']:
            values = [(r['params'][param], r['score']) for r in self.results]
            # Group by param value
            by_value = {}
            for v, s in values:
                if v not in by_value:
                    by_value[v] = []
                by_value[v].append(s)

            param_correlations[param] = {v: sum(scores)/len(scores) for v, scores in by_value.items()}

        return {
            'total_configs': len(self.results),
            'top5': top5,
            'bottom5': bottom5,
            'param_impact': param_correlations,
            'best_params': sorted_results[0]['params'],
            'best_score': sorted_results[0]['score'],
            'worst_score': sorted_results[-1]['score'],
            'avg_score': sum(r['score'] for r in self.results) / len(self.results),
        }

    def print_analysis(self, analysis: Dict):
        """Print analysis report."""
        print(f"\n{'='*70}")
        print("PARAMETER SWEEP ANALYSIS")
        print(f"{'='*70}")

        print(f"\nTotal Configurations Tested: {analysis['total_configs']}")
        print(f"Score Range: {analysis['worst_score']:.1f}% - {analysis['best_score']:.1f}%")
        print(f"Average Score: {analysis['avg_score']:.1f}%")

        print(f"\n--- TOP 5 CONFIGURATIONS ---")
        for i, config in enumerate(analysis['top5'], 1):
            m = config['metrics']
            print(f"\n#{i} Score: {config['score']:.1f}%")
            print(f"   Attack Win: {m['attack_win_rate']*100:.1f}% | Plant: {m['plant_rate']*100:.1f}% | Trade: {m['trade_rate']*100:.1f}%")
            print(f"   Params: atk={config['params']['attacker_base']}, peek={config['params']['peeker_bonus']}, def_site={config['params']['defender_site_bonus']}, smoke={config['params']['smoke_block_chance']}, plant={config['params']['base_plant_chance']}")

        print(f"\n--- PARAMETER IMPACT ANALYSIS ---")
        for param, values in analysis['param_impact'].items():
            print(f"\n{param}:")
            for val, avg_score in sorted(values.items()):
                bar = "█" * int(avg_score / 5)
                print(f"  {val:5.2f} → {avg_score:5.1f}% {bar}")

        print(f"\n--- OPTIMAL PARAMETERS ---")
        for k, v in analysis['best_params'].items():
            print(f"  {k}: {v}")

        print(f"\n{'='*70}")


async def main():
    sweep = ParameterSweep()

    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 20

    results = await sweep.run_sweep(rounds_per_config=rounds)

    analysis = sweep.analyze_results()
    sweep.print_analysis(analysis)

    # Save results
    output_file = Path(__file__).parent / 'param_sweep_results.json'
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'analysis': {
                'total_configs': analysis['total_configs'],
                'best_score': analysis['best_score'],
                'worst_score': analysis['worst_score'],
                'avg_score': analysis['avg_score'],
                'best_params': analysis['best_params'],
                'param_impact': analysis['param_impact'],
            },
            'all_results': results,
        }, f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    asyncio.run(main())
