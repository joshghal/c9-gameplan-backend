#!/usr/bin/env python3
"""
Parameter Tuning Script for V6 Simulation

Tests different parameter combinations to find optimal configuration.
"""

import asyncio
import json
import sys
import math
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple
from datetime import datetime
from enum import Enum
import itertools

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.simulation_engine import SimulationEngine, SimulatedPlayer
from app.services.weapon_system import WeaponDatabase
from app.services.pathfinding import AStarPathfinder
from app.services.combat_system_v2 import CombatSystemV2, CombatState


class Scenario(Enum):
    ECO_VS_FULL = "eco_vs_full"
    FULL_VS_ECO = "full_vs_eco"
    CLUTCH_1V1 = "clutch_1v1"
    CLUTCH_1V2 = "clutch_1v2"
    MAN_ADVANTAGE_4V5 = "man_advantage_4v5"
    MAN_ADVANTAGE_5V4 = "man_advantage_5v4"
    MAN_ADVANTAGE_3V5 = "man_advantage_3v5"


VCT_REFERENCE = {
    Scenario.ECO_VS_FULL: {'attack_win_rate': 0.15},
    Scenario.FULL_VS_ECO: {'attack_win_rate': 0.85},
    Scenario.CLUTCH_1V1: {'attack_win_rate': 0.50},
    Scenario.CLUTCH_1V2: {'attack_win_rate': 0.25},
    Scenario.MAN_ADVANTAGE_4V5: {'attack_win_rate': 0.35},
    Scenario.MAN_ADVANTAGE_5V4: {'attack_win_rate': 0.60},
    Scenario.MAN_ADVANTAGE_3V5: {'attack_win_rate': 0.15},
}

ROLE_HS_RATES = {
    "duelist": 0.233,
    "initiator": 0.190,
    "controller": 0.196,
    "sentinel": 0.133,
}

AGENT_ROLES = {
    "jett": "duelist", "raze": "duelist", "reyna": "duelist",
    "phoenix": "duelist", "neon": "duelist", "iso": "duelist",
    "sova": "initiator", "breach": "initiator", "skye": "initiator",
    "kayo": "initiator", "fade": "initiator", "gekko": "initiator",
    "brimstone": "controller", "omen": "controller", "viper": "controller",
    "astra": "controller", "harbor": "controller", "clove": "controller",
    "sage": "sentinel", "cypher": "sentinel", "killjoy": "sentinel",
    "chamber": "sentinel", "deadlock": "sentinel",
}

WEAPON_EFFECTIVENESS = {
    'vandal': 1.0, 'phantom': 1.0, 'spectre': 0.7, 'stinger': 0.6,
    'classic': 0.4, 'ghost': 0.5, 'sheriff': 0.6, 'frenzy': 0.4,
    'shorty': 0.3, 'operator': 1.2, 'guardian': 0.9, 'bulldog': 0.8,
    'marshal': 0.7, 'judge': 0.6, 'bucky': 0.5, 'ares': 0.8, 'odin': 0.9,
}


@dataclass
class KillEvent:
    timestamp_ms: int
    killer_id: str
    killer_side: str
    killer_pos: Tuple[float, float]
    victim_id: str
    victim_side: str


class MockDB:
    async def execute(self, *args, **kwargs): return None
    async def commit(self): pass
    async def rollback(self): pass


class ParameterizedSimulator:
    """Simulator with tunable parameters."""

    MAPS = ['ascent', 'bind', 'haven', 'split', 'icebox', 'breeze', 'pearl']
    AGENTS_ATTACK = ['jett', 'raze', 'omen', 'sova', 'killjoy']
    AGENTS_DEFENSE = ['chamber', 'cypher', 'viper', 'fade', 'sage']

    def __init__(self, params: Dict):
        self.params = params
        self.pathfinder = AStarPathfinder()
        self.combat_system = CombatSystemV2()

    def get_hs_rate(self, agent: str) -> float:
        role = AGENT_ROLES.get(agent, "duelist")
        return ROLE_HS_RATES.get(role, 0.20)

    async def test_scenarios(self, rounds_per_scenario: int = 100) -> Dict:
        """Run key scenarios and return accuracy."""
        results = {}
        good_count = 0

        for scenario in Scenario:
            result = await self._run_scenario(scenario, rounds_per_scenario)
            results[scenario.value] = result

            ref = VCT_REFERENCE.get(scenario, {})
            target = ref.get('attack_win_rate', 0.5) * 100
            actual = result['attack_win_rate'] * 100

            if abs(actual - target) < 15:
                good_count += 1

        accuracy = good_count / len(Scenario) * 100
        return {'accuracy': accuracy, 'results': results}

    async def _run_scenario(self, scenario: Scenario, num_rounds: int) -> Dict:
        attack_wins = 0
        rounds_per_map = max(1, num_rounds // len(self.MAPS))

        for map_name in self.MAPS:
            self.pathfinder.load_nav_grid(map_name)

            for _ in range(rounds_per_map):
                result = await self._simulate(scenario, map_name)
                if result and result['winner'] == 'attack':
                    attack_wins += 1

        total = rounds_per_map * len(self.MAPS)
        return {'attack_win_rate': attack_wins / max(1, total), 'total_rounds': total}

    async def _simulate(self, scenario: Scenario, map_name: str):
        try:
            db = MockDB()
            engine = SimulationEngine(db)
            engine.map_name = map_name
            map_data = engine.MAP_DATA.get(map_name, engine.DEFAULT_MAP_DATA)

            config = self._get_config(scenario, map_data)
            self._setup_players(engine, map_data, config)

            return await self._run_simulation(engine, map_data, config)
        except:
            return None

    def _get_config(self, scenario: Scenario, map_data: Dict) -> Dict:
        config = {
            'num_attackers': 5, 'num_defenders': 5,
            'attacker_weapons': ['vandal'] * 5, 'defender_weapons': ['vandal'] * 5,
            'attacker_armor': [50] * 5, 'defender_armor': [50] * 5,
            'spike_planted': False, 'spike_site': None,
            'start_time_ms': 0, 'is_retake': False, 'is_clutch': False,
        }

        if scenario == Scenario.ECO_VS_FULL:
            config['attacker_weapons'] = ['classic'] * 5
            config['attacker_armor'] = [0] * 5
        elif scenario == Scenario.FULL_VS_ECO:
            config['defender_weapons'] = ['classic'] * 5
            config['defender_armor'] = [0] * 5
        elif scenario == Scenario.CLUTCH_1V1:
            config['num_attackers'] = 1
            config['num_defenders'] = 1
            config['is_clutch'] = True
        elif scenario == Scenario.CLUTCH_1V2:
            config['num_attackers'] = 1
            config['num_defenders'] = 2
            config['is_clutch'] = True
        elif scenario == Scenario.MAN_ADVANTAGE_4V5:
            config['num_attackers'] = 4
        elif scenario == Scenario.MAN_ADVANTAGE_5V4:
            config['num_defenders'] = 4
        elif scenario == Scenario.MAN_ADVANTAGE_3V5:
            config['num_attackers'] = 3

        return config

    def _setup_players(self, engine, map_data, config):
        attack_spawns = map_data['spawns']['attack']
        defense_spawns = map_data['spawns']['defense']

        for i in range(config['num_attackers']):
            player_id = f"attack_{i}"
            spawn = attack_spawns[i % len(attack_spawns)]
            player = SimulatedPlayer(
                player_id=player_id, team_id="attack", side="attack",
                x=spawn[0], y=spawn[1],
                agent=self.AGENTS_ATTACK[i % len(self.AGENTS_ATTACK)],
                has_spike=(i == 0)
            )
            player.weapon = WeaponDatabase.get_weapon(
                config['attacker_weapons'][i % len(config['attacker_weapons'])])
            player.shield = config['attacker_armor'][i % len(config['attacker_armor'])]
            engine.players[player_id] = player
            engine.info_manager.initialize_player(player_id, "attack")

        for i in range(config['num_defenders']):
            player_id = f"defense_{i}"
            spawn = defense_spawns[i % len(defense_spawns)]
            player = SimulatedPlayer(
                player_id=player_id, team_id="defense", side="defense",
                x=spawn[0], y=spawn[1],
                agent=self.AGENTS_DEFENSE[i % len(self.AGENTS_DEFENSE)]
            )
            player.weapon = WeaponDatabase.get_weapon(
                config['defender_weapons'][i % len(config['defender_weapons'])])
            player.shield = config['defender_armor'][i % len(config['defender_armor'])]
            engine.players[player_id] = player
            engine.info_manager.initialize_player(player_id, "defense")

    async def _run_simulation(self, engine, map_data, config) -> Dict:
        time_ms = config['start_time_ms']
        max_time = 100000
        tick_ms = 128
        kills = []

        while time_ms < max_time:
            attack_alive = [p for p in engine.players.values() if p.side == 'attack' and p.is_alive]
            defense_alive = [p for p in engine.players.values() if p.side == 'defense' and p.is_alive]

            if not attack_alive or not defense_alive:
                break

            # Movement
            for player in engine.players.values():
                if not player.is_alive:
                    continue
                target = self._get_target(player, engine, map_data, config)
                dx = target[0] - player.x
                dy = target[1] - player.y
                dist = math.sqrt(dx*dx + dy*dy)
                speed = 0.015
                if dist > 0.01:
                    player.x += (dx / dist) * min(speed, dist)
                    player.y += (dy / dist) * min(speed, dist)
                    player.x = max(0.05, min(0.95, player.x))
                    player.y = max(0.05, min(0.95, player.y))
                    player.is_moving = True
                else:
                    player.is_moving = False

            # Combat with trade system
            for player_a in list(engine.players.values()):
                if not player_a.is_alive:
                    continue
                for player_b in list(engine.players.values()):
                    if not player_b.is_alive or player_b.side == player_a.side:
                        continue

                    dist = math.sqrt((player_a.x - player_b.x)**2 + (player_a.y - player_b.y)**2)
                    if dist > 0.3:
                        continue

                    if not self.pathfinder.has_line_of_sight(
                        (player_a.x, player_a.y), (player_b.x, player_b.y)):
                        continue

                    combat_chance = 0.02 + 0.08 * (1 - dist / 0.3)
                    if random.random() < combat_chance:
                        # IMPORTANT: Players stop moving when they engage in combat
                        # In real Valorant, you stop to aim and shoot (run and gun is inaccurate)
                        player_a.is_moving = False
                        player_b.is_moving = False
                        winner_id, loser_id = self._resolve_combat(
                            player_a, player_b, config, dist)

                        if winner_id == player_a.player_id:
                            winner, loser = player_a, player_b
                        else:
                            winner, loser = player_b, player_a

                        loser.is_alive = False
                        kill_event = KillEvent(
                            timestamp_ms=time_ms,
                            killer_id=winner.player_id,
                            killer_side=winner.side,
                            killer_pos=(winner.x, winner.y),
                            victim_id=loser.player_id,
                            victim_side=loser.side,
                        )
                        kills.append(kill_event)

                        # Trade check
                        trade_result = self._check_for_trade(kill_event, engine, time_ms)
                        if trade_result:
                            winner.is_alive = False

                        break

                if kills and kills[-1].timestamp_ms == time_ms:
                    break

            time_ms += tick_ms

        attack_alive = len([p for p in engine.players.values() if p.side == 'attack' and p.is_alive])
        defense_alive = len([p for p in engine.players.values() if p.side == 'defense' and p.is_alive])

        if attack_alive == 0:
            winner = 'defense'
        elif defense_alive == 0:
            winner = 'attack'
        else:
            winner = 'defense'

        return {'winner': winner, 'kills': len(kills)}

    def _check_for_trade(self, kill_event: KillEvent, engine, current_time_ms: int):
        potential_traders = [
            p for p in engine.players.values()
            if p.is_alive and p.side == kill_event.victim_side
            and p.player_id != kill_event.victim_id
        ]

        if not potential_traders:
            return None

        killer = engine.players.get(kill_event.killer_id)
        killer_weapon = killer.weapon.name.lower() if killer and killer.weapon else 'vandal'
        killer_pos = kill_event.killer_pos

        max_trade_dist = self.params.get('max_trade_distance', 0.20)
        base_trade_prob = self.params.get('base_trade_prob', 0.70)

        for trader in potential_traders:
            dist = math.sqrt((trader.x - killer_pos[0])**2 + (trader.y - killer_pos[1])**2)

            if dist > max_trade_dist:
                continue

            if not self.pathfinder.has_line_of_sight((trader.x, trader.y), killer_pos):
                continue

            # Distance factor
            dist_factor = 1.0 - (dist / max_trade_dist) * 0.5

            # Weapon factor
            trader_weapon = trader.weapon.name.lower() if trader.weapon else 'classic'
            trader_eff = WEAPON_EFFECTIVENESS.get(trader_weapon, 0.7)
            killer_eff = WEAPON_EFFECTIVENESS.get(killer_weapon, 1.0)
            weapon_factor = max(0.2, min(1.2, trader_eff / killer_eff))

            trade_prob = base_trade_prob * dist_factor * weapon_factor

            if random.random() < trade_prob:
                return {'trader_id': trader.player_id}

        return None

    def _resolve_combat(self, player_a, player_b, config, distance: float):
        """
        Resolve combat between two players.

        Position advantage is based on actual player SIDES, not loop order:
        - Defense players hold angles (have position) when spike NOT planted
        - Attack players hold site (have position) when spike IS planted
        """
        weapon_a = player_a.weapon.name.lower() if player_a.weapon else 'classic'
        weapon_b = player_b.weapon.name.lower() if player_b.weapon else 'classic'

        state_a = CombatState(
            health=100, armor=player_a.shield,
            is_moving=getattr(player_a, 'is_moving', False),
            distance_units=distance * 4000,
        )
        state_b = CombatState(
            health=100, armor=player_b.shield,
            is_moving=getattr(player_b, 'is_moving', False),
            distance_units=distance * 4000,
        )

        hs_rate_a = self.get_hs_rate(player_a.agent)
        hs_rate_b = self.get_hs_rate(player_b.agent)

        # Position based on actual sides, but PROBABILISTIC
        # Not every engagement has clear position advantage
        spike_planted = config.get('spike_planted', False)
        position_pct = self.params.get('position_advantage_pct', 0.30)

        # Determine if this engagement has a clear position holder
        has_clear_position = random.random() < position_pct

        if has_clear_position:
            # When spike NOT planted: defenders hold angles (have position)
            # When spike IS planted: attackers hold site (have position)
            player_a_has_position = (
                (player_a.side == 'defense' and not spike_planted) or
                (player_a.side == 'attack' and spike_planted)
            )
            player_b_has_position = (
                (player_b.side == 'defense' and not spike_planted) or
                (player_b.side == 'attack' and spike_planted)
            )
        else:
            # Open engagement, no one has position
            player_a_has_position = False
            player_b_has_position = False

        result = self.combat_system.resolve_duel(
            player_a.player_id, weapon_a, state_a, hs_rate_a,
            player_b.player_id, weapon_b, state_b, hs_rate_b,
            player_a_has_position=player_a_has_position,
            player_b_has_position=player_b_has_position,
        )

        return result.winner_id, result.loser_id

    def _get_target(self, player, engine, map_data, config):
        if player.side == 'attack':
            site = random.choice(list(map_data['sites'].keys()))
            site_data = map_data['sites'][site]
            return (site_data['center'][0] + random.uniform(-0.1, 0.1),
                    site_data['center'][1] + random.uniform(-0.1, 0.1))
        else:
            return (player.x + random.uniform(-0.02, 0.02),
                    player.y + random.uniform(-0.02, 0.02))


async def run_param_sweep():
    """Run parameter sweep to find optimal configuration."""
    print(f"\n{'='*70}")
    print("PARAMETER TUNING - FINDING OPTIMAL CONFIGURATION")
    print(f"{'='*70}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Parameter ranges to test
    param_grid = {
        'max_trade_distance': [0.10, 0.15],  # Smaller trade distance = less trading
        'base_trade_prob': [0.30, 0.50],  # Lower prob = less trading
        'position_advantage_pct': [0.0, 0.15],  # Low position advantage
    }

    # Generate all combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))

    print(f"\nTesting {len(combinations)} parameter combinations...")
    print(f"Scenarios: {len(Scenario)}, Rounds per scenario: 100")
    print()

    best_accuracy = 0
    best_params = None
    all_results = []

    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        print(f"[{i+1}/{len(combinations)}] Testing {params}...", end=" ", flush=True)

        simulator = ParameterizedSimulator(params)
        result = await simulator.test_scenarios(rounds_per_scenario=100)

        accuracy = result['accuracy']
        all_results.append({'params': params, 'accuracy': accuracy, 'results': result['results']})

        status = "★" if accuracy > best_accuracy else " "
        print(f"Accuracy: {accuracy:.0f}% {status}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = params

    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")

    print(f"\nBest configuration: {best_params}")
    print(f"Best accuracy: {best_accuracy:.0f}%")

    print("\nAll results (sorted by accuracy):")
    for res in sorted(all_results, key=lambda x: x['accuracy'], reverse=True):
        print(f"  {res['accuracy']:.0f}% - {res['params']}")

    # Save results
    output_file = Path(__file__).parent / 'param_tuning_results.json'
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'best_params': best_params,
            'best_accuracy': best_accuracy,
            'all_results': all_results,
        }, f, indent=2)

    print(f"\nResults saved to: {output_file}")
    return best_params, best_accuracy


async def main():
    best_params, best_accuracy = await run_param_sweep()

    # Run consistency test with best params
    print(f"\n{'='*70}")
    print("CONSISTENCY TEST - Running 5 trials with best parameters")
    print(f"{'='*70}")

    accuracies = []
    for trial in range(5):
        print(f"Trial {trial + 1}/5...", end=" ", flush=True)
        simulator = ParameterizedSimulator(best_params)
        result = await simulator.test_scenarios(rounds_per_scenario=100)
        accuracies.append(result['accuracy'])
        print(f"Accuracy: {result['accuracy']:.0f}%")

    avg_accuracy = sum(accuracies) / len(accuracies)
    std_dev = (sum((a - avg_accuracy)**2 for a in accuracies) / len(accuracies)) ** 0.5

    print(f"\nMean accuracy: {avg_accuracy:.1f}% ± {std_dev:.1f}%")
    print(f"Range: {min(accuracies):.0f}% - {max(accuracies):.0f}%")


if __name__ == '__main__':
    asyncio.run(main())
