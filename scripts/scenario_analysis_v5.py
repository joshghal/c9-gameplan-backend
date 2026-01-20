#!/usr/bin/env python3
"""
Scenario Analysis V5: TTK-Based Combat System

Philosophy: Economy outcomes EMERGE from weapon TTK, not hardcoded bonuses.

Key changes from V4:
- Replaced WEAPON_ADVANTAGE_MULTIPLIER with actual TTK calculation
- Combat resolution based on weapon damage, fire rate, accuracy
- Positioning affects timing, not flat bonuses
"""

import asyncio
import json
import sys
import math
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from enum import Enum

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.simulation_engine import SimulationEngine, SimulatedPlayer
from app.services.weapon_system import WeaponDatabase
from app.services.pathfinding import AStarPathfinder
from app.services.combat_system_v2 import CombatSystemV2, CombatState, WEAPON_DATABASE


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


VCT_REFERENCE = {
    Scenario.ATTACK_EXECUTE_FAST: {'attack_win_rate': 0.52},
    Scenario.ATTACK_EXECUTE_SLOW: {'attack_win_rate': 0.45},
    Scenario.POST_PLANT_DEFENSE: {'attack_win_rate': 0.65},
    Scenario.RETAKE: {'attack_win_rate': 0.65},
    Scenario.MID_CONTROL: {'attack_win_rate': 0.48},
    Scenario.DEFENDER_AGGRESSION: {'attack_win_rate': 0.55},
    Scenario.ECO_VS_FULL: {'attack_win_rate': 0.15},
    Scenario.FULL_VS_ECO: {'attack_win_rate': 0.85},
    Scenario.CLUTCH_1V1: {'attack_win_rate': 0.50},
    Scenario.CLUTCH_1V2: {'attack_win_rate': 0.25},
    Scenario.CLUTCH_2V1: {'attack_win_rate': 0.75},
    Scenario.MAN_ADVANTAGE_4V5: {'attack_win_rate': 0.35},
    Scenario.MAN_ADVANTAGE_5V4: {'attack_win_rate': 0.60},
    Scenario.MAN_ADVANTAGE_3V5: {'attack_win_rate': 0.15},
}


# Role-based headshot rates from VCT data
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


class MockDB:
    async def execute(self, *args, **kwargs): return None
    async def commit(self): pass
    async def rollback(self): pass


class TTKScenarioAnalyzer:
    """Scenario analyzer using TTK-based combat resolution."""

    MAPS = ['ascent', 'bind', 'haven', 'split', 'icebox', 'breeze', 'pearl']
    AGENTS_ATTACK = ['jett', 'raze', 'omen', 'sova', 'killjoy']
    AGENTS_DEFENSE = ['chamber', 'cypher', 'viper', 'fade', 'sage']

    def __init__(self):
        self.pathfinder = AStarPathfinder()
        self.combat_system = CombatSystemV2()
        self.results: Dict[Scenario, Dict] = {}

    def get_hs_rate(self, agent: str) -> float:
        """Get headshot rate based on agent role."""
        role = AGENT_ROLES.get(agent, "duelist")
        return ROLE_HS_RATES.get(role, 0.20)

    async def run_all_scenarios(self, rounds_per_scenario: int = 50):
        print(f"\n{'='*70}")
        print("SCENARIO ANALYSIS V5 - TTK-BASED COMBAT")
        print(f"{'='*70}")
        print("Key changes:")
        print("  - Combat resolved via TTK (time-to-kill) calculation")
        print("  - Weapon advantage emerges from damage/fire rate")
        print("  - Positioning affects timing, not flat bonuses")
        print("  - Role-based headshot rates from VCT data")
        print(f"\nTesting {len(Scenario)} scenarios with {rounds_per_scenario} rounds each")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        for scenario in Scenario:
            print(f"Testing {scenario.value}...", end=" ", flush=True)
            result = await self._run_scenario(scenario, rounds_per_scenario)
            self.results[scenario] = result

            ref = VCT_REFERENCE.get(scenario, {})
            target = ref.get('attack_win_rate', 0.5) * 100
            actual = result['attack_win_rate'] * 100
            diff = actual - target
            status = "✓" if abs(diff) < 15 else "✗"

            print(f"{status} Attack Win: {actual:.0f}% (target: {target:.0f}%, diff: {diff:+.0f}%)")

        return self.results

    async def _run_scenario(self, scenario: Scenario, num_rounds: int) -> Dict:
        attack_wins = 0
        plants = 0
        total_duration = 0
        total_kills = 0

        rounds_per_map = max(1, num_rounds // len(self.MAPS))

        for map_name in self.MAPS:
            self.pathfinder.load_nav_grid(map_name)

            for _ in range(rounds_per_map):
                result = await self._simulate(scenario, map_name)
                if result:
                    if result['winner'] == 'attack':
                        attack_wins += 1
                    if result['planted']:
                        plants += 1
                    total_duration += result['duration_ms']
                    total_kills += result['kills']

        total = rounds_per_map * len(self.MAPS)

        return {
            'attack_win_rate': attack_wins / max(1, total),
            'plant_rate': plants / max(1, total),
            'avg_duration_ms': total_duration / max(1, total),
            'avg_kills': total_kills / max(1, total),
            'total_rounds': total,
        }

    async def _simulate(self, scenario: Scenario, map_name: str) -> Optional[Dict]:
        try:
            db = MockDB()
            engine = SimulationEngine(db)
            engine.map_name = map_name
            map_data = engine.MAP_DATA.get(map_name, engine.DEFAULT_MAP_DATA)

            config = self._get_config(scenario, map_data)
            self._setup_players(engine, map_data, config)

            return await self._run_simulation(engine, map_data, scenario, config)

        except Exception as e:
            return None

    def _get_config(self, scenario: Scenario, map_data: Dict) -> Dict:
        config = {
            'num_attackers': 5,
            'num_defenders': 5,
            'attacker_weapons': ['vandal'] * 5,
            'defender_weapons': ['vandal'] * 5,
            'attacker_armor': [50] * 5,
            'defender_armor': [50] * 5,
            'spike_planted': False,
            'spike_site': None,
            'start_time_ms': 0,
            'is_retake': False,
            'is_clutch': False,
            'movement_speed': 'normal',
        }

        if scenario == Scenario.ATTACK_EXECUTE_FAST:
            config['movement_speed'] = 'fast'

        elif scenario == Scenario.ATTACK_EXECUTE_SLOW:
            config['movement_speed'] = 'slow'
            config['start_time_ms'] = 30000

        elif scenario == Scenario.POST_PLANT_DEFENSE:
            config['spike_planted'] = True
            config['spike_site'] = random.choice(list(map_data['sites'].keys()))
            config['start_time_ms'] = 60000
            config['is_retake'] = True

        elif scenario == Scenario.RETAKE:
            config['spike_planted'] = True
            config['spike_site'] = random.choice(list(map_data['sites'].keys()))
            config['start_time_ms'] = 65000
            config['is_retake'] = True

        elif scenario == Scenario.ECO_VS_FULL:
            config['attacker_weapons'] = ['classic'] * 5
            config['attacker_armor'] = [0] * 5

        elif scenario == Scenario.FULL_VS_ECO:
            config['defender_weapons'] = ['classic'] * 5
            config['defender_armor'] = [0] * 5

        elif scenario == Scenario.CLUTCH_1V1:
            config['num_attackers'] = 1
            config['num_defenders'] = 1
            config['is_clutch'] = True
            config['start_time_ms'] = 70000

        elif scenario == Scenario.CLUTCH_1V2:
            config['num_attackers'] = 1
            config['num_defenders'] = 2
            config['is_clutch'] = True
            config['start_time_ms'] = 70000

        elif scenario == Scenario.CLUTCH_2V1:
            config['num_attackers'] = 2
            config['num_defenders'] = 1
            config['is_clutch'] = True
            config['start_time_ms'] = 70000

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

            if config['spike_planted'] and config['spike_site']:
                site_data = map_data['sites'][config['spike_site']]
                spawn = (
                    site_data['center'][0] + random.uniform(-0.08, 0.08),
                    site_data['center'][1] + random.uniform(-0.08, 0.08)
                )
            else:
                spawn = attack_spawns[i % len(attack_spawns)]

            player = SimulatedPlayer(
                player_id=player_id, team_id="attack", side="attack",
                x=spawn[0], y=spawn[1],
                agent=self.AGENTS_ATTACK[i % len(self.AGENTS_ATTACK)],
                has_spike=(i == 0 and not config['spike_planted'])
            )
            weapon_name = config['attacker_weapons'][i % len(config['attacker_weapons'])]
            player.weapon = WeaponDatabase.get_weapon(weapon_name)
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
            weapon_name = config['defender_weapons'][i % len(config['defender_weapons'])]
            player.weapon = WeaponDatabase.get_weapon(weapon_name)
            player.shield = config['defender_armor'][i % len(config['defender_armor'])]
            engine.players[player_id] = player
            engine.info_manager.initialize_player(player_id, "defense")

        if config['spike_planted']:
            engine.spike_planted = True
            engine.spike_site = config['spike_site']
            engine.spike_plant_time = config['start_time_ms'] - 5000

    async def _run_simulation(self, engine, map_data, scenario, config) -> Dict:
        time_ms = config['start_time_ms']
        max_time = 100000
        tick_ms = 128

        kills = []

        while time_ms < max_time:
            attack_alive = [p for p in engine.players.values() if p.side == 'attack' and p.is_alive]
            defense_alive = [p for p in engine.players.values() if p.side == 'defense' and p.is_alive]

            if len(attack_alive) == 0 or len(defense_alive) == 0:
                break
            if engine.spike_planted and engine.spike_plant_time:
                if time_ms - engine.spike_plant_time >= 45000:
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
                if config.get('movement_speed') == 'fast':
                    speed = 0.020
                elif config.get('movement_speed') == 'slow':
                    speed = 0.010

                if dist > 0.01:
                    player.x += (dx / dist) * min(speed, dist)
                    player.y += (dy / dist) * min(speed, dist)
                    player.x = max(0.05, min(0.95, player.x))
                    player.y = max(0.05, min(0.95, player.y))
                    player.is_moving = True
                else:
                    player.is_moving = False

            # Combat using TTK system
            for attacker in list(engine.players.values()):
                if not attacker.is_alive:
                    continue
                for defender in list(engine.players.values()):
                    if not defender.is_alive or defender.side == attacker.side:
                        continue

                    dist = math.sqrt((attacker.x - defender.x)**2 + (attacker.y - defender.y)**2)
                    if dist > 0.3:
                        continue

                    if not self.pathfinder.has_line_of_sight(
                        (attacker.x, attacker.y), (defender.x, defender.y)
                    ):
                        continue

                    # Combat chance scales with distance
                    combat_chance = 0.02 + 0.08 * (1 - dist / 0.3)

                    if random.random() < combat_chance:
                        # Resolve using TTK-based combat
                        winner_id, loser_id = self._resolve_combat_ttk(
                            attacker, defender, config, dist
                        )

                        # Apply result
                        if winner_id == attacker.player_id:
                            defender.is_alive = False
                            kills.append({'time_ms': time_ms, 'killer_side': attacker.side})
                        else:
                            attacker.is_alive = False
                            kills.append({'time_ms': time_ms, 'killer_side': defender.side})
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
                            defenders_watching = sum(1 for d in engine.players.values()
                                                   if d.side == 'defense' and d.is_alive and
                                                   self.pathfinder.has_line_of_sight((carrier.x, carrier.y), (d.x, d.y)))

                            plant_chance = 0.04
                            if defenders_watching == 0:
                                plant_chance += 0.12
                            elif defenders_watching >= 2:
                                plant_chance *= 0.2

                            if random.random() < plant_chance:
                                engine.spike_planted = True
                                engine.spike_plant_time = time_ms
                                break

            time_ms += tick_ms

        # Determine winner
        attack_alive = len([p for p in engine.players.values() if p.side == 'attack' and p.is_alive])
        defense_alive = len([p for p in engine.players.values() if p.side == 'defense' and p.is_alive])

        if attack_alive == 0:
            winner = 'defense'
        elif defense_alive == 0:
            winner = 'attack'
        elif engine.spike_planted and engine.spike_plant_time and time_ms - engine.spike_plant_time >= 45000:
            winner = 'attack'
        else:
            winner = 'defense'

        return {
            'winner': winner,
            'kills': len(kills),
            'planted': engine.spike_planted,
            'duration_ms': time_ms - config['start_time_ms'],
        }

    def _resolve_combat_ttk(self, attacker, defender, config, distance: float) -> Tuple[str, str]:
        """
        Resolve combat using TTK-based system.

        Returns: (winner_id, loser_id)
        """
        # Get weapon names
        attacker_weapon = attacker.weapon.name.lower() if attacker.weapon else 'classic'
        defender_weapon = defender.weapon.name.lower() if defender.weapon else 'classic'

        # Build combat states
        attacker_state = CombatState(
            health=100,
            armor=attacker.shield,
            is_moving=getattr(attacker, 'is_moving', False),
            distance_units=distance * 4000,  # Convert to game units (0.3 map = ~12m)
        )

        defender_state = CombatState(
            health=100,
            armor=defender.shield,
            is_moving=getattr(defender, 'is_moving', False),
            distance_units=distance * 4000,
        )

        # Get role-based headshot rates
        attacker_hs = self.get_hs_rate(attacker.agent)
        defender_hs = self.get_hs_rate(defender.agent)

        # Determine positioning advantage
        # Defenders holding site have position, attackers pushing don't
        attacker_has_position = config.get('spike_planted', False)  # Post-plant attackers have position
        defender_has_position = not config.get('spike_planted', False) and not config.get('is_retake', False)

        # Resolve duel
        result = self.combat_system.resolve_duel(
            attacker.player_id, attacker_weapon, attacker_state, attacker_hs,
            defender.player_id, defender_weapon, defender_state, defender_hs,
            player_a_has_position=attacker_has_position,
            player_b_has_position=defender_has_position,
        )

        return result.winner_id, result.loser_id

    def _get_target(self, player, engine, map_data, config) -> Tuple[float, float]:
        if player.side == 'attack':
            if config.get('spike_planted'):
                return (player.x + random.uniform(-0.02, 0.02), player.y + random.uniform(-0.02, 0.02))
            else:
                site = random.choice(list(map_data['sites'].keys()))
                site_data = map_data['sites'][site]
                return (site_data['center'][0] + random.uniform(-0.1, 0.1),
                        site_data['center'][1] + random.uniform(-0.1, 0.1))
        else:
            if config.get('is_retake'):
                site_data = map_data['sites'][config['spike_site']]
                return (site_data['center'][0] + random.uniform(-0.1, 0.1),
                        site_data['center'][1] + random.uniform(-0.1, 0.1))
            else:
                return (player.x + random.uniform(-0.02, 0.02), player.y + random.uniform(-0.02, 0.02))

    def print_results(self):
        print(f"\n{'='*70}")
        print("V5 RESULTS - TTK-BASED COMBAT")
        print(f"{'='*70}")

        categories = {
            'Standard Attack': [Scenario.ATTACK_EXECUTE_FAST, Scenario.ATTACK_EXECUTE_SLOW],
            'Post-Plant': [Scenario.POST_PLANT_DEFENSE, Scenario.RETAKE],
            'Map Control': [Scenario.MID_CONTROL, Scenario.DEFENDER_AGGRESSION],
            'Economy': [Scenario.ECO_VS_FULL, Scenario.FULL_VS_ECO],
            'Clutch': [Scenario.CLUTCH_1V1, Scenario.CLUTCH_1V2, Scenario.CLUTCH_2V1],
            'Man Advantage': [Scenario.MAN_ADVANTAGE_4V5, Scenario.MAN_ADVANTAGE_5V4, Scenario.MAN_ADVANTAGE_3V5],
        }

        good_count = 0
        total_count = 0

        for category, scenarios in categories.items():
            print(f"\n{category}:")
            for scenario in scenarios:
                if scenario in self.results:
                    result = self.results[scenario]
                    ref = VCT_REFERENCE.get(scenario, {})
                    target = ref.get('attack_win_rate', 0.5) * 100
                    actual = result['attack_win_rate'] * 100
                    diff = actual - target

                    is_good = abs(diff) < 15
                    if is_good:
                        good_count += 1
                    total_count += 1

                    status = "✓" if is_good else "✗"
                    bar_len = int(actual / 5)
                    bar = "█" * bar_len + "░" * (20 - bar_len)

                    print(f"  {status} {scenario.value:25s} │ {bar} │ {actual:5.1f}% (target: {target:.0f}%, diff: {diff:+.1f}%)")

        accuracy = good_count / total_count * 100 if total_count > 0 else 0

        print(f"\n{'─'*70}")
        print(f"SUMMARY: {good_count}/{total_count} scenarios within 15% of target ({accuracy:.0f}% accuracy)")
        print(f"{'='*70}")

        return accuracy


async def main():
    analyzer = TTKScenarioAnalyzer()

    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 50

    await analyzer.run_all_scenarios(rounds_per_scenario=rounds)
    accuracy = analyzer.print_results()

    output_file = Path(__file__).parent / 'scenario_analysis_v5_results.json'
    json_results = {s.value: r for s, r in analyzer.results.items()}

    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'accuracy': accuracy,
            'version': 'v5_ttk_based',
            'key_changes': [
                'ttk_based_combat',
                'no_weapon_advantage_multiplier',
                'positioning_affects_timing',
                'role_based_headshot_rates',
            ],
            'results': json_results,
        }, f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    asyncio.run(main())
