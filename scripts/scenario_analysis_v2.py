#!/usr/bin/env python3
"""
Scenario Analysis V2 - With Fixes for Identified Weaknesses

Fixes implemented:
1. RETAKE: Defenders get coordination bonus, utility advantage, time pressure on attackers
2. MAN DISADVANTAGE: Numbers affect crossfire potential and trade ability
3. CLUTCH: Solo player gets clutch factor bonus (focus, no comms noise)
4. POST-PLANT: Attackers have time advantage but defenders have urgency focus
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


class MockDB:
    async def execute(self, *args, **kwargs): return None
    async def commit(self): pass
    async def rollback(self): pass


class ImprovedScenarioAnalyzer:
    """Scenario analyzer with fixes for identified weaknesses."""

    MAPS = ['ascent', 'bind', 'haven', 'split', 'icebox', 'breeze', 'pearl']
    AGENTS_ATTACK = ['jett', 'raze', 'omen', 'sova', 'killjoy']
    AGENTS_DEFENSE = ['chamber', 'cypher', 'viper', 'fade', 'sage']

    def __init__(self):
        self.pathfinder = AStarPathfinder()
        self.results: Dict[Scenario, Dict] = {}

    async def run_all_scenarios(self, rounds_per_scenario: int = 50):
        """Run all scenarios with improved combat model."""
        print(f"\n{'='*70}")
        print("IMPROVED SCENARIO ANALYSIS (V2)")
        print(f"{'='*70}")
        print(f"Testing {len(Scenario)} scenarios with {rounds_per_scenario} rounds each")
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
        """Run a scenario with improved mechanics."""
        attack_wins = 0
        plants = 0
        total_duration = 0
        total_kills = 0

        rounds_per_map = max(1, num_rounds // len(self.MAPS))

        for map_name in self.MAPS:
            self.pathfinder.load_nav_grid(map_name)

            for _ in range(rounds_per_map):
                result = await self._simulate_improved(scenario, map_name)
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

    async def _simulate_improved(self, scenario: Scenario, map_name: str) -> Optional[Dict]:
        """Simulate with improved combat mechanics."""
        try:
            db = MockDB()
            engine = SimulationEngine(db)
            engine.map_name = map_name
            map_data = engine.MAP_DATA.get(map_name, engine.DEFAULT_MAP_DATA)

            config = self._get_config(scenario, map_data)
            self._setup_players(engine, map_data, config)

            return await self._run_improved_loop(engine, map_data, scenario, config)

        except Exception as e:
            return None

    def _get_config(self, scenario: Scenario, map_data: Dict) -> Dict:
        """Get scenario configuration."""
        config = {
            'num_attackers': 5,
            'num_defenders': 5,
            'attacker_weapons': ['vandal'] * 5,
            'defender_weapons': ['vandal'] * 5,
            'spike_planted': False,
            'spike_site': None,
            'start_time_ms': 0,
            'is_retake': False,
            'is_clutch': False,
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

        elif scenario == Scenario.RETAKE:
            config['spike_planted'] = True
            config['spike_site'] = random.choice(list(map_data['sites'].keys()))
            config['start_time_ms'] = 65000
            config['is_retake'] = True

        elif scenario == Scenario.ECO_VS_FULL:
            config['attacker_weapons'] = ['classic'] * 5

        elif scenario == Scenario.FULL_VS_ECO:
            config['defender_weapons'] = ['classic'] * 5

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
        """Setup players for scenario."""
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
            player.weapon = WeaponDatabase.get_weapon(config['attacker_weapons'][i % len(config['attacker_weapons'])])
            player.shield = 50 if config['attacker_weapons'][0] == 'vandal' else 0
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
            player.weapon = WeaponDatabase.get_weapon(config['defender_weapons'][i % len(config['defender_weapons'])])
            player.shield = 50 if config['defender_weapons'][0] == 'vandal' else 0
            engine.players[player_id] = player
            engine.info_manager.initialize_player(player_id, "defense")

        if config['spike_planted']:
            engine.spike_planted = True
            engine.spike_site = config['spike_site']
            engine.spike_plant_time = config['start_time_ms'] - 5000

    async def _run_improved_loop(self, engine, map_data, scenario, config) -> Dict:
        """Run simulation with improved combat model."""
        time_ms = config['start_time_ms']
        max_time = 100000
        tick_ms = 128

        kills = []

        while time_ms < max_time:
            attack_alive = [p for p in engine.players.values() if p.side == 'attack' and p.is_alive]
            defense_alive = [p for p in engine.players.values() if p.side == 'defense' and p.is_alive]

            num_attackers = len(attack_alive)
            num_defenders = len(defense_alive)

            if num_attackers == 0 or num_defenders == 0:
                break
            if engine.spike_planted and engine.spike_plant_time:
                if time_ms - engine.spike_plant_time >= 45000:
                    break

            smoke_positions = engine.ability_system.get_active_smokes(time_ms)

            # Movement
            for player in engine.players.values():
                if not player.is_alive:
                    continue

                target = self._get_target(player, engine, map_data, config, scenario)

                dx = target[0] - player.x
                dy = target[1] - player.y
                dist = math.sqrt(dx*dx + dy*dy)

                speed = 0.015
                if config.get('movement_speed') == 'fast':
                    speed = 0.020
                elif config.get('movement_speed') == 'slow':
                    speed = 0.010

                if dist > 0.01:
                    player.prev_x = player.x
                    player.prev_y = player.y
                    player.x += (dx / dist) * min(speed, dist)
                    player.y += (dy / dist) * min(speed, dist)
                    player.x = max(0.05, min(0.95, player.x))
                    player.y = max(0.05, min(0.95, player.y))
                    player.moved_this_tick = True
                else:
                    player.moved_this_tick = False

            # IMPROVED COMBAT with scenario-specific modifiers
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

                    combat_chance = 0.01 + 0.09 * (1 - dist / 0.3)

                    if random.random() < combat_chance:
                        # ========================================
                        # IMPROVED COMBAT ADVANTAGE CALCULATION
                        # ========================================

                        # Base advantage (slight defender edge for holding)
                        adv = 0.45

                        # Peeker's advantage
                        if attacker.moved_this_tick and not defender.moved_this_tick:
                            adv += 0.10

                        # Weapon advantage
                        if attacker.weapon.name != 'Classic' and defender.weapon.name == 'Classic':
                            adv += 0.25
                        elif attacker.weapon.name == 'Classic' and defender.weapon.name != 'Classic':
                            adv -= 0.25

                        # ========================================
                        # FIX 1: NUMBERS ADVANTAGE (crossfire/trade potential)
                        # ========================================
                        # More teammates = better crossfire and trade potential
                        attacker_team_alive = num_attackers if attacker.side == 'attack' else num_defenders
                        defender_team_alive = num_defenders if attacker.side == 'attack' else num_attackers

                        numbers_diff = attacker_team_alive - defender_team_alive
                        # Each extra teammate gives +5% advantage (crossfire, trades)
                        adv += numbers_diff * 0.05

                        # ========================================
                        # FIX 2: RETAKE BONUS FOR DEFENDERS
                        # ========================================
                        if config['is_retake'] and defender.side == 'defense':
                            # Defenders in retake have:
                            # - Coordinated utility (flashes, recon)
                            # - Known spike location
                            # - Attackers have time pressure
                            adv -= 0.15  # Strong defender retake bonus

                            # Time pressure on attackers (less time = more pressure)
                            time_remaining = 45000 - (time_ms - engine.spike_plant_time) if engine.spike_plant_time else 45000
                            if time_remaining < 20000:
                                adv -= 0.10  # Attackers panicking

                        # ========================================
                        # FIX 3: POST-PLANT ATTACKER ADVANTAGE
                        # ========================================
                        if engine.spike_planted and attacker.side == 'attack' and not config['is_retake']:
                            # Attackers can play time, defenders must push
                            adv += 0.10

                        # ========================================
                        # FIX 4: CLUTCH FACTOR
                        # ========================================
                        if config['is_clutch']:
                            # Solo player gets "clutch factor" - focused, no comms noise
                            if attacker_team_alive == 1:
                                adv += 0.08  # Clutch factor for attacker
                            if defender_team_alive == 1:
                                adv -= 0.08  # Clutch factor for defender

                        # ========================================
                        # FIX 5: DEFENDER SITE ADVANTAGE
                        # ========================================
                        if defender.side == 'defense' and not engine.spike_planted:
                            for site_data in map_data['sites'].values():
                                site_x, site_y = site_data['center']
                                if math.sqrt((defender.x - site_x)**2 + (defender.y - site_y)**2) < site_data['radius'] + 0.1:
                                    adv -= 0.05
                                    break

                        # Clamp advantage
                        adv = max(0.15, min(0.85, adv))

                        # Resolve combat
                        if random.random() < adv:
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

    def _get_target(self, player, engine, map_data, config, scenario) -> Tuple[float, float]:
        """Get movement target."""
        if player.side == 'attack':
            if config['spike_planted']:
                # Hold position post-plant
                return (player.x + random.uniform(-0.02, 0.02), player.y + random.uniform(-0.02, 0.02))
            else:
                # Push to site
                site = random.choice(list(map_data['sites'].keys()))
                site_data = map_data['sites'][site]
                return (site_data['center'][0] + random.uniform(-0.1, 0.1),
                        site_data['center'][1] + random.uniform(-0.1, 0.1))
        else:
            if config['is_retake']:
                # Push to planted site
                site_data = map_data['sites'][config['spike_site']]
                return (site_data['center'][0] + random.uniform(-0.1, 0.1),
                        site_data['center'][1] + random.uniform(-0.1, 0.1))
            else:
                # Hold position
                return (player.x + random.uniform(-0.02, 0.02), player.y + random.uniform(-0.02, 0.02))

    def print_comparison(self, old_results: Dict = None):
        """Print comparison with old results."""
        print(f"\n{'='*70}")
        print("IMPROVED MODEL RESULTS")
        print(f"{'='*70}")

        categories = {
            'Standard Attack': [Scenario.ATTACK_EXECUTE_FAST, Scenario.ATTACK_EXECUTE_SLOW],
            'Post-Plant': [Scenario.POST_PLANT_DEFENSE, Scenario.RETAKE],
            'Map Control': [Scenario.MID_CONTROL, Scenario.DEFENDER_AGGRESSION],
            'Economy': [Scenario.ECO_VS_FULL, Scenario.FULL_VS_ECO],
            'Clutch': [Scenario.CLUTCH_1V1, Scenario.CLUTCH_1V2, Scenario.CLUTCH_2V1],
            'Man Advantage': [Scenario.MAN_ADVANTAGE_4V5, Scenario.MAN_ADVANTAGE_5V4, Scenario.MAN_ADVANTAGE_3V5],
        }

        improvements = 0
        regressions = 0

        for category, scenarios in categories.items():
            print(f"\n{category}:")
            for scenario in scenarios:
                if scenario in self.results:
                    result = self.results[scenario]
                    ref = VCT_REFERENCE.get(scenario, {})
                    target = ref.get('attack_win_rate', 0.5) * 100
                    actual = result['attack_win_rate'] * 100
                    diff = actual - target

                    status = "✓" if abs(diff) < 15 else "✗"
                    bar_len = int(actual / 5)
                    bar = "█" * bar_len + "░" * (20 - bar_len)

                    print(f"  {status} {scenario.value:25s} │ {bar} │ {actual:5.1f}% (target: {target:.0f}%, diff: {diff:+.1f}%)")

        # Summary
        good = sum(1 for s, r in self.results.items()
                   if abs(r['attack_win_rate'] - VCT_REFERENCE[s]['attack_win_rate']) < 0.15)
        total = len(self.results)

        print(f"\n{'─'*70}")
        print(f"SUMMARY: {good}/{total} scenarios within 15% of target ({good/total*100:.0f}% accuracy)")
        print(f"{'='*70}")


async def main():
    analyzer = ImprovedScenarioAnalyzer()

    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 42

    await analyzer.run_all_scenarios(rounds_per_scenario=rounds)
    analyzer.print_comparison()

    # Save results
    output_file = Path(__file__).parent / 'scenario_analysis_v2_results.json'
    json_results = {s.value: r for s, r in analyzer.results.items()}

    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': json_results,
        }, f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    asyncio.run(main())
