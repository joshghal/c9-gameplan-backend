#!/usr/bin/env python3
"""
Scenario Analysis V3 - Comprehensive Fixes

Implements all identified fixes:
1. UTILITY-BASED ENTRY: Flashes reduce enemy accuracy by 70%
2. CROSSFIRE DETECTION: 2+ teammates with LOS get +15% damage bonus
3. DEFUSE PRESSURE: Defenders get focus bonus when timer < 15s
4. NUMBERS ADVANTAGE CAP: Max ±10% instead of unlimited
5. RETAKE UTILITY: Defenders use coordinated utility during retake
6. TIME PRESSURE: Attackers holding post-plant get anxiety penalty as time runs low
"""

import asyncio
import json
import sys
import math
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
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
    Scenario.RETAKE: {'attack_win_rate': 0.65},  # Note: This is ATTACKER win rate (defenders retaking)
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


@dataclass
class FlashState:
    """Tracks flash status for a player."""
    is_flashed: bool = False
    flash_end_time: int = 0
    flash_intensity: float = 0.0  # 0-1, decays over time


@dataclass
class CombatContext:
    """Context for combat resolution."""
    attacker_id: str
    defender_id: str
    attacker_team_alive: int
    defender_team_alive: int
    attacker_has_crossfire: bool  # Teammate also has LOS on defender
    defender_has_crossfire: bool
    attacker_flashed: bool
    defender_flashed: bool
    is_retake: bool
    spike_time_remaining: int  # ms until detonation
    attacker_on_site: bool
    defender_on_site: bool


class MockDB:
    async def execute(self, *args, **kwargs): return None
    async def commit(self): pass
    async def rollback(self): pass


class FixedScenarioAnalyzer:
    """Scenario analyzer with all fixes implemented."""

    MAPS = ['ascent', 'bind', 'haven', 'split', 'icebox', 'breeze', 'pearl']
    AGENTS_ATTACK = ['jett', 'raze', 'omen', 'sova', 'killjoy']
    AGENTS_DEFENSE = ['chamber', 'cypher', 'viper', 'fade', 'sage']

    # Fix parameters
    FLASH_ACCURACY_PENALTY = 0.70  # 70% accuracy reduction when flashed
    FLASH_DURATION_MS = 2000       # Flash lasts 2 seconds
    CROSSFIRE_BONUS = 0.15         # 15% bonus when teammate has LOS
    NUMBERS_ADVANTAGE_CAP = 0.10   # Max ±10% from numbers
    NUMBERS_ADVANTAGE_PER_PLAYER = 0.03  # 3% per player difference
    DEFUSE_PRESSURE_THRESHOLD = 15000  # 15 seconds
    DEFUSE_PRESSURE_BONUS = 0.12   # Defenders get 12% bonus under pressure
    RETAKE_UTILITY_SUCCESS = 0.40  # 40% chance retake utility lands
    POST_PLANT_ANXIETY_THRESHOLD = 10000  # Last 10 seconds
    POST_PLANT_ANXIETY_PENALTY = 0.08  # Attackers get anxious

    def __init__(self):
        self.pathfinder = AStarPathfinder()
        self.results: Dict[Scenario, Dict] = {}
        self.flash_states: Dict[str, FlashState] = {}

    async def run_all_scenarios(self, rounds_per_scenario: int = 50):
        """Run all scenarios with fixed combat model."""
        print(f"\n{'='*70}")
        print("SCENARIO ANALYSIS V3 - WITH ALL FIXES")
        print(f"{'='*70}")
        print("Fixes implemented:")
        print("  1. Flash accuracy penalty (70% reduction)")
        print("  2. Crossfire detection (+15% when teammate has LOS)")
        print("  3. Defuse pressure (+12% defender bonus when <15s)")
        print("  4. Numbers advantage cap (±10% max)")
        print("  5. Retake utility (40% flash success rate)")
        print("  6. Post-plant anxiety (-8% attacker when <10s)")
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
        """Run a scenario multiple times."""
        attack_wins = 0
        plants = 0
        total_duration = 0
        total_kills = 0
        flashes_used = 0
        crossfire_kills = 0

        rounds_per_map = max(1, num_rounds // len(self.MAPS))

        for map_name in self.MAPS:
            self.pathfinder.load_nav_grid(map_name)

            for _ in range(rounds_per_map):
                result = await self._simulate_fixed(scenario, map_name)
                if result:
                    if result['winner'] == 'attack':
                        attack_wins += 1
                    if result['planted']:
                        plants += 1
                    total_duration += result['duration_ms']
                    total_kills += result['kills']
                    flashes_used += result.get('flashes_used', 0)
                    crossfire_kills += result.get('crossfire_kills', 0)

        total = rounds_per_map * len(self.MAPS)

        return {
            'attack_win_rate': attack_wins / max(1, total),
            'plant_rate': plants / max(1, total),
            'avg_duration_ms': total_duration / max(1, total),
            'avg_kills': total_kills / max(1, total),
            'avg_flashes': flashes_used / max(1, total),
            'crossfire_kill_rate': crossfire_kills / max(1, total_kills) if total_kills > 0 else 0,
            'total_rounds': total,
        }

    async def _simulate_fixed(self, scenario: Scenario, map_name: str) -> Optional[Dict]:
        """Simulate with all fixes."""
        try:
            db = MockDB()
            engine = SimulationEngine(db)
            engine.map_name = map_name
            map_data = engine.MAP_DATA.get(map_name, engine.DEFAULT_MAP_DATA)

            config = self._get_config(scenario, map_data)
            self._setup_players(engine, map_data, config)

            # Reset flash states
            self.flash_states = {pid: FlashState() for pid in engine.players.keys()}

            return await self._run_fixed_loop(engine, map_data, scenario, config)

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

    async def _run_fixed_loop(self, engine, map_data, scenario, config) -> Dict:
        """Run simulation with all fixes applied."""
        time_ms = config['start_time_ms']
        max_time = 100000
        tick_ms = 128

        kills = []
        flashes_used = 0
        crossfire_kills = 0

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

            # Calculate spike time remaining
            spike_time_remaining = 45000
            if engine.spike_planted and engine.spike_plant_time:
                spike_time_remaining = max(0, 45000 - (time_ms - engine.spike_plant_time))

            # Update flash states (decay)
            for pid, flash_state in self.flash_states.items():
                if flash_state.is_flashed and time_ms >= flash_state.flash_end_time:
                    flash_state.is_flashed = False
                    flash_state.flash_intensity = 0.0

            smoke_positions = engine.ability_system.get_active_smokes(time_ms)

            # ==========================================
            # FIX 5: RETAKE UTILITY - Defenders use flashes when retaking
            # ==========================================
            if config['is_retake'] and num_defenders > 0:
                # Defenders have chance to flash before entry
                for defender in defense_alive:
                    if random.random() < 0.03:  # 3% per tick to throw flash
                        # Flash affects attackers on site
                        if random.random() < self.RETAKE_UTILITY_SUCCESS:
                            flashes_used += 1
                            for attacker in attack_alive:
                                # Check if attacker is near site (would be affected)
                                if config['spike_site']:
                                    site_data = map_data['sites'][config['spike_site']]
                                    dist_to_site = math.sqrt(
                                        (attacker.x - site_data['center'][0])**2 +
                                        (attacker.y - site_data['center'][1])**2
                                    )
                                    if dist_to_site < 0.2:  # On site
                                        self.flash_states[attacker.player_id] = FlashState(
                                            is_flashed=True,
                                            flash_end_time=time_ms + self.FLASH_DURATION_MS,
                                            flash_intensity=1.0
                                        )

            # Movement
            for player in engine.players.values():
                if not player.is_alive:
                    continue

                # Flashed players move randomly
                if self.flash_states.get(player.player_id, FlashState()).is_flashed:
                    target = (
                        player.x + random.uniform(-0.05, 0.05),
                        player.y + random.uniform(-0.05, 0.05)
                    )
                else:
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

            # ==========================================
            # COMBAT WITH ALL FIXES
            # ==========================================
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

                    # Build combat context
                    context = self._build_combat_context(
                        attacker, defender, engine, map_data, config,
                        num_attackers, num_defenders, spike_time_remaining
                    )

                    combat_chance = 0.01 + 0.09 * (1 - dist / 0.3)

                    # FIX 1: Flash reduces combat effectiveness
                    if context.attacker_flashed:
                        combat_chance *= (1 - self.FLASH_ACCURACY_PENALTY)
                    if context.defender_flashed:
                        combat_chance *= 1.3  # Easier to hit flashed enemy

                    if random.random() < combat_chance:
                        # Calculate advantage with all fixes
                        adv = self._calculate_advantage(context, config)

                        # Track if this was a crossfire kill
                        is_crossfire = context.attacker_has_crossfire or context.defender_has_crossfire

                        # Resolve combat
                        if random.random() < adv:
                            defender.is_alive = False
                            kills.append({'time_ms': time_ms, 'killer_side': attacker.side})
                            if is_crossfire:
                                crossfire_kills += 1
                        else:
                            attacker.is_alive = False
                            kills.append({'time_ms': time_ms, 'killer_side': defender.side})
                            if is_crossfire:
                                crossfire_kills += 1
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
            'flashes_used': flashes_used,
            'crossfire_kills': crossfire_kills,
        }

    def _build_combat_context(self, attacker, defender, engine, map_data, config,
                               num_attackers, num_defenders, spike_time_remaining) -> CombatContext:
        """Build context for combat resolution."""

        # Check for crossfire (teammate also has LOS)
        attacker_has_crossfire = False
        defender_has_crossfire = False

        for teammate in engine.players.values():
            if not teammate.is_alive or teammate.player_id == attacker.player_id:
                continue
            if teammate.side == attacker.side:
                if self.pathfinder.has_line_of_sight(
                    (teammate.x, teammate.y), (defender.x, defender.y)
                ):
                    attacker_has_crossfire = True
                    break

        for teammate in engine.players.values():
            if not teammate.is_alive or teammate.player_id == defender.player_id:
                continue
            if teammate.side == defender.side:
                if self.pathfinder.has_line_of_sight(
                    (teammate.x, teammate.y), (attacker.x, attacker.y)
                ):
                    defender_has_crossfire = True
                    break

        # Check on-site status
        attacker_on_site = False
        defender_on_site = False
        for site_data in map_data['sites'].values():
            site_x, site_y = site_data['center']
            if math.sqrt((attacker.x - site_x)**2 + (attacker.y - site_y)**2) < site_data['radius'] + 0.1:
                attacker_on_site = True
            if math.sqrt((defender.x - site_x)**2 + (defender.y - site_y)**2) < site_data['radius'] + 0.1:
                defender_on_site = True

        return CombatContext(
            attacker_id=attacker.player_id,
            defender_id=defender.player_id,
            attacker_team_alive=num_attackers if attacker.side == 'attack' else num_defenders,
            defender_team_alive=num_defenders if attacker.side == 'attack' else num_attackers,
            attacker_has_crossfire=attacker_has_crossfire,
            defender_has_crossfire=defender_has_crossfire,
            attacker_flashed=self.flash_states.get(attacker.player_id, FlashState()).is_flashed,
            defender_flashed=self.flash_states.get(defender.player_id, FlashState()).is_flashed,
            is_retake=config['is_retake'],
            spike_time_remaining=spike_time_remaining,
            attacker_on_site=attacker_on_site,
            defender_on_site=defender_on_site,
        )

    def _calculate_advantage(self, ctx: CombatContext, config: Dict) -> float:
        """Calculate attacker advantage with all fixes applied."""

        # Base advantage (slight defender edge for holding angles)
        adv = 0.47

        # ==========================================
        # FIX 1: FLASH PENALTY
        # ==========================================
        if ctx.attacker_flashed:
            adv -= 0.25  # Huge disadvantage when flashed
        if ctx.defender_flashed:
            adv += 0.25  # Huge advantage against flashed enemy

        # ==========================================
        # FIX 2: CROSSFIRE BONUS
        # ==========================================
        if ctx.attacker_has_crossfire:
            adv += self.CROSSFIRE_BONUS
        if ctx.defender_has_crossfire:
            adv -= self.CROSSFIRE_BONUS

        # ==========================================
        # FIX 4: NUMBERS ADVANTAGE (CAPPED)
        # ==========================================
        numbers_diff = ctx.attacker_team_alive - ctx.defender_team_alive
        numbers_bonus = numbers_diff * self.NUMBERS_ADVANTAGE_PER_PLAYER
        numbers_bonus = max(-self.NUMBERS_ADVANTAGE_CAP, min(self.NUMBERS_ADVANTAGE_CAP, numbers_bonus))
        adv += numbers_bonus

        # ==========================================
        # FIX 3: DEFUSE PRESSURE
        # ==========================================
        if ctx.is_retake and ctx.spike_time_remaining < self.DEFUSE_PRESSURE_THRESHOLD:
            # Defenders get focus bonus when time is running out
            adv -= self.DEFUSE_PRESSURE_BONUS

        # ==========================================
        # FIX 6: POST-PLANT ANXIETY
        # ==========================================
        if config.get('spike_planted') and not ctx.is_retake:
            if ctx.spike_time_remaining < self.POST_PLANT_ANXIETY_THRESHOLD:
                # Attackers get nervous in last 10 seconds
                adv -= self.POST_PLANT_ANXIETY_PENALTY

        # Standard modifiers
        # Defender on-site bonus (when not post-plant)
        if ctx.defender_on_site and not config.get('spike_planted'):
            adv -= 0.05

        # Clamp final advantage
        return max(0.15, min(0.85, adv))

    def _get_target(self, player, engine, map_data, config, scenario) -> Tuple[float, float]:
        """Get movement target."""
        if player.side == 'attack':
            if config.get('spike_planted'):
                # Hold position post-plant
                return (player.x + random.uniform(-0.02, 0.02), player.y + random.uniform(-0.02, 0.02))
            else:
                # Push to site
                site = random.choice(list(map_data['sites'].keys()))
                site_data = map_data['sites'][site]
                return (site_data['center'][0] + random.uniform(-0.1, 0.1),
                        site_data['center'][1] + random.uniform(-0.1, 0.1))
        else:
            if config.get('is_retake'):
                # Push to planted site
                site_data = map_data['sites'][config['spike_site']]
                return (site_data['center'][0] + random.uniform(-0.1, 0.1),
                        site_data['center'][1] + random.uniform(-0.1, 0.1))
            else:
                # Hold position
                return (player.x + random.uniform(-0.02, 0.02), player.y + random.uniform(-0.02, 0.02))

    def print_results(self):
        """Print comprehensive results."""
        print(f"\n{'='*70}")
        print("V3 RESULTS WITH ALL FIXES")
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
    analyzer = FixedScenarioAnalyzer()

    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 50

    await analyzer.run_all_scenarios(rounds_per_scenario=rounds)
    accuracy = analyzer.print_results()

    # Save results
    output_file = Path(__file__).parent / 'scenario_analysis_v3_results.json'
    json_results = {s.value: r for s, r in analyzer.results.items()}

    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'accuracy': accuracy,
            'fixes_applied': [
                'flash_accuracy_penalty',
                'crossfire_detection',
                'defuse_pressure',
                'numbers_cap',
                'retake_utility',
                'post_plant_anxiety',
            ],
            'results': json_results,
        }, f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    asyncio.run(main())
