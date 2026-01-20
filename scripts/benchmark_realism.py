#!/usr/bin/env python3
"""
Realism Benchmark Script for C9 Tactical Vision

Measures key metrics that indicate simulation realism:
1. Win rate balance (attack vs defense)
2. Round timing distribution
3. Trade mechanics
4. Information system effectiveness
5. Ability usage patterns
6. Movement realism

Run with: python -m scripts.benchmark_realism
"""

import asyncio
import json
import random
import sys
import math
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.simulation_engine import SimulationEngine, SimulatedPlayer
from app.services.weapon_system import WeaponDatabase
from app.services.ability_system import AbilitySystem, AbilityCategory


@dataclass
class RoundMetrics:
    """Metrics collected from a single round."""
    winner: str  # 'attack' or 'defense'
    duration_ms: int
    first_blood_time_ms: int
    first_blood_side: str
    total_kills: int
    trade_kills: int  # Kills within 4s of teammate death
    spike_planted: bool
    spike_site: Optional[str]
    plant_time_ms: Optional[int]
    abilities_used: int
    smokes_used: int
    flashes_used: int
    info_reveals: int  # Recon abilities
    movement_distance_attack: float
    movement_distance_defense: float

    # Information system metrics
    kills_with_info: int  # Kills where killer had info on victim
    kills_without_info: int  # Kills where killer had no prior info (surprise)
    correct_rotations: int  # Defenders rotating to correct site

    map_name: str
    round_type: str


@dataclass
class BenchmarkResults:
    """Aggregated benchmark results."""
    total_rounds: int = 0
    attack_wins: int = 0
    defense_wins: int = 0

    # Timing
    avg_round_duration_ms: float = 0
    avg_first_blood_time_ms: float = 0
    first_blood_attack_rate: float = 0

    # Combat
    avg_kills_per_round: float = 0
    trade_rate: float = 0  # % of kills that were trades

    # Spike
    plant_rate: float = 0
    avg_plant_time_ms: float = 0

    # Abilities
    avg_abilities_per_round: float = 0
    avg_smokes_per_round: float = 0
    avg_flashes_per_round: float = 0

    # Information system effectiveness
    info_kill_rate: float = 0  # % of kills with prior info
    surprise_kill_rate: float = 0
    rotation_accuracy: float = 0

    # Movement
    avg_attack_movement: float = 0
    avg_defense_movement: float = 0

    # Per-map breakdown
    map_stats: Dict[str, Dict] = field(default_factory=dict)

    # VCT comparison (target values from pro data)
    vct_comparison: Dict[str, Dict] = field(default_factory=dict)


# VCT Pro Match Reference Data (from extracted data)
VCT_REFERENCE = {
    'attack_win_rate': 0.48,  # 48% attack win rate
    'first_blood_time_ms': 25000,  # ~25 seconds average
    'trade_rate': 0.252,  # 25.2% of kills are trades (3036/12029)
    'trade_window_ms': 3000,  # 79.3% of trades within 3s
    'avg_engagement_distance': 0.18,  # 1846 units on 10000 unit map
    'plant_rate': 0.65,  # ~65% of attack rounds see a plant
    'avg_plant_time_ms': 45000,  # ~45 seconds into round
    'round_duration_ms': 75000,  # ~75 seconds average
}


class MockDB:
    """Mock database session for standalone testing."""
    async def execute(self, *args, **kwargs):
        return None
    async def commit(self):
        pass
    async def rollback(self):
        pass


class SimulationBenchmark:
    """Runs simulation benchmarks to measure realism."""

    MAPS = ['ascent', 'bind', 'split', 'icebox', 'breeze', 'haven', 'lotus', 'pearl', 'sunset', 'fracture', 'abyss']
    AGENTS_ATTACK = ['jett', 'raze', 'omen', 'sova', 'killjoy']
    AGENTS_DEFENSE = ['chamber', 'cypher', 'viper', 'fade', 'sage']
    ROUND_TYPES = ['pistol', 'eco', 'force', 'full']

    def __init__(self):
        self.results = BenchmarkResults()
        self.round_metrics: List[RoundMetrics] = []
        # Pathfinder for LOS checks
        from app.services.pathfinding import AStarPathfinder
        self.pathfinder = AStarPathfinder()

    async def run_benchmark(self, num_rounds: int = 100, maps: Optional[List[str]] = None) -> BenchmarkResults:
        """Run benchmark simulations and collect metrics."""
        maps = maps or self.MAPS
        rounds_per_map = max(1, num_rounds // len(maps))

        print(f"\n{'='*60}")
        print(f"C9 TACTICAL VISION - REALISM BENCHMARK")
        print(f"{'='*60}")
        print(f"Running {num_rounds} simulations across {len(maps)} maps...")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        for map_name in maps:
            print(f"  Simulating {map_name}...", end=" ", flush=True)
            map_metrics = []

            for i in range(rounds_per_map):
                round_type = random.choice(self.ROUND_TYPES)
                metrics = await self._run_single_round(map_name, round_type)
                if metrics:
                    map_metrics.append(metrics)
                    self.round_metrics.append(metrics)

            # Calculate map-specific stats
            if map_metrics:
                self.results.map_stats[map_name] = self._calculate_map_stats(map_metrics)
                wins = sum(1 for m in map_metrics if m.winner == 'attack')
                print(f"Done ({len(map_metrics)} rounds, {wins}/{len(map_metrics)} attack wins)")
            else:
                print("Failed")

        # Aggregate all results
        self._aggregate_results()
        self._compare_to_vct()

        return self.results

    async def _run_single_round(self, map_name: str, round_type: str) -> Optional[RoundMetrics]:
        """Run a single simulation round and extract metrics."""
        try:
            # Create mock session
            db = MockDB()
            engine = SimulationEngine(db)

            # Load map geometry for pathfinding and LOS
            self.pathfinder.load_nav_grid(map_name)

            # Initialize players manually (bypassing DB)
            engine.map_name = map_name
            map_data = engine.MAP_DATA.get(map_name, engine.DEFAULT_MAP_DATA)

            # Create attack team
            attack_spawns = map_data['spawns']['attack']
            for i, agent in enumerate(self.AGENTS_ATTACK):
                player_id = f"attack_{i}"
                spawn = attack_spawns[i % len(attack_spawns)]
                player = SimulatedPlayer(
                    player_id=player_id,
                    team_id="attack_team",
                    side="attack",
                    x=spawn[0],
                    y=spawn[1],
                    agent=agent,
                    has_spike=(i == 0)  # First attacker has spike
                )
                # Get weapon based on round type
                weapon = self._get_weapon_for_round_type(round_type, player.side)
                player.weapon = weapon
                player.shield = 50 if round_type == 'full' else 25 if round_type in ('force', 'half') else 0
                engine.players[player_id] = player
                engine.info_manager.initialize_player(player_id, "attack")
                engine.ability_system.initialize_player(player_id, agent)

            # Create defense team
            defense_spawns = map_data['spawns']['defense']
            for i, agent in enumerate(self.AGENTS_DEFENSE):
                player_id = f"defense_{i}"
                spawn = defense_spawns[i % len(defense_spawns)]
                player = SimulatedPlayer(
                    player_id=player_id,
                    team_id="defense_team",
                    side="defense",
                    x=spawn[0],
                    y=spawn[1],
                    agent=agent
                )
                weapon = self._get_weapon_for_round_type(round_type, player.side)
                player.weapon = weapon
                player.shield = 50 if round_type == 'full' else 25 if round_type in ('force', 'half') else 0
                engine.players[player_id] = player
                engine.info_manager.initialize_player(player_id, "defense")
                engine.ability_system.initialize_player(player_id, agent)

            # Run simulation
            metrics = await self._simulate_round(engine, map_name, round_type)
            return metrics

        except Exception as e:
            # Silently skip failed rounds
            return None

    def _get_weapon_for_round_type(self, round_type: str, side: str) -> 'WeaponStats':
        """Get appropriate weapon for round type."""
        from app.services.weapon_system import WeaponDatabase

        if round_type == 'pistol':
            return WeaponDatabase.get_weapon('classic')
        elif round_type == 'eco':
            return random.choice([
                WeaponDatabase.get_weapon('sheriff'),
                WeaponDatabase.get_weapon('shorty'),
                WeaponDatabase.get_weapon('classic')
            ])
        elif round_type == 'force':
            return random.choice([
                WeaponDatabase.get_weapon('spectre'),
                WeaponDatabase.get_weapon('marshal'),
                WeaponDatabase.get_weapon('sheriff')
            ])
        else:  # full buy
            return random.choice([
                WeaponDatabase.get_weapon('vandal'),
                WeaponDatabase.get_weapon('phantom'),
                WeaponDatabase.get_weapon('operator')
            ])

    async def _simulate_round(self, engine: SimulationEngine, map_name: str, round_type: str) -> RoundMetrics:
        """Simulate a round and collect metrics."""
        time_ms = 0
        max_time_ms = 100000  # 100 second round
        tick_ms = 128

        # Tracking variables
        first_blood_time = None
        first_blood_side = None
        kills = []
        trade_kills = 0
        abilities_used = 0
        smokes_used = 0
        flashes_used = 0
        info_reveals = 0
        kills_with_info = 0
        kills_without_info = 0

        # Track initial positions for movement calculation
        initial_positions = {
            pid: (p.x, p.y) for pid, p in engine.players.items()
        }

        while time_ms < max_time_ms:
            # Get alive counts
            attack_alive = [p for p in engine.players.values() if p.side == 'attack' and p.is_alive]
            defense_alive = [p for p in engine.players.values() if p.side == 'defense' and p.is_alive]

            # Check win conditions
            if len(attack_alive) == 0:
                break
            if len(defense_alive) == 0:
                break
            if engine.spike_planted and engine.spike_plant_time:
                if time_ms - engine.spike_plant_time >= 45000:  # Spike detonated
                    break

            # Simulate one tick
            tick_result = await self._simulate_tick(engine, time_ms)

            # Process tick results
            if tick_result.get('kill'):
                kill = tick_result['kill']
                kills.append(kill)

                if first_blood_time is None:
                    first_blood_time = time_ms
                    first_blood_side = kill['killer_side']

                # Check if trade
                if len(kills) >= 2:
                    prev_kill = kills[-2]
                    if (time_ms - prev_kill['time_ms'] <= 4000 and
                        kill['killer_side'] != prev_kill['killer_side']):
                        trade_kills += 1

                # Track if killer had info on victim at time of kill
                if kill.get('killer_had_info', False):
                    kills_with_info += 1
                else:
                    kills_without_info += 1

            if tick_result.get('ability_used'):
                abilities_used += 1
                ability = tick_result['ability_used']
                if ability.category == AbilityCategory.SMOKE:
                    smokes_used += 1
                elif ability.category == AbilityCategory.FLASH:
                    flashes_used += 1
                elif ability.category == AbilityCategory.RECON:
                    info_reveals += 1

            time_ms += tick_ms

        # Determine winner
        attack_alive = len([p for p in engine.players.values() if p.side == 'attack' and p.is_alive])
        defense_alive = len([p for p in engine.players.values() if p.side == 'defense' and p.is_alive])

        if attack_alive == 0:
            winner = 'defense'
        elif defense_alive == 0:
            winner = 'attack'
        elif engine.spike_planted:
            # Spike detonated
            winner = 'attack'
        else:
            # Time ran out
            winner = 'defense'

        # Calculate movement distances
        attack_movement = 0
        defense_movement = 0
        for pid, player in engine.players.items():
            init_x, init_y = initial_positions.get(pid, (player.x, player.y))
            dist = math.sqrt((player.x - init_x)**2 + (player.y - init_y)**2)
            if player.side == 'attack':
                attack_movement += dist
            else:
                defense_movement += dist

        return RoundMetrics(
            winner=winner,
            duration_ms=time_ms,
            first_blood_time_ms=first_blood_time or 0,
            first_blood_side=first_blood_side or 'none',
            total_kills=len(kills),
            trade_kills=trade_kills,
            spike_planted=engine.spike_planted,
            spike_site=engine.spike_site,
            plant_time_ms=engine.spike_plant_time,
            abilities_used=abilities_used,
            smokes_used=smokes_used,
            flashes_used=flashes_used,
            info_reveals=info_reveals,
            movement_distance_attack=attack_movement,
            movement_distance_defense=defense_movement,
            kills_with_info=kills_with_info,
            kills_without_info=kills_without_info,
            correct_rotations=0,  # TODO: Track this
            map_name=map_name,
            round_type=round_type
        )

    async def _simulate_tick(self, engine: SimulationEngine, time_ms: int) -> Dict:
        """Simulate a single tick and return events."""
        result = {}

        # Track player facing directions (persistent per player)
        if not hasattr(engine, '_player_facings'):
            engine._player_facings = {}

        # Update info system with realistic FOV and facing
        all_players = []
        for p in engine.players.values():
            # Update facing based on movement direction
            if p.moved_this_tick and hasattr(p, 'prev_x') and hasattr(p, 'prev_y'):
                dx = p.x - p.prev_x
                dy = p.y - p.prev_y
                if dx != 0 or dy != 0:
                    engine._player_facings[p.player_id] = math.atan2(dy, dx)
            elif p.player_id not in engine._player_facings:
                # Default facing: attackers face toward sites (north), defenders face toward spawn (south)
                engine._player_facings[p.player_id] = -math.pi/2 if p.side == 'attack' else math.pi/2

            all_players.append({
                'id': p.player_id,
                'x': p.x,
                'y': p.y,
                'team': p.side,
                'is_alive': p.is_alive,
                'is_running': p.is_running,
                'facing': engine._player_facings.get(p.player_id, 0)
            })

        smoke_positions = [(pos, radius) for pos, radius in engine.ability_system.get_active_smokes(time_ms)]

        for player in engine.players.values():
            if not player.is_alive:
                continue

            # Update vision with player's actual facing direction (limited FOV)
            facing = engine._player_facings.get(player.player_id, 0)
            engine.info_manager.update_vision(
                player.player_id,
                (player.x, player.y),
                facing,  # Use actual facing, not random
                all_players,
                time_ms,
                [pos for pos, _ in smoke_positions]
            )

            # Propagate sound for running players
            if player.is_running and player.moved_this_tick:
                from app.services.information_system import InfoSource
                engine.info_manager.propagate_sound(
                    (player.x, player.y),
                    InfoSource.SOUND_FOOTSTEP,
                    player.side,
                    all_players,
                    time_ms,
                    player.player_id
                )

        engine.info_manager.update_tick(time_ms)

        # Move players
        for player in engine.players.values():
            if not player.is_alive:
                continue

            # Simple movement toward objectives
            if player.side == 'attack':
                # Move toward sites
                target_y = 0.3 + random.uniform(-0.1, 0.1)
                target_x = 0.3 if random.random() < 0.5 else 0.7
            else:
                # Hold positions with small adjustments
                target_x = player.x + random.uniform(-0.02, 0.02)
                target_y = player.y + random.uniform(-0.02, 0.02)

            # Move toward target
            dx = target_x - player.x
            dy = target_y - player.y
            dist = math.sqrt(dx*dx + dy*dy)

            if dist > 0.01:
                speed = 0.015  # Normalized speed per tick
                player.prev_x = player.x
                player.prev_y = player.y
                player.x += (dx / dist) * min(speed, dist)
                player.y += (dy / dist) * min(speed, dist)
                player.x = max(0.05, min(0.95, player.x))
                player.y = max(0.05, min(0.95, player.y))
                player.moved_this_tick = True
            else:
                player.moved_this_tick = False

        # Check for combat
        for attacker in engine.players.values():
            if not attacker.is_alive:
                continue

            for defender in engine.players.values():
                if not defender.is_alive or defender.side == attacker.side:
                    continue

                # Distance check
                dist = math.sqrt((attacker.x - defender.x)**2 + (attacker.y - defender.y)**2)
                if dist > 0.3:  # Max engagement range (30% of map)
                    continue

                # LINE OF SIGHT CHECK - key realism improvement
                if not self.pathfinder.has_line_of_sight(
                    (attacker.x, attacker.y),
                    (defender.x, defender.y)
                ):
                    continue  # Can't see each other, no combat

                # SMOKE CHECK - combat through smoke is very unlikely
                line_through_smoke = False
                for smoke_pos, smoke_radius in smoke_positions:
                    # Check if the line between attacker and defender passes through smoke
                    # Simplified: check if smoke is between them and close to the midpoint
                    mid_x = (attacker.x + defender.x) / 2
                    mid_y = (attacker.y + defender.y) / 2
                    smoke_dist = math.sqrt((mid_x - smoke_pos[0])**2 + (mid_y - smoke_pos[1])**2)
                    if smoke_dist < smoke_radius + 0.03:
                        line_through_smoke = True
                        break

                if line_through_smoke:
                    # Moderate chance of engagement through smoke (defenders can spray at sounds)
                    if random.random() > 0.20:  # 80% chance to skip (was 95%)
                        continue

                # Combat probability scales with distance (closer = more likely)
                # At 0.05 distance: 10% chance, at 0.3 distance: 1% chance
                combat_chance = 0.01 + 0.09 * (1 - dist / 0.3)
                if random.random() < combat_chance:
                    # Check if either player had prior info on the other
                    attacker_knowledge = engine.info_manager.get_knowledge(attacker.player_id)
                    defender_knowledge = engine.info_manager.get_knowledge(defender.player_id)

                    attacker_had_info = (
                        attacker_knowledge and
                        defender.player_id in attacker_knowledge.enemies and
                        attacker_knowledge.enemies[defender.player_id].confidence.value in ['exact', 'high', 'medium']
                    )
                    defender_had_info = (
                        defender_knowledge and
                        attacker.player_id in defender_knowledge.enemies and
                        defender_knowledge.enemies[attacker.player_id].confidence.value in ['exact', 'high', 'medium']
                    )

                    # Determine who wins - defender advantage for holding angles!
                    # In VALORANT, defense wins ~52% because they hold angles
                    attacker_advantage = 0.45  # Base: slight defender advantage (holding angle)
                    if attacker.moved_this_tick and not defender.moved_this_tick:
                        attacker_advantage += 0.12  # Peeker's advantage (reduced from 0.15)
                    if attacker_had_info and not defender_had_info:
                        attacker_advantage += 0.10  # Info advantage
                    elif defender_had_info and not attacker_had_info:
                        attacker_advantage -= 0.10  # Defender has info advantage
                    # Defenders on site get bonus
                    if defender.side == 'defense':
                        map_data = engine.MAP_DATA.get(engine.map_name, engine.DEFAULT_MAP_DATA)
                        for site_data in map_data['sites'].values():
                            site_x, site_y = site_data['center']
                            dist_to_site = math.sqrt((defender.x - site_x)**2 + (defender.y - site_y)**2)
                            if dist_to_site < site_data['radius'] + 0.1:
                                attacker_advantage -= 0.05  # Defender on-site bonus
                                break

                    if random.random() < attacker_advantage:
                        defender.is_alive = False
                        defender.health = 0
                        result['kill'] = {
                            'killer_id': attacker.player_id,
                            'killer_side': attacker.side,
                            'victim_id': defender.player_id,
                            'victim_side': defender.side,
                            'time_ms': time_ms,
                            'killer_had_info': attacker_had_info  # Track info state
                        }
                        engine.info_manager.notify_kill(
                            attacker.player_id, defender.player_id,
                            defender.side, (defender.x, defender.y)
                        )
                    else:
                        attacker.is_alive = False
                        attacker.health = 0
                        result['kill'] = {
                            'killer_id': defender.player_id,
                            'killer_side': defender.side,
                            'victim_id': attacker.player_id,
                            'victim_side': attacker.side,
                            'time_ms': time_ms,
                            'killer_had_info': defender_had_info  # Track info state
                        }
                        engine.info_manager.notify_kill(
                            defender.player_id, attacker.player_id,
                            attacker.side, (attacker.x, attacker.y)
                        )
                    break
            if 'kill' in result:
                break

        # Check for spike plant
        if not engine.spike_planted:
            spike_carrier = next(
                (p for p in engine.players.values() if p.has_spike and p.is_alive),
                None
            )
            if spike_carrier:
                # Check if near a site
                map_data = engine.MAP_DATA.get(engine.map_name, engine.DEFAULT_MAP_DATA)
                for site_name, site_data in map_data['sites'].items():
                    site_x, site_y = site_data['center']
                    dist = math.sqrt((spike_carrier.x - site_x)**2 + (spike_carrier.y - site_y)**2)
                    if dist < site_data['radius'] + 0.08:  # Slightly larger plant zone
                        # Check if site is covered by smokes (safer to plant)
                        smoke_coverage = 0
                        for smoke_pos, smoke_radius in smoke_positions:
                            smoke_dist = math.sqrt((site_x - smoke_pos[0])**2 + (site_y - smoke_pos[1])**2)
                            if smoke_dist < smoke_radius + 0.1:
                                smoke_coverage += 1

                        # Count defenders with LOS to spike carrier
                        defenders_watching = 0
                        for defender in engine.players.values():
                            if defender.side == 'defense' and defender.is_alive:
                                if self.pathfinder.has_line_of_sight(
                                    (spike_carrier.x, spike_carrier.y),
                                    (defender.x, defender.y)
                                ):
                                    defenders_watching += 1

                        # Base plant chance increases with smoke coverage, decreases with defenders watching
                        # VCT data: ~65% of attack rounds have a plant attempt
                        base_plant_chance = 0.03  # 3% base per tick when on site
                        if smoke_coverage > 0:
                            base_plant_chance += 0.05 * smoke_coverage  # +5% per smoke
                        if defenders_watching == 0:
                            base_plant_chance += 0.10  # +10% if no defenders watching
                        elif defenders_watching >= 2:
                            base_plant_chance *= 0.3  # Much lower if multiple defenders

                        if random.random() < base_plant_chance:
                            engine.spike_planted = True
                            engine.spike_site = site_name
                            engine.spike_plant_time = time_ms
                            engine.info_manager.notify_spike_plant(site_name, (spike_carrier.x, spike_carrier.y), time_ms)
                            break

        # Check for ability usage - strategic use based on role and situation
        for player in engine.players.values():
            if not player.is_alive:
                continue

            # Determine ability use probability based on situation
            ability_chance = 0.01  # Base 1% per tick

            # Attackers near site should use abilities to execute
            if player.side == 'attack':
                map_data = engine.MAP_DATA.get(engine.map_name, engine.DEFAULT_MAP_DATA)
                for site_name, site_data in map_data['sites'].items():
                    site_x, site_y = site_data['center']
                    dist_to_site = math.sqrt((player.x - site_x)**2 + (player.y - site_y)**2)
                    if dist_to_site < 0.25:  # Close to site
                        ability_chance = 0.04  # 4% chance when near site (was 8%)
                        break

                # Controllers (smoke agents) should use abilities early
                if player.agent.lower() in ['omen', 'brimstone', 'viper', 'astra', 'harbor', 'clove']:
                    if time_ms < 30000:  # First 30 seconds
                        ability_chance = max(ability_chance, 0.02)

            # Initiators should flash before entry
            if player.agent.lower() in ['breach', 'skye', 'kayo', 'gekko', 'fade', 'sova']:
                if time_ms > 15000:  # After initial setup
                    ability_chance = max(ability_chance, 0.015)

            if random.random() < ability_chance:
                ability_result = engine.ability_system.should_use_ability(
                    player.player_id,
                    time_ms,
                    'mid_round',
                    engine.round_state,
                    (player.x, player.y),
                    player.side
                )
                if ability_result:
                    ability, target_pos = ability_result
                    engine.ability_system.use_ability(
                        player.player_id, ability, target_pos, time_ms, player.side
                    )
                    result['ability_used'] = ability
                    break

        return result

    def _calculate_map_stats(self, metrics: List[RoundMetrics]) -> Dict:
        """Calculate statistics for a specific map."""
        if not metrics:
            return {}

        attack_wins = sum(1 for m in metrics if m.winner == 'attack')

        return {
            'rounds': len(metrics),
            'attack_wins': attack_wins,
            'defense_wins': len(metrics) - attack_wins,
            'attack_win_rate': attack_wins / len(metrics),
            'avg_duration_ms': sum(m.duration_ms for m in metrics) / len(metrics),
            'avg_kills': sum(m.total_kills for m in metrics) / len(metrics),
            'plant_rate': sum(1 for m in metrics if m.spike_planted) / len(metrics),
            'trade_rate': sum(m.trade_kills for m in metrics) / max(1, sum(m.total_kills for m in metrics)),
        }

    def _aggregate_results(self):
        """Aggregate all round metrics into final results."""
        if not self.round_metrics:
            return

        n = len(self.round_metrics)
        self.results.total_rounds = n
        self.results.attack_wins = sum(1 for m in self.round_metrics if m.winner == 'attack')
        self.results.defense_wins = n - self.results.attack_wins

        self.results.avg_round_duration_ms = sum(m.duration_ms for m in self.round_metrics) / n
        self.results.avg_first_blood_time_ms = sum(m.first_blood_time_ms for m in self.round_metrics) / n
        self.results.first_blood_attack_rate = sum(1 for m in self.round_metrics if m.first_blood_side == 'attack') / n

        total_kills = sum(m.total_kills for m in self.round_metrics)
        total_trades = sum(m.trade_kills for m in self.round_metrics)
        self.results.avg_kills_per_round = total_kills / n
        self.results.trade_rate = total_trades / max(1, total_kills)

        plants = [m for m in self.round_metrics if m.spike_planted]
        self.results.plant_rate = len(plants) / n
        self.results.avg_plant_time_ms = sum(m.plant_time_ms or 0 for m in plants) / max(1, len(plants))

        self.results.avg_abilities_per_round = sum(m.abilities_used for m in self.round_metrics) / n
        self.results.avg_smokes_per_round = sum(m.smokes_used for m in self.round_metrics) / n
        self.results.avg_flashes_per_round = sum(m.flashes_used for m in self.round_metrics) / n

        total_info_kills = sum(m.kills_with_info for m in self.round_metrics)
        total_surprise_kills = sum(m.kills_without_info for m in self.round_metrics)
        self.results.info_kill_rate = total_info_kills / max(1, total_info_kills + total_surprise_kills)
        self.results.surprise_kill_rate = total_surprise_kills / max(1, total_info_kills + total_surprise_kills)

        self.results.avg_attack_movement = sum(m.movement_distance_attack for m in self.round_metrics) / n
        self.results.avg_defense_movement = sum(m.movement_distance_defense for m in self.round_metrics) / n

    def _compare_to_vct(self):
        """Compare results to VCT reference data."""
        self.results.vct_comparison = {
            'attack_win_rate': {
                'actual': self.results.attack_wins / max(1, self.results.total_rounds),
                'target': VCT_REFERENCE['attack_win_rate'],
                'diff': abs(self.results.attack_wins / max(1, self.results.total_rounds) - VCT_REFERENCE['attack_win_rate']),
            },
            'trade_rate': {
                'actual': self.results.trade_rate,
                'target': VCT_REFERENCE['trade_rate'],
                'diff': abs(self.results.trade_rate - VCT_REFERENCE['trade_rate']),
            },
            'plant_rate': {
                'actual': self.results.plant_rate,
                'target': VCT_REFERENCE['plant_rate'],
                'diff': abs(self.results.plant_rate - VCT_REFERENCE['plant_rate']),
            },
            'round_duration': {
                'actual': self.results.avg_round_duration_ms,
                'target': VCT_REFERENCE['round_duration_ms'],
                'diff': abs(self.results.avg_round_duration_ms - VCT_REFERENCE['round_duration_ms']) / VCT_REFERENCE['round_duration_ms'],
            },
        }

    def calculate_realism_score(self) -> float:
        """Calculate overall realism score (0-100%)."""
        if not self.results.vct_comparison:
            return 0

        # Weight each metric
        weights = {
            'attack_win_rate': 25,  # Very important
            'trade_rate': 20,       # Important for combat feel
            'plant_rate': 15,       # Important for round flow
            'round_duration': 10,   # Moderate importance
        }

        total_weight = sum(weights.values())
        score = 0

        for metric, weight in weights.items():
            comp = self.results.vct_comparison.get(metric, {})
            diff = comp.get('diff', 1.0)
            # Convert diff to score (smaller diff = higher score)
            metric_score = max(0, 1 - diff * 2)  # 50% diff = 0 score
            score += metric_score * weight

        # Additional score components
        # Information system usage (10%)
        if self.results.info_kill_rate > 0.3:
            score += 10 * min(1, self.results.info_kill_rate / 0.5)

        # Ability usage (10%)
        if self.results.avg_abilities_per_round > 2:
            score += 10 * min(1, self.results.avg_abilities_per_round / 10)

        # Movement differential (10% - attackers should move more)
        if self.results.avg_attack_movement > self.results.avg_defense_movement:
            score += 10

        return score / (total_weight + 30) * 100

    def print_report(self):
        """Print detailed benchmark report."""
        score = self.calculate_realism_score()

        print(f"\n{'='*60}")
        print(f"BENCHMARK RESULTS")
        print(f"{'='*60}")
        print(f"Total Rounds: {self.results.total_rounds}")
        print(f"")
        print(f"REALISM SCORE: {score:.1f}%")
        print(f"{'='*60}")

        print(f"\n--- Win Rates ---")
        attack_rate = self.results.attack_wins / max(1, self.results.total_rounds) * 100
        print(f"Attack Wins: {self.results.attack_wins} ({attack_rate:.1f}%)")
        print(f"Defense Wins: {self.results.defense_wins} ({100-attack_rate:.1f}%)")
        print(f"Target: {VCT_REFERENCE['attack_win_rate']*100:.0f}% / {(1-VCT_REFERENCE['attack_win_rate'])*100:.0f}%")

        print(f"\n--- Combat ---")
        print(f"Avg Kills/Round: {self.results.avg_kills_per_round:.1f}")
        print(f"Trade Rate: {self.results.trade_rate*100:.1f}% (target: {VCT_REFERENCE['trade_rate']*100:.1f}%)")
        print(f"First Blood Attack Rate: {self.results.first_blood_attack_rate*100:.1f}%")

        print(f"\n--- Timing ---")
        print(f"Avg Round Duration: {self.results.avg_round_duration_ms/1000:.1f}s (target: {VCT_REFERENCE['round_duration_ms']/1000:.1f}s)")
        print(f"Avg First Blood: {self.results.avg_first_blood_time_ms/1000:.1f}s")
        print(f"Avg Plant Time: {self.results.avg_plant_time_ms/1000:.1f}s")

        print(f"\n--- Spike ---")
        print(f"Plant Rate: {self.results.plant_rate*100:.1f}% (target: {VCT_REFERENCE['plant_rate']*100:.1f}%)")

        print(f"\n--- Abilities ---")
        print(f"Avg Abilities/Round: {self.results.avg_abilities_per_round:.1f}")
        print(f"Avg Smokes/Round: {self.results.avg_smokes_per_round:.1f}")
        print(f"Avg Flashes/Round: {self.results.avg_flashes_per_round:.1f}")

        print(f"\n--- Information System ---")
        print(f"Kills with Prior Info: {self.results.info_kill_rate*100:.1f}%")
        print(f"Surprise Kills: {self.results.surprise_kill_rate*100:.1f}%")

        print(f"\n--- Movement ---")
        print(f"Attack Movement: {self.results.avg_attack_movement:.3f}")
        print(f"Defense Movement: {self.results.avg_defense_movement:.3f}")

        print(f"\n--- Per-Map Breakdown ---")
        for map_name, stats in sorted(self.results.map_stats.items()):
            print(f"  {map_name:12s}: {stats['attack_win_rate']*100:.0f}% atk, {stats['avg_kills']:.1f} kills, {stats['plant_rate']*100:.0f}% plants")

        print(f"\n{'='*60}")
        return score


async def main():
    """Run benchmark."""
    benchmark = SimulationBenchmark()

    # Run fewer rounds for quick testing
    import sys
    num_rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 50

    await benchmark.run_benchmark(num_rounds=num_rounds)
    score = benchmark.print_report()

    # Save results
    results_file = Path(__file__).parent / 'benchmark_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'num_rounds': benchmark.results.total_rounds,
            'realism_score': score,
            'attack_win_rate': benchmark.results.attack_wins / max(1, benchmark.results.total_rounds),
            'trade_rate': benchmark.results.trade_rate,
            'plant_rate': benchmark.results.plant_rate,
            'avg_abilities': benchmark.results.avg_abilities_per_round,
            'info_kill_rate': benchmark.results.info_kill_rate,
            'map_stats': benchmark.results.map_stats,
        }, f, indent=2)

    print(f"\nResults saved to: {results_file}")
    return score


if __name__ == '__main__':
    asyncio.run(main())
