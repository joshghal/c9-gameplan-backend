#!/usr/bin/env python3
"""
Test impact of fixing logic fallacies.

Compares:
1. Current implementation (with fallacies)
2. Fixed: Randomized iteration order
3. Fixed: Trade reaction time delay
"""

import asyncio
import random
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.simulation_engine import SimulationEngine, SimulatedPlayer
from app.services.weapon_system import WeaponDatabase
from app.services.pathfinding import AStarPathfinder
from app.services.combat_system_v2 import CombatSystemV2, CombatState


class Scenario(Enum):
    ECO_VS_FULL = "eco_vs_full"
    FULL_VS_ECO = "full_vs_eco"
    MAN_ADVANTAGE_4V5 = "man_advantage_4v5"
    MAN_ADVANTAGE_5V4 = "man_advantage_5v4"
    CLUTCH_1V2 = "clutch_1v2"


VCT_REFERENCE = {
    Scenario.ECO_VS_FULL: 0.15,
    Scenario.FULL_VS_ECO: 0.85,
    Scenario.MAN_ADVANTAGE_4V5: 0.35,
    Scenario.MAN_ADVANTAGE_5V4: 0.60,
    Scenario.CLUTCH_1V2: 0.25,
}

AGENT_ROLES = {
    "jett": "duelist", "raze": "duelist", "omen": "controller",
    "sova": "initiator", "killjoy": "sentinel", "chamber": "sentinel",
    "cypher": "sentinel", "viper": "controller", "fade": "initiator", "sage": "sentinel",
}
ROLE_HS_RATES = {"duelist": 0.233, "initiator": 0.190, "controller": 0.196, "sentinel": 0.133}


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


class FallacyTestSimulator:
    """Simulator with configurable fallacy fixes."""

    MAPS = ['ascent', 'bind', 'haven', 'split', 'icebox', 'breeze', 'pearl']
    AGENTS_ATTACK = ['jett', 'raze', 'omen', 'sova', 'killjoy']
    AGENTS_DEFENSE = ['chamber', 'cypher', 'viper', 'fade', 'sage']

    def __init__(self, fix_iteration_order=False, fix_trade_timing=False):
        self.fix_iteration_order = fix_iteration_order
        self.fix_trade_timing = fix_trade_timing
        self.pathfinder = AStarPathfinder()
        self.combat_system = CombatSystemV2()

    def get_hs_rate(self, agent: str) -> float:
        role = AGENT_ROLES.get(agent, "duelist")
        return ROLE_HS_RATES.get(role, 0.20)

    async def run_scenarios(self, rounds: int = 100) -> Dict:
        results = {}
        for scenario in Scenario:
            win_rate = await self._run_scenario(scenario, rounds)
            target = VCT_REFERENCE[scenario]
            diff = abs(win_rate - target)
            results[scenario.value] = {
                'win_rate': win_rate,
                'target': target,
                'diff': diff,
                'pass': diff < 0.15,
            }
        return results

    async def _run_scenario(self, scenario: Scenario, num_rounds: int) -> float:
        attack_wins = 0
        rounds_per_map = max(1, num_rounds // len(self.MAPS))

        for map_name in self.MAPS:
            self.pathfinder.load_nav_grid(map_name)
            for _ in range(rounds_per_map):
                result = await self._simulate(scenario, map_name)
                if result and result['winner'] == 'attack':
                    attack_wins += 1

        return attack_wins / (rounds_per_map * len(self.MAPS))

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
            'spike_planted': False,
        }

        if scenario == Scenario.ECO_VS_FULL:
            config['attacker_weapons'] = ['classic'] * 5
            config['attacker_armor'] = [0] * 5
        elif scenario == Scenario.FULL_VS_ECO:
            config['defender_weapons'] = ['classic'] * 5
            config['defender_armor'] = [0] * 5
        elif scenario == Scenario.MAN_ADVANTAGE_4V5:
            config['num_attackers'] = 4
        elif scenario == Scenario.MAN_ADVANTAGE_5V4:
            config['num_defenders'] = 4
        elif scenario == Scenario.CLUTCH_1V2:
            config['num_attackers'] = 1
            config['num_defenders'] = 2

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
            player.weapon = WeaponDatabase.get_weapon(config['attacker_weapons'][i])
            player.shield = config['attacker_armor'][i]
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
            player.weapon = WeaponDatabase.get_weapon(config['defender_weapons'][i])
            player.shield = config['defender_armor'][i]
            engine.players[player_id] = player
            engine.info_manager.initialize_player(player_id, "defense")

    async def _run_simulation(self, engine, map_data, config) -> Dict:
        time_ms = 0
        max_time = 100000
        tick_ms = 128
        kills = []
        pending_trades = []  # For trade timing fix

        while time_ms < max_time:
            attack_alive = [p for p in engine.players.values() if p.side == 'attack' and p.is_alive]
            defense_alive = [p for p in engine.players.values() if p.side == 'defense' and p.is_alive]

            if not attack_alive or not defense_alive:
                break

            # Movement
            for player in engine.players.values():
                if not player.is_alive:
                    continue
                target = self._get_target(player, map_data)
                dx, dy = target[0] - player.x, target[1] - player.y
                dist = math.sqrt(dx*dx + dy*dy)
                if dist > 0.01:
                    player.x += (dx / dist) * min(0.015, dist)
                    player.y += (dy / dist) * min(0.015, dist)
                    player.x = max(0.05, min(0.95, player.x))
                    player.y = max(0.05, min(0.95, player.y))

            # Process pending trades (FIX: trade timing)
            if self.fix_trade_timing:
                resolved_trades = []
                for trade in pending_trades:
                    if time_ms >= trade['execute_time']:
                        killer = engine.players.get(trade['killer_id'])
                        trader = engine.players.get(trade['trader_id'])
                        if killer and killer.is_alive and trader and trader.is_alive:
                            # Trade executes - killer dies
                            killer.is_alive = False
                            kills.append(KillEvent(
                                timestamp_ms=time_ms,
                                killer_id=trader.player_id,
                                killer_side=trader.side,
                                killer_pos=(trader.x, trader.y),
                                victim_id=killer.player_id,
                                victim_side=killer.side,
                            ))
                        resolved_trades.append(trade)
                for t in resolved_trades:
                    pending_trades.remove(t)

            # Combat - FIX: randomize iteration order
            players_list = list(engine.players.values())
            if self.fix_iteration_order:
                random.shuffle(players_list)

            for player_a in players_list:
                if not player_a.is_alive:
                    continue

                opponents = [p for p in players_list if p.is_alive and p.side != player_a.side]
                if self.fix_iteration_order:
                    random.shuffle(opponents)

                for player_b in opponents:
                    if not player_b.is_alive:
                        continue

                    dist = math.sqrt((player_a.x - player_b.x)**2 + (player_a.y - player_b.y)**2)
                    if dist > 0.3:
                        continue

                    if not self.pathfinder.has_line_of_sight(
                        (player_a.x, player_a.y), (player_b.x, player_b.y)):
                        continue

                    combat_chance = 0.02 + 0.08 * (1 - dist / 0.3)
                    if random.random() < combat_chance:
                        winner_id, loser_id = self._resolve_combat(player_a, player_b, config, dist)

                        if winner_id == player_a.player_id:
                            winner, loser = player_a, player_b
                        else:
                            winner, loser = player_b, player_a

                        loser.is_alive = False
                        kills.append(KillEvent(
                            timestamp_ms=time_ms,
                            killer_id=winner.player_id,
                            killer_side=winner.side,
                            killer_pos=(winner.x, winner.y),
                            victim_id=loser.player_id,
                            victim_side=loser.side,
                        ))

                        # Trade check
                        trade_result = self._check_trade(kills[-1], engine, time_ms)
                        if trade_result:
                            if self.fix_trade_timing:
                                # FIX: Delay trade execution by reaction time
                                pending_trades.append({
                                    'killer_id': winner.player_id,
                                    'trader_id': trade_result['trader_id'],
                                    'execute_time': time_ms + trade_result['reaction_time_ms'],
                                })
                            else:
                                # Original: instant trade
                                winner.is_alive = False

                        break
                if kills and kills[-1].timestamp_ms == time_ms:
                    break

            time_ms += tick_ms

        attack_alive = len([p for p in engine.players.values() if p.side == 'attack' and p.is_alive])
        return {'winner': 'attack' if attack_alive > 0 else 'defense'}

    def _check_trade(self, kill: KillEvent, engine, time_ms) -> Optional[Dict]:
        traders = [p for p in engine.players.values()
                   if p.is_alive and p.side == kill.victim_side and p.player_id != kill.victim_id]

        for trader in traders:
            dist = math.sqrt((trader.x - kill.killer_pos[0])**2 + (trader.y - kill.killer_pos[1])**2)
            if dist > 0.15:
                continue
            if not self.pathfinder.has_line_of_sight((trader.x, trader.y), kill.killer_pos):
                continue

            # Weapon factor
            trader_weapon = trader.weapon.name.lower() if trader.weapon else 'classic'
            weapon_eff = {'vandal': 1.0, 'phantom': 1.0, 'classic': 0.4}.get(trader_weapon, 0.7)

            trade_prob = 0.50 * (1 - dist / 0.15) * weapon_eff
            if random.random() < trade_prob:
                reaction_time = max(150, random.gauss(250, 50))
                return {'trader_id': trader.player_id, 'reaction_time_ms': int(reaction_time)}

        return None

    def _resolve_combat(self, player_a, player_b, config, distance):
        weapon_a = player_a.weapon.name.lower() if player_a.weapon else 'classic'
        weapon_b = player_b.weapon.name.lower() if player_b.weapon else 'classic'

        state_a = CombatState(health=100, armor=player_a.shield, is_moving=False, distance_units=distance * 4000)
        state_b = CombatState(health=100, armor=player_b.shield, is_moving=False, distance_units=distance * 4000)

        hs_a = self.get_hs_rate(player_a.agent)
        hs_b = self.get_hs_rate(player_b.agent)

        # Probabilistic position advantage
        has_position = random.random() < 0.15
        a_pos = has_position and player_a.side == 'defense'
        b_pos = has_position and player_b.side == 'defense'

        result = self.combat_system.resolve_duel(
            player_a.player_id, weapon_a, state_a, hs_a,
            player_b.player_id, weapon_b, state_b, hs_b,
            player_a_has_position=a_pos, player_b_has_position=b_pos,
        )
        return result.winner_id, result.loser_id

    def _get_target(self, player, map_data):
        if player.side == 'attack':
            site = random.choice(list(map_data['sites'].keys()))
            center = map_data['sites'][site]['center']
            return (center[0] + random.uniform(-0.1, 0.1), center[1] + random.uniform(-0.1, 0.1))
        return (player.x + random.uniform(-0.02, 0.02), player.y + random.uniform(-0.02, 0.02))


async def main():
    print("=" * 70)
    print("FALLACY IMPACT TEST")
    print("=" * 70)
    print("\nTesting 3 configurations:")
    print("  1. Current (with fallacies)")
    print("  2. Fix: Randomized iteration order")
    print("  3. Fix: Both iteration + trade timing")
    print()

    configs = [
        ("Current (fallacies)", False, False),
        ("Fix: Random iteration", True, False),
        ("Fix: Both fixes", True, True),
    ]

    all_results = {}
    rounds = 150

    for name, fix_iter, fix_trade in configs:
        print(f"Testing '{name}'...", flush=True)
        sim = FallacyTestSimulator(fix_iteration_order=fix_iter, fix_trade_timing=fix_trade)
        results = await sim.run_scenarios(rounds)
        all_results[name] = results

        passing = sum(1 for r in results.values() if r['pass'])
        print(f"  Accuracy: {passing}/{len(results)} scenarios pass")

    # Compare results
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"\n{'Scenario':<20} {'Target':>8} | {'Current':>10} | {'Fix Iter':>10} | {'Both Fix':>10}")
    print("-" * 70)

    for scenario in Scenario:
        target = VCT_REFERENCE[scenario] * 100
        current = all_results["Current (fallacies)"][scenario.value]['win_rate'] * 100
        fix_iter = all_results["Fix: Random iteration"][scenario.value]['win_rate'] * 100
        fix_both = all_results["Fix: Both fixes"][scenario.value]['win_rate'] * 100

        # Calculate change
        iter_change = fix_iter - current
        both_change = fix_both - current

        print(f"{scenario.value:<20} {target:>7.0f}% | {current:>9.1f}% | {fix_iter:>7.1f}% ({iter_change:+.1f}) | {fix_both:>7.1f}% ({both_change:+.1f})")

    # Summary
    print("\n" + "-" * 70)
    current_pass = sum(1 for r in all_results["Current (fallacies)"].values() if r['pass'])
    iter_pass = sum(1 for r in all_results["Fix: Random iteration"].values() if r['pass'])
    both_pass = sum(1 for r in all_results["Fix: Both fixes"].values() if r['pass'])

    print(f"ACCURACY:           Current: {current_pass}/5  |  Fix Iter: {iter_pass}/5  |  Both: {both_pass}/5")


if __name__ == '__main__':
    asyncio.run(main())
