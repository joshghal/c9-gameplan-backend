#!/usr/bin/env python3
"""
C9 Tactical Vision - Simulation Demo

Run with: python3 tests/run_simulation_demo.py

Features:
- Spike plant (4 seconds)
- Spike defuse (7 seconds, or 3.5 for half)
- Spike explosion (45 seconds after plant)
- Round timeout (100 seconds = 1:40)
- Win conditions: elimination, spike exploded, spike defused, timeout
"""

import sys
import os
import random
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.services.weapon_system import WeaponDatabase, WeaponStats, ArmorStats
from app.services.economy_engine import EconomyEngine, BuyType, TeamEconomy, Loadout
from app.services.round_state import RoundState, WinProbabilityCalculator
from app.services.behavior_adaptation import BehaviorAdapter, PlayerTendencies
from app.services.strategy_coordinator import StrategyCoordinator, Role
from app.services.ability_system import AbilityDatabase, AbilitySystem


@dataclass
class SimPlayer:
    """Simplified player for demo simulation."""
    id: str
    team: str  # 'attack' or 'defense'
    agent: str
    health: int = 100
    shield: int = 50
    weapon: WeaponStats = None
    armor: ArmorStats = None
    is_alive: bool = True
    role: Role = None
    position: Tuple[float, float] = (0.5, 0.5)
    kills: int = 0
    is_planting: bool = False
    is_defusing: bool = False
    plant_start_ms: int = 0
    has_spike: bool = False


def run_demo_simulation(
    attack_agents: List[str] = None,
    defense_agents: List[str] = None,
    round_type: str = 'full',
    verbose: bool = True
):
    """
    Run a demo tactical simulation round with full spike mechanics.

    Args:
        attack_agents: List of 5 agent names for attack team
        defense_agents: List of 5 agent names for defense team
        round_type: 'pistol', 'eco', 'force', 'half', or 'full'
        verbose: Print detailed output

    Returns:
        Dict with simulation results
    """

    # Default team compositions
    if attack_agents is None:
        attack_agents = ['jett', 'raze', 'omen', 'sova', 'killjoy']
    if defense_agents is None:
        defense_agents = ['chamber', 'viper', 'fade', 'cypher', 'sage']

    if verbose:
        print("=" * 60)
        print("C9 TACTICAL VISION - ROUND SIMULATION")
        print("=" * 60)
        print("\n⏱️  ROUND RULES:")
        print("   • Round time: 100 seconds (1:40)")
        print("   • Spike plant: 4 seconds")
        print("   • Spike defuse: 7 seconds")
        print("   • Spike explosion: 45 seconds after plant")

    # === PHASE 1: ECONOMY & LOADOUTS ===
    buy_type = BuyType[round_type.upper()]

    credit_map = {
        BuyType.PISTOL: 800,
        BuyType.ECO: 2000,
        BuyType.FORCE: 3000,
        BuyType.HALF: 4000,
        BuyType.FULL: 5000,
    }

    base_credits = credit_map[buy_type]
    atk_economy = TeamEconomy(credits=[base_credits + random.randint(-200, 500) for _ in range(5)])
    def_economy = TeamEconomy(credits=[base_credits + random.randint(-200, 500) for _ in range(5)])

    atk_loadouts = EconomyEngine.generate_team_loadout(
        team_economy=atk_economy, round_num=5, side='attack', forced_buy_type=buy_type
    )
    def_loadouts = EconomyEngine.generate_team_loadout(
        team_economy=def_economy, round_num=5, side='defense', forced_buy_type=buy_type
    )

    if verbose:
        print(f"\n📊 Round Type: {round_type.upper()}")
        print(f"\n🔴 ATTACK TEAM (${atk_economy.total_credits} total)")
        for i, (agent, loadout) in enumerate(zip(attack_agents, atk_loadouts)):
            print(f"   {agent.capitalize():12} → {loadout.weapon.name:10} + {loadout.armor.name:12} (${loadout.total_value})")

        print(f"\n🔵 DEFENSE TEAM (${def_economy.total_credits} total)")
        for i, (agent, loadout) in enumerate(zip(defense_agents, def_loadouts)):
            print(f"   {agent.capitalize():12} → {loadout.weapon.name:10} + {loadout.armor.name:12} (${loadout.total_value})")

    # === PHASE 2: CREATE PLAYERS ===
    attack_players = []
    defense_players = []

    for i, (agent, loadout) in enumerate(zip(attack_agents, atk_loadouts)):
        player = SimPlayer(
            id=f"atk_{i}",
            team='attack',
            agent=agent,
            weapon=loadout.weapon,
            armor=loadout.armor,
            shield=loadout.armor.shield_value,
            has_spike=(i == 0)  # First attacker has spike
        )
        attack_players.append(player)

    for i, (agent, loadout) in enumerate(zip(defense_agents, def_loadouts)):
        defense_players.append(SimPlayer(
            id=f"def_{i}",
            team='defense',
            agent=agent,
            weapon=loadout.weapon,
            armor=loadout.armor,
            shield=loadout.armor.shield_value
        ))

    # === PHASE 3: STRATEGY SELECTION ===
    coordinator = StrategyCoordinator()

    atk_strategy = coordinator.select_strategy(
        team_id='attack_team', map_name='ascent', side='attack',
        round_type=round_type, team_credits=atk_economy.total_credits, round_number=5
    )
    def_strategy = coordinator.select_strategy(
        team_id='defense_team', map_name='ascent', side='defense',
        round_type=round_type, team_credits=def_economy.total_credits, round_number=5
    )

    if verbose:
        print(f"\n🎯 STRATEGIES")
        print(f"   Attack: {atk_strategy.name}")
        print(f"   Defense: {def_strategy.name}")

    # === PHASE 4: SIMULATE ROUND ===
    round_state = RoundState()
    round_state.attack_buy_type = round_type
    round_state.defense_buy_type = round_type

    events = []
    time_ms = 0
    TICK_MS = 500

    if verbose:
        print(f"\n⏱️  ROUND EVENTS")
        print("-" * 60)

    spike_carrier = attack_players[0]  # First attacker has spike
    defuser = None
    round_end = None

    while round_end is None:
        atk_alive = [p for p in attack_players if p.is_alive]
        def_alive = [p for p in defense_players if p.is_alive]

        round_state.attack_alive = len(atk_alive)
        round_state.defense_alive = len(def_alive)

        # === CHECK SPIKE EXPLOSION ===
        if round_state.check_spike_explosion(time_ms):
            if verbose:
                print(f"   [{time_ms/1000:.1f}s] 💥 SPIKE EXPLODED!")
            break

        # === CHECK DEFUSE COMPLETE ===
        if round_state.check_defuse_complete(time_ms):
            if verbose:
                print(f"   [{time_ms/1000:.1f}s] 🛡️  SPIKE DEFUSED!")
            break

        # === CHECK ROUND END CONDITIONS ===
        round_end = round_state.get_round_end_condition(time_ms)
        if round_end:
            break

        # === SPIKE PLANT LOGIC ===
        if not round_state.spike_planted and atk_alive:
            # Find spike carrier (or transfer if dead)
            if not spike_carrier.is_alive:
                alive_attackers = [p for p in attack_players if p.is_alive]
                if alive_attackers:
                    spike_carrier = alive_attackers[0]
                    spike_carrier.has_spike = True

            # Try to plant after 25 seconds if site is "clear enough"
            if time_ms > 25000 and spike_carrier.is_alive:
                if spike_carrier.is_planting:
                    # Check if plant completes
                    plant_duration = time_ms - spike_carrier.plant_start_ms
                    if plant_duration >= round_state.SPIKE_PLANT_TIME_MS:
                        site = random.choice(['A', 'B'])
                        round_state.plant_spike(time_ms, site)
                        spike_carrier.is_planting = False
                        if verbose:
                            print(f"   [{time_ms/1000:.1f}s] 💣 SPIKE PLANTED on {site}!")
                            time_remaining = round_state.get_time_remaining(time_ms)
                            print(f"            ⏰ {time_remaining.get('spike_time_ms', 0)/1000:.0f}s until explosion")
                else:
                    # Start planting with some probability
                    if random.random() < 0.02:  # 2% per tick after 25s
                        spike_carrier.is_planting = True
                        spike_carrier.plant_start_ms = time_ms
                        if verbose:
                            print(f"   [{time_ms/1000:.1f}s] 🔧 {spike_carrier.agent.upper()} planting spike...")

        # === DEFUSE LOGIC ===
        if round_state.spike_planted and not round_state.spike_defused and def_alive:
            if round_state.spike_defuser_id:
                # Check if defuser is still alive
                current_defuser = next((p for p in defense_players if p.id == round_state.spike_defuser_id), None)
                if not current_defuser or not current_defuser.is_alive:
                    round_state.cancel_defuse()
                    if verbose:
                        print(f"   [{time_ms/1000:.1f}s] ❌ Defuse cancelled!")
            else:
                # Defenders need to clear site before defusing
                # Lower chance to start defuse when attackers are alive (they're watching)
                time_remaining = round_state.get_time_remaining(time_ms)
                spike_time = time_remaining.get('spike_time_ms', 0)

                # Base defuse chance depends on situation
                base_chance = 0.008

                # Urgency increases as time runs out
                defuse_urgency = max(0, 1.0 - (spike_time / 45000))

                # More attackers = harder to defuse (they're watching the spike)
                attacker_penalty = len(atk_alive) * 0.006

                # More defenders = can trade/cover for the defuser
                defender_bonus = len(def_alive) * 0.008

                # If attackers are outnumbered, easier to defuse
                if len(def_alive) > len(atk_alive):
                    defender_bonus += 0.02

                defuse_chance = base_chance + (0.03 * defuse_urgency) + defender_bonus - attacker_penalty
                defuse_chance = max(0.002, defuse_chance)

                if random.random() < defuse_chance:
                    defuser = random.choice(def_alive)
                    round_state.start_defuse(time_ms, defuser.id)
                    defuser.is_defusing = True
                    if verbose:
                        print(f"   [{time_ms/1000:.1f}s] 🔧 {defuser.agent.upper()} defusing... (7s needed, {spike_time/1000:.0f}s remaining)")

        # === COMBAT SIMULATION ===
        # Combat rate varies by phase
        combat_chance = 0.03  # Base rate

        if not round_state.spike_planted:
            # Pre-plant: fighting for site control
            if time_ms > 20000:
                combat_chance = 0.04  # Increased when executing onto site
        else:
            # Post-plant: retake scenario
            combat_chance = 0.05  # More frequent fights during retake

        # CRITICAL: If defender is defusing, attacker checks the spike
        defusing_player = next((p for p in def_alive if p.is_defusing), None)
        if defusing_player and atk_alive:
            # Catch rate depends on:
            # - More attackers = more angles covered = higher catch rate
            # - More defenders = they can smoke/flash/trade = lower catch rate
            base_catch_rate = 0.15  # Base 15% per tick

            # Each attacker adds coverage
            attacker_bonus = len(atk_alive) * 0.05

            # Defenders can cover with abilities/trades (smoke defuse, flash peek)
            defender_reduction = (len(def_alive) - 1) * 0.08  # Other defenders covering

            catch_rate = base_catch_rate + attacker_bonus - defender_reduction
            catch_rate = max(0.05, min(0.5, catch_rate))  # Clamp between 5-50%

            if random.random() < catch_rate:
                attacker = random.choice(atk_alive)
                defender = defusing_player
                # This is a "peek the spike" fight
                distance = random.uniform(8, 25)  # Close range on site

                # Attacker has advantage but defenders can trade
                atk_win_prob = 0.70  # Attacker advantage (defuser can't shoot)
                # More defenders = better chance to trade
                if len(def_alive) >= 2:
                    atk_win_prob -= 0.15  # Teammate ready to trade
                if len(def_alive) >= 3:
                    atk_win_prob -= 0.10  # Multiple angles covered

                def_win_prob = 1.0 - atk_win_prob

                if random.random() < atk_win_prob:
                    defender.is_alive = False
                    attacker.kills += 1
                    round_state.cancel_defuse()
                    round_state.defense_alive -= 1
                    if verbose:
                        print(f"   [{time_ms/1000:.1f}s] 💀 {attacker.agent.upper()} CATCHES {defender.agent.upper()} DEFUSING! @ {distance:.0f}m")
                        win_prob = WinProbabilityCalculator.calculate_win_probability(round_state, time_ms)
                        print(f"            📊 {round_state.attack_alive}v{round_state.defense_alive} | ATK {win_prob['attack']*100:.0f}% / DEF {win_prob['defense']*100:.0f}%")
                else:
                    # Defender somehow wins (trades/teammate covers)
                    attacker.is_alive = False
                    defender.kills += 1
                    round_state.attack_alive -= 1
                    if verbose:
                        print(f"   [{time_ms/1000:.1f}s] 💀 {defender.agent.upper()} TRADED while defusing! → {attacker.agent.upper()}")
                        win_prob = WinProbabilityCalculator.calculate_win_probability(round_state, time_ms)
                        print(f"            📊 {round_state.attack_alive}v{round_state.defense_alive} | ATK {win_prob['attack']*100:.0f}% / DEF {win_prob['defense']*100:.0f}%")
                time_ms += TICK_MS
                continue

        if atk_alive and def_alive and random.random() < combat_chance:
            attacker = random.choice(atk_alive)
            defender = random.choice(def_alive)

            # Cancel planting/defusing if in combat
            if attacker.is_planting:
                attacker.is_planting = False
                if verbose:
                    print(f"   [{time_ms/1000:.1f}s] ⚠️  {attacker.agent.upper()} plant interrupted!")

            if defender.is_defusing:
                round_state.cancel_defuse()
                defender.is_defusing = False
                if verbose:
                    print(f"   [{time_ms/1000:.1f}s] ⚠️  {defender.agent.upper()} defuse interrupted!")

            distance = random.uniform(10, 40)

            # Players who are planting/defusing are VERY vulnerable (can't shoot back)
            attacker_hs_rate = 0.25
            defender_hs_rate = 0.25

            if attacker.is_planting:
                attacker_hs_rate = 0.0  # Can't fight while planting
            if defender.is_defusing:
                defender_hs_rate = 0.0  # Can't fight while defusing

            atk_win_prob, def_win_prob = WeaponDatabase.calculate_kill_probability(
                attacker_weapon=attacker.weapon,
                defender_weapon=defender.weapon,
                distance_meters=distance,
                attacker_headshot_rate=attacker_hs_rate,
                defender_headshot_rate=defender_hs_rate,
                attacker_armor=attacker.armor,
                defender_armor=defender.armor
            )

            # Massive advantage when enemy is planting/defusing (free kill)
            if defender.is_defusing:
                atk_win_prob = min(0.95, atk_win_prob * 2.5)  # Defuser is sitting duck
                def_win_prob = max(0.05, def_win_prob * 0.2)
            if attacker.is_planting:
                def_win_prob = min(0.95, def_win_prob * 2.5)  # Planter is sitting duck
                atk_win_prob = max(0.05, atk_win_prob * 0.2)

            if random.random() < atk_win_prob / (atk_win_prob + def_win_prob):
                defender.is_alive = False
                attacker.kills += 1

                if round_state.first_blood_team is None:
                    round_state.first_blood_team = 'attack'
                    round_state.first_blood_time_ms = time_ms

                round_state.defense_alive -= 1

                if verbose:
                    print(f"   [{time_ms/1000:.1f}s] 💀 {attacker.agent.upper()} ({attacker.weapon.name}) → {defender.agent.upper()} @ {distance:.0f}m")
                    win_prob = WinProbabilityCalculator.calculate_win_probability(round_state, time_ms)
                    print(f"            📊 {round_state.attack_alive}v{round_state.defense_alive} | ATK {win_prob['attack']*100:.0f}% / DEF {win_prob['defense']*100:.0f}%")
            else:
                attacker.is_alive = False
                defender.kills += 1

                if round_state.first_blood_team is None:
                    round_state.first_blood_team = 'defense'
                    round_state.first_blood_time_ms = time_ms

                round_state.attack_alive -= 1

                if verbose:
                    print(f"   [{time_ms/1000:.1f}s] 💀 {defender.agent.upper()} ({defender.weapon.name}) → {attacker.agent.upper()} @ {distance:.0f}m")
                    win_prob = WinProbabilityCalculator.calculate_win_probability(round_state, time_ms)
                    print(f"            📊 {round_state.attack_alive}v{round_state.defense_alive} | ATK {win_prob['attack']*100:.0f}% / DEF {win_prob['defense']*100:.0f}%")

        time_ms += TICK_MS

        # Safety check for very long rounds
        if time_ms > 200000:
            round_end = {'winner': 'defense', 'reason': 'safety_timeout'}
            break

    # === DETERMINE FINAL RESULT ===
    if round_end is None:
        round_end = round_state.get_round_end_condition(time_ms)

    if round_end is None:
        # Fallback determination
        if round_state.spike_exploded:
            round_end = {'winner': 'attack', 'reason': 'spike_exploded'}
        elif round_state.spike_defused:
            round_end = {'winner': 'defense', 'reason': 'spike_defused'}
        elif round_state.attack_alive == 0:
            round_end = {'winner': 'defense', 'reason': 'elimination'}
        elif round_state.defense_alive == 0:
            round_end = {'winner': 'attack', 'reason': 'elimination'}
        else:
            round_end = {'winner': 'defense', 'reason': 'timeout'}

    winner = round_end['winner']
    reason = round_end['reason']

    # Format reason for display
    reason_display = {
        'elimination': 'Team Eliminated',
        'elimination_post_plant': 'Defenders Eliminated (Post-Plant)',
        'spike_exploded': 'Spike Exploded',
        'spike_defused': 'Spike Defused',
        'timeout': 'Time Ran Out (No Plant)',
        'elimination_tie': 'Mutual Elimination',
    }.get(reason, reason)

    if verbose:
        print("-" * 60)
        print(f"\n🏆 ROUND RESULT")
        print(f"   Winner: {'🔴 ATTACK' if winner == 'attack' else '🔵 DEFENSE'}")
        print(f"   Reason: {reason_display}")
        print(f"   Final: {round_state.attack_alive}v{round_state.defense_alive} @ {time_ms/1000:.1f}s")

        if round_state.first_blood_team:
            print(f"   First Blood: {'Attack' if round_state.first_blood_team == 'attack' else 'Defense'} @ {round_state.first_blood_time_ms/1000:.1f}s")

        if round_state.spike_planted:
            print(f"   Spike: Planted @ {round_state.spike_plant_time_ms/1000:.1f}s on {round_state.spike_site}")
            if round_state.spike_defused:
                print(f"          Defused!")
            elif round_state.spike_exploded:
                print(f"          Exploded!")

        print(f"\n📈 SCOREBOARD")
        print("   Attack Team:")
        for p in attack_players:
            status = "✓" if p.is_alive else "✗"
            spike_icon = " 💣" if p.has_spike and not round_state.spike_planted else ""
            print(f"      {status} {p.agent.capitalize():10} - {p.kills} kills{spike_icon}")
        print("   Defense Team:")
        for p in defense_players:
            status = "✓" if p.is_alive else "✗"
            print(f"      {status} {p.agent.capitalize():10} - {p.kills} kills")

    return {
        'winner': winner,
        'reason': reason,
        'attack_alive': round_state.attack_alive,
        'defense_alive': round_state.defense_alive,
        'first_blood': round_state.first_blood_team,
        'spike_planted': round_state.spike_planted,
        'spike_defused': round_state.spike_defused,
        'spike_exploded': round_state.spike_exploded,
        'round_time_ms': time_ms,
        'attack_kills': sum(p.kills for p in attack_players),
        'defense_kills': sum(p.kills for p in defense_players),
    }


def run_batch_simulation(num_rounds: int = 100, round_type: str = 'full'):
    """Run multiple simulations to gather statistics."""
    print(f"\n{'='*60}")
    print(f"BATCH SIMULATION: {num_rounds} rounds ({round_type.upper()} buy)")
    print("="*60)

    results = {
        'attack_wins': 0,
        'defense_wins': 0,
        'reasons': {},
        'first_blood_wins': 0,
        'spike_plants': 0,
        'spike_defuses': 0,
        'spike_explosions': 0,
        'timeouts': 0,
    }

    for i in range(num_rounds):
        result = run_demo_simulation(round_type=round_type, verbose=False)

        if result['winner'] == 'attack':
            results['attack_wins'] += 1
        else:
            results['defense_wins'] += 1

        reason = result['reason']
        results['reasons'][reason] = results['reasons'].get(reason, 0) + 1

        if result['first_blood'] == result['winner']:
            results['first_blood_wins'] += 1

        if result['spike_planted']:
            results['spike_plants'] += 1
        if result['spike_defused']:
            results['spike_defuses'] += 1
        if result['spike_exploded']:
            results['spike_explosions'] += 1
        if result['reason'] == 'timeout':
            results['timeouts'] += 1

    print(f"\n📊 WIN RATES ({num_rounds} rounds)")
    print(f"   Attack:  {results['attack_wins']:3d} ({results['attack_wins']/num_rounds*100:.1f}%)")
    print(f"   Defense: {results['defense_wins']:3d} ({results['defense_wins']/num_rounds*100:.1f}%)")

    print(f"\n📈 WIN CONDITIONS")
    for reason, count in sorted(results['reasons'].items(), key=lambda x: -x[1]):
        reason_display = {
            'elimination': 'Elimination',
            'elimination_post_plant': 'Elim (Post-Plant)',
            'spike_exploded': 'Spike Exploded',
            'spike_defused': 'Spike Defused',
            'timeout': 'Timeout',
        }.get(reason, reason)
        print(f"   {reason_display:20} {count:3d} ({count/num_rounds*100:.1f}%)")

    print(f"\n💣 SPIKE STATISTICS")
    print(f"   Spike Plants:     {results['spike_plants']:3d} ({results['spike_plants']/num_rounds*100:.1f}%)")
    if results['spike_plants'] > 0:
        print(f"   → Explosions:     {results['spike_explosions']:3d} ({results['spike_explosions']/results['spike_plants']*100:.1f}% of plants)")
        print(f"   → Defuses:        {results['spike_defuses']:3d} ({results['spike_defuses']/results['spike_plants']*100:.1f}% of plants)")

    print(f"\n🎯 FIRST BLOOD IMPACT")
    print(f"   FB Team Won:      {results['first_blood_wins']:3d} ({results['first_blood_wins']/num_rounds*100:.1f}%)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='C9 Tactical Vision Simulation Demo')
    parser.add_argument('--type', '-t', choices=['pistol', 'eco', 'force', 'half', 'full'],
                        default='full', help='Round type (default: full)')
    parser.add_argument('--batch', '-b', type=int, default=0,
                        help='Run batch of N simulations (default: 0 = single)')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Quiet mode (less output)')

    args = parser.parse_args()

    if args.batch > 0:
        run_batch_simulation(num_rounds=args.batch, round_type=args.type)
    else:
        run_demo_simulation(round_type=args.type, verbose=not args.quiet)
