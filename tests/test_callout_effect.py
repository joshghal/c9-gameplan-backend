#!/usr/bin/env python3
"""
Test to verify death callouts actually affect player knowledge.
Shows the knowledge state before and after a kill.
"""

import asyncio
import sys
import os
import uuid
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.simulation_engine import SimulationEngine
from app.services.information_system import InfoSource


@dataclass
class MockSession:
    id: str
    attack_team_id: str
    defense_team_id: str
    map_name: str
    current_time_ms: int = 0
    status: str = "created"
    phase: str = "opening"


async def test_callout_effect():
    """Run simulation and show how knowledge changes after kills."""
    print("=" * 70)
    print("TESTING: Death Callouts & Gunfire Sound Effect on Knowledge")
    print("=" * 70)

    engine = SimulationEngine()
    session = MockSession(
        id=str(uuid.uuid4()),
        attack_team_id="cloud9",
        defense_team_id="g2",
        map_name="ascent"
    )

    await engine.initialize(session, round_type="full")

    last_kill_count = 0

    for tick in range(500):
        state = await engine.advance(session, ticks=1)
        session.current_time_ms = state.current_time_ms

        # Count kills
        kill_events = [e for e in state.events if e.event_type == 'kill']

        if len(kill_events) > last_kill_count:
            # New kill happened!
            new_kill = kill_events[-1]
            killer_id = new_kill.details.get('killer_id')
            victim_id = new_kill.player_id

            # Find victim's team
            victim = engine.players.get(victim_id)
            if victim:
                victim_team = victim.side

                print(f"\n[{state.current_time_ms/1000:.1f}s] KILL: {killer_id} -> {victim_id}")
                print(f"  Victim team: {victim_team}")

                # Check surviving teammates' knowledge of killer
                print(f"  Victim's surviving teammates' knowledge of killer {killer_id}:")

                for pid, knowledge in engine.info_manager.player_knowledge.items():
                    if knowledge.team == victim_team and pid != victim_id:
                        player = engine.players.get(pid)
                        if player and player.is_alive:
                            if killer_id in knowledge.enemies:
                                enemy_info = knowledge.enemies[killer_id]
                                print(f"    {pid}: KNOWS killer at ({enemy_info.last_known_x:.2f}, {enemy_info.last_known_y:.2f})")
                                print(f"           Source: {enemy_info.source.value}, Age: {state.current_time_ms - enemy_info.last_seen_ms}ms")
                            else:
                                print(f"    {pid}: Does NOT know killer position yet")

                # Check if enemies heard gunfire
                killer = engine.players.get(killer_id)
                if killer:
                    print(f"  Enemies who heard gunfire from killer at ({killer.x:.2f}, {killer.y:.2f}):")

                    for pid, knowledge in engine.info_manager.player_knowledge.items():
                        if knowledge.team != killer.side:
                            player = engine.players.get(pid)
                            if player and player.is_alive and pid != victim_id:
                                if killer_id in knowledge.enemies:
                                    enemy_info = knowledge.enemies[killer_id]
                                    if enemy_info.source == InfoSource.SOUND_GUNFIRE:
                                        print(f"    {pid}: Heard gunfire, knows approx position")
                                    elif enemy_info.source == InfoSource.VISION:
                                        print(f"    {pid}: Has DIRECT VISION of killer")
                                    elif enemy_info.source == InfoSource.CALLOUT:
                                        print(f"    {pid}: Got CALLOUT about killer")

            last_kill_count = len(kill_events)

        if state.status == 'completed':
            break

    print("\n" + "=" * 70)
    print("SUMMARY: Knowledge System Flow")
    print("=" * 70)
    print("""
1. When a kill happens:
   - notify_kill() queues a death callout for victim's teammates
   - propagate_sound() alerts enemies within 60% map range

2. After 500ms (CALLOUT_DELAY_MS):
   - Teammates receive callout via process_callouts()
   - Killer position added to their knowledge.enemies

3. In combat:
   - _get_information_advantage() checks knowledge.enemies
   - If player knew about enemy (within 3s): 20% faster reaction
   - This affects combat outcome probabilities
""")


if __name__ == "__main__":
    asyncio.run(test_callout_effect())
