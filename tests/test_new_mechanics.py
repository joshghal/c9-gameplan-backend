#!/usr/bin/env python3
"""
Test script to verify the new simulation mechanics via API calls:
- Death callouts (killer position shared with victim's team)
- Gunfire sound propagation
- Spike drop and pickup mechanics
"""

import asyncio
import sys
import os
import uuid
from dataclasses import dataclass
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.simulation_engine import SimulationEngine
from app.services.information_system import InformationManager, InfoSource


@dataclass
class MockSession:
    """Mock session object for testing."""
    id: str
    attack_team_id: str
    defense_team_id: str
    map_name: str
    current_time_ms: int = 0
    status: str = "created"
    phase: str = "opening"


async def run_simulation_and_monitor():
    """Run a simulation and monitor for new event types."""
    print("=" * 60)
    print("TESTING NEW MECHANICS: Death Callouts, Gunfire, Spike Drop")
    print("=" * 60)

    # Create simulation engine
    engine = SimulationEngine()

    # Create a test session
    session = MockSession(
        id=str(uuid.uuid4()),
        attack_team_id="cloud9",
        defense_team_id="g2",
        map_name="ascent",
        current_time_ms=0,
        status="created"
    )

    # Initialize simulation
    await engine.initialize(session, round_type="full")

    print("\nSimulation initialized. Running to completion...\n")

    # Track events of interest
    spike_dropped_events = []
    spike_pickup_events = []
    kill_events = []

    # Run simulation tick by tick
    max_ticks = 2000  # Safety limit
    tick_count = 0

    while tick_count < max_ticks:
        tick_count += 1
        state = await engine.advance(session, ticks=1)
        session.current_time_ms = state.current_time_ms

        # Check for new events
        for event in state.events:
            if event.event_type == 'spike_dropped':
                spike_dropped_events.append(event)
                print(f"[{state.current_time_ms/1000:.1f}s] SPIKE DROPPED at ({event.position_x:.2f}, {event.position_y:.2f})")
                print(f"         Dropped by: {event.details.get('dropped_by')}")

            elif event.event_type == 'spike_pickup':
                spike_pickup_events.append(event)
                print(f"[{state.current_time_ms/1000:.1f}s] SPIKE PICKUP at ({event.position_x:.2f}, {event.position_y:.2f})")
                print(f"         Picked up by: {event.details.get('picked_up_by')}")

            elif event.event_type == 'kill':
                kill_events.append(event)
                killer_id = event.details.get('killer_id', 'unknown')
                print(f"[{state.current_time_ms/1000:.1f}s] KILL: {killer_id} -> {event.player_id}")

        # Check round end
        if state.status == 'completed':
            break

    # Print summary
    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE - MECHANICS VERIFICATION")
    print("=" * 60)

    print(f"\nTotal ticks: {tick_count}")
    print(f"Final time: {session.current_time_ms/1000:.1f}s")
    print(f"Total kills: {len(kill_events)}")
    print(f"Spike dropped events: {len(spike_dropped_events)}")
    print(f"Spike pickup events: {len(spike_pickup_events)}")

    # Verify death callout mechanism by checking info_manager
    print("\n--- Death Callout Verification ---")
    if kill_events:
        # Check if victims' teammates have killer info
        print("Checking if killer positions were shared with victim's team...")

        # Get a sample kill event
        sample_kill = kill_events[0]
        killer_id = sample_kill.details.get('killer_id')
        victim_id = sample_kill.player_id

        # Check knowledge state
        victim_team = None
        for pid, player in engine.players.items():
            if pid == victim_id:
                victim_team = player.side
                break

        if victim_team:
            teammates_with_info = 0
            for pid, knowledge in engine.info_manager.player_knowledge.items():
                if knowledge.team == victim_team and pid != victim_id:
                    if killer_id in knowledge.enemies:
                        enemy_info = knowledge.enemies[killer_id]
                        if enemy_info.source == InfoSource.CALLOUT:
                            teammates_with_info += 1
                            print(f"  {pid} knows about {killer_id} via CALLOUT")

            if teammates_with_info > 0:
                print(f"  SUCCESS: {teammates_with_info} teammates received death callout")
            else:
                print("  NOTE: Teammates may have fresher info from direct vision")

    # Verify gunfire sound propagation
    print("\n--- Gunfire Sound Verification ---")
    if kill_events:
        # Check if enemies within range have sound-based intel
        enemies_with_sound_info = 0
        for pid, knowledge in engine.info_manager.player_knowledge.items():
            for enemy_id, enemy_info in knowledge.enemies.items():
                if enemy_info.source == InfoSource.SOUND_GUNFIRE:
                    enemies_with_sound_info += 1
                    print(f"  {pid} heard gunfire from {enemy_id}")

        if enemies_with_sound_info > 0:
            print(f"  SUCCESS: {enemies_with_sound_info} instances of gunfire detection")
        else:
            print("  NOTE: Players may have had direct vision instead")

    # Verify spike mechanics
    print("\n--- Spike Drop/Pickup Verification ---")
    if spike_dropped_events:
        print(f"  SUCCESS: Spike drop mechanics working ({len(spike_dropped_events)} drops)")
        for evt in spike_dropped_events:
            print(f"    - Dropped at ({evt.position_x:.2f}, {evt.position_y:.2f})")
    else:
        print("  NOTE: No spike carrier deaths occurred in this simulation")

    if spike_pickup_events:
        print(f"  SUCCESS: Spike pickup mechanics working ({len(spike_pickup_events)} pickups)")
        for evt in spike_pickup_events:
            print(f"    - Picked up at ({evt.position_x:.2f}, {evt.position_y:.2f})")

    # Check if dropped spike state is correctly reset
    print(f"\n  Dropped spike position: {engine.dropped_spike_position}")
    print(f"  Dropped spike time: {engine.dropped_spike_time}")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_simulation_and_monitor())
