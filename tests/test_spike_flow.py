"""Test spike flow improvements.

Validates that the spike flow fix achieves target statistics:
- ~35% of rounds should end in spike outcomes (plant→defuse or plant→explode)
- Previously: 99.6% elimination wins
- Target: 65% elimination wins, 35% spike outcomes
"""

import asyncio
import sys
import os
from uuid import uuid4

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from unittest.mock import MagicMock, AsyncMock
from app.services.simulation_engine import SimulationEngine


async def run_spike_flow_test(num_rounds: int = 100):
    """Run multiple simulation rounds and measure spike outcomes."""
    print(f"Running {num_rounds} simulation rounds to test spike flow...")
    print("-" * 60)

    results = {
        'total_rounds': 0,
        'spike_planted': 0,
        'spike_defused': 0,
        'spike_exploded': 0,
        'elimination_attack': 0,
        'elimination_defense': 0,
        'timeout': 0,
        'execute_started': 0,
        'site_control': 0,
        'avg_plant_time_ms': [],
        'total_kills': 0,
        'attack_kills': 0,
        'defense_kills': 0,
    }

    for i in range(num_rounds):
        # Create mock database session
        mock_db = MagicMock()

        # Initialize engine
        engine = SimulationEngine(mock_db)

        # Mock the required methods
        async def mock_load_team(team_id):
            return {
                'players': [
                    {'id': f'{team_id}_p{j}', 'agent': 'Jett'} for j in range(5)
                ]
            }

        async def mock_load_map(map_name):
            return {'width': 1.0, 'height': 1.0}

        engine._load_team = mock_load_team
        engine._load_map_config = mock_load_map

        # Mock pattern matcher
        engine.pattern_matcher.predict_team_positions = AsyncMock(return_value=[])

        # Create mock session with proper UUID
        mock_session = MagicMock()
        mock_session.id = uuid4()
        mock_session.status = 'running'
        mock_session.attack_team_id = 'attack_team'
        mock_session.defense_team_id = 'defense_team'
        mock_session.map_name = 'ascent'
        mock_session.current_time_ms = 0

        try:
            # Initialize simulation
            await engine.initialize(mock_session, round_type='full')

            # Run simulation for full round (100s = 781 ticks at 128ms per tick)
            max_ticks = 800
            for tick in range(max_ticks):
                state = await engine.advance(mock_session, ticks=1)

                # Check round end
                if engine._check_round_end(mock_session.current_time_ms):
                    break

                mock_session.current_time_ms = state.current_time_ms

            # Analyze results
            results['total_rounds'] += 1

            # Check events for what happened
            event_types = [e.event_type for e in engine.events]

            if 'spike_plant' in event_types:
                results['spike_planted'] += 1

                # Find plant time
                plant_event = next(e for e in engine.events if e.event_type == 'spike_plant')
                results['avg_plant_time_ms'].append(plant_event.timestamp_ms)

                if 'spike_defuse' in event_types:
                    results['spike_defused'] += 1
                elif engine.spike_planted:
                    # Check if spike exploded
                    time_since_plant = mock_session.current_time_ms - (engine.spike_plant_time or 0)
                    if time_since_plant >= engine.SPIKE_TIME_MS:
                        results['spike_exploded'] += 1

            if 'execute_start' in event_types:
                results['execute_started'] += 1

            if 'site_control' in event_types:
                results['site_control'] += 1

            # Count kills
            kills = [e for e in engine.events if e.event_type == 'kill']
            results['total_kills'] += len(kills)
            for kill in kills:
                killer_id = kill.details.get('killer_id', '')
                killer = engine.players.get(killer_id)
                if killer and killer.side == 'attack':
                    results['attack_kills'] += 1
                else:
                    results['defense_kills'] += 1

            # Check final state
            alive_attack = sum(1 for p in engine.players.values() if p.side == 'attack' and p.is_alive)
            alive_defense = sum(1 for p in engine.players.values() if p.side == 'defense' and p.is_alive)

            if alive_attack == 0 and not engine.spike_planted:
                results['elimination_defense'] += 1
            elif alive_defense == 0 and not engine.spike_planted:
                results['elimination_attack'] += 1
            elif not engine.spike_planted and mock_session.current_time_ms >= engine.ROUND_TIME_MS:
                results['timeout'] += 1

        except Exception as e:
            print(f"Error in round {i}: {e}")
            continue

        # Progress update
        if (i + 1) % 20 == 0:
            print(f"Completed {i + 1}/{num_rounds} rounds...")

    # Calculate statistics
    print("\n" + "=" * 60)
    print("SPIKE FLOW TEST RESULTS")
    print("=" * 60)

    total = results['total_rounds']
    if total == 0:
        print("No rounds completed!")
        return results

    spike_outcomes = results['spike_defused'] + results['spike_exploded']
    elimination_outcomes = results['elimination_attack'] + results['elimination_defense']

    print(f"\nTotal Rounds: {total}")
    print(f"\nRound Outcomes:")
    print(f"  Spike Planted:     {results['spike_planted']:3d} ({100*results['spike_planted']/total:.1f}%)")
    print(f"  - Defused:         {results['spike_defused']:3d} ({100*results['spike_defused']/total:.1f}%)")
    print(f"  - Exploded:        {results['spike_exploded']:3d} ({100*results['spike_exploded']/total:.1f}%)")
    print(f"  Elimination (Atk): {results['elimination_attack']:3d} ({100*results['elimination_attack']/total:.1f}%)")
    print(f"  Elimination (Def): {results['elimination_defense']:3d} ({100*results['elimination_defense']/total:.1f}%)")
    print(f"  Timeout:           {results['timeout']:3d} ({100*results['timeout']/total:.1f}%)")

    print(f"\nSpike Outcome Rate: {100*spike_outcomes/total:.1f}% (Target: ~35%)")
    print(f"Elimination Rate:   {100*elimination_outcomes/total:.1f}% (Target: ~65%)")

    if results['avg_plant_time_ms']:
        avg_plant = sum(results['avg_plant_time_ms']) / len(results['avg_plant_time_ms'])
        print(f"\nAverage Plant Time: {avg_plant/1000:.1f}s (Target: ~56s)")

    print(f"\nExecute System:")
    print(f"  Executes Started:  {results['execute_started']:3d} ({100*results['execute_started']/total:.1f}%)")
    print(f"  Site Control:      {results['site_control']:3d} ({100*results['site_control']/total:.1f}%)")

    avg_kills = results['total_kills'] / total if total > 0 else 0
    print(f"\nCombat Statistics:")
    print(f"  Total Kills:       {results['total_kills']:3d} ({avg_kills:.1f} per round, Target: ~7.5)")
    print(f"  Attack Kills:      {results['attack_kills']:3d}")
    print(f"  Defense Kills:     {results['defense_kills']:3d}")

    # Assessment
    print("\n" + "-" * 60)
    spike_rate = 100 * spike_outcomes / total
    if spike_rate >= 25 and spike_rate <= 45:
        print("PASS: Spike outcome rate is within target range (25-45%)")
    elif spike_rate > 10:
        print(f"IMPROVED: Spike outcome rate ({spike_rate:.1f}%) is better than before (0.4%)")
    else:
        print(f"NEEDS WORK: Spike outcome rate ({spike_rate:.1f}%) is still too low")

    return results


if __name__ == '__main__':
    asyncio.run(run_spike_flow_test(100))
