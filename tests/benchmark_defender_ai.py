"""
Benchmark: Defender AI Movement Quality
Runs N simulations per map, records win rates, kill counts, spike plants,
round durations. Save results to JSON for before/after comparison.

Usage:
    python tests/benchmark_defender_ai.py --label baseline --runs 20
    # ... apply changes ...
    python tests/benchmark_defender_ai.py --label improved --runs 20
    python tests/benchmark_defender_ai.py --compare baseline improved
"""

import asyncio
import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.services.simulation_engine import SimulationEngine


@dataclass
class DemoSession:
    id: str
    attack_team_id: str
    defense_team_id: str
    map_name: str
    round_type: str
    status: str = "created"
    current_time_ms: int = 0
    phase: str = "opening"


MAPS = [
    "ascent", "bind", "split", "icebox", "breeze",
    "fracture", "pearl", "sunset", "haven", "lotus",
    "abyss",
]

RESULTS_DIR = Path(__file__).parent / "benchmark_results"


async def run_single(map_name: str) -> dict:
    """Run one full simulation on a map, return stats."""
    engine = SimulationEngine(db=None)
    session = DemoSession(
        id=str(uuid4()),
        attack_team_id="cloud9",
        defense_team_id="g2",
        map_name=map_name,
        round_type="full",
    )

    state = await engine.initialize(session, round_type="full")

    winner = "defense"
    final_time = 0
    spike_planted = False

    for _ in range(1000):
        state = await engine.advance(session, 5)
        session.current_time_ms = state.current_time_ms
        attack_alive = sum(1 for p in state.positions if p.side == "attack" and p.is_alive)
        defense_alive = sum(1 for p in state.positions if p.side == "defense" and p.is_alive)
        final_time = state.current_time_ms
        spike_planted = state.spike_planted

        if attack_alive == 0 or defense_alive == 0 or state.current_time_ms >= 100000:
            if defense_alive == 0:
                winner = "attack"
            elif attack_alive == 0:
                winner = "defense"
            elif spike_planted:
                winner = "attack"
            break

    kills = []
    if state.events:
        kills = [e for e in state.events if hasattr(e, 'event_type') and e.event_type == 'kill']

    attack_kills = 0
    defense_kills = 0
    for k in kills:
        details = k.details if hasattr(k, 'details') else {}
        if isinstance(details, dict):
            if details.get('attacker_side') == 'attack':
                attack_kills += 1
            elif details.get('attacker_side') == 'defense':
                defense_kills += 1

    return {
        "map": map_name,
        "winner": winner,
        "duration_ms": final_time,
        "spike_planted": spike_planted,
        "attack_kills": attack_kills,
        "defense_kills": defense_kills,
        "total_kills": len(kills),
        "attack_alive": attack_alive,
        "defense_alive": defense_alive,
    }


async def run_benchmark(runs_per_map: int, label: str):
    """Run benchmark across all maps."""
    RESULTS_DIR.mkdir(exist_ok=True)
    all_results = {}
    total_start = time.time()

    print(f"\n{'='*60}")
    print(f"  DEFENDER AI BENCHMARK — '{label}'")
    print(f"  {runs_per_map} runs × {len(MAPS)} maps = {runs_per_map * len(MAPS)} total")
    print(f"{'='*60}\n")

    for map_name in MAPS:
        print(f"  [{map_name.upper():>10}] Running {runs_per_map} simulations...", end=" ", flush=True)
        map_start = time.time()

        results = []
        for i in range(runs_per_map):
            try:
                result = await run_single(map_name)
                results.append(result)
            except Exception as e:
                print(f"\n    ERROR on run {i+1}: {e}")
                continue

        elapsed = time.time() - map_start

        # Aggregate
        if results:
            atk_wins = sum(1 for r in results if r["winner"] == "attack")
            def_wins = sum(1 for r in results if r["winner"] == "defense")
            spike_rate = sum(1 for r in results if r["spike_planted"]) / len(results)
            avg_duration = sum(r["duration_ms"] for r in results) / len(results)
            avg_atk_kills = sum(r["attack_kills"] for r in results) / len(results)
            avg_def_kills = sum(r["defense_kills"] for r in results) / len(results)

            all_results[map_name] = {
                "runs": len(results),
                "attack_wins": atk_wins,
                "defense_wins": def_wins,
                "attack_win_rate": round(atk_wins / len(results), 3),
                "defense_win_rate": round(def_wins / len(results), 3),
                "spike_plant_rate": round(spike_rate, 3),
                "avg_duration_ms": round(avg_duration),
                "avg_attack_kills": round(avg_atk_kills, 2),
                "avg_defense_kills": round(avg_def_kills, 2),
                "individual_runs": results,
            }

            print(f"ATK {atk_wins}/{len(results)} ({atk_wins/len(results)*100:.0f}%)  "
                  f"DEF {def_wins}/{len(results)} ({def_wins/len(results)*100:.0f}%)  "
                  f"plant {spike_rate*100:.0f}%  "
                  f"avg {avg_duration/1000:.1f}s  "
                  f"({elapsed:.1f}s)")

    # Global summary
    total_runs = sum(m["runs"] for m in all_results.values())
    total_atk = sum(m["attack_wins"] for m in all_results.values())
    total_def = sum(m["defense_wins"] for m in all_results.values())
    total_elapsed = time.time() - total_start

    summary = {
        "label": label,
        "total_runs": total_runs,
        "global_attack_win_rate": round(total_atk / total_runs, 3) if total_runs else 0,
        "global_defense_win_rate": round(total_def / total_runs, 3) if total_runs else 0,
        "total_attack_wins": total_atk,
        "total_defense_wins": total_def,
        "elapsed_seconds": round(total_elapsed, 1),
    }

    print(f"\n{'='*60}")
    print(f"  SUMMARY — '{label}'")
    print(f"  Total: {total_runs} runs in {total_elapsed:.1f}s")
    print(f"  Attack win rate:  {total_atk}/{total_runs} ({summary['global_attack_win_rate']*100:.1f}%)")
    print(f"  Defense win rate: {total_def}/{total_runs} ({summary['global_defense_win_rate']*100:.1f}%)")
    print(f"  VCT target: ~53% attack / ~47% defense")
    print(f"{'='*60}\n")

    output = {"summary": summary, "per_map": all_results}
    output_path = RESULTS_DIR / f"{label}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"  Results saved to: {output_path}\n")
    return output


def compare_results(label_a: str, label_b: str):
    """Compare two benchmark runs side by side."""
    path_a = RESULTS_DIR / f"{label_a}.json"
    path_b = RESULTS_DIR / f"{label_b}.json"

    if not path_a.exists():
        print(f"ERROR: {path_a} not found")
        return
    if not path_b.exists():
        print(f"ERROR: {path_b} not found")
        return

    with open(path_a) as f:
        data_a = json.load(f)
    with open(path_b) as f:
        data_b = json.load(f)

    sa = data_a["summary"]
    sb = data_b["summary"]

    print(f"\n{'='*70}")
    print(f"  COMPARISON: '{label_a}' vs '{label_b}'")
    print(f"{'='*70}\n")

    # Global
    print(f"  {'METRIC':<25} {'[' + label_a + ']':>15} {'[' + label_b + ']':>15} {'DELTA':>10}")
    print(f"  {'-'*65}")

    atk_a = sa["global_attack_win_rate"] * 100
    atk_b = sb["global_attack_win_rate"] * 100
    def_a = sa["global_defense_win_rate"] * 100
    def_b = sb["global_defense_win_rate"] * 100

    print(f"  {'Attack Win Rate':<25} {atk_a:>14.1f}% {atk_b:>14.1f}% {atk_b - atk_a:>+9.1f}%")
    print(f"  {'Defense Win Rate':<25} {def_a:>14.1f}% {def_b:>14.1f}% {def_b - def_a:>+9.1f}%")
    print()

    # Per map
    print(f"  {'MAP':<12} {'ATK% [' + label_a + ']':>14} {'ATK% [' + label_b + ']':>14} {'DEF Δ':>8} {'PLANT% Δ':>10} {'DUR Δ':>8}")
    print(f"  {'-'*65}")

    all_maps = set(list(data_a.get("per_map", {}).keys()) + list(data_b.get("per_map", {}).keys()))
    for m in sorted(all_maps):
        ma = data_a.get("per_map", {}).get(m, {})
        mb = data_b.get("per_map", {}).get(m, {})
        if not ma or not mb:
            continue

        atk_a_m = ma["attack_win_rate"] * 100
        atk_b_m = mb["attack_win_rate"] * 100
        def_delta = (mb["defense_win_rate"] - ma["defense_win_rate"]) * 100
        plant_delta = (mb["spike_plant_rate"] - ma["spike_plant_rate"]) * 100
        dur_delta = (mb["avg_duration_ms"] - ma["avg_duration_ms"]) / 1000

        print(f"  {m:<12} {atk_a_m:>13.0f}% {atk_b_m:>13.0f}% {def_delta:>+7.0f}% {plant_delta:>+9.0f}% {dur_delta:>+7.1f}s")

    print(f"\n  VCT benchmark target: ~53% attack / ~47% defense")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Defender AI")
    parser.add_argument("--label", type=str, help="Label for this benchmark run (e.g. 'baseline', 'improved')")
    parser.add_argument("--runs", type=int, default=20, help="Simulations per map (default: 20)")
    parser.add_argument("--compare", nargs=2, metavar=("LABEL_A", "LABEL_B"), help="Compare two saved results")

    args = parser.parse_args()

    if args.compare:
        compare_results(args.compare[0], args.compare[1])
    elif args.label:
        asyncio.run(run_benchmark(args.runs, args.label))
    else:
        parser.print_help()
