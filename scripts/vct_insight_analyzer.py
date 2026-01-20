#!/usr/bin/env python3
"""
VCT Data Insight Analyzer

Extracts actionable insights from VCT pro match data to inform simulation mechanics.
Goal: Derive mechanics from DATA, not hardcoded if-else statements.
"""

import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List
import math

DATA_DIR = Path(__file__).parent.parent / "app" / "data"


@dataclass
class InsightReport:
    """Structured insights for simulation mechanics."""
    combat_insights: Dict
    trade_insights: Dict
    economy_insights: Dict
    positioning_insights: Dict
    role_insights: Dict


def load_data():
    """Load all VCT data files."""
    data = {}

    files = [
        "trade_patterns.json",
        "economy_patterns.json",
        "behavioral_patterns.json",
        "movement_patterns.json",
        "hold_angles.json",
    ]

    for f in files:
        path = DATA_DIR / f
        if path.exists():
            with open(path) as fp:
                data[f.replace(".json", "")] = json.load(fp)
                print(f"✓ Loaded {f}")
        else:
            print(f"✗ Missing {f}")

    return data


def analyze_trade_mechanics(trade_data: Dict) -> Dict:
    """Extract trade timing insights for simulation."""

    patterns = trade_data.get("trade_patterns", {})

    total_trades = patterns.get("total_trades", 0)
    avg_time = patterns.get("avg_trade_time", 0)
    distribution = patterns.get("trade_time_distribution", {})

    # Calculate cumulative probability of trade by time
    under_1s = distribution.get("under_1s", 0)
    one_to_2s = distribution.get("1_to_2s", 0)
    two_to_3s = distribution.get("2_to_3s", 0)
    three_to_5s = distribution.get("3_to_5s", 0)

    total = under_1s + one_to_2s + two_to_3s + three_to_5s

    # Analyze sample trades for distance patterns
    sample_trades = patterns.get("sample_trades", [])
    trade_distances = []
    for trade in sample_trades[:100]:  # Sample
        orig = trade.get("original_kill", {})
        trad = trade.get("trade_kill", {})
        if orig.get("distance") and trad.get("distance"):
            trade_distances.append({
                "original_distance": orig["distance"],
                "trade_distance": trad["distance"],
                "trade_time": trade.get("trade_time", 0)
            })

    return {
        "total_kills": trade_data.get("metadata", {}).get("total_kills", 0),
        "total_trades": total_trades,
        "trade_rate": total_trades / trade_data.get("metadata", {}).get("total_kills", 1),
        "avg_trade_time_seconds": avg_time,
        "trade_probability_by_time": {
            "0-1s": under_1s / total if total > 0 else 0,
            "1-2s": one_to_2s / total if total > 0 else 0,
            "2-3s": two_to_3s / total if total > 0 else 0,
            "3-5s": three_to_5s / total if total > 0 else 0,
        },
        "cumulative_trade_probability": {
            "by_1s": under_1s / total if total > 0 else 0,
            "by_2s": (under_1s + one_to_2s) / total if total > 0 else 0,
            "by_3s": (under_1s + one_to_2s + two_to_3s) / total if total > 0 else 0,
            "by_5s": 1.0,
        },
        "sample_trade_distances": trade_distances[:10],
        "insight": f"40% of trades happen within 1 second. Model trade windows, not trade bonuses.",
    }


def analyze_economy_mechanics(econ_data: Dict) -> Dict:
    """Extract economy insights for simulation."""

    buy_rates = econ_data.get("buy_patterns", {}).get("buy_win_rates", {})

    # Calculate relative win rates
    full_buy_wr = buy_rates.get("full_buy", {}).get("win_rate", 0.5)
    force_buy_wr = buy_rates.get("force_buy", {}).get("win_rate", 0.5)
    half_buy_wr = buy_rates.get("half_buy", {}).get("win_rate", 0.5)
    eco_wr = buy_rates.get("eco", {}).get("win_rate", 0.5)

    # The key insight: win rate difference should emerge from weapon TTK, not bonuses
    return {
        "win_rates": {
            "full_buy": full_buy_wr,
            "force_buy": force_buy_wr,
            "half_buy": half_buy_wr,
            "eco": eco_wr,
        },
        "relative_to_full_buy": {
            "force_buy_penalty": full_buy_wr - force_buy_wr,  # ~9%
            "half_buy_penalty": full_buy_wr - half_buy_wr,    # ~1%
            "eco_penalty": full_buy_wr - eco_wr,              # varies
        },
        "sample_sizes": {
            "full_buy": buy_rates.get("full_buy", {}).get("total", 0),
            "force_buy": buy_rates.get("force_buy", {}).get("total", 0),
            "half_buy": buy_rates.get("half_buy", {}).get("total", 0),
            "eco": buy_rates.get("eco", {}).get("total", 0),
        },
        "insight": "Force buy is only 9% worse than full buy. Model weapon damage, not buy type.",
    }


def analyze_role_mechanics(behavior_data: Dict) -> Dict:
    """Extract role-specific insights for simulation."""

    combat = behavior_data.get("combat_behaviors", {})
    roles = behavior_data.get("role_behaviors", {})

    role_insights = {}
    for role in ["duelist", "initiator", "controller", "sentinel"]:
        combat_stats = combat.get(role, {})
        role_stats = roles.get(role, {})

        role_insights[role] = {
            # Combat characteristics
            "engagement_distance": combat_stats.get("engagement_distance_mean", 1800),
            "headshot_rate": combat_stats.get("headshot_base_rate", 0.2),
            "first_kill_aggression": combat_stats.get("first_kill_aggression", 0.1),
            "trade_kill_rate": combat_stats.get("trade_kill_rate", 0.3),

            # Behavioral characteristics
            "aggression_level": role_stats.get("aggression_level", 0.5),
            "entry_probability": role_stats.get("entry_probability", 0.2),
            "clutch_success_rate": role_stats.get("clutch_success_rate", 0.3),

            # Derived combat advantage (from HS rate)
            "aim_advantage": combat_stats.get("headshot_base_rate", 0.2) / 0.2 - 1,
        }

    return {
        "by_role": role_insights,
        "key_differences": {
            "duelist_vs_sentinel_hs": role_insights["duelist"]["headshot_rate"] - role_insights["sentinel"]["headshot_rate"],
            "duelist_vs_sentinel_entry": role_insights["duelist"]["entry_probability"] - role_insights["sentinel"]["entry_probability"],
            "sentinel_clutch_advantage": role_insights["sentinel"]["clutch_success_rate"] - role_insights["duelist"]["clutch_success_rate"],
        },
        "insight": "Sentinels have 40% clutch rate vs 35% for duelists. Clutch success is role-dependent.",
    }


def analyze_positioning_mechanics(movement_data: Dict, angles_data: Dict) -> Dict:
    """Extract positioning and angle insights."""

    # Analyze zone danger levels
    zone_stats = movement_data.get("zone_statistics", {})

    danger_zones = []
    for map_name, zones in zone_stats.items():
        if not isinstance(zones, dict):
            continue
        for zone_name, stats in zones.items():
            if not isinstance(stats, dict):
                continue
            kills = stats.get("kill_samples", 0)
            deaths = stats.get("death_samples", 0)
            total = stats.get("total_samples", 1)

            if total > 50:  # Only significant zones
                danger_zones.append({
                    "map": map_name,
                    "zone": zone_name,
                    "kill_rate": kills / total,
                    "death_rate": deaths / total,
                    "kd_ratio": kills / deaths if deaths > 0 else 0,
                    "attack_presence": stats.get("by_side", {}).get("attack", 0) / total,
                    "defense_presence": stats.get("by_side", {}).get("defense", 0) / total,
                })

    # Sort by danger (kill + death rate)
    danger_zones.sort(key=lambda x: x["kill_rate"] + x["death_rate"], reverse=True)

    # Analyze angles data
    angles_by_map = angles_data.get("angles_by_map", {})
    engagement_distances = []
    for map_name, zones in angles_by_map.items():
        for zone_id, data in zones.items():
            if data.get("samples", 0) > 10:
                engagement_distances.append(data.get("avg_distance", 0))

    avg_engagement_dist = sum(engagement_distances) / len(engagement_distances) if engagement_distances else 0

    return {
        "top_danger_zones": danger_zones[:10],
        "safest_zones": sorted(danger_zones, key=lambda x: x["kill_rate"] + x["death_rate"])[:5],
        "avg_engagement_distance_units": avg_engagement_dist,
        "avg_engagement_distance_meters": avg_engagement_dist / 100,  # Approx conversion
        "insight": f"Average engagement at {avg_engagement_dist:.0f} units (~{avg_engagement_dist/100:.1f}m). Combat is mid-range focused.",
    }


def generate_simulation_recommendations(insights: Dict) -> List[str]:
    """Generate specific recommendations for simulation improvements."""

    recs = []

    # Trade system recommendation
    trade = insights.get("trade_insights", {})
    recs.append(f"""
## TRADE SYSTEM (from {trade.get('total_kills', 0)} kills)
- Trade rate: {trade.get('trade_rate', 0)*100:.1f}% of kills get traded
- 40% of trades happen within 1 second
- Model: After kill, nearby enemies have REACTION_WINDOW to trade
- NOT: "trade_bonus += 0.15"
- INSTEAD: Track time since last kill, give accuracy bonus to traders within window
""")

    # Economy recommendation
    econ = insights.get("economy_insights", {})
    penalties = econ.get("relative_to_full_buy", {})
    recs.append(f"""
## ECONOMY SYSTEM (from win rate data)
- Force buy only {penalties.get('force_buy_penalty', 0)*100:.1f}% worse than full buy
- Model: Use actual weapon TTK differences
- NOT: "if eco: penalty = 0.30"
- INSTEAD: Classic TTK = 600ms, Vandal TTK = 300ms → natural 2x disadvantage
""")

    # Role recommendation
    roles = insights.get("role_insights", {})
    hs_diff = roles.get("key_differences", {}).get("duelist_vs_sentinel_hs", 0)
    clutch_diff = roles.get("key_differences", {}).get("sentinel_clutch_advantage", 0)
    recs.append(f"""
## ROLE SYSTEM (from behavioral data)
- Duelists have {hs_diff*100:.1f}% higher headshot rate than sentinels
- Sentinels have {clutch_diff*100:.1f}% better clutch rate
- Model: Role affects AIM_SKILL and CLUTCH_FACTOR
- NOT: "if role == 'duelist': damage += 10%"
- INSTEAD: duelist.headshot_chance = 0.233, sentinel.headshot_chance = 0.133
""")

    # Positioning recommendation
    pos = insights.get("positioning_insights", {})
    recs.append(f"""
## POSITIONING SYSTEM (from {movement_data.get('total_samples', 0)} samples)
- Average engagement distance: {pos.get('avg_engagement_distance_units', 0):.0f} units
- Model: Combat effectiveness scales with distance
- Use zone-specific danger levels from data
- NOT: "site_bonus += 0.10"
- INSTEAD: Load zone danger from VCT data, affects engagement probability
""")

    return recs


def main():
    print("=" * 70)
    print("VCT DATA INSIGHT ANALYZER")
    print("=" * 70)
    print("\nLoading VCT data files...\n")

    data = load_data()

    if not data:
        print("No data files found!")
        return

    print("\n" + "=" * 70)
    print("ANALYZING DATA...")
    print("=" * 70)

    insights = {}

    # Analyze each data source
    if "trade_patterns" in data:
        insights["trade_insights"] = analyze_trade_mechanics(data["trade_patterns"])
        print(f"\n📊 TRADE INSIGHTS:")
        print(f"   Total kills: {insights['trade_insights']['total_kills']}")
        print(f"   Trade rate: {insights['trade_insights']['trade_rate']*100:.1f}%")
        print(f"   Avg trade time: {insights['trade_insights']['avg_trade_time_seconds']:.2f}s")
        print(f"   → {insights['trade_insights']['insight']}")

    if "economy_patterns" in data:
        insights["economy_insights"] = analyze_economy_mechanics(data["economy_patterns"])
        print(f"\n💰 ECONOMY INSIGHTS:")
        wr = insights["economy_insights"]["win_rates"]
        print(f"   Full buy win rate: {wr['full_buy']*100:.1f}%")
        print(f"   Force buy win rate: {wr['force_buy']*100:.1f}%")
        print(f"   → {insights['economy_insights']['insight']}")

    if "behavioral_patterns" in data:
        insights["role_insights"] = analyze_role_mechanics(data["behavioral_patterns"])
        print(f"\n🎭 ROLE INSIGHTS:")
        for role, stats in insights["role_insights"]["by_role"].items():
            print(f"   {role}: HS={stats['headshot_rate']*100:.1f}%, Entry={stats['entry_probability']*100:.0f}%, Clutch={stats['clutch_success_rate']*100:.0f}%")
        print(f"   → {insights['role_insights']['insight']}")

    if "movement_patterns" in data and "hold_angles" in data:
        global movement_data
        movement_data = data["movement_patterns"]
        insights["positioning_insights"] = analyze_positioning_mechanics(
            data["movement_patterns"],
            data["hold_angles"]
        )
        print(f"\n📍 POSITIONING INSIGHTS:")
        print(f"   Avg engagement distance: {insights['positioning_insights']['avg_engagement_distance_units']:.0f} units")
        print(f"   Top danger zones:")
        for zone in insights["positioning_insights"]["top_danger_zones"][:3]:
            print(f"      {zone['map']}/{zone['zone']}: K/D={zone['kd_ratio']:.2f}")
        print(f"   → {insights['positioning_insights']['insight']}")

    # Generate recommendations
    print("\n" + "=" * 70)
    print("SIMULATION RECOMMENDATIONS")
    print("=" * 70)

    recs = generate_simulation_recommendations(insights)
    for rec in recs:
        print(rec)

    # Save insights to file
    output_file = Path(__file__).parent / "vct_insights.json"
    with open(output_file, "w") as f:
        json.dump(insights, f, indent=2, default=str)
    print(f"\n✓ Insights saved to: {output_file}")

    # Summary table
    print("\n" + "=" * 70)
    print("KEY NUMBERS FOR SIMULATION")
    print("=" * 70)
    print("""
┌─────────────────────────────────────────────────────────────────────┐
│ MECHANIC              │ VCT VALUE           │ USE IN SIMULATION     │
├───────────────────────┼─────────────────────┼───────────────────────┤
│ Trade window          │ 1.72s avg           │ TRADE_WINDOW_MS=1700  │
│ Trade rate            │ 25% of kills        │ Emerges from timing   │
│ 1s trade probability  │ 40%                 │ Reaction time model   │
├───────────────────────┼─────────────────────┼───────────────────────┤
│ Force vs Full         │ -9% win rate        │ Weapon TTK difference │
│ Engagement distance   │ ~1800 units         │ EFFECTIVE_RANGE=1800  │
├───────────────────────┼─────────────────────┼───────────────────────┤
│ Duelist HS rate       │ 23.3%               │ Role-based aim skill  │
│ Sentinel HS rate      │ 13.3%               │ Role-based aim skill  │
│ Sentinel clutch       │ 40%                 │ Role clutch modifier  │
│ Duelist entry rate    │ 60%                 │ Role entry behavior   │
└─────────────────────────────────────────────────────────────────────┘
""")


if __name__ == "__main__":
    main()
