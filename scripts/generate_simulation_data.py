#!/usr/bin/env python3
"""
Generate simulation-ready data from extracted patterns.

Creates:
- Player behavior profiles for AI
- Position heatmaps per role
- Zone transition probabilities
- Engagement statistics for combat model

Usage:
    python scripts/generate_simulation_data.py
"""

import json
import math
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

GRID_DATA_DIR = Path(__file__).parent.parent.parent / "grid_data"
PROCESSED_DIR = GRID_DATA_DIR / "processed"
OUTPUT_DIR = Path(__file__).parent.parent / "app" / "data"

# Role classification based on agents
ROLE_AGENTS = {
    "duelist": ["jett", "reyna", "raze", "phoenix", "yoru", "neon", "iso"],
    "initiator": ["sova", "breach", "skye", "kayo", "gekko", "fade"],
    "controller": ["omen", "brimstone", "astra", "viper", "harbor", "clove"],
    "sentinel": ["sage", "cypher", "killjoy", "chamber", "deadlock", "vyse"],
}

AGENT_TO_ROLE = {}
for role, agents in ROLE_AGENTS.items():
    for agent in agents:
        AGENT_TO_ROLE[agent] = role


@dataclass
class PlayerBehavior:
    """Simulation-ready player behavior profile."""
    id: str
    name: str
    team: str

    # Combat behavior
    aggression: float  # 0-1 based on first kill rate
    accuracy: float    # 0-1 based on headshot rate
    clutch_factor: float  # 0-1 based on K/D in disadvantage

    # Role preference
    primary_role: str
    role_distribution: Dict[str, float]

    # Weapon preference
    primary_weapon: str
    weapon_preference: Dict[str, float]

    # Positioning
    avg_engagement_distance: float
    preferred_distance_category: str  # "close", "medium", "long"

    # Economy
    avg_loadout: float
    eco_efficiency: float  # kills per credit spent

    def to_dict(self):
        return asdict(self)


@dataclass
class RolePattern:
    """Patterns for a specific role."""
    role: str

    # Average stats
    avg_kills_per_round: float
    avg_deaths_per_round: float
    avg_first_kill_rate: float
    avg_headshot_rate: float

    # Positioning
    avg_engagement_distance: float
    position_variance: float  # How much they move around

    # Weapon usage
    weapon_distribution: Dict[str, float]

    def to_dict(self):
        return asdict(self)


@dataclass
class CombatParameters:
    """Parameters for the combat model."""
    # Distance brackets (in game units)
    close_range_max: float = 1200
    medium_range_max: float = 2200
    long_range_max: float = 4000

    # Headshot rates by distance
    close_range_hs_rate: float = 0.30
    medium_range_hs_rate: float = 0.25
    long_range_hs_rate: float = 0.18

    # Weapon modifiers
    weapon_accuracy_modifier: Dict[str, float] = None
    weapon_effective_range: Dict[str, float] = None

    def __post_init__(self):
        if self.weapon_accuracy_modifier is None:
            self.weapon_accuracy_modifier = {}
        if self.weapon_effective_range is None:
            self.weapon_effective_range = {}

    def to_dict(self):
        return asdict(self)


def load_profiles() -> dict:
    """Load extracted player profiles."""
    with open(PROCESSED_DIR / "full_profiles.json", 'r') as f:
        return json.load(f)


def calculate_aggression(player: dict) -> float:
    """Calculate aggression score (0-1) from first kill rate and attack kills."""
    fk = player.get('first_kills', 0)
    rounds = max(1, player.get('rounds_played', 1))
    fk_rate = fk / rounds

    atk_kills = player.get('attack_kills', 0)
    total_kills = max(1, player.get('kills', 1))
    atk_ratio = atk_kills / total_kills

    # Combine: more first kills + more attack kills = more aggressive
    return min(1.0, (fk_rate * 2 + atk_ratio) / 2)


def calculate_role_from_agents(agents: dict) -> Tuple[str, Dict[str, float]]:
    """Determine role distribution from agent usage."""
    if not agents:
        return "duelist", {"duelist": 1.0}

    role_counts = Counter()
    total = sum(agents.values())

    for agent, count in agents.items():
        role = AGENT_TO_ROLE.get(agent, "duelist")
        role_counts[role] += count

    role_dist = {role: count/total for role, count in role_counts.items()}
    primary = max(role_dist.keys(), key=lambda r: role_dist[r])

    return primary, role_dist


def calculate_weapon_preference(weapons: dict) -> Tuple[str, Dict[str, float]]:
    """Calculate weapon preference distribution."""
    if not weapons:
        return "vandal", {"vandal": 1.0}

    total = sum(weapons.values())
    pref = {w: c/total for w, c in weapons.items()}
    primary = max(pref.keys(), key=lambda w: pref[w])

    return primary, pref


def categorize_distance(avg_dist: float) -> str:
    """Categorize average engagement distance."""
    if avg_dist < 1400:
        return "close"
    elif avg_dist < 2000:
        return "medium"
    else:
        return "long"


def generate_player_behaviors(profiles: dict) -> List[PlayerBehavior]:
    """Generate PlayerBehavior objects from profiles."""
    behaviors = []

    for pid, player in profiles.get('players', {}).items():
        name = player.get('name', 'Unknown')
        team = player.get('team_name', 'Unknown')

        # Skip players with very few kills
        if player.get('kills', 0) < 50:
            continue

        # Calculate metrics
        aggression = calculate_aggression(player)
        accuracy = player.get('headshot_rate', 0.2)
        kd = player.get('kd_ratio', 1.0)
        clutch = min(1.0, kd * 0.5)  # Simplified

        # Role
        primary_role, role_dist = calculate_role_from_agents(player.get('agents_played', {}))

        # Weapons
        primary_weapon, weapon_pref = calculate_weapon_preference(player.get('weapon_kills', {}))

        # Distance
        avg_dist = player.get('avg_kill_distance', 1700)
        dist_category = categorize_distance(avg_dist)

        # Economy
        avg_loadout = player.get('avg_loadout_value', 3000)
        kills = max(1, player.get('kills', 1))
        eco_eff = kills / max(1, avg_loadout)  # Kills per credit

        behavior = PlayerBehavior(
            id=pid,
            name=name,
            team=team,
            aggression=round(aggression, 3),
            accuracy=round(accuracy, 3),
            clutch_factor=round(clutch, 3),
            primary_role=primary_role,
            role_distribution=role_dist,
            primary_weapon=primary_weapon,
            weapon_preference=weapon_pref,
            avg_engagement_distance=round(avg_dist, 0),
            preferred_distance_category=dist_category,
            avg_loadout=round(avg_loadout, 0),
            eco_efficiency=round(eco_eff * 1000, 2)  # Per 1000 credits
        )

        behaviors.append(behavior)

    return behaviors


def generate_role_patterns(profiles: dict) -> List[RolePattern]:
    """Generate patterns per role."""
    role_data = defaultdict(lambda: {
        'kills': [], 'deaths': [], 'fk_rate': [], 'hs_rate': [],
        'distances': [], 'weapons': Counter()
    })

    for pid, player in profiles.get('players', {}).items():
        if player.get('kills', 0) < 50:
            continue

        primary_role, _ = calculate_role_from_agents(player.get('agents_played', {}))

        rounds = max(1, player.get('rounds_played', 1))

        role_data[primary_role]['kills'].append(player.get('kills', 0) / rounds)
        role_data[primary_role]['deaths'].append(player.get('deaths', 0) / rounds)
        role_data[primary_role]['fk_rate'].append(player.get('first_kills', 0) / rounds)
        role_data[primary_role]['hs_rate'].append(player.get('headshot_rate', 0.2))
        role_data[primary_role]['distances'].append(player.get('avg_kill_distance', 1700))

        for weapon, count in player.get('weapon_kills', {}).items():
            role_data[primary_role]['weapons'][weapon] += count

    patterns = []
    for role, data in role_data.items():
        if not data['kills']:
            continue

        # Normalize weapon distribution
        total_weapon = sum(data['weapons'].values())
        weapon_dist = {w: c/total_weapon for w, c in data['weapons'].most_common(5)}

        # Calculate position variance from distance spread
        if len(data['distances']) > 1:
            mean_dist = sum(data['distances']) / len(data['distances'])
            variance = sum((d - mean_dist)**2 for d in data['distances']) / len(data['distances'])
            pos_variance = math.sqrt(variance)
        else:
            pos_variance = 0

        pattern = RolePattern(
            role=role,
            avg_kills_per_round=round(sum(data['kills']) / len(data['kills']), 3),
            avg_deaths_per_round=round(sum(data['deaths']) / len(data['deaths']), 3),
            avg_first_kill_rate=round(sum(data['fk_rate']) / len(data['fk_rate']), 3),
            avg_headshot_rate=round(sum(data['hs_rate']) / len(data['hs_rate']), 3),
            avg_engagement_distance=round(sum(data['distances']) / len(data['distances']), 0),
            position_variance=round(pos_variance, 0),
            weapon_distribution=weapon_dist
        )
        patterns.append(pattern)

    return patterns


def generate_combat_parameters(profiles: dict) -> CombatParameters:
    """Generate combat model parameters from data."""

    # Load combat patterns
    with open(PROCESSED_DIR / "combat_patterns.json", 'r') as f:
        combat = json.load(f)

    weapon_distances = combat.get('weapon_distances', {})

    # Build weapon accuracy modifiers (relative to vandal baseline)
    vandal_dist = weapon_distances.get('vandal', {}).get('mean', 1800)
    accuracy_mod = {}
    for weapon, data in weapon_distances.items():
        dist = data.get('mean', 1800)
        # Closer range weapons have higher accuracy at close range
        accuracy_mod[weapon] = round(vandal_dist / dist, 2)

    # Effective ranges
    effective_range = {}
    for weapon, data in weapon_distances.items():
        # Effective range is roughly the mean distance + 1 std
        effective_range[weapon] = round(data.get('mean', 1800) * 1.3, 0)

    return CombatParameters(
        weapon_accuracy_modifier=accuracy_mod,
        weapon_effective_range=effective_range
    )


def export_simulation_data():
    """Export all simulation-ready data."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading profiles...")
    profiles = load_profiles()

    print("Generating player behaviors...")
    behaviors = generate_player_behaviors(profiles)

    # Separate C9 players
    c9_behaviors = [b for b in behaviors if b.team == "Cloud9"]
    other_behaviors = [b for b in behaviors if b.team != "Cloud9"]

    print(f"  C9 players: {len(c9_behaviors)}")
    print(f"  Other players: {len(other_behaviors)}")

    print("Generating role patterns...")
    role_patterns = generate_role_patterns(profiles)

    print("Generating combat parameters...")
    combat_params = generate_combat_parameters(profiles)

    # Export
    simulation_data = {
        "generated": "2026-01-19",
        "source": "33 VCT matches (Feb 2024 - Aug 2025)",
        "total_kills_analyzed": profiles.get('total_kills', 0),

        "player_behaviors": {
            "cloud9": [b.to_dict() for b in c9_behaviors],
            "opponents": [b.to_dict() for b in other_behaviors],
        },

        "role_patterns": [p.to_dict() for p in role_patterns],

        "combat_parameters": combat_params.to_dict(),
    }

    with open(OUTPUT_DIR / "simulation_profiles.json", 'w') as f:
        json.dump(simulation_data, f, indent=2)

    print(f"\nExported to {OUTPUT_DIR / 'simulation_profiles.json'}")

    # Print summary
    print("\n" + "="*60)
    print("SIMULATION DATA SUMMARY")
    print("="*60)

    print("\n### C9 PLAYER BEHAVIORS ###\n")
    print(f"{'Player':<12} {'Role':<12} {'Aggr':<6} {'Acc':<6} {'Weapon':<10} {'Dist'}")
    print("-" * 60)
    for b in c9_behaviors:
        print(f"{b.name:<12} {b.primary_role:<12} {b.aggression:<6.2f} {b.accuracy:<6.2f} "
              f"{b.primary_weapon:<10} {b.preferred_distance_category}")

    print("\n### ROLE PATTERNS ###\n")
    for p in role_patterns:
        print(f"{p.role}:")
        print(f"  K/R: {p.avg_kills_per_round:.2f}, D/R: {p.avg_deaths_per_round:.2f}")
        print(f"  FK rate: {p.avg_first_kill_rate:.2f}, HS rate: {p.avg_headshot_rate:.2f}")
        print(f"  Avg distance: {p.avg_engagement_distance:.0f}")
        print()

    print("### WEAPON EFFECTIVE RANGES ###\n")
    for weapon in ['vandal', 'phantom', 'operator', 'sheriff', 'spectre']:
        if weapon in combat_params.weapon_effective_range:
            print(f"  {weapon}: {combat_params.weapon_effective_range[weapon]:.0f} units")


if __name__ == "__main__":
    export_simulation_data()
