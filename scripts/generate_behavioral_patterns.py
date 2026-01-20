#!/usr/bin/env python3
"""
Generate comprehensive behavioral patterns for simulation AI.

Combines all extracted data into simulation-ready behavioral models:
- Combat engagement patterns (when/where to fight)
- Role-specific decision trees
- Economy decision patterns
- Round phase behaviors
- Agent-specific ability usage timing

Usage:
    python scripts/generate_behavioral_patterns.py
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any
from dataclasses import dataclass, asdict

GRID_DATA_DIR = Path(__file__).parent.parent.parent / "grid_data"
PROCESSED_DIR = GRID_DATA_DIR / "processed"
OUTPUT_DIR = Path(__file__).parent.parent / "app" / "data"


@dataclass
class CombatBehavior:
    """Combat engagement behavior parameters."""
    engagement_distance_mean: float
    engagement_distance_std: float
    headshot_base_rate: float
    first_kill_aggression: float
    trade_kill_rate: float  # How often team trades a kill
    time_to_kill_avg: float  # ms

    def to_dict(self):
        return asdict(self)


@dataclass
class RoleBehavior:
    """Role-specific behavior patterns."""
    role: str

    # Combat style
    aggression_level: float  # 0-1
    entry_probability: float  # Likelihood to entry frag
    clutch_success_rate: float

    # Positioning
    preferred_zones: List[str]
    avg_distance_from_site: float

    # Ability usage
    ability_timing: Dict[str, float]  # Phase -> usage rate

    # Weapons
    weapon_preference: Dict[str, float]

    def to_dict(self):
        return asdict(self)


@dataclass
class EconomyBehavior:
    """Economy decision patterns."""
    # Buy thresholds
    full_buy_threshold: int  # Credits needed for full buy
    force_buy_threshold: int  # Credits where team forces
    eco_threshold: int  # Credits where team ecos

    # Weapon choices by economy
    eco_weapon_preference: Dict[str, float]
    force_weapon_preference: Dict[str, float]
    full_buy_weapon_preference: Dict[str, float]

    def to_dict(self):
        return asdict(self)


@dataclass
class RoundPhaseBehavior:
    """Behavior by round phase (early/mid/late)."""
    phase: str

    # Movement
    movement_speed: float  # Relative speed 0-1
    rotation_probability: float

    # Combat
    engagement_likelihood: float
    peek_frequency: float

    # Utility usage
    utility_usage_rate: float

    def to_dict(self):
        return asdict(self)


@dataclass
class AgentBehavior:
    """Agent-specific ability usage patterns."""
    agent: str
    role: str

    # Ability usage timing (phase -> ability -> usage rate)
    ability_timing: Dict[str, Dict[str, float]]

    # Signature ability cooldown management
    signature_hold_rate: float  # How often they hold signature

    # Ultimate economy
    ult_usage_round_threshold: float  # Round importance to use ult

    def to_dict(self):
        return asdict(self)


def load_all_data():
    """Load all processed data."""
    data = {}

    files = [
        'full_profiles.json',
        'combat_patterns.json',
        'movement_patterns.json',
        'opponent_profiles.json',
        'c9_profiles.json'
    ]

    for filename in files:
        filepath = PROCESSED_DIR / filename
        if filepath.exists():
            with open(filepath) as f:
                key = filename.replace('.json', '').replace('_', '-')
                data[key] = json.load(f)

    return data


def generate_combat_behaviors(data: dict) -> Dict[str, CombatBehavior]:
    """Generate combat behaviors by role."""
    combat = data.get('combat-patterns', {})
    profiles = data.get('full-profiles', {})

    # Role-specific combat stats
    role_stats = defaultdict(lambda: {
        'distances': [],
        'headshots': [],
        'kills': [],
        'first_kills': [],
        'deaths': []
    })

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

    for pid, player in profiles.get('players', {}).items():
        # Determine role from most played agents
        agents = player.get('agents_played', {})
        if not agents:
            continue

        role_counts = defaultdict(int)
        for agent, count in agents.items():
            role = AGENT_TO_ROLE.get(agent.lower(), 'duelist')
            role_counts[role] += count

        primary_role = max(role_counts.keys(), key=lambda r: role_counts[r])

        role_stats[primary_role]['distances'].append(player.get('avg_kill_distance', 1700))
        role_stats[primary_role]['headshots'].append(player.get('headshot_rate', 0.2))
        role_stats[primary_role]['kills'].append(player.get('kills', 0))
        role_stats[primary_role]['first_kills'].append(player.get('first_kills', 0))
        role_stats[primary_role]['deaths'].append(player.get('deaths', 0))

    # Calculate combat behaviors
    behaviors = {}
    for role, stats in role_stats.items():
        if not stats['distances']:
            continue

        avg_dist = sum(stats['distances']) / len(stats['distances'])
        dist_variance = sum((d - avg_dist)**2 for d in stats['distances']) / len(stats['distances'])
        dist_std = dist_variance ** 0.5

        avg_hs = sum(stats['headshots']) / len(stats['headshots'])

        total_kills = sum(stats['kills'])
        total_fk = sum(stats['first_kills'])
        total_deaths = sum(stats['deaths'])

        fk_rate = total_fk / total_kills if total_kills > 0 else 0.1
        trade_rate = min(0.7, total_kills / max(1, total_deaths) * 0.3)

        behaviors[role] = CombatBehavior(
            engagement_distance_mean=round(avg_dist, 0),
            engagement_distance_std=round(dist_std, 0),
            headshot_base_rate=round(avg_hs, 3),
            first_kill_aggression=round(fk_rate, 3),
            trade_kill_rate=round(trade_rate, 3),
            time_to_kill_avg=300  # Average TTK estimate
        )

    return behaviors


def generate_role_behaviors(data: dict) -> Dict[str, RoleBehavior]:
    """Generate role-specific behavior patterns."""
    movement = data.get('movement-patterns', {})
    profiles = data.get('full-profiles', {})

    role_heatmaps = movement.get('role_heatmaps', {})
    phase_patterns = movement.get('phase_patterns', {})

    behaviors = {}

    # Default role behaviors based on game knowledge + data
    role_defaults = {
        'duelist': {
            'aggression': 0.8,
            'entry_prob': 0.6,
            'clutch': 0.35,
            'distance': 1600,
        },
        'initiator': {
            'aggression': 0.5,
            'entry_prob': 0.3,
            'clutch': 0.30,
            'distance': 1800,
        },
        'controller': {
            'aggression': 0.3,
            'entry_prob': 0.1,
            'clutch': 0.25,
            'distance': 1900,
        },
        'sentinel': {
            'aggression': 0.2,
            'entry_prob': 0.05,
            'clutch': 0.40,
            'distance': 1850,
        },
    }

    for role, defaults in role_defaults.items():
        # Get zone preferences from first map with data
        preferred_zones = []
        for map_name, roles in role_heatmaps.items():
            if role in roles:
                zones = list(roles[role].items())[:3]
                preferred_zones = [z for z, _ in zones]
                break

        # Default weapon preferences
        weapon_pref = {
            'duelist': {'vandal': 0.5, 'phantom': 0.3, 'operator': 0.1, 'sheriff': 0.1},
            'initiator': {'phantom': 0.4, 'vandal': 0.4, 'sheriff': 0.1, 'ghost': 0.1},
            'controller': {'vandal': 0.45, 'phantom': 0.4, 'odin': 0.05, 'sheriff': 0.1},
            'sentinel': {'phantom': 0.5, 'vandal': 0.3, 'judge': 0.1, 'sheriff': 0.1},
        }

        behaviors[role] = RoleBehavior(
            role=role,
            aggression_level=defaults['aggression'],
            entry_probability=defaults['entry_prob'],
            clutch_success_rate=defaults['clutch'],
            preferred_zones=preferred_zones or ['mid', 'a_site', 'b_site'],
            avg_distance_from_site=defaults['distance'],
            ability_timing={
                'early': 0.3,
                'mid': 0.5,
                'late': 0.2,
            },
            weapon_preference=weapon_pref.get(role, {'vandal': 0.5, 'phantom': 0.5})
        )

    return behaviors


def generate_economy_behaviors(data: dict) -> EconomyBehavior:
    """Generate economy decision patterns."""
    profiles = data.get('full-profiles', {})

    # Calculate average loadout values
    loadout_values = []
    for pid, player in profiles.get('players', {}).items():
        if player.get('avg_loadout_value'):
            loadout_values.append(player['avg_loadout_value'])

    avg_loadout = sum(loadout_values) / len(loadout_values) if loadout_values else 3500

    return EconomyBehavior(
        full_buy_threshold=3900,
        force_buy_threshold=2000,
        eco_threshold=1500,
        eco_weapon_preference={
            'classic': 0.4,
            'ghost': 0.3,
            'sheriff': 0.2,
            'shorty': 0.1,
        },
        force_weapon_preference={
            'spectre': 0.3,
            'stinger': 0.25,
            'bulldog': 0.2,
            'marshal': 0.15,
            'sheriff': 0.1,
        },
        full_buy_weapon_preference={
            'vandal': 0.45,
            'phantom': 0.35,
            'operator': 0.1,
            'odin': 0.05,
            'guardian': 0.05,
        }
    )


def generate_phase_behaviors(data: dict) -> Dict[str, RoundPhaseBehavior]:
    """Generate round phase behaviors."""
    movement = data.get('movement-patterns', {})
    phase_patterns = movement.get('phase_patterns', {})

    behaviors = {}

    phase_defaults = {
        'early': {
            'movement_speed': 0.9,
            'rotation_prob': 0.1,
            'engagement': 0.3,
            'peek_freq': 0.4,
            'utility': 0.5,
        },
        'mid': {
            'movement_speed': 0.7,
            'rotation_prob': 0.3,
            'engagement': 0.6,
            'peek_freq': 0.6,
            'utility': 0.3,
        },
        'late': {
            'movement_speed': 0.5,
            'rotation_prob': 0.5,
            'engagement': 0.8,
            'peek_freq': 0.8,
            'utility': 0.2,
        },
    }

    for phase, defaults in phase_defaults.items():
        behaviors[phase] = RoundPhaseBehavior(
            phase=phase,
            movement_speed=defaults['movement_speed'],
            rotation_probability=defaults['rotation_prob'],
            engagement_likelihood=defaults['engagement'],
            peek_frequency=defaults['peek_freq'],
            utility_usage_rate=defaults['utility']
        )

    return behaviors


def generate_agent_behaviors(data: dict) -> Dict[str, AgentBehavior]:
    """Generate agent-specific ability patterns."""
    profiles = data.get('full-profiles', {})
    c9 = data.get('c9-profiles', {})

    # Build ability usage patterns from player data
    agent_abilities = defaultdict(lambda: defaultdict(int))

    for pid, player in profiles.get('players', {}).items():
        for ability, count in player.get('top_abilities', {}).items():
            # Find which agent this ability belongs to
            agent = ability_to_agent(ability)
            if agent:
                agent_abilities[agent][ability] += count

    # Also add from c9 profiles
    for player in c9.get('players', []):
        for ability, count in player.get('top_abilities', {}).items():
            agent = ability_to_agent(ability)
            if agent:
                agent_abilities[agent][ability] += count

    behaviors = {}

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

    for agent, abilities in agent_abilities.items():
        total = sum(abilities.values())
        if total == 0:
            continue

        ability_rates = {a: c/total for a, c in abilities.items()}

        # Create phase-based timing (approximate from general patterns)
        ability_timing = {
            'early': {},
            'mid': {},
            'late': {},
        }

        # Distribute abilities across phases based on type
        for ability, rate in ability_rates.items():
            if is_recon_ability(ability):
                ability_timing['early'][ability] = rate * 0.5
                ability_timing['mid'][ability] = rate * 0.35
                ability_timing['late'][ability] = rate * 0.15
            elif is_flash_ability(ability):
                ability_timing['early'][ability] = rate * 0.3
                ability_timing['mid'][ability] = rate * 0.5
                ability_timing['late'][ability] = rate * 0.2
            elif is_smoke_ability(ability):
                ability_timing['early'][ability] = rate * 0.4
                ability_timing['mid'][ability] = rate * 0.4
                ability_timing['late'][ability] = rate * 0.2
            else:
                ability_timing['early'][ability] = rate * 0.33
                ability_timing['mid'][ability] = rate * 0.34
                ability_timing['late'][ability] = rate * 0.33

        behaviors[agent] = AgentBehavior(
            agent=agent,
            role=AGENT_TO_ROLE.get(agent, 'duelist'),
            ability_timing=ability_timing,
            signature_hold_rate=0.3,  # 30% hold rate
            ult_usage_round_threshold=0.7  # Use ult on important rounds
        )

    return behaviors


def ability_to_agent(ability: str) -> str:
    """Map ability name to agent."""
    ABILITY_TO_AGENT = {
        "nanoswarm": "killjoy", "alarmbot": "killjoy", "turret": "killjoy", "lockdown": "killjoy",
        "trapwire": "cypher", "cybercage": "cypher", "spycam": "cypher", "neuraltheft": "cypher",
        "darkcover": "omen", "shroudedstep": "omen", "paranoia": "omen", "fromtheshadows": "omen",
        "cloudburst": "jett", "updraft": "jett", "tailwind": "jett", "bladestorm": "jett",
        "blastpack": "raze", "boombot": "raze", "paintshells": "raze", "showstopper": "raze",
        "reconbolt": "sova", "shockbolt": "sova", "owldrone": "sova", "hunter'sfury": "sova",
        "skysmoke": "brimstone", "stimbeacon": "brimstone", "incendiary": "brimstone",
        "toxicscreen": "viper", "poisoncloud": "viper", "snakebite": "viper",
        "faultline": "breach", "flashpoint": "breach", "aftershock": "breach",
        "barrierorb": "sage", "sloworb": "sage", "healingorb": "sage",
        "headhunter": "chamber", "trademark": "chamber", "rendezvous": "chamber",
        "guidinglight": "skye", "trailblazer": "skye", "regrowth": "skye",
        "flashdrive": "kayo", "zeropoint": "kayo", "fragbolt": "kayo",
        "prowler": "fade", "seize": "fade", "haunt": "fade",
        "stars": "astra", "nebula": "astra", "novapulse": "astra",
        "highgear": "neon", "relaybolt": "neon", "fastelane": "neon",
        "gatecrash": "yoru", "blindside": "yoru", "fakeout": "yoru",
        "leer": "reyna", "devour": "reyna", "dismiss": "reyna",
        "dizzy": "gekko", "mosh": "gekko", "wingman": "gekko",
        "cascade": "harbor", "cove": "harbor", "hightide": "harbor",
        "ruse": "clove", "pickmeup": "clove", "meddle": "clove",
        "undercut": "iso", "contingency": "iso",
        "gravnet": "deadlock", "sonicsensor": "deadlock",
        "razorvine": "vyse", "shear": "vyse", "arcrose": "vyse",
        "nebuladissipate": "omen", "guidedsalvo": "kayo",
    }
    return ABILITY_TO_AGENT.get(ability.lower().replace(' ', '').replace('-', ''), '')


def is_recon_ability(ability: str) -> bool:
    """Check if ability is reconnaissance."""
    recon = ['reconbolt', 'owldrone', 'haunt', 'trailblazer', 'turret', 'spycam', 'sonicsensor']
    return ability.lower() in recon


def is_flash_ability(ability: str) -> bool:
    """Check if ability is a flash/blind."""
    flashes = ['flashpoint', 'blindside', 'guidinglight', 'flashdrive', 'paranoia', 'leer', 'dizzy', 'curveball']
    return ability.lower() in flashes


def is_smoke_ability(ability: str) -> bool:
    """Check if ability is smoke/vision block."""
    smokes = ['darkcover', 'skysmoke', 'nebula', 'cloudburst', 'toxicscreen', 'cascade', 'ruse']
    return ability.lower() in smokes


def generate_behavioral_patterns():
    """Generate all behavioral patterns."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading all extracted data...")
    data = load_all_data()

    print("Generating combat behaviors...")
    combat_behaviors = generate_combat_behaviors(data)

    print("Generating role behaviors...")
    role_behaviors = generate_role_behaviors(data)

    print("Generating economy behaviors...")
    economy_behavior = generate_economy_behaviors(data)

    print("Generating phase behaviors...")
    phase_behaviors = generate_phase_behaviors(data)

    print("Generating agent behaviors...")
    agent_behaviors = generate_agent_behaviors(data)

    # Build output
    output = {
        "generated": "2026-01-19",
        "description": "Behavioral patterns for simulation AI, extracted from 33 VCT matches",

        "combat_behaviors": {
            role: behavior.to_dict()
            for role, behavior in combat_behaviors.items()
        },

        "role_behaviors": {
            role: behavior.to_dict()
            for role, behavior in role_behaviors.items()
        },

        "economy_behavior": economy_behavior.to_dict(),

        "phase_behaviors": {
            phase: behavior.to_dict()
            for phase, behavior in phase_behaviors.items()
        },

        "agent_behaviors": {
            agent: behavior.to_dict()
            for agent, behavior in agent_behaviors.items()
        },
    }

    # Save
    with open(PROCESSED_DIR / "behavioral_patterns.json", 'w') as f:
        json.dump(output, f, indent=2)

    with open(OUTPUT_DIR / "behavioral_patterns.json", 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nExported to:")
    print(f"  - {PROCESSED_DIR / 'behavioral_patterns.json'}")
    print(f"  - {OUTPUT_DIR / 'behavioral_patterns.json'}")

    # Print summary
    print("\n" + "="*60)
    print("BEHAVIORAL PATTERNS SUMMARY")
    print("="*60)

    print("\n### COMBAT BEHAVIORS BY ROLE ###")
    for role, behavior in combat_behaviors.items():
        print(f"\n{role.upper()}:")
        print(f"  Engagement Distance: {behavior.engagement_distance_mean:.0f} ± {behavior.engagement_distance_std:.0f}")
        print(f"  Headshot Rate: {behavior.headshot_base_rate:.1%}")
        print(f"  First Kill Aggression: {behavior.first_kill_aggression:.1%}")

    print("\n### ROLE BEHAVIORS ###")
    for role, behavior in role_behaviors.items():
        print(f"\n{role.upper()}:")
        print(f"  Aggression: {behavior.aggression_level:.1%}")
        print(f"  Entry Probability: {behavior.entry_probability:.1%}")
        print(f"  Clutch Success: {behavior.clutch_success_rate:.1%}")
        print(f"  Preferred Zones: {', '.join(behavior.preferred_zones)}")

    print("\n### ECONOMY THRESHOLDS ###")
    print(f"  Full Buy: ${economy_behavior.full_buy_threshold}")
    print(f"  Force Buy: ${economy_behavior.force_buy_threshold}")
    print(f"  Eco: ${economy_behavior.eco_threshold}")

    print("\n### PHASE BEHAVIORS ###")
    for phase, behavior in phase_behaviors.items():
        print(f"\n{phase.upper()}:")
        print(f"  Movement Speed: {behavior.movement_speed:.1%}")
        print(f"  Engagement Likelihood: {behavior.engagement_likelihood:.1%}")
        print(f"  Utility Usage: {behavior.utility_usage_rate:.1%}")

    print(f"\n### AGENT BEHAVIORS ###")
    print(f"Agents with behavior profiles: {len(agent_behaviors)}")
    top_agents = sorted(agent_behaviors.keys())[:8]
    print(f"Sample: {', '.join(top_agents)}")


if __name__ == "__main__":
    generate_behavioral_patterns()
