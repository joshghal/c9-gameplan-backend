#!/usr/bin/env python3
"""
Extract movement patterns and zone transitions from GRID match data.

Creates:
- Position heatmaps per role/agent
- Zone transition probabilities
- Site preference patterns
- Round phase positioning (early, mid, late)

Usage:
    python scripts/extract_movement_patterns.py
"""

import json
import math
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict, field

GRID_DATA_DIR = Path(__file__).parent.parent.parent / "grid_data"
PROCESSED_DIR = GRID_DATA_DIR / "processed"
OUTPUT_DIR = Path(__file__).parent.parent / "app" / "data"

# Map bounds from GRID data (used for normalization)
MAP_BOUNDS = {
    'ascent': {'min': {'x': -6000, 'y': -7000}, 'max': {'x': 9000, 'y': 10000}},
    'bind': {'min': {'x': -5000, 'y': -6000}, 'max': {'x': 8000, 'y': 9000}},
    'haven': {'min': {'x': -3700, 'y': -14100}, 'max': {'x': 6000, 'y': 2000}},
    'split': {'min': {'x': -5000, 'y': -6500}, 'max': {'x': 9000, 'y': 8500}},
    'icebox': {'min': {'x': -5000, 'y': -6000}, 'max': {'x': 8000, 'y': 9000}},
    'breeze': {'min': {'x': -6000, 'y': -8000}, 'max': {'x': 10000, 'y': 8000}},
    'fracture': {'min': {'x': -7000, 'y': -7500}, 'max': {'x': 8000, 'y': 8000}},
    'pearl': {'min': {'x': -5000, 'y': -6000}, 'max': {'x': 9000, 'y': 9000}},
    'lotus': {'min': {'x': -6000, 'y': -7000}, 'max': {'x': 9000, 'y': 8000}},
    'sunset': {'min': {'x': -5000, 'y': -6000}, 'max': {'x': 9000, 'y': 9000}},
    'abyss': {'min': {'x': -5500, 'y': -6500}, 'max': {'x': 8500, 'y': 8500}},
}

# Zone definitions for each map (normalized coordinates 0-1)
# Each zone has center and radius for proximity matching
MAP_ZONES = {
    'haven': {
        'a_site': {'center': (0.15, 0.25), 'radius': 0.1},
        'b_site': {'center': (0.5, 0.18), 'radius': 0.1},
        'c_site': {'center': (0.85, 0.25), 'radius': 0.1},
        'a_long': {'center': (0.08, 0.5), 'radius': 0.08},
        'a_short': {'center': (0.25, 0.4), 'radius': 0.08},
        'b_window': {'center': (0.45, 0.35), 'radius': 0.06},
        'c_long': {'center': (0.92, 0.5), 'radius': 0.08},
        'c_link': {'center': (0.75, 0.4), 'radius': 0.08},
        'garage': {'center': (0.58, 0.45), 'radius': 0.1},
        'mid': {'center': (0.5, 0.5), 'radius': 0.12},
        't_spawn': {'center': (0.5, 0.85), 'radius': 0.1},
        'ct_spawn': {'center': (0.5, 0.15), 'radius': 0.1},
    },
    'ascent': {
        'a_site': {'center': (0.25, 0.25), 'radius': 0.1},
        'b_site': {'center': (0.75, 0.25), 'radius': 0.1},
        'a_main': {'center': (0.15, 0.45), 'radius': 0.08},
        'a_tree': {'center': (0.3, 0.4), 'radius': 0.06},
        'b_main': {'center': (0.85, 0.45), 'radius': 0.08},
        'b_market': {'center': (0.7, 0.35), 'radius': 0.06},
        'mid': {'center': (0.5, 0.45), 'radius': 0.1},
        'mid_top': {'center': (0.5, 0.3), 'radius': 0.08},
        'catwalk': {'center': (0.55, 0.4), 'radius': 0.06},
        't_spawn': {'center': (0.5, 0.85), 'radius': 0.1},
        'ct_spawn': {'center': (0.5, 0.12), 'radius': 0.1},
    },
    'bind': {
        'a_site': {'center': (0.2, 0.3), 'radius': 0.1},
        'b_site': {'center': (0.8, 0.3), 'radius': 0.1},
        'a_short': {'center': (0.25, 0.5), 'radius': 0.08},
        'a_bath': {'center': (0.12, 0.45), 'radius': 0.08},
        'b_long': {'center': (0.85, 0.55), 'radius': 0.08},
        'b_garden': {'center': (0.7, 0.45), 'radius': 0.08},
        'hookah': {'center': (0.65, 0.35), 'radius': 0.06},
        't_spawn': {'center': (0.5, 0.85), 'radius': 0.1},
        'ct_spawn': {'center': (0.5, 0.15), 'radius': 0.1},
    },
    'split': {
        'a_site': {'center': (0.2, 0.2), 'radius': 0.1},
        'b_site': {'center': (0.8, 0.2), 'radius': 0.1},
        'a_main': {'center': (0.15, 0.45), 'radius': 0.08},
        'a_ramp': {'center': (0.25, 0.35), 'radius': 0.06},
        'b_main': {'center': (0.85, 0.45), 'radius': 0.08},
        'b_heaven': {'center': (0.75, 0.25), 'radius': 0.06},
        'mid': {'center': (0.5, 0.5), 'radius': 0.1},
        'vents': {'center': (0.55, 0.35), 'radius': 0.08},
        'mail': {'center': (0.45, 0.4), 'radius': 0.06},
        't_spawn': {'center': (0.5, 0.9), 'radius': 0.1},
        'ct_spawn': {'center': (0.5, 0.1), 'radius': 0.1},
    },
    'lotus': {
        'a_site': {'center': (0.18, 0.28), 'radius': 0.1},
        'b_site': {'center': (0.5, 0.2), 'radius': 0.1},
        'c_site': {'center': (0.82, 0.28), 'radius': 0.1},
        'a_main': {'center': (0.12, 0.5), 'radius': 0.08},
        'a_rubble': {'center': (0.25, 0.4), 'radius': 0.06},
        'b_upper': {'center': (0.45, 0.35), 'radius': 0.08},
        'b_lower': {'center': (0.55, 0.35), 'radius': 0.08},
        'c_main': {'center': (0.88, 0.5), 'radius': 0.08},
        'c_mound': {'center': (0.75, 0.4), 'radius': 0.06},
        't_spawn': {'center': (0.5, 0.85), 'radius': 0.1},
        'ct_spawn': {'center': (0.5, 0.15), 'radius': 0.1},
    },
}

# Default zones for maps without specific definitions
DEFAULT_ZONES = {
    'a_site': {'center': (0.25, 0.25), 'radius': 0.1},
    'b_site': {'center': (0.75, 0.25), 'radius': 0.1},
    'mid': {'center': (0.5, 0.5), 'radius': 0.15},
    't_spawn': {'center': (0.5, 0.85), 'radius': 0.1},
    'ct_spawn': {'center': (0.5, 0.15), 'radius': 0.1},
}

# Role classification
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

# Ability to agent mapping for agent inference
ABILITY_TO_AGENT = {
    "nanoswarm": "killjoy", "alarmbot": "killjoy", "turret": "killjoy", "lockdown": "killjoy",
    "trapwire": "cypher", "cybercage": "cypher", "spycam": "cypher", "neuraltheft": "cypher",
    "darkcover": "omen", "shroudedstep": "omen", "paranoia": "omen", "fromtheshadows": "omen",
    "cloudburst": "jett", "updraft": "jett", "tailwind": "jett", "bladestorm": "jett",
    "blastpack": "raze", "boombot": "raze", "paintshells": "raze", "showstopper": "raze",
    "reconbolt": "sova", "shockbolt": "sova", "owldrone": "sova", "hunter'sfury": "sova",
    "skysmoke": "brimstone", "stimbeacon": "brimstone", "incendiary": "brimstone", "orbitalstrike": "brimstone",
    "toxicscreen": "viper", "poisoncloud": "viper", "snakebite": "viper", "viper'spit": "viper",
    "faultline": "breach", "flashpoint": "breach", "aftershock": "breach", "rollingthunder": "breach",
    "barrierorb": "sage", "sloworb": "sage", "healingorb": "sage", "resurrection": "sage",
    "headhunter": "chamber", "trademark": "chamber", "rendezvous": "chamber", "tourdeforce": "chamber",
    "guidinglight": "skye", "trailblazer": "skye", "regrowth": "skye", "seekers": "skye",
    "flashdrive": "kayo", "zeropoint": "kayo", "fragbolt": "kayo", "nullcmd": "kayo",
    "prowler": "fade", "seize": "fade", "haunt": "fade", "nightfall": "fade",
    "starspower": "astra", "nebula": "astra", "novapulse": "astra", "cosmicDivide": "astra",
    "highgear": "neon", "relaybolt": "neon", "fastelane": "neon", "overdrive": "neon",
    "gatecrash": "yoru", "blindside": "yoru", "fakeout": "yoru", "dimensionaldrift": "yoru",
    "leer": "reyna", "devour": "reyna", "dismiss": "reyna", "empress": "reyna",
    "dizzy": "gekko", "mosh": "gekko", "wingman": "gekko", "thrash": "gekko",
    "cascade": "harbor", "cove": "harbor", "hightide": "harbor", "reckoning": "harbor",
    "meddle": "clove", "ruse": "clove", "pickmeup": "clove", "notdeadyet": "clove",
    "undercut": "iso", "contingency": "iso", "doubleTap": "iso", "killcontract": "iso",
    "gravnet": "deadlock", "sonicsensor": "deadlock", "barrierMesh": "deadlock", "annihilation": "deadlock",
    "razorvine": "vyse", "shear": "vyse", "arcrose": "vyse", "steelsurge": "vyse",
    "blaze": "phoenix", "curveball": "phoenix", "hotwands": "phoenix", "runItBack": "phoenix",
    "guidedsalvo": "kayo",
    "nebuladissipate": "omen",
    "stars": "astra",
}


@dataclass
class PositionSample:
    """A single position sample with context."""
    x: float
    y: float
    normalized_x: float
    normalized_y: float
    zone: str
    map_name: str
    player_id: str
    player_name: str
    team: str
    agent: str
    role: str
    side: str  # 'attack' or 'defense'
    round_phase: str  # 'early', 'mid', 'late'
    event_type: str


@dataclass
class ZoneStats:
    """Statistics for a specific zone."""
    zone_name: str
    total_samples: int = 0
    kill_samples: int = 0
    death_samples: int = 0
    ability_samples: int = 0
    by_role: Dict[str, int] = field(default_factory=dict)
    by_side: Dict[str, int] = field(default_factory=dict)
    by_phase: Dict[str, int] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


@dataclass
class TransitionStats:
    """Zone transition statistics."""
    from_zone: str
    to_zone: str
    count: int
    probability: float

    def to_dict(self):
        return asdict(self)


def normalize_position(x: float, y: float, map_name: str) -> Tuple[float, float]:
    """Normalize raw GRID coordinates to 0-1 range."""
    bounds = MAP_BOUNDS.get(map_name.lower(), MAP_BOUNDS['ascent'])

    min_x = bounds['min']['x']
    max_x = bounds['max']['x']
    min_y = bounds['min']['y']
    max_y = bounds['max']['y']

    norm_x = (x - min_x) / (max_x - min_x) if max_x != min_x else 0.5
    norm_y = (y - min_y) / (max_y - min_y) if max_y != min_y else 0.5

    # Clamp to 0-1
    norm_x = max(0, min(1, norm_x))
    norm_y = max(0, min(1, norm_y))

    return norm_x, norm_y


def get_zone(norm_x: float, norm_y: float, map_name: str) -> str:
    """Determine which zone a normalized position belongs to."""
    zones = MAP_ZONES.get(map_name.lower(), DEFAULT_ZONES)

    closest_zone = 'unknown'
    min_dist = float('inf')

    for zone_name, zone_data in zones.items():
        cx, cy = zone_data['center']
        radius = zone_data['radius']

        dist = math.sqrt((norm_x - cx)**2 + (norm_y - cy)**2)

        # Check if within zone radius
        if dist <= radius and dist < min_dist:
            min_dist = dist
            closest_zone = zone_name

    # If no zone matched, find nearest
    if closest_zone == 'unknown':
        for zone_name, zone_data in zones.items():
            cx, cy = zone_data['center']
            dist = math.sqrt((norm_x - cx)**2 + (norm_y - cy)**2)
            if dist < min_dist:
                min_dist = dist
                closest_zone = zone_name

    return closest_zone


def determine_round_phase(round_time_ms: int) -> str:
    """Determine round phase based on time elapsed."""
    if round_time_ms < 15000:  # First 15 seconds
        return 'early'
    elif round_time_ms < 60000:  # 15-60 seconds
        return 'mid'
    else:  # After 60 seconds
        return 'late'


def extract_positions_from_file(filepath: Path) -> List[PositionSample]:
    """Extract position samples from a JSONL file."""
    samples = []
    player_agents = {}  # Track agent per player per round
    team_names = {}  # Map team ID to team name
    current_map = None
    current_map_bounds = None
    current_round = 0
    round_start_time = 0

    print(f"  Processing {filepath.name}...")

    with open(filepath, 'r') as f:
        for line in f:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Track map and team names from various sources
            for e in event.get('events', []):
                # Check target.state.map for game events
                target_state = e.get('target', {}).get('state', {})
                if target_state.get('map', {}).get('name'):
                    current_map = target_state['map']['name'].lower()
                    current_map_bounds = target_state['map'].get('bounds')

                # Check seriesState.games for map info
                for game in e.get('seriesState', {}).get('games', []):
                    if game.get('map', {}).get('name'):
                        current_map = game['map']['name'].lower()
                        current_map_bounds = game['map'].get('bounds')
                        break

                # Track team names from actor/target in team events
                actor = e.get('actor', {})
                if actor.get('type') == 'team':
                    actor_state = actor.get('state', {})
                    team_id = actor_state.get('id')
                    team_name = actor_state.get('name')
                    if team_id and team_name:
                        team_names[str(team_id)] = team_name

            # Track rounds and process events
            for e in event.get('events', []):
                if e.get('type') == 'round':
                    current_round = e.get('roundNumber', current_round)
                    round_start_time = event.get('timestamp', 0)
                    player_agents = {}  # Reset agent tracking per round

                # Track agent from ability usage
                if e.get('type') == 'player-used-ability':
                    actor = e.get('actor', {})
                    player_id = str(actor.get('id', ''))
                    ability = e.get('ability', {}).get('id', '').lower().replace(' ', '').replace('-', '')

                    if ability in ABILITY_TO_AGENT and player_id:
                        player_agents[player_id] = ABILITY_TO_AGENT[ability]

                # Extract positions from kill events
                if e.get('type') == 'player-killed-player':
                    actor = e.get('actor', {})
                    target = e.get('target', {})

                    # Position is at actor.state.game.position
                    actor_state = actor.get('state', {})
                    actor_game = actor_state.get('game', {})
                    target_state = target.get('state', {})
                    target_game = target_state.get('game', {})

                    actor_pos = actor_game.get('position')
                    target_pos = target_game.get('position')

                    event_time = event.get('timestamp', 0)
                    round_time = event_time - round_start_time if round_start_time > 0 else 0
                    phase = determine_round_phase(round_time)

                    # Actor (killer) position
                    if actor_pos and actor_pos.get('x') is not None and current_map:
                        x, y = actor_pos['x'], actor_pos['y']
                        norm_x, norm_y = normalize_position(x, y, current_map)
                        zone = get_zone(norm_x, norm_y, current_map)

                        player_id = str(actor.get('id', ''))
                        agent = player_agents.get(player_id, 'unknown')
                        role = AGENT_TO_ROLE.get(agent, 'unknown')
                        team_id = str(actor_state.get('teamId', ''))
                        team = team_names.get(team_id, 'Unknown')
                        side = actor_state.get('side', 'attacker')
                        # Normalize side naming
                        side = 'attack' if side in ('attacker', 'attack') else 'defense'

                        samples.append(PositionSample(
                            x=x, y=y,
                            normalized_x=norm_x, normalized_y=norm_y,
                            zone=zone, map_name=current_map,
                            player_id=player_id,
                            player_name=actor_state.get('name', 'Unknown'),
                            team=team,
                            agent=agent, role=role,
                            side=side, round_phase=phase,
                            event_type='kill'
                        ))

                    # Target (victim) position
                    if target_pos and target_pos.get('x') is not None and current_map:
                        x, y = target_pos['x'], target_pos['y']
                        norm_x, norm_y = normalize_position(x, y, current_map)
                        zone = get_zone(norm_x, norm_y, current_map)

                        player_id = str(target.get('id', ''))
                        agent = player_agents.get(player_id, 'unknown')
                        role = AGENT_TO_ROLE.get(agent, 'unknown')
                        target_team_id = str(target_state.get('teamId', ''))
                        target_team = team_names.get(target_team_id, 'Unknown')
                        target_side = target_state.get('side', 'defender')
                        target_side = 'attack' if target_side in ('attacker', 'attack') else 'defense'

                        samples.append(PositionSample(
                            x=x, y=y,
                            normalized_x=norm_x, normalized_y=norm_y,
                            zone=zone, map_name=current_map,
                            player_id=player_id,
                            player_name=target_state.get('name', 'Unknown'),
                            team=target_team,
                            agent=agent, role=role,
                            side=target_side, round_phase=phase,
                            event_type='death'
                        ))

                # Extract positions from ability usage
                if e.get('type') == 'player-used-ability':
                    ability_actor = e.get('actor', {})
                    ability_state = ability_actor.get('state', {})
                    ability_game = ability_state.get('game', {})
                    pos = ability_game.get('position')

                    if pos and pos.get('x') is not None and current_map:
                        x, y = pos['x'], pos['y']
                        norm_x, norm_y = normalize_position(x, y, current_map)
                        zone = get_zone(norm_x, norm_y, current_map)

                        player_id = str(ability_actor.get('id', ''))
                        agent = player_agents.get(player_id, 'unknown')
                        role = AGENT_TO_ROLE.get(agent, 'unknown')
                        ability_team_id = str(ability_state.get('teamId', ''))
                        ability_team = team_names.get(ability_team_id, 'Unknown')
                        ability_side = ability_state.get('side', 'attacker')
                        ability_side = 'attack' if ability_side in ('attacker', 'attack') else 'defense'

                        event_time = event.get('timestamp', 0)
                        round_time = event_time - round_start_time if round_start_time > 0 else 0
                        phase = determine_round_phase(round_time)

                        samples.append(PositionSample(
                            x=x, y=y,
                            normalized_x=norm_x, normalized_y=norm_y,
                            zone=zone, map_name=current_map,
                            player_id=player_id,
                            player_name=ability_state.get('name', 'Unknown'),
                            team=ability_team,
                            agent=agent, role=role,
                            side=ability_side, round_phase=phase,
                            event_type='ability'
                        ))

    return samples


def calculate_zone_stats(samples: List[PositionSample]) -> Dict[str, Dict[str, ZoneStats]]:
    """Calculate zone statistics per map."""
    stats_by_map = defaultdict(lambda: defaultdict(lambda: ZoneStats(zone_name='')))

    for sample in samples:
        map_stats = stats_by_map[sample.map_name]
        zone_stats = map_stats[sample.zone]
        zone_stats.zone_name = sample.zone

        zone_stats.total_samples += 1

        if sample.event_type == 'kill':
            zone_stats.kill_samples += 1
        elif sample.event_type == 'death':
            zone_stats.death_samples += 1
        elif sample.event_type == 'ability':
            zone_stats.ability_samples += 1

        # By role
        if sample.role not in zone_stats.by_role:
            zone_stats.by_role[sample.role] = 0
        zone_stats.by_role[sample.role] += 1

        # By side
        if sample.side not in zone_stats.by_side:
            zone_stats.by_side[sample.side] = 0
        zone_stats.by_side[sample.side] += 1

        # By phase
        if sample.round_phase not in zone_stats.by_phase:
            zone_stats.by_phase[sample.round_phase] = 0
        zone_stats.by_phase[sample.round_phase] += 1

    return stats_by_map


def calculate_transitions(samples: List[PositionSample]) -> Dict[str, List[TransitionStats]]:
    """Calculate zone transition probabilities per map."""
    # Group samples by player and round (approximated by sequential order)
    player_sequences = defaultdict(list)

    for sample in samples:
        key = (sample.map_name, sample.player_id)
        player_sequences[key].append(sample)

    # Count transitions
    transitions_by_map = defaultdict(Counter)
    from_zone_counts = defaultdict(Counter)

    for (map_name, player_id), sequence in player_sequences.items():
        for i in range(len(sequence) - 1):
            from_zone = sequence[i].zone
            to_zone = sequence[i + 1].zone

            if from_zone != to_zone:  # Only count actual transitions
                transitions_by_map[map_name][(from_zone, to_zone)] += 1
                from_zone_counts[map_name][from_zone] += 1

    # Calculate probabilities
    result = {}
    for map_name, transitions in transitions_by_map.items():
        trans_list = []
        for (from_zone, to_zone), count in transitions.most_common(50):  # Top 50 transitions
            total_from = from_zone_counts[map_name][from_zone]
            prob = count / total_from if total_from > 0 else 0

            trans_list.append(TransitionStats(
                from_zone=from_zone,
                to_zone=to_zone,
                count=count,
                probability=round(prob, 3)
            ))
        result[map_name] = trans_list

    return result


def calculate_role_heatmaps(samples: List[PositionSample]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Calculate position heatmaps per role per map."""
    role_zones = defaultdict(lambda: defaultdict(Counter))

    for sample in samples:
        if sample.role != 'unknown':
            role_zones[sample.map_name][sample.role][sample.zone] += 1

    # Normalize to probabilities
    result = {}
    for map_name, roles in role_zones.items():
        result[map_name] = {}
        for role, zones in roles.items():
            total = sum(zones.values())
            result[map_name][role] = {zone: round(count/total, 3) for zone, count in zones.most_common(10)}

    return result


def calculate_phase_patterns(samples: List[PositionSample]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Calculate zone preferences by round phase per map."""
    phase_zones = defaultdict(lambda: defaultdict(Counter))

    for sample in samples:
        phase_zones[sample.map_name][sample.round_phase][sample.zone] += 1

    # Normalize to probabilities
    result = {}
    for map_name, phases in phase_zones.items():
        result[map_name] = {}
        for phase, zones in phases.items():
            total = sum(zones.values())
            result[map_name][phase] = {zone: round(count/total, 3) for zone, count in zones.most_common(10)}

    return result


def calculate_c9_patterns(samples: List[PositionSample]) -> Dict[str, any]:
    """Calculate C9-specific patterns."""
    c9_samples = [s for s in samples if s.team == 'Cloud9']

    # Player zone preferences
    player_zones = defaultdict(Counter)
    for sample in c9_samples:
        player_zones[sample.player_name][sample.zone] += 1

    player_prefs = {}
    for player, zones in player_zones.items():
        total = sum(zones.values())
        player_prefs[player] = {zone: round(count/total, 3) for zone, count in zones.most_common(5)}

    # Attack vs defense positioning
    attack_zones = Counter()
    defense_zones = Counter()
    for sample in c9_samples:
        if sample.side == 'attack':
            attack_zones[sample.zone] += 1
        else:
            defense_zones[sample.zone] += 1

    atk_total = sum(attack_zones.values()) or 1
    def_total = sum(defense_zones.values()) or 1

    return {
        'player_zone_preferences': player_prefs,
        'attack_zones': {z: round(c/atk_total, 3) for z, c in attack_zones.most_common(10)},
        'defense_zones': {z: round(c/def_total, 3) for z, c in defense_zones.most_common(10)},
        'total_samples': len(c9_samples)
    }


def extract_movement_patterns():
    """Main extraction function."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Find all JSONL files
    jsonl_files = list(GRID_DATA_DIR.glob("events_*.jsonl"))
    print(f"Found {len(jsonl_files)} JSONL files")

    # Extract all positions
    all_samples = []
    for filepath in jsonl_files:
        samples = extract_positions_from_file(filepath)
        all_samples.extend(samples)
        print(f"    -> {len(samples)} position samples")

    print(f"\nTotal position samples: {len(all_samples)}")

    # Calculate statistics
    print("\nCalculating zone statistics...")
    zone_stats = calculate_zone_stats(all_samples)

    print("Calculating zone transitions...")
    transitions = calculate_transitions(all_samples)

    print("Calculating role heatmaps...")
    role_heatmaps = calculate_role_heatmaps(all_samples)

    print("Calculating phase patterns...")
    phase_patterns = calculate_phase_patterns(all_samples)

    print("Calculating C9 patterns...")
    c9_patterns = calculate_c9_patterns(all_samples)

    # Build output
    movement_data = {
        "generated": "2026-01-19",
        "total_samples": len(all_samples),
        "maps_analyzed": list(zone_stats.keys()),

        "zone_statistics": {
            map_name: {zone: stats.to_dict() for zone, stats in zones.items()}
            for map_name, zones in zone_stats.items()
        },

        "zone_transitions": {
            map_name: [t.to_dict() for t in trans]
            for map_name, trans in transitions.items()
        },

        "role_heatmaps": role_heatmaps,
        "phase_patterns": phase_patterns,
        "c9_patterns": c9_patterns,
    }

    # Save to processed dir
    with open(PROCESSED_DIR / "movement_patterns.json", 'w') as f:
        json.dump(movement_data, f, indent=2)

    # Also save to app data dir for simulation
    with open(OUTPUT_DIR / "movement_patterns.json", 'w') as f:
        json.dump(movement_data, f, indent=2)

    print(f"\nExported to:")
    print(f"  - {PROCESSED_DIR / 'movement_patterns.json'}")
    print(f"  - {OUTPUT_DIR / 'movement_patterns.json'}")

    # Print summary
    print("\n" + "="*60)
    print("MOVEMENT PATTERNS SUMMARY")
    print("="*60)

    print(f"\nMaps analyzed: {', '.join(movement_data['maps_analyzed'])}")
    print(f"Total position samples: {len(all_samples)}")

    print("\n### ZONE STATISTICS BY MAP ###")
    for map_name, zones in zone_stats.items():
        print(f"\n{map_name.upper()}:")
        sorted_zones = sorted(zones.items(), key=lambda x: x[1].total_samples, reverse=True)[:5]
        for zone_name, stats in sorted_zones:
            print(f"  {zone_name}: {stats.total_samples} samples (K:{stats.kill_samples} D:{stats.death_samples})")

    print("\n### TOP ZONE TRANSITIONS ###")
    for map_name, trans in list(transitions.items())[:3]:
        print(f"\n{map_name.upper()}:")
        for t in trans[:5]:
            print(f"  {t.from_zone} -> {t.to_zone}: {t.count} ({t.probability:.1%})")

    print("\n### ROLE HEATMAPS (sample) ###")
    if role_heatmaps:
        sample_map = list(role_heatmaps.keys())[0]
        print(f"\n{sample_map.upper()}:")
        for role, zones in role_heatmaps[sample_map].items():
            top_zones = list(zones.items())[:3]
            zones_str = ', '.join(f"{z}:{p:.0%}" for z, p in top_zones)
            print(f"  {role}: {zones_str}")

    print("\n### C9 PLAYER ZONE PREFERENCES ###")
    for player, zones in list(c9_patterns.get('player_zone_preferences', {}).items())[:5]:
        top_zones = list(zones.items())[:3]
        zones_str = ', '.join(f"{z}:{p:.0%}" for z, p in top_zones)
        print(f"  {player}: {zones_str}")


if __name__ == "__main__":
    extract_movement_patterns()
