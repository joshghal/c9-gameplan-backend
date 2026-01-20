#!/usr/bin/env python3
"""
Generate detailed opponent team and player profiles.

Creates profiles for all non-C9 teams with:
- Team-level statistics and play style metrics
- Individual player profiles with strengths/weaknesses
- Comparison metrics against C9

Usage:
    python scripts/generate_opponent_profiles.py
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict, field

GRID_DATA_DIR = Path(__file__).parent.parent.parent / "grid_data"
PROCESSED_DIR = GRID_DATA_DIR / "processed"
OUTPUT_DIR = Path(__file__).parent.parent / "app" / "data"

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


@dataclass
class PlayerProfile:
    """Individual player profile."""
    id: str
    name: str
    team: str

    # Core stats
    kills: int
    deaths: int
    kd_ratio: float
    headshot_rate: float
    first_kill_rate: float
    first_death_rate: float

    # Play style metrics (0-1 scale)
    aggression: float
    consistency: float
    clutch_potential: float

    # Role info
    primary_role: str
    role_distribution: Dict[str, float]

    # Weapon preferences
    primary_weapon: str
    weapon_preference: Dict[str, float]

    # Performance
    avg_kill_distance: float
    avg_loadout_value: float
    attack_kills: int
    defense_kills: int

    def to_dict(self):
        return asdict(self)


@dataclass
class TeamProfile:
    """Team-level profile."""
    id: str
    name: str

    # Roster
    players: List[str]
    player_count: int

    # Aggregate stats
    total_kills: int
    total_deaths: int
    avg_kd: float
    avg_headshot_rate: float
    avg_first_kill_rate: float

    # Team play style (0-1 scale)
    team_aggression: float  # How aggressive the team plays
    role_balance: float     # How balanced the role distribution is
    star_reliance: float    # How much they rely on top fraggers

    # Role composition
    role_distribution: Dict[str, int]

    # Preferred strategies (inferred)
    preferred_agents: List[str]
    preferred_weapons: List[str]

    # Performance metrics
    attack_preference: float  # % kills on attack
    defense_preference: float # % kills on defense

    def to_dict(self):
        return asdict(self)


def load_profiles() -> dict:
    """Load full profiles."""
    with open(PROCESSED_DIR / "full_profiles.json") as f:
        return json.load(f)


def calculate_role_distribution(agents: dict) -> Tuple[str, Dict[str, float]]:
    """Calculate role distribution from agent usage."""
    if not agents:
        return "duelist", {"duelist": 1.0}

    role_counts = defaultdict(int)
    total = sum(agents.values())

    for agent, count in agents.items():
        role = AGENT_TO_ROLE.get(agent.lower(), "duelist")
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


def calculate_aggression(player: dict) -> float:
    """Calculate aggression score."""
    fk = player.get('first_kills', 0)
    total_kills = max(1, player.get('kills', 1))
    fk_rate = fk / total_kills

    atk_kills = player.get('attack_kills', 0)
    atk_ratio = atk_kills / total_kills

    # Higher first kill rate + higher attack kills = more aggressive
    return min(1.0, (fk_rate * 1.5 + atk_ratio) / 2)


def calculate_consistency(player: dict) -> float:
    """Calculate consistency score based on K/D variance."""
    kd = player.get('kd_ratio', 1.0)
    # Higher K/D = more consistent, cap at 1.0
    return min(1.0, kd * 0.7)


def calculate_clutch_potential(player: dict) -> float:
    """Calculate clutch potential."""
    kd = player.get('kd_ratio', 1.0)
    hs_rate = player.get('headshot_rate', 0.2)
    # Good aim + positive K/D = clutch potential
    return min(1.0, (kd * 0.4 + hs_rate * 2))


def generate_player_profile(player: dict) -> PlayerProfile:
    """Generate a player profile from raw data."""
    total_kills = max(1, player.get('kills', 1))
    total_rounds = max(1, player.get('rounds_played', 1))

    fk = player.get('first_kills', 0)
    fd = player.get('first_deaths', 0)

    primary_role, role_dist = calculate_role_distribution(
        player.get('agents_played', {})
    )
    primary_weapon, weapon_pref = calculate_weapon_preference(
        player.get('weapon_kills', {})
    )

    return PlayerProfile(
        id=player.get('id', ''),
        name=player.get('name', 'Unknown'),
        team=player.get('team_name', 'Unknown'),
        kills=player.get('kills', 0),
        deaths=player.get('deaths', 0),
        kd_ratio=player.get('kd_ratio', 1.0),
        headshot_rate=player.get('headshot_rate', 0.0),
        first_kill_rate=round(fk / total_kills, 3) if total_kills > 0 else 0,
        first_death_rate=round(fd / player.get('deaths', 1), 3) if player.get('deaths', 0) > 0 else 0,
        aggression=round(calculate_aggression(player), 3),
        consistency=round(calculate_consistency(player), 3),
        clutch_potential=round(calculate_clutch_potential(player), 3),
        primary_role=primary_role,
        role_distribution={k: round(v, 3) for k, v in role_dist.items()},
        primary_weapon=primary_weapon,
        weapon_preference={k: round(v, 3) for k, v in list(weapon_pref.items())[:5]},
        avg_kill_distance=player.get('avg_kill_distance', 1700),
        avg_loadout_value=player.get('avg_loadout_value', 3000),
        attack_kills=player.get('attack_kills', 0),
        defense_kills=player.get('defense_kills', 0)
    )


def generate_team_profile(team_name: str, players: List[dict]) -> TeamProfile:
    """Generate a team profile from player data."""
    player_profiles = [generate_player_profile(p) for p in players]

    # Aggregate stats
    total_kills = sum(p.kills for p in player_profiles)
    total_deaths = sum(p.deaths for p in player_profiles)

    avg_kd = sum(p.kd_ratio for p in player_profiles) / len(player_profiles) if player_profiles else 1.0
    avg_hs = sum(p.headshot_rate for p in player_profiles) / len(player_profiles) if player_profiles else 0.2
    avg_fk = sum(p.first_kill_rate for p in player_profiles) / len(player_profiles) if player_profiles else 0.1

    # Team aggression
    team_aggression = sum(p.aggression for p in player_profiles) / len(player_profiles) if player_profiles else 0.5

    # Role distribution
    role_counts = defaultdict(int)
    for p in player_profiles:
        role_counts[p.primary_role] += 1

    # Role balance: 1.0 if all 4 roles present, lower otherwise
    role_balance = min(1.0, len(role_counts) / 4)

    # Star reliance: how much of the kills come from top 2 players
    if len(player_profiles) >= 2:
        sorted_players = sorted(player_profiles, key=lambda p: p.kills, reverse=True)
        top2_kills = sorted_players[0].kills + sorted_players[1].kills
        star_reliance = top2_kills / total_kills if total_kills > 0 else 0.5
    else:
        star_reliance = 1.0

    # Preferred agents (top 5 across team)
    agent_counts = defaultdict(int)
    for p in players:
        for agent, count in p.get('agents_played', {}).items():
            agent_counts[agent] += count
    preferred_agents = [a for a, _ in sorted(agent_counts.items(), key=lambda x: -x[1])[:5]]

    # Preferred weapons
    weapon_counts = defaultdict(int)
    for p in players:
        for weapon, count in p.get('weapon_kills', {}).items():
            weapon_counts[weapon] += count
    preferred_weapons = [w for w, _ in sorted(weapon_counts.items(), key=lambda x: -x[1])[:3]]

    # Attack/defense preference
    atk_kills = sum(p.attack_kills for p in player_profiles)
    def_kills = sum(p.defense_kills for p in player_profiles)
    total_sided_kills = atk_kills + def_kills
    attack_pref = atk_kills / total_sided_kills if total_sided_kills > 0 else 0.5
    defense_pref = def_kills / total_sided_kills if total_sided_kills > 0 else 0.5

    return TeamProfile(
        id=players[0].get('team_id', '') if players else '',
        name=team_name,
        players=[p.name for p in player_profiles],
        player_count=len(player_profiles),
        total_kills=total_kills,
        total_deaths=total_deaths,
        avg_kd=round(avg_kd, 2),
        avg_headshot_rate=round(avg_hs, 3),
        avg_first_kill_rate=round(avg_fk, 3),
        team_aggression=round(team_aggression, 3),
        role_balance=round(role_balance, 2),
        star_reliance=round(star_reliance, 3),
        role_distribution=dict(role_counts),
        preferred_agents=preferred_agents,
        preferred_weapons=preferred_weapons,
        attack_preference=round(attack_pref, 3),
        defense_preference=round(defense_pref, 3)
    )


def generate_opponent_profiles():
    """Generate all opponent profiles."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading profiles...")
    profiles = load_profiles()

    # Group players by team
    teams = defaultdict(list)
    for pid, player in profiles.get('players', {}).items():
        team = player.get('team_name', 'Unknown')
        player['id'] = pid
        teams[team].append(player)

    print(f"Found {len(teams)} teams")

    # Generate profiles
    c9_profile = None
    opponent_profiles = []
    player_profiles_by_team = {}

    for team_name, players in teams.items():
        team_profile = generate_team_profile(team_name, players)
        player_profs = [generate_player_profile(p) for p in players]

        if team_name == "Cloud9":
            c9_profile = team_profile
            player_profiles_by_team["Cloud9"] = player_profs
        else:
            opponent_profiles.append(team_profile)
            player_profiles_by_team[team_name] = player_profs

    # Sort opponents by total kills (most data first)
    opponent_profiles.sort(key=lambda t: t.total_kills, reverse=True)

    # Build output
    output = {
        "generated": "2026-01-19",
        "total_teams": len(teams),
        "total_players": len(profiles.get('players', {})),

        "cloud9": {
            "team": c9_profile.to_dict() if c9_profile else None,
            "players": [p.to_dict() for p in player_profiles_by_team.get("Cloud9", [])]
        },

        "opponents": {
            "teams": [t.to_dict() for t in opponent_profiles],
            "players_by_team": {
                team: [p.to_dict() for p in players]
                for team, players in player_profiles_by_team.items()
                if team != "Cloud9"
            }
        },

        # Quick lookup for simulation
        "player_lookup": {
            p['id']: p
            for team_players in player_profiles_by_team.values()
            for p in [pp.to_dict() for pp in team_players]
        }
    }

    # Save
    with open(PROCESSED_DIR / "opponent_profiles.json", 'w') as f:
        json.dump(output, f, indent=2)

    with open(OUTPUT_DIR / "opponent_profiles.json", 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nExported to:")
    print(f"  - {PROCESSED_DIR / 'opponent_profiles.json'}")
    print(f"  - {OUTPUT_DIR / 'opponent_profiles.json'}")

    # Print summary
    print("\n" + "="*60)
    print("OPPONENT PROFILES SUMMARY")
    print("="*60)

    print("\n### CLOUD9 ###")
    if c9_profile:
        print(f"Players: {', '.join(c9_profile.players)}")
        print(f"Avg K/D: {c9_profile.avg_kd}")
        print(f"Team Aggression: {c9_profile.team_aggression:.1%}")
        print(f"Role Balance: {c9_profile.role_balance:.1%}")
        print(f"Attack/Defense: {c9_profile.attack_preference:.1%}/{c9_profile.defense_preference:.1%}")

    print("\n### OPPONENT TEAMS ###")
    print(f"{'Team':<20} {'Players':<8} {'Kills':<8} {'Avg K/D':<8} {'Aggr':<8} {'Star%':<8}")
    print("-" * 60)
    for team in opponent_profiles[:10]:
        print(f"{team.name:<20} {team.player_count:<8} {team.total_kills:<8} "
              f"{team.avg_kd:<8.2f} {team.team_aggression:<8.1%} {team.star_reliance:<8.1%}")

    print("\n### TOP OPPONENT PLAYERS (by K/D) ###")
    all_opponent_players = [
        p for team, players in player_profiles_by_team.items()
        if team != "Cloud9"
        for p in players
    ]
    top_players = sorted(all_opponent_players, key=lambda p: p.kd_ratio, reverse=True)[:10]

    print(f"{'Player':<15} {'Team':<18} {'K/D':<6} {'HS%':<6} {'Role':<12} {'Aggr':<6}")
    print("-" * 70)
    for p in top_players:
        print(f"{p.name:<15} {p.team:<18} {p.kd_ratio:<6.2f} {p.headshot_rate:<6.1%} "
              f"{p.primary_role:<12} {p.aggression:<6.1%}")


if __name__ == "__main__":
    generate_opponent_profiles()
