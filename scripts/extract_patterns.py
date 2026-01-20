#!/usr/bin/env python3
"""
Comprehensive GRID Data Pattern Extractor

Extracts detailed patterns for simulation:
- Player combat stats (kills, deaths, headshots, weapons)
- Agent usage and ability patterns
- Position heatmaps and zone preferences
- Team compositions and strategies
- Round-by-round economy patterns

Usage:
    python scripts/extract_patterns.py
"""

import os
import sys
import json
import math
from pathlib import Path
from collections import defaultdict, Counter
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime

# Configuration
GRID_DATA_DIR = Path(__file__).parent.parent.parent / "grid_data"
OUTPUT_DIR = GRID_DATA_DIR / "processed"

# Agent ability mappings
ABILITY_TO_AGENT = {
    # Jett
    "cloudburst": "jett", "updraft": "jett", "tailwind": "jett", "bladestorm": "jett",
    # Reyna
    "leer": "reyna", "devour": "reyna", "dismiss": "reyna", "empress": "reyna",
    # Raze
    "boombot": "raze", "blastpack": "raze", "paintshells": "raze", "showstopper": "raze",
    # Phoenix
    "curveball": "phoenix", "hothand": "phoenix", "blaze": "phoenix", "runnitback": "phoenix",
    # Yoru
    "fakeout": "yoru", "blindside": "yoru", "gatecrash": "yoru", "dimensionaldrift": "yoru",
    # Neon
    "fastlane": "neon", "relay bolt": "neon", "highgear": "neon", "overdrive": "neon",
    # Iso
    "undercut": "iso", "doubleTap": "iso", "contingency": "iso", "killcontract": "iso",
    # Sage
    "sloworb": "sage", "barrierorb": "sage", "healingorb": "sage", "resurrection": "sage",
    # Skye
    "guidinglight": "skye", "trailblazer": "skye", "regrowth": "skye", "seekers": "skye",
    # KAY/O
    "frag/ment": "kayo", "flash/drive": "kayo", "zero/point": "kayo", "null/cmd": "kayo",
    "fragment": "kayo", "flashdrive": "kayo", "zeropoint": "kayo", "nullcmd": "kayo",
    # Breach
    "aftershock": "breach", "flashpoint": "breach", "faultline": "breach", "rollingthunder": "breach",
    # Gekko
    "dizzy": "gekko", "wingman": "gekko", "moshpit": "gekko", "thrash": "gekko",
    # Deadlock
    "gravnet": "deadlock", "sonicSensor": "deadlock", "barrierMesh": "deadlock", "annihilation": "deadlock",
    # Sova
    "owldrone": "sova", "shockbolt": "sova", "reconbolt": "sova", "huntersfury": "sova",
    # Fade
    "prowler": "fade", "seize": "fade", "haunt": "fade", "nightfall": "fade",
    # Cypher
    "trapwire": "cypher", "cybercage": "cypher", "spycam": "cypher", "neuraltheft": "cypher",
    # Killjoy
    "nanoswarm": "killjoy", "alarmbot": "killjoy", "turret": "killjoy", "lockdown": "killjoy",
    # Chamber
    "trademark": "chamber", "headhunter": "chamber", "rendezvous": "chamber", "tourdeforce": "chamber",
    # Clove
    "pick-me-up": "clove", "meddle": "clove", "ruse": "clove", "notdeadyet": "clove",
    # Omen
    "shroudedstep": "omen", "paranoia": "omen", "darkcover": "omen", "fromshadows": "omen",
    # Brimstone
    "stim beacon": "brimstone", "incendiary": "brimstone", "skysmoke": "brimstone", "orbitalstrike": "brimstone",
    "stimbeacon": "brimstone",
    # Astra
    "gravity well": "astra", "nova pulse": "astra", "nebula": "astra", "astralform": "astra", "cosmicDivide": "astra",
    # Viper
    "snakebite": "viper", "poisoncloud": "viper", "toxicscreen": "viper", "viperpit": "viper",
    # Harbor
    "cascade": "harbor", "cove": "harbor", "hightide": "harbor", "reckoning": "harbor",
    # Vyse
    "shear": "vyse", "arc rose": "vyse", "razorvine": "vyse", "steelsurge": "vyse",
}

# Team mappings
TEAM_NAMES = {
    "79": "Cloud9", "77": "Sentinels", "92": "NRG", "95": "G2", "94": "MIBR",
    "96": "Leviatán", "98": "FURIA", "99": "KRÜ", "93": "LOUD", "91": "100 Thieves",
    "97": "Evil Geniuses", "81": "FUT", "1079": "Shopify Rebellion", "1611": "2GAME",
    "281": "The Guard", "48457": "Rex Regum Qeon", "337": "EDward Gaming",
    "3412": "Fnatic", "53367": "Acend",
}

# Weapon info
WEAPON_DAMAGE = {
    "classic": {"body": 26, "head": 78, "type": "sidearm"},
    "shorty": {"body": 12, "head": 36, "type": "sidearm"},
    "frenzy": {"body": 26, "head": 78, "type": "sidearm"},
    "ghost": {"body": 30, "head": 105, "type": "sidearm"},
    "sheriff": {"body": 55, "head": 159, "type": "sidearm"},
    "stinger": {"body": 27, "head": 67, "type": "smg"},
    "spectre": {"body": 26, "head": 66, "type": "smg"},
    "bucky": {"body": 20, "head": 44, "type": "shotgun"},
    "judge": {"body": 17, "head": 34, "type": "shotgun"},
    "bulldog": {"body": 35, "head": 116, "type": "rifle"},
    "guardian": {"body": 65, "head": 195, "type": "rifle"},
    "phantom": {"body": 39, "head": 156, "type": "rifle"},
    "vandal": {"body": 40, "head": 160, "type": "rifle"},
    "marshal": {"body": 101, "head": 202, "type": "sniper"},
    "operator": {"body": 150, "head": 255, "type": "sniper"},
    "outlaw": {"body": 140, "head": 238, "type": "sniper"},
    "ares": {"body": 30, "head": 72, "type": "heavy"},
    "odin": {"body": 38, "head": 95, "type": "heavy"},
}


@dataclass
class PlayerStats:
    """Complete player statistics."""
    id: str
    name: str
    team_id: str
    team_name: str = ""

    # Core stats
    kills: int = 0
    deaths: int = 0
    assists: int = 0
    headshots: int = 0
    body_shots: int = 0
    leg_shots: int = 0

    # First blood
    first_kills: int = 0
    first_deaths: int = 0

    # Multi-kills
    double_kills: int = 0
    triple_kills: int = 0
    quadra_kills: int = 0
    ace_kills: int = 0

    # Side stats
    attack_kills: int = 0
    attack_deaths: int = 0
    defense_kills: int = 0
    defense_deaths: int = 0

    # Weapon kills
    weapon_kills: Dict[str, int] = field(default_factory=dict)

    # Agent stats
    agents_played: Dict[str, int] = field(default_factory=dict)
    abilities_used: Dict[str, int] = field(default_factory=dict)

    # Position data
    kill_positions: List[Tuple[float, float]] = field(default_factory=list)
    death_positions: List[Tuple[float, float]] = field(default_factory=list)
    kill_distances: List[float] = field(default_factory=list)

    # Economy
    avg_loadout_value: float = 0
    loadout_samples: int = 0

    # Round tracking
    rounds_played: int = 0
    matches_played: int = 0

    def add_loadout(self, value: int):
        n = self.loadout_samples
        self.avg_loadout_value = (self.avg_loadout_value * n + value) / (n + 1)
        self.loadout_samples += 1

    @property
    def kd_ratio(self) -> float:
        return self.kills / max(1, self.deaths)

    @property
    def headshot_rate(self) -> float:
        total = self.headshots + self.body_shots + self.leg_shots
        return self.headshots / max(1, total)

    @property
    def avg_kill_distance(self) -> float:
        if not self.kill_distances:
            return 0
        return sum(self.kill_distances) / len(self.kill_distances)

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "name": self.name,
            "team_id": self.team_id,
            "team_name": self.team_name,
            "kills": self.kills,
            "deaths": self.deaths,
            "assists": self.assists,
            "kd_ratio": round(self.kd_ratio, 2),
            "headshots": self.headshots,
            "headshot_rate": round(self.headshot_rate, 3),
            "first_kills": self.first_kills,
            "first_deaths": self.first_deaths,
            "attack_kills": self.attack_kills,
            "attack_deaths": self.attack_deaths,
            "defense_kills": self.defense_kills,
            "defense_deaths": self.defense_deaths,
            "weapon_kills": dict(sorted(self.weapon_kills.items(), key=lambda x: -x[1])[:10]),
            "agents_played": dict(sorted(self.agents_played.items(), key=lambda x: -x[1])),
            "top_abilities": dict(sorted(self.abilities_used.items(), key=lambda x: -x[1])[:10]),
            "avg_kill_distance": round(self.avg_kill_distance, 1),
            "avg_loadout_value": round(self.avg_loadout_value, 0),
            "rounds_played": self.rounds_played,
            "matches_played": self.matches_played,
        }
        return d


@dataclass
class TeamStats:
    """Team-level statistics."""
    id: str
    name: str

    matches: int = 0
    wins: int = 0
    rounds_played: int = 0
    rounds_won: int = 0

    attack_rounds: int = 0
    attack_wins: int = 0
    defense_rounds: int = 0
    defense_wins: int = 0

    first_bloods: int = 0
    first_deaths: int = 0

    plants: int = 0
    defuses: int = 0

    # Agent compositions
    compositions: List[Set[str]] = field(default_factory=list)

    # Map performance
    map_rounds: Dict[str, int] = field(default_factory=dict)
    map_wins: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "matches": self.matches,
            "wins": self.wins,
            "rounds_played": self.rounds_played,
            "rounds_won": self.rounds_won,
            "round_win_rate": round(self.rounds_won / max(1, self.rounds_played), 3),
            "attack_win_rate": round(self.attack_wins / max(1, self.attack_rounds), 3),
            "defense_win_rate": round(self.defense_wins / max(1, self.defense_rounds), 3),
            "first_blood_rate": round(self.first_bloods / max(1, self.rounds_played), 3),
            "plants": self.plants,
            "defuses": self.defuses,
            "map_performance": {
                m: {"rounds": self.map_rounds.get(m, 0), "wins": self.map_wins.get(m, 0)}
                for m in self.map_rounds
            }
        }


class PatternExtractor:
    """Extracts patterns from GRID data."""

    def __init__(self):
        self.players: Dict[str, PlayerStats] = {}
        self.teams: Dict[str, TeamStats] = {}

        # Current match state
        self.current_series_id: str = ""
        self.current_game_id: str = ""
        self.current_round: int = 0
        self.round_kills: Dict[str, int] = defaultdict(int)  # player_id -> kills this round

        # Tracking sets
        self.processed_series: Set[str] = set()

        # Patterns
        self.position_patterns: Dict[str, List[Tuple[float, float, str]]] = defaultdict(list)  # player -> [(x, y, context)]
        self.engagement_distances: List[float] = []
        self.weapon_engagement_distances: Dict[str, List[float]] = defaultdict(list)

    def get_player(self, player_id: str, name: str = "", team_id: str = "") -> PlayerStats:
        if player_id not in self.players:
            self.players[player_id] = PlayerStats(
                id=player_id,
                name=name,
                team_id=team_id,
                team_name=TEAM_NAMES.get(team_id, f"Team {team_id}")
            )
        p = self.players[player_id]
        if name and not p.name:
            p.name = name
        if team_id and not p.team_id:
            p.team_id = team_id
            p.team_name = TEAM_NAMES.get(team_id, f"Team {team_id}")
        return p

    def get_team(self, team_id: str) -> TeamStats:
        if team_id not in self.teams:
            self.teams[team_id] = TeamStats(
                id=team_id,
                name=TEAM_NAMES.get(team_id, f"Team {team_id}")
            )
        return self.teams[team_id]

    def process_kill(self, event: dict, envelope: dict):
        """Process kill event."""
        actor = event.get('actor', {})
        target = event.get('target', {})

        actor_state = actor.get('state', {})
        target_state = target.get('state', {})
        actor_delta = actor.get('stateDelta', {}).get('round', {})

        killer_id = actor_state.get('id', '')
        victim_id = target_state.get('id', '')

        if not killer_id or not victim_id:
            return

        killer = self.get_player(killer_id, actor_state.get('name', ''), actor_state.get('teamId', ''))
        victim = self.get_player(victim_id, target_state.get('name', ''), target_state.get('teamId', ''))

        # Basic stats
        killer.kills += 1
        victim.deaths += 1

        # First kill
        if actor_delta.get('firstKill'):
            killer.first_kills += 1
            victim.first_deaths += 1

            # Team first blood
            if killer.team_id:
                self.get_team(killer.team_id).first_bloods += 1
            if victim.team_id:
                self.get_team(victim.team_id).first_deaths += 1

        # Side stats
        side = actor_state.get('side', '')
        if side == 'attacker':
            killer.attack_kills += 1
        elif side == 'defender':
            killer.defense_kills += 1

        victim_side = target_state.get('side', '')
        if victim_side == 'attacker':
            victim.attack_deaths += 1
        elif victim_side == 'defender':
            victim.defense_deaths += 1

        # Weapon
        weapon_kills = actor_delta.get('weaponKills', {})
        if weapon_kills:
            weapon = max(weapon_kills.keys(), key=lambda k: weapon_kills[k])
            killer.weapon_kills[weapon] = killer.weapon_kills.get(weapon, 0) + 1

        # Positions
        killer_game = actor_state.get('game', {})
        victim_game = target_state.get('game', {})

        if 'position' in killer_game:
            pos = killer_game['position']
            killer.kill_positions.append((pos['x'], pos['y']))
            self.position_patterns[killer_id].append((pos['x'], pos['y'], 'kill'))

        if 'position' in victim_game:
            pos = victim_game['position']
            victim.death_positions.append((pos['x'], pos['y']))
            self.position_patterns[victim_id].append((pos['x'], pos['y'], 'death'))

        # Distance
        if 'position' in killer_game and 'position' in victim_game:
            dx = killer_game['position']['x'] - victim_game['position']['x']
            dy = killer_game['position']['y'] - victim_game['position']['y']
            distance = math.sqrt(dx*dx + dy*dy)
            killer.kill_distances.append(distance)
            self.engagement_distances.append(distance)

            if weapon_kills:
                weapon = max(weapon_kills.keys(), key=lambda k: weapon_kills[k])
                self.weapon_engagement_distances[weapon].append(distance)

        # Multi-kill tracking
        self.round_kills[killer_id] += 1

        # Loadout value
        if 'loadoutValue' in killer_game:
            killer.add_loadout(killer_game['loadoutValue'])

    def process_damage(self, event: dict, envelope: dict):
        """Process damage event for headshot tracking."""
        actor_delta = event.get('actor', {}).get('stateDelta', {})

        # Check for headshot in damage targets
        game_delta = actor_delta.get('game', {})
        damage_targets = game_delta.get('damageDealtTargets', [])

        actor_id = event.get('actor', {}).get('state', {}).get('id', '')
        if not actor_id:
            return

        player = self.get_player(actor_id)

        for target in damage_targets:
            target_name = target.get('target', {}).get('name', '').lower()
            if target_name == 'head':
                player.headshots += 1
            elif target_name == 'body':
                player.body_shots += 1
            elif target_name == 'leg':
                player.leg_shots += 1

    def process_ability(self, event: dict, envelope: dict):
        """Process ability usage for agent inference."""
        actor_state = event.get('actor', {}).get('state', {})
        actor_id = actor_state.get('id', '')

        if not actor_id:
            return

        player = self.get_player(actor_id, actor_state.get('name', ''), actor_state.get('teamId', ''))

        # Get ability name
        ability_id = ''
        action = event.get('action', {})
        if isinstance(action, dict):
            ability_info = action.get('ability', {})
            if isinstance(ability_info, dict):
                ability_id = ability_info.get('id', '')
            elif isinstance(ability_info, str):
                ability_id = ability_info

        # Also check target
        target = event.get('target', {})
        if target.get('type') == 'ability':
            ability_id = target.get('state', {}).get('id', '') or target.get('id', '')

        if not ability_id:
            # Try stateDelta
            abilities = actor_state.get('game', {}).get('abilities', [])
            for ab in abilities:
                if ab.get('ready') == False:  # Just used
                    ability_id = ab.get('id', '')
                    break

        if ability_id:
            ability_id = ability_id.lower().replace(' ', '').replace('/', '').replace('-', '')
            player.abilities_used[ability_id] = player.abilities_used.get(ability_id, 0) + 1

            # Infer agent
            agent = ABILITY_TO_AGENT.get(ability_id)
            if agent:
                player.agents_played[agent] = player.agents_played.get(agent, 0) + 1

        # Track position during ability use
        game_state = actor_state.get('game', {})
        if 'position' in game_state:
            pos = game_state['position']
            self.position_patterns[actor_id].append((pos['x'], pos['y'], f'ability:{ability_id}'))

    def process_round_end(self, event: dict, envelope: dict):
        """Process round end for multi-kill tracking."""
        # Check for multi-kills
        for player_id, kills in self.round_kills.items():
            if player_id in self.players:
                player = self.players[player_id]
                if kills >= 5:
                    player.ace_kills += 1
                elif kills >= 4:
                    player.quadra_kills += 1
                elif kills >= 3:
                    player.triple_kills += 1
                elif kills >= 2:
                    player.double_kills += 1

        # Reset round tracking
        self.round_kills.clear()

    def process_round_won(self, event: dict, envelope: dict):
        """Process round win."""
        actor_state = event.get('actor', {}).get('state', {})
        team_id = actor_state.get('id', '')

        if team_id:
            team = self.get_team(team_id)
            team.rounds_won += 1

    def process_plant(self, event: dict, envelope: dict):
        """Process spike plant."""
        actor_state = event.get('actor', {}).get('state', {})
        team_id = actor_state.get('teamId', '')

        if team_id:
            self.get_team(team_id).plants += 1

    def process_defuse(self, event: dict, envelope: dict):
        """Process spike defuse."""
        actor_state = event.get('actor', {}).get('state', {})
        team_id = actor_state.get('teamId', '')

        if team_id:
            self.get_team(team_id).defuses += 1

    def process_file(self, filepath: Path, verbose: bool = False):
        """Process a single JSONL file."""
        series_id = filepath.stem.split('_')[1]

        if series_id in self.processed_series:
            return 0

        events_processed = 0

        with open(filepath, 'r') as f:
            for line in f:
                try:
                    envelope = json.loads(line)
                    self.current_series_id = envelope.get('seriesId', '')

                    for event in envelope.get('events', []):
                        etype = event.get('type', '')

                        if etype == 'player-killed-player':
                            self.process_kill(event, envelope)
                            events_processed += 1
                        elif etype == 'player-damaged-player':
                            self.process_damage(event, envelope)
                            events_processed += 1
                        elif etype == 'player-used-ability':
                            self.process_ability(event, envelope)
                            events_processed += 1
                        elif etype == 'game-ended-round':
                            self.process_round_end(event, envelope)
                        elif etype == 'team-won-round':
                            self.process_round_won(event, envelope)
                            events_processed += 1
                        elif etype == 'player-completed-plantBomb':
                            self.process_plant(event, envelope)
                            events_processed += 1
                        elif etype == 'player-completed-defuseBomb':
                            self.process_defuse(event, envelope)
                            events_processed += 1

                except Exception as e:
                    if verbose:
                        print(f"Error: {e}")

        self.processed_series.add(series_id)

        # Update match counts
        for player in self.players.values():
            if player.team_id:
                player.matches_played = len(self.processed_series)

        return events_processed

    def process_all(self, verbose: bool = False):
        """Process all JSONL files."""
        files = sorted(GRID_DATA_DIR.glob("events_*_grid.jsonl"))
        print(f"Processing {len(files)} match files...")

        total = 0
        for i, f in enumerate(files):
            if verbose:
                print(f"[{i+1}/{len(files)}] {f.name}")
            events = self.process_file(f, verbose)
            total += events
            if verbose:
                print(f"  → {events} events")

        print(f"\nTotal events: {total}")
        print(f"Players: {len(self.players)}")
        print(f"Teams: {len(self.teams)}")

    def get_c9_players(self) -> List[PlayerStats]:
        """Get Cloud9 players sorted by kills."""
        c9 = [p for p in self.players.values() if p.team_id == "79"]
        return sorted(c9, key=lambda p: p.kills, reverse=True)

    def generate_report(self):
        """Generate analysis report."""
        print("\n" + "="*70)
        print("C9 TACTICAL VISION - PATTERN ANALYSIS REPORT")
        print("="*70)

        # C9 Players
        print("\n### CLOUD9 PLAYER PROFILES ###\n")
        c9 = self.get_c9_players()

        print(f"{'Player':<12} {'K':<6} {'D':<6} {'K/D':<6} {'HS%':<6} {'FK':<4} {'AvgDist':<8} {'TopWeapon':<10} {'Agent'}")
        print("-" * 85)

        for p in c9[:10]:
            top_weapon = max(p.weapon_kills.keys(), key=lambda k: p.weapon_kills[k]) if p.weapon_kills else "-"
            top_agent = max(p.agents_played.keys(), key=lambda k: p.agents_played[k]) if p.agents_played else "-"
            print(f"{p.name:<12} {p.kills:<6} {p.deaths:<6} {p.kd_ratio:<6.2f} "
                  f"{p.headshot_rate*100:<5.1f}% {p.first_kills:<4} {p.avg_kill_distance:<8.0f} "
                  f"{top_weapon:<10} {top_agent}")

        # Weapon analysis
        print("\n### WEAPON ENGAGEMENT DISTANCES ###\n")
        for weapon in ['vandal', 'phantom', 'operator', 'sheriff', 'spectre']:
            distances = self.weapon_engagement_distances.get(weapon, [])
            if distances:
                avg = sum(distances) / len(distances)
                print(f"  {weapon}: avg {avg:.0f} units ({len(distances)} kills)")

        # Team summary
        print("\n### TEAM SUMMARY ###\n")
        for tid in ['79', '77', '92', '95', '97']:  # C9, SEN, NRG, G2, EG
            if tid in self.teams:
                t = self.teams[tid]
                print(f"  {t.name}: {t.rounds_won} rounds won, "
                      f"FB rate: {t.first_bloods/max(1,t.rounds_played)*100:.1f}%, "
                      f"Plants: {t.plants}, Defuses: {t.defuses}")

    def export(self, output_dir: Path):
        """Export all data."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Export full profiles
        profiles = {
            "generated": datetime.now().isoformat(),
            "matches_processed": len(self.processed_series),
            "total_kills": sum(p.kills for p in self.players.values()),
            "players": {pid: p.to_dict() for pid, p in self.players.items()},
            "teams": {tid: t.to_dict() for tid, t in self.teams.items()},
        }

        with open(output_dir / "full_profiles.json", 'w') as f:
            json.dump(profiles, f, indent=2)

        # Export C9 detailed
        c9_data = {
            "team": "Cloud9",
            "team_id": "79",
            "generated": datetime.now().isoformat(),
            "players": [p.to_dict() for p in self.get_c9_players()],
            "team_stats": self.teams.get("79", TeamStats(id="79", name="Cloud9")).to_dict(),
        }

        with open(output_dir / "c9_profiles.json", 'w') as f:
            json.dump(c9_data, f, indent=2)

        # Export patterns for simulation
        patterns = {
            "engagement_distance": {
                "mean": sum(self.engagement_distances) / max(1, len(self.engagement_distances)),
                "samples": len(self.engagement_distances),
            },
            "weapon_distances": {
                w: {"mean": sum(d)/len(d), "samples": len(d)}
                for w, d in self.weapon_engagement_distances.items()
                if d
            },
        }

        with open(output_dir / "combat_patterns.json", 'w') as f:
            json.dump(patterns, f, indent=2)

        print(f"\nExported to {output_dir}/")


def main():
    extractor = PatternExtractor()
    extractor.process_all(verbose=True)
    extractor.generate_report()
    extractor.export(OUTPUT_DIR)


if __name__ == "__main__":
    main()
