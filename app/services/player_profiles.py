"""
Player Profile System

Combines GRID data with synthetic profiles to create realistic player
profiles for simulation. Integrates with the combat model system.
"""

import json
import os
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from pathlib import Path

from .grid_parser import GRIDMatchParser, parse_grid_data, PlayerProfile as GridProfile
from .combat_model import PlayerCombatProfile


logger = logging.getLogger(__name__)


# Base directory for GRID data
GRID_DATA_DIR = Path(__file__).parent.parent.parent.parent / "grid_data"


@dataclass
class ValorantPlayerProfile:
    """Comprehensive player profile for simulation use."""
    player_id: str
    player_name: str
    team: str
    role: str  # duelist, initiator, controller, sentinel, flex

    # Combat characteristics
    headshot_rate: float = 0.25  # Realistic range: 0.18-0.32
    reaction_time_ms: float = 180.0  # Pro range: 150-200ms
    crosshair_placement: float = 0.75  # 0-1 scale
    spray_control: float = 0.70
    counter_strafe_skill: float = 0.80

    # Playstyle
    aggression: float = 0.50  # 0 = passive, 1 = aggressive
    entry_frequency: float = 0.15  # How often they entry
    trade_awareness: float = 0.60
    utility_usage: float = 0.50

    # Consistency
    clutch_factor: float = 1.0  # >1 = better under pressure
    consistency: float = 0.75  # Performance variance

    # Preferred engagement
    avg_kill_distance: float = 15.0  # meters

    # Agent pool
    agents_played: Dict[str, int] = field(default_factory=dict)
    primary_agent: str = ""

    # Data source
    source: str = "synthetic"  # "grid", "synthetic", or "hybrid"
    matches_analyzed: int = 0

    def to_combat_profile(self) -> PlayerCombatProfile:
        """Convert to combat model profile."""
        return PlayerCombatProfile(
            base_reaction_ms=self.reaction_time_ms,
            reaction_variance=40 - (self.consistency * 20),
            crosshair_placement=self.crosshair_placement,
            headshot_rate=self.headshot_rate,
            spray_control=self.spray_control,
            first_shot_discipline=0.5 + (self.utility_usage * 0.3) + (1.0 - self.aggression) * 0.2,
            counter_strafe_skill=self.counter_strafe_skill,
            clutch_factor=self.clutch_factor,
            peek_aggression=self.aggression
        )


class PlayerProfileManager:
    """Manages player profiles from various sources."""

    # Known team rosters with role information
    TEAM_ROSTERS = {
        'cloud9': {
            'zellsis': {'role': 'flex', 'agents': ['Raze', 'Sova', 'KAY/O', 'Skye']},
            'Xeppaa': {'role': 'initiator', 'agents': ['Sova', 'Fade', 'KAY/O', 'Breach']},
            'jakee': {'role': 'duelist', 'agents': ['Jett', 'Raze', 'Neon']},
            'leaf': {'role': 'controller', 'agents': ['Omen', 'Viper', 'Harbor', 'Astra']},
            'vanity': {'role': 'igl', 'agents': ['Omen', 'Astra', 'Viper', 'Brimstone']},
            # Historical roster members
            'OXY': {'role': 'flex', 'agents': ['Raze', 'Sova', 'Skye']},
            'wippie': {'role': 'sentinel', 'agents': ['Killjoy', 'Cypher', 'Chamber']},
        },
        'sentinels': {
            'TenZ': {'role': 'duelist', 'agents': ['Jett', 'Raze', 'Reyna']},
            'zekken': {'role': 'duelist', 'agents': ['Raze', 'Neon', 'Jett']},
            'Sacy': {'role': 'initiator', 'agents': ['Sova', 'Fade', 'Breach']},
            'johnqt': {'role': 'igl', 'agents': ['Omen', 'Astra', 'Viper']},
            'pANcada': {'role': 'controller', 'agents': ['Astra', 'Omen', 'Harbor']},
        },
        'g2': {
            'icy': {'role': 'duelist', 'agents': ['Jett', 'Raze', 'Neon']},
            'valyn': {'role': 'igl', 'agents': ['Omen', 'Astra', 'Harbor']},
            'leaf': {'role': 'flex', 'agents': ['Raze', 'Sova', 'Skye']},
            'trent': {'role': 'initiator', 'agents': ['Sova', 'Fade', 'Breach']},
            'JonahP': {'role': 'sentinel', 'agents': ['Killjoy', 'Cypher', 'Sage']},
        },
        'nrg': {
            's0m': {'role': 'duelist', 'agents': ['Jett', 'Raze', 'ISO']},
            'FNS': {'role': 'igl', 'agents': ['Omen', 'Brimstone', 'Astra']},
            'crashies': {'role': 'initiator', 'agents': ['Sova', 'Fade', 'KAY/O']},
            'ardiis': {'role': 'flex', 'agents': ['Chamber', 'Jett', 'Raze']},
            'Victor': {'role': 'entry', 'agents': ['Raze', 'Neon', 'Phoenix']},
        },
        'mibr': {
            'frz': {'role': 'controller', 'agents': ['Omen', 'Astra', 'Viper']},
            'jzz': {'role': 'duelist', 'agents': ['Jett', 'Raze', 'Neon']},
            'rglmeister': {'role': 'sentinel', 'agents': ['Killjoy', 'Cypher']},
            'artziN': {'role': 'initiator', 'agents': ['Sova', 'Fade', 'Skye']},
            'mazin': {'role': 'flex', 'agents': ['Raze', 'Chamber', 'Jett']},
        }
    }

    # Role-based baseline stats
    ROLE_BASELINES = {
        'duelist': {
            'headshot_rate': 0.27,
            'reaction_time_ms': 165,
            'crosshair_placement': 0.80,
            'aggression': 0.78,
            'entry_frequency': 0.35,
        },
        'initiator': {
            'headshot_rate': 0.24,
            'reaction_time_ms': 175,
            'crosshair_placement': 0.75,
            'aggression': 0.55,
            'entry_frequency': 0.20,
        },
        'controller': {
            'headshot_rate': 0.22,
            'reaction_time_ms': 180,
            'crosshair_placement': 0.72,
            'aggression': 0.42,
            'entry_frequency': 0.12,
        },
        'sentinel': {
            'headshot_rate': 0.23,
            'reaction_time_ms': 178,
            'crosshair_placement': 0.73,
            'aggression': 0.38,
            'entry_frequency': 0.10,
        },
        'flex': {
            'headshot_rate': 0.25,
            'reaction_time_ms': 172,
            'crosshair_placement': 0.76,
            'aggression': 0.52,
            'entry_frequency': 0.22,
        },
        'igl': {
            'headshot_rate': 0.22,
            'reaction_time_ms': 180,
            'crosshair_placement': 0.72,
            'aggression': 0.45,
            'entry_frequency': 0.12,
        },
        'entry': {
            'headshot_rate': 0.26,
            'reaction_time_ms': 168,
            'crosshair_placement': 0.78,
            'aggression': 0.82,
            'entry_frequency': 0.38,
        }
    }

    def __init__(self):
        self.profiles: Dict[str, ValorantPlayerProfile] = {}
        self.grid_parsers: Dict[str, GRIDMatchParser] = {}

    def load_grid_data(self, team_name: str = None) -> int:
        """Load GRID data from the data directory.

        Returns number of matches loaded.
        """
        if not GRID_DATA_DIR.exists():
            logger.warning(f"GRID data directory not found: {GRID_DATA_DIR}")
            return 0

        matches_loaded = 0
        for filepath in GRID_DATA_DIR.glob("*.jsonl"):
            try:
                parser = parse_grid_data(str(filepath))
                self.grid_parsers[filepath.stem] = parser

                # Build profiles from this match
                grid_profiles = parser.build_profiles()
                for name, grid_profile in grid_profiles.items():
                    self._integrate_grid_profile(name, grid_profile, parser)

                matches_loaded += 1
            except Exception as e:
                logger.error(f"Error loading {filepath}: {e}")

        logger.info(f"Loaded {matches_loaded} GRID matches, {len(self.profiles)} player profiles")
        return matches_loaded

    def _integrate_grid_profile(
        self,
        player_name: str,
        grid_profile: GridProfile,
        parser: GRIDMatchParser
    ) -> None:
        """Integrate a GRID profile into our profile system."""
        name_lower = player_name.lower()

        # Find role from roster data
        role = 'flex'
        team = grid_profile.team
        agents = []

        for team_name, roster in self.TEAM_ROSTERS.items():
            for roster_name, info in roster.items():
                if roster_name.lower() == name_lower:
                    role = info.get('role', 'flex')
                    agents = info.get('agents', [])
                    team = team_name.replace('_', ' ').title()
                    break

        # Get role baseline
        baseline = self.ROLE_BASELINES.get(role, self.ROLE_BASELINES['flex'])

        # Blend GRID data with baselines
        # Headshot rate: use GRID data but sanity check (cap at 40%)
        hs_rate = min(0.40, grid_profile.avg_headshot_rate)
        if hs_rate < 0.15:  # Too low, use baseline
            hs_rate = baseline['headshot_rate']

        # Reaction time from entry patterns
        reaction_ms = grid_profile.estimated_reaction_ms
        if reaction_ms < 140 or reaction_ms > 250:
            reaction_ms = baseline['reaction_time_ms']

        # Create or update profile
        if name_lower in self.profiles:
            existing = self.profiles[name_lower]
            # Blend with existing data
            existing.headshot_rate = (existing.headshot_rate + hs_rate) / 2
            existing.matches_analyzed += 1
            existing.source = 'hybrid'
        else:
            profile = ValorantPlayerProfile(
                player_id=grid_profile.player_id,
                player_name=player_name,
                team=team,
                role=role,
                headshot_rate=hs_rate,
                reaction_time_ms=reaction_ms,
                crosshair_placement=min(0.92, 0.4 + hs_rate * 1.3),
                spray_control=baseline.get('crosshair_placement', 0.75),
                counter_strafe_skill=0.70 + (grid_profile.aggression * 0.15),
                aggression=grid_profile.aggression,
                entry_frequency=grid_profile.entry_frequency,
                trade_awareness=0.60,
                utility_usage=0.5 if role != 'duelist' else 0.35,
                clutch_factor=1.0,
                consistency=grid_profile.consistency if hasattr(grid_profile, 'consistency') else 0.75,
                avg_kill_distance=grid_profile.avg_kill_distance,
                agents_played={a: 1 for a in agents},
                primary_agent=agents[0] if agents else '',
                source='grid',
                matches_analyzed=1
            )
            self.profiles[name_lower] = profile

    def create_synthetic_profile(
        self,
        player_name: str,
        team: str,
        role: str,
        skill_tier: str = 'pro'
    ) -> ValorantPlayerProfile:
        """Create a synthetic profile for a player without GRID data."""
        baseline = self.ROLE_BASELINES.get(role, self.ROLE_BASELINES['flex'])

        # Skill tier adjustments
        tier_mods = {
            'elite': {'reaction': 0.92, 'accuracy': 1.10},
            'pro': {'reaction': 1.0, 'accuracy': 1.0},
            'semi_pro': {'reaction': 1.10, 'accuracy': 0.90},
            'ranked': {'reaction': 1.25, 'accuracy': 0.75},
        }
        mods = tier_mods.get(skill_tier, tier_mods['pro'])

        # Get agent pool from roster
        agents = []
        for team_name, roster in self.TEAM_ROSTERS.items():
            for roster_name, info in roster.items():
                if roster_name.lower() == player_name.lower():
                    agents = info.get('agents', [])
                    break

        profile = ValorantPlayerProfile(
            player_id=f'synthetic_{player_name.lower()}',
            player_name=player_name,
            team=team,
            role=role,
            headshot_rate=baseline['headshot_rate'] * mods['accuracy'],
            reaction_time_ms=baseline['reaction_time_ms'] * mods['reaction'],
            crosshair_placement=baseline['crosshair_placement'] * mods['accuracy'],
            spray_control=0.70 * mods['accuracy'],
            counter_strafe_skill=0.75,
            aggression=baseline['aggression'],
            entry_frequency=baseline['entry_frequency'],
            trade_awareness=0.60,
            utility_usage=0.5,
            clutch_factor=1.0,
            consistency=0.75,
            avg_kill_distance=15.0,
            agents_played={a: 1 for a in agents},
            primary_agent=agents[0] if agents else '',
            source='synthetic',
            matches_analyzed=0
        )

        return profile

    def get_team_profiles(self, team_name: str) -> Dict[str, ValorantPlayerProfile]:
        """Get all profiles for a team."""
        team_profiles = {}
        team_lower = team_name.lower().replace(' ', '')

        # First, check existing profiles
        for name, profile in self.profiles.items():
            if profile.team.lower().replace(' ', '') == team_lower:
                team_profiles[name] = profile

        # If we have roster data but no profiles, create synthetic ones
        if team_lower in self.TEAM_ROSTERS:
            roster = self.TEAM_ROSTERS[team_lower]
            for player_name, info in roster.items():
                if player_name.lower() not in team_profiles:
                    profile = self.create_synthetic_profile(
                        player_name=player_name,
                        team=team_name,
                        role=info.get('role', 'flex'),
                        skill_tier='pro'
                    )
                    team_profiles[player_name.lower()] = profile
                    self.profiles[player_name.lower()] = profile

        return team_profiles

    def get_c9_profiles(self) -> Dict[str, ValorantPlayerProfile]:
        """Get C9 player profiles."""
        return self.get_team_profiles('cloud9')

    def get_opponent_profiles(self, team_name: str) -> Dict[str, ValorantPlayerProfile]:
        """Get opponent team profiles."""
        return self.get_team_profiles(team_name)

    def export_profiles(self, filepath: str, team: str = None) -> None:
        """Export profiles to JSON."""
        profiles_to_export = self.profiles
        if team:
            profiles_to_export = self.get_team_profiles(team)

        data = {name: asdict(p) for name, p in profiles_to_export.items()}

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Exported {len(data)} profiles to {filepath}")

    def load_profiles(self, filepath: str) -> int:
        """Load profiles from JSON."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        for name, p_dict in data.items():
            self.profiles[name] = ValorantPlayerProfile(**p_dict)

        logger.info(f"Loaded {len(data)} profiles from {filepath}")
        return len(data)


# Singleton instance
profile_manager = PlayerProfileManager()


def get_c9_combat_profiles() -> Dict[str, PlayerCombatProfile]:
    """Get C9 player combat profiles for simulation."""
    profile_manager.load_grid_data()
    c9_profiles = profile_manager.get_c9_profiles()

    combat_profiles = {}
    for name, profile in c9_profiles.items():
        combat_profiles[name] = profile.to_combat_profile()

    return combat_profiles


def get_team_combat_profiles(team_name: str) -> Dict[str, PlayerCombatProfile]:
    """Get team combat profiles for simulation."""
    profiles = profile_manager.get_team_profiles(team_name)

    combat_profiles = {}
    for name, profile in profiles.items():
        combat_profiles[name] = profile.to_combat_profile()

    return combat_profiles


if __name__ == "__main__":
    # Test the profile manager
    print("Loading GRID data...")
    profile_manager.load_grid_data()

    print("\n" + "=" * 60)
    print("C9 PROFILES")
    print("=" * 60)

    c9 = profile_manager.get_c9_profiles()
    for name, profile in c9.items():
        print(f"\n{profile.player_name} ({profile.role}):")
        print(f"  Source: {profile.source}")
        print(f"  HS Rate: {profile.headshot_rate*100:.1f}%")
        print(f"  Reaction: {profile.reaction_time_ms:.0f}ms")
        print(f"  Aggression: {profile.aggression:.2f}")
        print(f"  Entry Rate: {profile.entry_frequency:.2f}")

        combat = profile.to_combat_profile()
        print(f"  Combat Profile:")
        print(f"    - Crosshair: {combat.crosshair_placement:.2f}")
        print(f"    - Spray Control: {combat.spray_control:.2f}")
