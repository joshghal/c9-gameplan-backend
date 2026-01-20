"""
Data Loader Service for extracted VCT match data.

Loads and provides access to:
- Player profiles with real stats
- Team profiles and strategies
- Behavioral patterns for AI
- Movement patterns and zone transitions
- Combat parameters

Usage:
    from app.services.data_loader import DataLoader

    # Load all data
    data = DataLoader()

    # Get player profile
    profile = data.get_player_profile("OXY")

    # Get behavioral parameters for a role
    behavior = data.get_role_behavior("duelist")

    # Get zone transition probabilities
    transitions = data.get_zone_transitions("haven")
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import random

# Data directory
DATA_DIR = Path(__file__).parent.parent / "data"


@dataclass
class PlayerProfile:
    """Player profile with extracted stats."""
    id: str
    name: str
    team: str

    # Combat stats
    kills: int
    deaths: int
    kd_ratio: float
    headshot_rate: float
    first_kill_rate: float

    # Behavior metrics
    aggression: float
    consistency: float
    clutch_potential: float

    # Role info
    primary_role: str

    # Weapon preferences
    primary_weapon: str
    weapon_preference: Dict[str, float]

    # Positioning
    avg_kill_distance: float
    attack_kills: int
    defense_kills: int


@dataclass
class TeamProfile:
    """Team profile with aggregate stats."""
    id: str
    name: str
    players: List[str]

    # Stats
    avg_kd: float
    avg_headshot_rate: float
    team_aggression: float
    star_reliance: float

    # Preferred playstyle
    preferred_agents: List[str]
    preferred_weapons: List[str]
    attack_preference: float


@dataclass
class RoleBehavior:
    """Role-specific behavior parameters."""
    role: str
    aggression_level: float
    entry_probability: float
    clutch_success_rate: float
    preferred_zones: List[str]
    weapon_preference: Dict[str, float]


@dataclass
class CombatParams:
    """Combat engagement parameters."""
    engagement_distance_mean: float
    engagement_distance_std: float
    headshot_base_rate: float
    first_kill_aggression: float


class DataLoader:
    """Loads and provides access to extracted VCT match data."""

    _instance = None

    def __new__(cls):
        """Singleton pattern for shared data access."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    def __init__(self):
        if not self._loaded:
            self._load_all_data()
            self._loaded = True

    def _load_all_data(self):
        """Load all data files."""
        self._simulation_profiles = self._load_json("simulation_profiles.json")
        self._movement_patterns = self._load_json("movement_patterns.json")
        self._opponent_profiles = self._load_json("opponent_profiles.json")
        self._behavioral_patterns = self._load_json("behavioral_patterns.json")

        # VCT-derived realism data (primary source)
        self._hold_angles = self._load_json("hold_angles.json")
        self._trade_patterns = self._load_json("trade_patterns.json")
        self._economy_patterns = self._load_json("economy_patterns.json")
        self._vct_zones = self._load_json("vct_zone_definitions.json")

        # Note: position_trajectories.json is large (51MB), load on demand
        self._trajectories = None

        # Build lookup indices
        self._build_indices()

    def _load_json(self, filename: str) -> dict:
        """Load a JSON file from the data directory."""
        filepath = DATA_DIR / filename
        if filepath.exists():
            with open(filepath) as f:
                return json.load(f)
        return {}

    def _build_indices(self):
        """Build lookup indices for quick access."""
        # Player lookup by name (case-insensitive)
        self._player_by_name = {}
        self._player_by_id = {}

        # From simulation profiles
        for category in ['cloud9', 'opponents']:
            players = self._simulation_profiles.get('player_behaviors', {}).get(category, [])
            for p in players:
                name_key = p.get('name', '').lower()
                self._player_by_name[name_key] = p
                self._player_by_id[p.get('id', '')] = p

        # From opponent profiles
        player_lookup = self._opponent_profiles.get('player_lookup', {})
        for pid, p in player_lookup.items():
            name_key = p.get('name', '').lower()
            self._player_by_name[name_key] = p
            self._player_by_id[pid] = p

        # Team lookup
        self._team_by_name = {}
        c9 = self._opponent_profiles.get('cloud9', {}).get('team')
        if c9:
            self._team_by_name['cloud9'] = c9

        for team in self._opponent_profiles.get('opponents', {}).get('teams', []):
            name_key = team.get('name', '').lower()
            self._team_by_name[name_key] = team

    def get_player_profile(self, name_or_id: str) -> Optional[PlayerProfile]:
        """Get a player profile by name or ID."""
        # Try by ID first
        p = self._player_by_id.get(str(name_or_id))
        if not p:
            # Try by name
            p = self._player_by_name.get(name_or_id.lower())

        if not p:
            return None

        return PlayerProfile(
            id=p.get('id', ''),
            name=p.get('name', 'Unknown'),
            team=p.get('team', 'Unknown'),
            kills=p.get('kills', 0),
            deaths=p.get('deaths', 0),
            kd_ratio=p.get('kd_ratio', 1.0),
            headshot_rate=p.get('headshot_rate', 0) or p.get('accuracy', 0.2),
            first_kill_rate=p.get('first_kill_rate', 0.1),
            aggression=p.get('aggression', 0.5),
            consistency=p.get('consistency', 0.5),
            clutch_potential=p.get('clutch_potential', p.get('clutch_factor', 0.5)),
            primary_role=p.get('primary_role', 'duelist'),
            primary_weapon=p.get('primary_weapon', 'vandal'),
            weapon_preference=p.get('weapon_preference', {}),
            avg_kill_distance=p.get('avg_engagement_distance', p.get('avg_kill_distance', 1700)),
            attack_kills=p.get('attack_kills', 0),
            defense_kills=p.get('defense_kills', 0)
        )

    def get_team_profile(self, team_name: str) -> Optional[TeamProfile]:
        """Get a team profile by name."""
        t = self._team_by_name.get(team_name.lower())
        if not t:
            return None

        return TeamProfile(
            id=t.get('id', ''),
            name=t.get('name', 'Unknown'),
            players=t.get('players', []),
            avg_kd=t.get('avg_kd', 1.0),
            avg_headshot_rate=t.get('avg_headshot_rate', 0.2),
            team_aggression=t.get('team_aggression', 0.5),
            star_reliance=t.get('star_reliance', 0.5),
            preferred_agents=t.get('preferred_agents', []),
            preferred_weapons=t.get('preferred_weapons', []),
            attack_preference=t.get('attack_preference', 0.5)
        )

    def get_role_behavior(self, role: str) -> RoleBehavior:
        """Get behavior parameters for a role."""
        behaviors = self._behavioral_patterns.get('role_behaviors', {})
        b = behaviors.get(role.lower(), {})

        return RoleBehavior(
            role=role,
            aggression_level=b.get('aggression_level', 0.5),
            entry_probability=b.get('entry_probability', 0.3),
            clutch_success_rate=b.get('clutch_success_rate', 0.3),
            preferred_zones=b.get('preferred_zones', ['mid', 'a_site', 'b_site']),
            weapon_preference=b.get('weapon_preference', {'vandal': 0.5, 'phantom': 0.5})
        )

    def get_combat_params(self, role: str) -> CombatParams:
        """Get combat parameters for a role."""
        combat = self._behavioral_patterns.get('combat_behaviors', {})
        c = combat.get(role.lower(), {})

        return CombatParams(
            engagement_distance_mean=c.get('engagement_distance_mean', 1750),
            engagement_distance_std=c.get('engagement_distance_std', 200),
            headshot_base_rate=c.get('headshot_base_rate', 0.2),
            first_kill_aggression=c.get('first_kill_aggression', 0.1)
        )

    def get_zone_transitions(self, map_name: str) -> List[Dict[str, Any]]:
        """Get zone transition probabilities for a map."""
        transitions = self._movement_patterns.get('zone_transitions', {})
        return transitions.get(map_name.lower(), [])

    def get_zone_stats(self, map_name: str) -> Dict[str, Dict]:
        """Get zone statistics for a map."""
        stats = self._movement_patterns.get('zone_statistics', {})
        return stats.get(map_name.lower(), {})

    def get_role_heatmap(self, map_name: str, role: str) -> Dict[str, float]:
        """Get zone heatmap for a role on a map."""
        heatmaps = self._movement_patterns.get('role_heatmaps', {})
        map_data = heatmaps.get(map_name.lower(), {})
        return map_data.get(role.lower(), {})

    def get_phase_behavior(self, phase: str) -> Dict[str, float]:
        """Get behavior parameters for a round phase."""
        phases = self._behavioral_patterns.get('phase_behaviors', {})
        return phases.get(phase.lower(), {})

    def get_economy_behavior(self) -> Dict[str, Any]:
        """Get economy decision parameters."""
        return self._behavioral_patterns.get('economy_behavior', {
            'full_buy_threshold': 3900,
            'force_buy_threshold': 2000,
            'eco_threshold': 1500
        })

    def get_agent_behavior(self, agent: str) -> Dict[str, Any]:
        """Get agent-specific ability patterns."""
        agents = self._behavioral_patterns.get('agent_behaviors', {})
        return agents.get(agent.lower(), {})

    def get_c9_player(self, name: str) -> Optional[PlayerProfile]:
        """Get a Cloud9 player profile by name."""
        # Try exact match first
        profile = self.get_player_profile(name)
        if profile and profile.team == 'Cloud9':
            return profile

        # Try partial match
        for key, p in self._player_by_name.items():
            if name.lower() in key and p.get('team') == 'Cloud9':
                return self.get_player_profile(p.get('name'))

        return None

    def get_all_c9_players(self) -> List[PlayerProfile]:
        """Get all Cloud9 player profiles."""
        players = []
        for name, p in self._player_by_name.items():
            if p.get('team') == 'Cloud9':
                profile = self.get_player_profile(p.get('name'))
                if profile:
                    players.append(profile)
        return players

    def get_player_tendencies(self, player_name_or_id: str) -> Tuple[float, float, float]:
        """Get player tendencies (aggression, clutch_factor, trade_awareness).

        Returns tuple of (aggression, clutch_factor, trade_awareness) for use
        in PlayerTendencies initialization.
        """
        profile = self.get_player_profile(player_name_or_id)

        if profile:
            return (
                profile.aggression,
                profile.clutch_potential,
                min(0.9, profile.kd_ratio * 0.5)  # Trade awareness from K/D
            )

        # Return random defaults if player not found
        return (
            random.uniform(0.3, 0.7),
            random.uniform(0.3, 0.7),
            random.uniform(0.4, 0.8)
        )

    def get_headshot_rate(self, player_name_or_id: str, role: str = None) -> float:
        """Get headshot rate for a player, falling back to role average."""
        profile = self.get_player_profile(player_name_or_id)

        if profile and profile.headshot_rate > 0:
            return profile.headshot_rate

        # Fall back to role average
        if role:
            combat = self.get_combat_params(role)
            return combat.headshot_base_rate

        # Default
        return 0.25

    # === ZONE DATA ACCESS (VCT-derived) ===

    def get_zone_bounds(self, map_name: str, zone_name: str) -> Optional[List[float]]:
        """Get zone coordinate bounds from VCT-derived zones.

        Returns [x_min, x_max, y_min, y_max] in game units.
        """
        zone = self.get_vct_zone(map_name, zone_name)
        if zone:
            return zone.get('bounds')
        return None

    def get_all_zone_definitions(self, map_name: str) -> Dict[str, Dict]:
        """Get all zone definitions for a map with coordinate bounds."""
        return self.get_all_vct_zones(map_name)

    def get_transition_timing(self, map_name: str, from_zone: str, to_zone: str) -> Dict[str, float]:
        """Get zone transition timing data from movement patterns.

        Returns dict with avg_time_ms, avg_speed, avg_distance.
        """
        transitions = self._movement_patterns.get('zone_transitions', {})
        map_trans = transitions.get(map_name.lower(), [])

        # Search for matching transition
        for trans in map_trans:
            if trans.get('from_zone') == from_zone and trans.get('to_zone') == to_zone:
                return {
                    'avg_time_ms': trans.get('avg_time_ms', 0),
                    'avg_speed': trans.get('avg_speed', 0),
                    'avg_distance': trans.get('avg_distance', 0),
                    'probability': trans.get('probability', 0),
                }

        return {'avg_time_ms': 0, 'avg_speed': 0, 'avg_distance': 0, 'probability': 0}

    def get_all_transitions_for_map(self, map_name: str) -> List[Dict]:
        """Get all zone transitions for a map with timing data."""
        transitions = self._movement_patterns.get('zone_transitions', {})
        return transitions.get(map_name.lower(), [])

    def get_unified_player(self, name: str) -> Optional[Dict]:
        """Get player data. Redirects to standard player lookup."""
        profile = self.get_player_profile(name)
        if profile:
            return {
                'name': profile.name,
                'team': profile.team,
                'kd_ratio': profile.kd_ratio,
                'headshot_rate': profile.headshot_rate,
                'aggression': profile.aggression,
            }
        return None

    def get_unified_combat_params(self) -> Dict[str, Any]:
        """Get combat parameters from VCT trade patterns."""
        ttk = self.get_time_to_kill()
        distance_dmg = self.get_distance_damage()

        return {
            'avg_ttk': ttk.get('avg_ttk', 1.5),
            'avg_hits_to_kill': ttk.get('avg_hits_to_kill', 4),
            'distance_damage': distance_dmg,
        }

    def get_unified_economy(self) -> Dict[str, Any]:
        """Get economy data from VCT economy patterns."""
        buy_rates = self.get_buy_win_rates()
        thresholds = self._economy_patterns.get('metadata', {}).get('thresholds', {})

        return {
            'buy_win_rates': buy_rates,
            'thresholds': thresholds,
        }

    # === NEW: VCT-DERIVED REALISM DATA ===

    def get_hold_angle(self, map_name: str, zone: str) -> Optional[Dict]:
        """Get hold angle data for a zone.

        Returns:
            {
                'mean_angle': float (radians),
                'mean_angle_degrees': float,
                'std_angle': float,
                'samples': int,
                'source': 'vct' or 'henrik_fallback'
            }
        """
        angles = self._hold_angles.get('angles_by_map', {})
        map_angles = angles.get(map_name.lower(), {})
        return map_angles.get(zone)

    def get_all_hold_angles(self, map_name: str) -> Dict[str, Dict]:
        """Get all hold angles for a map."""
        angles = self._hold_angles.get('angles_by_map', {})
        return angles.get(map_name.lower(), {})

    def get_trade_timing(self) -> Dict[str, Any]:
        """Get trade timing patterns.

        Returns:
            {
                'total_trades': int,
                'avg_trade_time': float (seconds),
                'trade_time_distribution': {...}
            }
        """
        return self._trade_patterns.get('trade_patterns', {})

    def get_time_to_kill(self) -> Dict[str, Any]:
        """Get time-to-kill patterns.

        Returns:
            {
                'avg_ttk': float (seconds),
                'avg_hits_to_kill': float,
                'ttk_distribution': {...}
            }
        """
        return self._trade_patterns.get('time_to_kill', {})

    def get_distance_damage(self) -> Dict[str, Dict]:
        """Get damage by distance patterns.

        Returns dict with buckets: close_0_500, short_500_1000, etc.
        Each has avg_damage and samples.
        """
        return self._trade_patterns.get('distance_damage', {})

    def get_map_engagement_stats(self, map_name: str) -> Optional[Dict]:
        """Get engagement statistics for a map.

        Returns:
            {
                'kills': int,
                'damage_events': int,
                'avg_engagement_distance': float,
                'avg_damage_per_event': float
            }
        """
        by_map = self._trade_patterns.get('by_map', {})
        return by_map.get(map_name.lower())

    def get_buy_win_rates(self) -> Dict[str, Dict]:
        """Get win rates by buy type.

        Returns dict with eco, force_buy, half_buy, full_buy keys.
        Each has win_rate, wins, total.
        """
        return self._economy_patterns.get('buy_patterns', {}).get('buy_win_rates', {})

    def get_economy_by_side(self) -> Dict[str, Dict]:
        """Get economy patterns by attacker/defender side."""
        return self._economy_patterns.get('by_side', {})

    def get_economy_by_map(self, map_name: str) -> Optional[Dict]:
        """Get economy patterns for a map.

        Returns:
            {
                'avg_attacker_loadout': float,
                'avg_defender_loadout': float,
                'buy_distribution': {...}
            }
        """
        by_map = self._economy_patterns.get('by_map', {})
        return by_map.get(map_name.lower())

    def get_vct_zone(self, map_name: str, zone: str) -> Optional[Dict]:
        """Get VCT-derived zone definition.

        Returns:
            {
                'bounds': [x_min, x_max, y_min, y_max],
                'position_count': int
            }
        """
        maps = self._vct_zones.get('maps', {})
        map_zones = maps.get(map_name.lower(), {})
        return map_zones.get(zone)

    def get_all_vct_zones(self, map_name: str) -> Dict[str, Dict]:
        """Get all VCT-derived zones for a map."""
        maps = self._vct_zones.get('maps', {})
        return maps.get(map_name.lower(), {})

    def get_zone_for_position(self, map_name: str, x: float, y: float) -> Optional[str]:
        """Find which VCT zone contains a position."""
        zones = self.get_all_vct_zones(map_name)
        for zone_name, zone_data in zones.items():
            bounds = zone_data.get('bounds', [])
            if len(bounds) == 4:
                x_min, x_max, y_min, y_max = bounds
                if x_min <= x <= x_max and y_min <= y <= y_max:
                    return zone_name
        return None

    def get_trajectories(self, map_name: str = None) -> Dict:
        """Get position trajectories (loads on first access - 51MB file).

        Args:
            map_name: Optional filter by map

        Returns:
            Full trajectories data or filtered by map
        """
        if self._trajectories is None:
            self._trajectories = self._load_json("position_trajectories.json")

        if map_name:
            by_map = self._trajectories.get('trajectories_by_map', {})
            return by_map.get(map_name.lower(), [])

        return self._trajectories

    def get_round_trajectory(self, map_name: str, round_num: int) -> Optional[Dict]:
        """Get trajectory data for a specific round.

        Returns:
            {
                'round_num': int,
                'game_id': str,
                'player_trajectories': {
                    'player_name': [
                        {'clock': int, 'x': float, 'y': float, 'alive': bool},
                        ...
                    ]
                }
            }
        """
        rounds = self.get_trajectories(map_name)
        for r in rounds:
            if r.get('round_num') == round_num:
                return r
        return None

    # === SPIKE CARRIER DATA (VCT-extracted) ===

    def get_spike_carrier_stats(self) -> Dict[str, Any]:
        """Get spike carrier patterns extracted from VCT data.

        Returns:
            {
                'metadata': {
                    'total_rounds': int,
                    'total_plants': int,
                    'plant_rate': float,
                    'carrier_death_rate': float,
                    'avg_plant_time_ms': int
                },
                'player_spike_stats': {...},
                'role_spike_stats': {...},
                'role_carrier_defaults': {...}
            }
        """
        if not hasattr(self, '_spike_carrier_patterns'):
            self._spike_carrier_patterns = self._load_json("spike_carrier_patterns.json")
        return self._spike_carrier_patterns

    def get_player_spike_stats(self, player_name: str) -> Optional[Dict]:
        """Get spike carrier stats for a specific player.

        Args:
            player_name: Player name (case-insensitive)

        Returns:
            Player spike stats dict or None if not found
        """
        patterns = self.get_spike_carrier_stats()
        player_stats = patterns.get('player_spike_stats', {})
        return player_stats.get(player_name) or player_stats.get(player_name.upper())

    def get_role_spike_defaults(self, role: str) -> Optional[Dict]:
        """Get spike carrier defaults for a role.

        Args:
            role: Role name (duelist, initiator, controller, sentinel)

        Returns:
            Role carrier defaults dict
        """
        patterns = self.get_spike_carrier_stats()
        defaults = patterns.get('role_carrier_defaults', {})
        return defaults.get(role.lower())

    def get_spike_validation_targets(self) -> Dict[str, Dict]:
        """Get VCT validation targets for spike carrier behavior.

        Returns:
            Dict of validation targets with 'target', 'tolerance', 'description'
        """
        patterns = self.get_spike_carrier_stats()
        return patterns.get('vct_validation_targets', {
            'plant_rate': {'target': 0.65, 'tolerance': 0.05},
            'carrier_death_rate': {'target': 0.12, 'tolerance': 0.04},
            'avg_plant_time_ms': {'target': 56000, 'tolerance': 8000},
        })


# Global singleton instance
_data_loader: Optional[DataLoader] = None


def get_data_loader() -> DataLoader:
    """Get the global DataLoader instance."""
    global _data_loader
    if _data_loader is None:
        _data_loader = DataLoader()
    return _data_loader
