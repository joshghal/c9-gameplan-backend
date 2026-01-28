"""
Opponent Realism Service

Provides opponent team movement, positioning, and coordination patterns
based on VCT data extraction from professional teams.

Supports per-TEAM profiles similar to C9's structure:
- teams/{team}/{team}_opening_setups.json - Position clusters per player/map/side
- teams/{team}/{team}_distance_preferences.json - Engagement distances per player
- teams/{team}/{team}_movement_models.json - KDE heatmaps per player/map/side

Available Teams:
- sentinels, nrg, loud, evil_geniuses, g2_esports, 100_thieves
- furia, mibr, leviatan_esports, kru_esports, 2game_esports

Implements:
- Opening Setups - Default positions by player/map/side from VCT data
- Distance Preferences - Preferred engagement distances per player
- Movement Model - KDE-based position sampling from VCT data
"""

import json
import random
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np

# Data directory
DATA_DIR = Path(__file__).parent.parent / "data"
TEAMS_DIR = DATA_DIR / "teams"

# Available teams (directory names)
AVAILABLE_TEAMS = {
    "sentinels", "nrg", "loud", "evil_geniuses", "g2_esports",
    "100_thieves", "furia", "mibr", "leviatan_esports",
    "kru_esports", "2game_esports", "cloud9"
}

# Default team to use if none specified
DEFAULT_TEAM = "sentinels"


@dataclass
class Position:
    """A 2D position with optional metadata."""
    x: float
    y: float
    confidence: float = 1.0


class OpponentRealismService:
    """
    Service for team-specific opponent simulation realism.

    Supports loading profiles for specific VCT teams:
    - sentinels, nrg, loud, evil_geniuses, g2_esports, etc.

    Provides methods to:
    1. Get opening positions for opponent team players
    2. Calculate optimal combat positions based on distance preferences
    3. Sample movement positions from player-specific KDE models
    """

    _instances = {}  # team_name -> instance

    def __new__(cls, team_name: str = None):
        """Factory pattern - one instance per team."""
        team = (team_name or DEFAULT_TEAM).lower()
        if team not in cls._instances:
            instance = super().__new__(cls)
            instance._team_name = team
            instance._loaded = False
            cls._instances[team] = instance
        return cls._instances[team]

    def __init__(self, team_name: str = None):
        if not self._loaded:
            self._load_team_data()
            self._loaded = True

    def _load_team_data(self):
        """Load team-specific profile data."""
        team_dir = TEAMS_DIR / self._team_name

        if not team_dir.exists():
            print(f"OpponentRealism: Team '{self._team_name}' not found, using default")
            team_dir = TEAMS_DIR / DEFAULT_TEAM
            self._team_name = DEFAULT_TEAM

        # Load team profile files
        self._opening_setups = self._load_json(team_dir / f"{self._team_name}_opening_setups.json")
        self._distance_prefs = self._load_json(team_dir / f"{self._team_name}_distance_preferences.json")
        self._movement_models = self._load_json(team_dir / f"{self._team_name}_movement_models.json")

        # Build lookup indices
        self._setups = self._opening_setups.get("setups", {})
        self._prefs = self._distance_prefs.get("preferences", {})
        self._models = self._movement_models.get("models", {})

        # Get list of team players
        self._players = list(self._setups.keys())
        self._roster = self._players.copy()

        print(f"OpponentRealism: Loaded {self._team_name.upper()} ({len(self._players)} players)")

    def _load_json(self, filepath: Path) -> dict:
        """Load a JSON file."""
        if filepath.exists():
            with open(filepath) as f:
                return json.load(f)
        return {}

    @property
    def team_name(self) -> str:
        """Get current team name."""
        return self._team_name

    @property
    def roster(self) -> List[str]:
        """Get team roster."""
        return self._roster.copy()

    @classmethod
    def get_available_teams(cls) -> List[str]:
        """Get list of available teams."""
        return sorted(AVAILABLE_TEAMS)

    # =========================================================================
    # PLAYER SELECTION
    # =========================================================================

    def get_random_player(self, map_name: str = None, side: str = None) -> Optional[str]:
        """Get a random opponent player, optionally filtered by map/side data."""
        candidates = self._players.copy()

        if map_name and side:
            # Filter to players with data for this map/side
            map_name = map_name.lower()
            side = side.lower()
            candidates = [
                p for p in candidates
                if map_name in self._setups.get(p, {})
                and side in self._setups.get(p, {}).get(map_name, {})
            ]

        if not candidates:
            return None

        return random.choice(candidates)

    def get_players_for_map(self, map_name: str, side: str) -> List[str]:
        """Get list of players with data for a specific map/side."""
        map_name = map_name.lower()
        side = side.lower()

        return [
            p for p in self._players
            if map_name in self._setups.get(p, {})
            and side in self._setups.get(p, {}).get(map_name, {})
        ]

    # =========================================================================
    # OPENING SETUPS
    # =========================================================================

    def get_opening_position(self, player_name: str, map_name: str,
                            side: str) -> Optional[Position]:
        """
        Get opponent player's opening position for a map/side.

        Args:
            player_name: Player name
            map_name: Map name (e.g., 'lotus', 'bind')
            side: 'attack' or 'defense'

        Returns:
            Position with x, y in VCT game coordinates, or None
        """
        player = player_name.lower() if player_name else None
        map_name = map_name.lower()
        side = side.lower()

        # Try exact player match
        setup = self._setups.get(player, {}).get(map_name, {}).get(side)

        # Fallback: get any player with this map/side data
        if not setup:
            candidates = self.get_players_for_map(map_name, side)
            if candidates:
                fallback_player = random.choice(candidates)
                setup = self._setups.get(fallback_player, {}).get(map_name, {}).get(side)

        if not setup:
            return None

        positions = setup.get("positions", [])
        if not positions:
            return None

        # Sample from positions weighted by weight
        if len(positions) == 1:
            pos = positions[0]
        else:
            weights = [p.get("weight", 1.0) for p in positions]
            pos = random.choices(positions, weights=weights, k=1)[0]

        # Add noise based on std deviation
        std_x = pos.get("std_x", 500)
        std_y = pos.get("std_y", 500)

        # Sample with noise
        x = pos["x"] + random.gauss(0, std_x / 3)
        y = pos["y"] + random.gauss(0, std_y / 3)

        # Convert to minimap coordinates (0-1)
        # Use simple normalization - can be refined per map
        minimap_x = self._vct_to_minimap_x(y, map_name)  # SWAPPED
        minimap_y = self._vct_to_minimap_y(x, map_name)  # SWAPPED

        return Position(
            x=minimap_x,
            y=minimap_y,
            confidence=setup.get("confidence", 0.7)
        )

    def get_opening_position_by_index(self, map_name: str, side: str,
                                      role_index: int) -> Optional[Position]:
        """
        Get opening position for a role index (0-4).

        Distributes players across different positions.
        """
        map_name = map_name.lower()
        side = side.lower()

        # Get players with data for this map/side
        candidates = self.get_players_for_map(map_name, side)
        if not candidates:
            return None

        # Select player based on index
        player = candidates[role_index % len(candidates)]
        return self.get_opening_position(player, map_name, side)

    # =========================================================================
    # DISTANCE PREFERENCES
    # =========================================================================

    def get_preferred_distance(self, player_name: str = None) -> Tuple[float, float]:
        """
        Get player's preferred engagement distance.

        Args:
            player_name: Player name (optional)

        Returns:
            Tuple of (mean_distance, std_distance) in game units
        """
        if player_name:
            player = player_name.lower()
            pref = self._prefs.get(player, {})
            if pref:
                return (
                    pref.get("mean_distance", 1700),
                    pref.get("std_distance", 255)
                )

        # Default fallback
        return (1700, 255)

    def get_optimal_combat_position(self, player_name: str,
                                    current_pos: Tuple[float, float],
                                    enemy_positions: List[Tuple[float, float]],
                                    map_bounds: Tuple[float, float, float, float] = (0, 1, 0, 1)
                                    ) -> Optional[Position]:
        """
        Calculate optimal position to engage at preferred distance.

        Args:
            player_name: Player name
            current_pos: Current (x, y) in normalized coords
            enemy_positions: List of enemy (x, y) positions
            map_bounds: (x_min, x_max, y_min, y_max)

        Returns:
            Optimal Position to move towards
        """
        if not enemy_positions:
            return None

        pref_distance, std_distance = self.get_preferred_distance(player_name)
        # Convert to normalized units (assume 10000 game units = 1.0 normalized)
        pref_dist_norm = pref_distance / 10000.0
        std_dist_norm = std_distance / 10000.0

        # Find nearest enemy
        nearest_enemy = min(enemy_positions,
                           key=lambda e: self._distance(current_pos, e))
        dist_to_enemy = self._distance(current_pos, nearest_enemy)

        # If already at good distance, stay put
        if abs(dist_to_enemy - pref_dist_norm) < std_dist_norm:
            return Position(x=current_pos[0], y=current_pos[1], confidence=0.9)

        # Calculate direction to/from enemy
        dx = nearest_enemy[0] - current_pos[0]
        dy = nearest_enemy[1] - current_pos[1]
        dist = max(0.001, math.sqrt(dx*dx + dy*dy))

        # Unit vector
        ux = dx / dist
        uy = dy / dist

        # Target position at preferred distance
        if dist_to_enemy > pref_dist_norm:
            # Move closer
            move_dist = min(0.03, dist_to_enemy - pref_dist_norm)
            target_x = current_pos[0] + ux * move_dist
            target_y = current_pos[1] + uy * move_dist
        else:
            # Move away
            move_dist = min(0.03, pref_dist_norm - dist_to_enemy)
            target_x = current_pos[0] - ux * move_dist
            target_y = current_pos[1] - uy * move_dist

        # Clamp to map bounds
        x_min, x_max, y_min, y_max = map_bounds
        target_x = max(x_min, min(x_max, target_x))
        target_y = max(y_min, min(y_max, target_y))

        return Position(x=target_x, y=target_y, confidence=0.8)

    # =========================================================================
    # MOVEMENT MODEL
    # =========================================================================

    def sample_position(self, player_name: str, map_name: str,
                       side: str) -> Optional[Position]:
        """
        Sample a position from player's movement model (KDE).

        Args:
            player_name: Player name
            map_name: Map name
            side: 'attack' or 'defense'

        Returns:
            Sampled Position in normalized coordinates (0-1)
        """
        player = player_name.lower() if player_name else None
        map_name = map_name.lower()
        side = side.lower()

        # Try exact player
        model = self._models.get(player, {}).get(map_name, {}).get(side, {}).get('all')

        # Fallback: get any player with this map/side data
        if not model:
            candidates = [
                p for p in self._players
                if map_name in self._models.get(p, {})
                and side in self._models.get(p, {}).get(map_name, {})
            ]
            if candidates:
                fallback_player = random.choice(candidates)
                model = self._models.get(fallback_player, {}).get(map_name, {}).get(side, {}).get('all')

        if not model:
            return None

        heatmap = model.get("heatmap", [])
        if not heatmap:
            return None

        heatmap = np.array(heatmap)
        resolution = model.get("grid_resolution", 50)

        # Flatten and sample
        flat = heatmap.flatten()
        if flat.sum() == 0:
            return None

        probs = flat / flat.sum()
        idx = np.random.choice(len(probs), p=probs)

        # Convert to grid coordinates
        grid_y = idx // resolution
        grid_x = idx % resolution

        # Convert to normalized position with noise
        cell_size = 1.0 / resolution
        x = (grid_x + random.random()) * cell_size
        y = (grid_y + random.random()) * cell_size

        return Position(
            x=x,
            y=y,
            confidence=model.get("confidence", 0.7)
        )

    def get_movement_target(self, map_name: str, side: str,
                           current_pos: Tuple[float, float],
                           player_name: str = None,
                           max_move: float = 0.05) -> Optional[Position]:
        """
        Get next movement target based on model.

        Samples from KDE but biases towards positions near current location.

        Args:
            map_name: Map name
            side: 'attack' or 'defense'
            current_pos: Current (x, y) in normalized coords
            player_name: Optional specific player
            max_move: Maximum movement per step

        Returns:
            Target Position
        """
        # If no specific player, pick random one with data
        if not player_name:
            player_name = self.get_random_player(map_name, side)

        # Sample multiple candidates
        candidates = []
        for _ in range(10):
            pos = self.sample_position(player_name, map_name, side)
            if pos:
                candidates.append((pos.x, pos.y))

        if not candidates:
            return None

        # Weight by distance (prefer nearby)
        weights = []
        for cx, cy in candidates:
            dist = self._distance(current_pos, (cx, cy))
            weight = math.exp(-dist / (max_move * 3))
            weights.append(weight)

        total = sum(weights)
        if total == 0:
            return None

        probs = [w / total for w in weights]
        idx = random.choices(range(len(candidates)), weights=probs, k=1)[0]
        target = candidates[idx]

        # Limit movement distance
        dx = target[0] - current_pos[0]
        dy = target[1] - current_pos[1]
        dist = math.sqrt(dx*dx + dy*dy)

        if dist > max_move:
            scale = max_move / dist
            target = (
                current_pos[0] + dx * scale,
                current_pos[1] + dy * scale
            )

        return Position(x=target[0], y=target[1], confidence=0.7)

    # =========================================================================
    # COORDINATE CONVERSION
    # =========================================================================

    def _vct_to_minimap_x(self, vct_y: float, map_name: str) -> float:
        """Convert VCT Y coordinate to minimap X (SWAPPED)."""
        # Simple normalization - can be refined per map
        # Typical VCT range: -10000 to +12000
        normalized = (vct_y + 5000) / 15000.0
        return max(0.05, min(0.95, normalized))

    def _vct_to_minimap_y(self, vct_x: float, map_name: str) -> float:
        """Convert VCT X coordinate to minimap Y (SWAPPED)."""
        normalized = (vct_x + 5000) / 15000.0
        return max(0.05, min(0.95, normalized))

    # =========================================================================
    # UTILITY
    # =========================================================================

    def _distance(self, pos1: Tuple[float, float],
                 pos2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance."""
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        return math.sqrt(dx*dx + dy*dy)

    def get_player_count(self) -> int:
        """Get number of opponent players with data."""
        return len(self._players)

    def get_available_maps(self) -> List[str]:
        """Get list of maps with data."""
        maps = set()
        for player_data in self._setups.values():
            maps.update(player_data.keys())
        return list(maps)

    def get_stats(self) -> dict:
        """Get statistics about loaded data."""
        return {
            'n_players': len(self._players),
            'n_setups': len(self._setups),
            'n_preferences': len(self._prefs),
            'n_movement_models': len(self._models),
            'maps': self.get_available_maps(),
        }


# Team service instances
_opponent_services: Dict[str, OpponentRealismService] = {}
_current_team: str = DEFAULT_TEAM


def get_opponent_realism(team_name: str = None) -> OpponentRealismService:
    """
    Get OpponentRealismService instance for a specific team.

    Args:
        team_name: Team name (e.g., 'sentinels', 'nrg', 'loud')
                   If None, uses current/default team

    Returns:
        OpponentRealismService for the specified team
    """
    global _current_team
    team = (team_name or _current_team).lower()

    if team not in _opponent_services:
        _opponent_services[team] = OpponentRealismService(team)

    if team_name:
        _current_team = team

    return _opponent_services[team]


def set_opponent_team(team_name: str) -> OpponentRealismService:
    """
    Set the current opponent team for simulations.

    Args:
        team_name: Team name (e.g., 'sentinels', 'nrg', 'loud')

    Returns:
        OpponentRealismService for the specified team
    """
    global _current_team
    team = team_name.lower()
    _current_team = team
    return get_opponent_realism(team)


def get_available_teams() -> List[str]:
    """Get list of available opponent teams."""
    return sorted(AVAILABLE_TEAMS)


def get_current_team() -> str:
    """Get current opponent team name."""
    return _current_team


# ============================================================================
# INTEGRATION HELPERS FOR SIMULATION ENGINE
# ============================================================================

def get_opponent_opening_position(map_name: str, side: str,
                                  role_index: int = 0,
                                  team_name: str = None) -> Optional[Tuple[float, float]]:
    """
    Get opening position for an opponent player.

    Args:
        map_name: Map name
        side: 'attack' or 'defense'
        role_index: Player index (0-4)
        team_name: Optional team name

    Returns:
        (x, y) in normalized coords or None
    """
    service = get_opponent_realism(team_name)
    pos = service.get_opening_position_by_index(map_name, side, role_index)
    if pos:
        return (pos.x, pos.y)
    return None


def get_opponent_movement_target(map_name: str, side: str,
                                current_pos: Tuple[float, float],
                                player_name: str = None,
                                max_move: float = 0.03,
                                team_name: str = None) -> Optional[Tuple[float, float]]:
    """
    Get movement target for opponent player.

    Args:
        map_name: Map name
        side: 'attack' or 'defense'
        current_pos: Current (x, y)
        player_name: Optional specific player
        max_move: Max movement distance
        team_name: Optional team name

    Returns:
        Target (x, y) or None
    """
    service = get_opponent_realism(team_name)
    pos = service.get_movement_target(map_name, side, current_pos, player_name, max_move)
    if pos:
        return (pos.x, pos.y)
    return None
