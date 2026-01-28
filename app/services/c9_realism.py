"""
C9 Realism Service

Provides C9-specific movement, positioning, and coordination patterns
based on VCT data extraction.

Implements:
- P0: Opening Setups - C9's actual default positions by map/side
- P1: Combat Positioning - Position players at preferred engagement distances
- P2: Movement Model - KDE-based position sampling from VCT data

Methodology: docs/C9_SIMULATION_METHODOLOGY.md
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

# C9 Current Roster
C9_ROSTER = ["OXY", "v1c", "Xeppaa", "neT", "mitch"]

# =============================================================================
# VCT COORDINATE CONVERSION
# =============================================================================
# VCT data uses raw game coordinates (roughly -10000 to +12000 range)
# Minimap uses normalized 0-1 coordinates.
#
# CRITICAL: In VALORANT, coordinates are SWAPPED:
#   minimapX = gameY * xMultiplier + xScalarAdd
#   minimapY = gameX * yMultiplier + yScalarAdd
#
# The values below are estimated from reference points:
# - Attack/Defense spawn positions on minimap
# - Site positions on minimap
# - Comparing with VCT raw coordinate ranges per map
#
# Format: {map_name: (x_mult, x_add, y_mult, y_add)}
MAP_COORDINATE_CONVERSION = {
    # Lotus: Attack spawn (0.50, 0.85), Defense spawn (0.62, 0.18)
    # VCT X range: ~2800-8500, VCT Y range: ~-2600 to 4500
    'lotus': (0.00007, 0.30, -0.00006, 0.95),

    # Ascent: Attack spawn (0.85, 0.58), Defense spawn (0.15, 0.42)
    # VCT X range: ~-2300 to 4000, VCT Y range: ~-10000 to -2000
    'ascent': (0.00008, 0.65, 0.00005, 1.0),

    # Bind: Attack spawn (0.55, 0.88), Defense spawn (0.55, 0.12)
    # VCT X range: ~6000-12500, VCT Y range: ~-2300 to 2500
    'bind': (0.00006, 0.30, -0.00005, 0.95),

    # Haven: Attack spawn (0.88, 0.55), Defense spawn (0.12, 0.40)
    # VCT X range: ~-1000 to 6500, VCT Y range: ~-11000 to -4500
    'haven': (0.00008, 0.80, 0.00005, 1.0),

    # Split: Attack spawn (0.85, 0.55), Defense spawn (0.12, 0.55)
    # VCT X range: ~1500-5500, VCT Y range: ~-8000 to -2000
    'split': (0.00008, 0.50, 0.00008, 1.1),

    # Icebox: Attack spawn (0.10, 0.58), Defense spawn (0.90, 0.55)
    # VCT X range: ~-6500 to -2000, VCT Y range: ~-1200 to 5000
    'icebox': (0.00008, 0.60, 0.00008, 0.15),

    # Pearl: VCT X range: ~2000-9000, VCT Y range: ~-2500 to 2200
    'pearl': (0.00006, 0.30, -0.00005, 0.85),

    # Fracture: VCT X range: ~6000-8700, VCT Y range: ~-4000 to -1300
    'fracture': (0.00008, 0.20, 0.00006, 0.80),

    # Sunset, Abyss, Corrode - estimated
    'sunset': (0.00007, 0.50, -0.00005, 0.50),
    'abyss': (0.00007, 0.50, -0.00005, 0.50),
    'corrode': (0.00007, 0.50, -0.00005, 0.50),
}


def convert_vct_to_minimap(vct_x: float, vct_y: float, map_name: str) -> Tuple[float, float]:
    """
    Convert VCT game coordinates to normalized minimap coordinates.

    VALORANT coordinate system is SWAPPED:
    - minimapX depends on gameY
    - minimapY depends on gameX

    Args:
        vct_x: X coordinate from VCT data (game units)
        vct_y: Y coordinate from VCT data (game units)
        map_name: Map name for conversion parameters

    Returns:
        (minimap_x, minimap_y) in 0-1 normalized coordinates
    """
    map_name = map_name.lower()
    if map_name not in MAP_COORDINATE_CONVERSION:
        # Fallback: simple division (less accurate)
        return (vct_x / 10000.0, vct_y / 10000.0)

    x_mult, x_add, y_mult, y_add = MAP_COORDINATE_CONVERSION[map_name]

    # SWAPPED: minimapX uses gameY, minimapY uses gameX
    minimap_x = vct_y * x_mult + x_add
    minimap_y = vct_x * y_mult + y_add

    # Clamp to valid range
    minimap_x = max(0.05, min(0.95, minimap_x))
    minimap_y = max(0.05, min(0.95, minimap_y))

    return (minimap_x, minimap_y)


@dataclass
class Position:
    """A 2D position with optional metadata."""
    x: float
    y: float
    confidence: float = 1.0


class C9RealismService:
    """
    Service for C9-specific simulation realism.

    Provides methods to:
    1. Get opening positions for C9 players
    2. Calculate optimal combat positions based on distance preferences
    3. Sample movement positions from player-specific KDE models
    """

    _instance = None

    def __new__(cls):
        """Singleton pattern for shared data access."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    def __init__(self):
        if not self._loaded:
            self._load_data()
            self._loaded = True

    def _load_data(self):
        """Load all C9 pattern data files."""
        self._opening_setups = self._load_json("c9_opening_setups.json")
        self._distance_prefs = self._load_json("c9_distance_preferences.json")
        self._movement_models = self._load_json("c9_movement_models.json")

        # Build lookup indices
        self._setups = self._opening_setups.get("setups", {})
        self._prefs = self._distance_prefs.get("preferences", {})
        self._models = self._movement_models.get("models", {})

    def _load_json(self, filename: str) -> dict:
        """Load a JSON file from the data directory."""
        filepath = DATA_DIR / filename
        if filepath.exists():
            with open(filepath) as f:
                return json.load(f)
        return {}

    # =========================================================================
    # P0: OPENING SETUPS
    # =========================================================================

    def get_opening_position(self, player_name: str, map_name: str,
                            side: str) -> Optional[Position]:
        """
        Get C9 player's opening position for a map/side.

        Args:
            player_name: Player name (e.g., "OXY", "Xeppaa")
            map_name: Map name (e.g., "lotus", "bind")
            side: "attack" or "defense"

        Returns:
            Position with x, y in NORMALIZED MINIMAP coordinates (0-1), or None
        """
        # Normalize inputs
        player = self._normalize_player_name(player_name)
        map_name_lower = map_name.lower()
        side = side.lower()

        setup = self._setups.get(player, {}).get(map_name_lower, {}).get(side)
        if not setup:
            return None

        positions = setup.get("positions", [])
        if not positions:
            return None

        # Sample from positions weighted by cluster weight
        if len(positions) == 1:
            pos = positions[0]
        else:
            weights = [p.get("weight", 1.0) for p in positions]
            pos = random.choices(positions, weights=weights, k=1)[0]

        # Add noise based on std deviation (captures uncertainty)
        std_x = pos.get("std_x", 100)
        std_y = pos.get("std_y", 100)

        # Sample with reduced noise (1/3 of std for more consistency)
        vct_x = pos["x"] + random.gauss(0, std_x / 3)
        vct_y = pos["y"] + random.gauss(0, std_y / 3)

        # Convert from VCT game coordinates to normalized minimap coordinates
        minimap_x, minimap_y = convert_vct_to_minimap(vct_x, vct_y, map_name_lower)

        return Position(
            x=minimap_x,
            y=minimap_y,
            confidence=setup.get("confidence", 0.7)
        )

    def get_team_opening_positions(self, map_name: str,
                                   side: str) -> Dict[str, Position]:
        """
        Get opening positions for entire C9 roster.

        Args:
            map_name: Map name
            side: "attack" or "defense"

        Returns:
            Dict mapping player name to Position
        """
        positions = {}
        for player in C9_ROSTER:
            pos = self.get_opening_position(player, map_name, side)
            if pos:
                positions[player] = pos
        return positions

    # =========================================================================
    # P1: COMBAT POSITIONING
    # =========================================================================

    def get_preferred_distance(self, player_name: str) -> Tuple[float, float]:
        """
        Get player's preferred engagement distance.

        Args:
            player_name: Player name

        Returns:
            Tuple of (mean_distance, std_distance) in game units
        """
        player = self._normalize_player_name(player_name)
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
                                    map_bounds: Tuple[float, float, float, float] = (0, 10000, 0, 10000)
                                    ) -> Optional[Position]:
        """
        Calculate optimal position to engage at preferred distance.

        Args:
            player_name: Player name
            current_pos: Current (x, y) position
            enemy_positions: List of enemy (x, y) positions
            map_bounds: (x_min, x_max, y_min, y_max) for valid positions

        Returns:
            Optimal Position to move towards, or None if no valid position
        """
        if not enemy_positions:
            return None

        pref_distance, std_distance = self.get_preferred_distance(player_name)

        # Find nearest enemy
        nearest_enemy = min(enemy_positions,
                           key=lambda e: self._distance(current_pos, e))
        dist_to_enemy = self._distance(current_pos, nearest_enemy)

        # If already at good distance, stay put
        if abs(dist_to_enemy - pref_distance) < std_distance:
            return Position(x=current_pos[0], y=current_pos[1], confidence=0.9)

        # Calculate direction to/from enemy
        dx = nearest_enemy[0] - current_pos[0]
        dy = nearest_enemy[1] - current_pos[1]
        dist = max(0.1, math.sqrt(dx*dx + dy*dy))

        # Unit vector
        ux = dx / dist
        uy = dy / dist

        # Target position at preferred distance
        if dist_to_enemy > pref_distance:
            # Move closer
            move_dist = min(300, dist_to_enemy - pref_distance)  # Cap movement
            target_x = current_pos[0] + ux * move_dist
            target_y = current_pos[1] + uy * move_dist
        else:
            # Move away
            move_dist = min(300, pref_distance - dist_to_enemy)
            target_x = current_pos[0] - ux * move_dist
            target_y = current_pos[1] - uy * move_dist

        # Clamp to map bounds
        x_min, x_max, y_min, y_max = map_bounds
        target_x = max(x_min, min(x_max, target_x))
        target_y = max(y_min, min(y_max, target_y))

        return Position(x=target_x, y=target_y, confidence=0.8)

    # =========================================================================
    # P2: MOVEMENT MODEL
    # =========================================================================

    def sample_position(self, player_name: str, map_name: str,
                       side: str, phase: str = "all") -> Optional[Position]:
        """
        Sample a position from player's movement model (KDE).

        Args:
            player_name: Player name
            map_name: Map name
            side: "attack" or "defense"
            phase: Round phase (default "all")

        Returns:
            Sampled Position in normalized coordinates (0-1)
        """
        player = self._normalize_player_name(player_name)
        map_name = map_name.lower()
        side = side.lower()

        model = self._models.get(player, {}).get(map_name, {}).get(side, {}).get(phase)
        if not model:
            return None

        heatmap = model.get("heatmap", [])
        if not heatmap:
            return None

        # Convert heatmap to numpy array
        heatmap = np.array(heatmap)
        resolution = model.get("grid_resolution", 50)

        # Flatten and sample index
        flat = heatmap.flatten()
        if flat.sum() == 0:
            return None

        # Normalize
        probs = flat / flat.sum()

        # Sample index
        idx = np.random.choice(len(probs), p=probs)

        # Convert to grid coordinates
        grid_y = idx // resolution
        grid_x = idx % resolution

        # Convert to normalized position with some noise
        cell_size = 1.0 / resolution
        x = (grid_x + random.random()) * cell_size
        y = (grid_y + random.random()) * cell_size

        return Position(
            x=x,
            y=y,
            confidence=model.get("confidence", 0.8) if model.get("n_samples", 0) >= 30 else 0.5
        )

    def get_position_probability(self, player_name: str, map_name: str,
                                side: str, position: Tuple[float, float],
                                phase: str = "all") -> float:
        """
        Get probability density at a position from player's model.

        Args:
            player_name: Player name
            map_name: Map name
            side: "attack" or "defense"
            position: (x, y) in normalized coordinates (0-1)
            phase: Round phase

        Returns:
            Probability density (0-1, higher = more likely position)
        """
        player = self._normalize_player_name(player_name)
        map_name = map_name.lower()
        side = side.lower()

        model = self._models.get(player, {}).get(map_name, {}).get(side, {}).get(phase)
        if not model:
            return 0.0

        heatmap = model.get("heatmap", [])
        if not heatmap:
            return 0.0

        heatmap = np.array(heatmap)
        resolution = model.get("grid_resolution", 50)

        # Convert position to grid index
        x, y = position
        grid_x = int(min(resolution - 1, max(0, x * resolution)))
        grid_y = int(min(resolution - 1, max(0, y * resolution)))

        return float(heatmap[grid_y, grid_x])

    def get_movement_target(self, player_name: str, map_name: str,
                           side: str, current_pos: Tuple[float, float],
                           max_move: float = 0.05) -> Optional[Position]:
        """
        Get next movement target based on player's model.

        Samples from KDE but biases towards positions near current location.

        Args:
            player_name: Player name
            map_name: Map name
            side: "attack" or "defense"
            current_pos: Current (x, y) in normalized coords
            max_move: Maximum movement per step (normalized)

        Returns:
            Target Position
        """
        # Sample multiple candidates from model
        candidates = []
        for _ in range(10):
            pos = self.sample_position(player_name, map_name, side)
            if pos:
                candidates.append((pos.x, pos.y))

        if not candidates:
            return None

        # Weight by distance (prefer nearby but still probabilistic)
        weights = []
        for cx, cy in candidates:
            dist = self._distance(current_pos, (cx, cy))
            # Exponential decay weight
            weight = math.exp(-dist / (max_move * 2))
            weights.append(weight)

        # Normalize weights
        total = sum(weights)
        if total == 0:
            return None

        probs = [w / total for w in weights]

        # Sample
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
    # UTILITY METHODS
    # =========================================================================

    def _normalize_player_name(self, name: str) -> str:
        """Normalize player name to canonical form."""
        name_lower = name.lower()

        for canonical in C9_ROSTER:
            if canonical.lower() == name_lower:
                return canonical

        return name

    def _distance(self, pos1: Tuple[float, float],
                 pos2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two positions."""
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        return math.sqrt(dx*dx + dy*dy)

    def is_c9_player(self, player_name: str) -> bool:
        """Check if player is in C9 roster."""
        return self._normalize_player_name(player_name) in C9_ROSTER

    def get_roster(self) -> List[str]:
        """Get C9 roster list."""
        return C9_ROSTER.copy()


# Singleton accessor
def get_c9_realism() -> C9RealismService:
    """Get singleton instance of C9RealismService."""
    return C9RealismService()


# ============================================================================
# INTEGRATION HELPERS FOR SIMULATION ENGINE
# ============================================================================

def apply_c9_opening_positions(players: Dict, map_name: str, side: str) -> Dict:
    """
    Apply C9 opening positions to player dict.

    Args:
        players: Dict of player_id -> SimulatedPlayer
        map_name: Map name
        side: "attack" or "defense"

    Returns:
        Modified players dict
    """
    service = get_c9_realism()

    for player_id, player in players.items():
        # Get player name from player object
        player_name = getattr(player, 'name', None) or player_id

        if service.is_c9_player(player_name):
            pos = service.get_opening_position(player_name, map_name, side)
            if pos:
                # Convert from game units to normalized if needed
                # Assuming 10000 unit map
                player.x = pos.x / 10000.0 if pos.x > 1 else pos.x
                player.y = pos.y / 10000.0 if pos.y > 1 else pos.y

    return players


def get_c9_combat_position(player_name: str, current_pos: Tuple[float, float],
                          enemy_positions: List[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    """
    Get optimal combat position for C9 player.

    Args:
        player_name: Player name
        current_pos: Current (x, y)
        enemy_positions: List of enemy positions

    Returns:
        Target (x, y) or None
    """
    service = get_c9_realism()

    if not service.is_c9_player(player_name):
        return None

    pos = service.get_optimal_combat_position(
        player_name, current_pos, enemy_positions
    )

    if pos:
        return (pos.x, pos.y)
    return None


def sample_c9_movement(player_name: str, map_name: str, side: str,
                      current_pos: Tuple[float, float]) -> Optional[Tuple[float, float]]:
    """
    Sample movement target for C9 player from KDE model.

    Args:
        player_name: Player name
        map_name: Map name
        side: "attack" or "defense"
        current_pos: Current (x, y) in normalized coords

    Returns:
        Target (x, y) in normalized coords or None
    """
    service = get_c9_realism()

    if not service.is_c9_player(player_name):
        return None

    pos = service.get_movement_target(
        player_name, map_name, side, current_pos
    )

    if pos:
        return (pos.x, pos.y)
    return None
