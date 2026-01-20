"""
Position Sampler - Get realistic player positions from observed data.

Two modes:
1. DATA-DRIVEN: Sample from actual observed player positions (default)
2. FIGMA MASK: Use manually drawn walkable areas from Figma export

Usage:
    from app.services.position_sampler import PositionSampler

    sampler = PositionSampler('ascent')

    # Get position for attacker during execute phase
    x, y = sampler.sample_position(side='attack', phase='execute')

    # Check if position is valid (if using Figma mask)
    is_valid = sampler.is_walkable(x, y)
"""

import json
import random
import math
from pathlib import Path
from typing import Tuple, Optional, List
import numpy as np

DATA_DIR = Path(__file__).parent.parent / "data"


class PositionSampler:
    """Sample realistic player positions from observed match data."""

    _patterns_cache = None
    _figma_masks_cache = {}

    def __init__(self, map_name: str):
        self.map_name = map_name.lower()
        self._load_patterns()
        self._load_figma_mask()

    def _load_patterns(self):
        """Load position patterns (shared across instances)."""
        if PositionSampler._patterns_cache is None:
            patterns_file = DATA_DIR / "position_patterns.json"
            if patterns_file.exists():
                with open(patterns_file) as f:
                    PositionSampler._patterns_cache = json.load(f)
            else:
                PositionSampler._patterns_cache = {}

        self.patterns = PositionSampler._patterns_cache.get('maps', {}).get(self.map_name, {})

        # Pre-compute walkable cells list for fast sampling
        self.walkable_cells = self.patterns.get('walkable_cells', [])

        # Load heatmaps as numpy arrays for weighted sampling
        if 'attack_heatmap' in self.patterns:
            self.attack_heatmap = np.array(self.patterns['attack_heatmap'])
            self.defense_heatmap = np.array(self.patterns['defense_heatmap'])
        else:
            self.attack_heatmap = None
            self.defense_heatmap = None

    def _load_figma_mask(self):
        """Load walkable mask from v4 JSON (primary) or PNG fallback."""
        if self.map_name in PositionSampler._figma_masks_cache:
            self.figma_mask = PositionSampler._figma_masks_cache[self.map_name]
            self._map_context = PositionSampler._figma_masks_cache.get(f"{self.map_name}_ctx")
            return

        # Try v4 JSON masks first (recommended)
        try:
            try:
                from app.services.map_context import get_map_context
            except ImportError:
                from backend.app.services.map_context import get_map_context
            self._map_context = get_map_context()
            mask = self._map_context.get_walkable_mask(self.map_name, grid_size=100)
            self.figma_mask = mask.astype(bool)
            PositionSampler._figma_masks_cache[self.map_name] = self.figma_mask
            PositionSampler._figma_masks_cache[f"{self.map_name}_ctx"] = self._map_context
            return
        except Exception as e:
            print(f"PositionSampler: Could not load v4 mask for {self.map_name}: {e}")
            self._map_context = None

        # Fallback to PNG file
        mask_file = DATA_DIR / "figma_masks" / f"{self.map_name}_walkable.png"
        if mask_file.exists():
            try:
                from PIL import Image
                img = Image.open(mask_file).convert('L')  # Grayscale
                img = img.resize((100, 100), Image.NEAREST)
                mask = np.array(img) > 128
                self.figma_mask = mask
                PositionSampler._figma_masks_cache[self.map_name] = mask
            except ImportError:
                self.figma_mask = None
        else:
            self.figma_mask = None

    def sample_position(
        self,
        side: str = 'attack',
        phase: str = 'execute',
        near_position: Tuple[float, float] = None,
        spread: float = 0.05
    ) -> Tuple[float, float]:
        """
        Sample a realistic position for a player.

        Args:
            side: 'attack' or 'defense'
            phase: 'setup', 'map_control', 'execute', 'post_plant'
            near_position: If provided, sample near this position
            spread: How far from near_position to sample (default 0.05 = 5% of map)

        Returns:
            (x, y) tuple in normalized 0-1 coordinates
        """
        # If near_position provided, sample around it
        if near_position is not None:
            return self._sample_near(near_position, spread)

        # Try phase centroids first (most realistic)
        phase_data = self.patterns.get('phase_centroids', {}).get(phase, {}).get(side)
        if phase_data and phase_data.get('samples', 0) > 10:
            centroid = phase_data['centroid']
            std = phase_data['std']
            x = random.gauss(centroid[0], std[0])
            y = random.gauss(centroid[1], std[1])
            return self._clamp_and_validate(x, y)

        # Fallback to heatmap sampling
        return self._sample_from_heatmap(side)

    def _sample_from_heatmap(self, side: str = 'attack') -> Tuple[float, float]:
        """Sample position weighted by heatmap density."""
        heatmap = self.attack_heatmap if side == 'attack' else self.defense_heatmap

        if heatmap is None or heatmap.sum() == 0:
            # Fallback to random within bounds
            return self._random_fallback()

        # Flatten and sample weighted by density
        flat = heatmap.flatten()
        if flat.sum() > 0:
            flat = flat / flat.sum()  # Normalize to probabilities
            idx = np.random.choice(len(flat), p=flat)
            grid_size = heatmap.shape[0]
            cell_y, cell_x = divmod(idx, grid_size)

            # Random position within cell
            cell_size = 1.0 / grid_size
            x = (cell_x + random.random()) * cell_size
            y = (cell_y + random.random()) * cell_size
            return (x, y)

        return self._random_fallback()

    def _sample_near(self, position: Tuple[float, float], spread: float) -> Tuple[float, float]:
        """Sample a position near the given position."""
        x = position[0] + random.gauss(0, spread)
        y = position[1] + random.gauss(0, spread)
        return self._clamp_and_validate(x, y)

    def _clamp_and_validate(self, x: float, y: float) -> Tuple[float, float]:
        """Clamp position and validate against Figma mask if available."""
        # Basic clamp to 0-1
        x = max(0.05, min(0.95, x))
        y = max(0.05, min(0.95, y))

        # If Figma mask available, find nearest walkable
        if self.figma_mask is not None and not self.is_walkable(x, y):
            return self._find_nearest_walkable(x, y)

        return (x, y)

    def _random_fallback(self) -> Tuple[float, float]:
        """Random position as last resort."""
        return (random.uniform(0.1, 0.9), random.uniform(0.1, 0.9))

    def is_walkable(self, x: float, y: float) -> bool:
        """
        Check if position is walkable (only works with Figma mask).

        Without Figma mask, always returns True.
        """
        if self.figma_mask is None:
            return True

        grid_size = self.figma_mask.shape[0]
        cell_x = min(grid_size - 1, max(0, int(x * grid_size)))
        cell_y = min(grid_size - 1, max(0, int(y * grid_size)))
        return bool(self.figma_mask[cell_y, cell_x])

    def _find_nearest_walkable(self, x: float, y: float, max_search: int = 10) -> Tuple[float, float]:
        """Find nearest walkable position using spiral search."""
        if self.figma_mask is None:
            return (x, y)

        grid_size = self.figma_mask.shape[0]
        cell_x = int(x * grid_size)
        cell_y = int(y * grid_size)

        # Spiral search outward
        for radius in range(1, max_search):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if abs(dx) != radius and abs(dy) != radius:
                        continue  # Only check perimeter

                    nx, ny = cell_x + dx, cell_y + dy
                    if 0 <= nx < grid_size and 0 <= ny < grid_size:
                        if self.figma_mask[ny, nx]:
                            # Found walkable cell
                            return ((nx + 0.5) / grid_size, (ny + 0.5) / grid_size)

        # Couldn't find walkable, return original
        return (x, y)

    def sample_team_positions(
        self,
        side: str,
        phase: str,
        num_players: int = 5,
        min_spacing: float = 0.03
    ) -> List[Tuple[float, float]]:
        """
        Sample positions for an entire team with minimum spacing.

        Args:
            side: 'attack' or 'defense'
            phase: Round phase
            num_players: Number of players (default 5)
            min_spacing: Minimum distance between players (default 0.03 = 3% of map)

        Returns:
            List of (x, y) positions
        """
        positions = []

        for i in range(num_players):
            max_attempts = 20
            for _ in range(max_attempts):
                if i == 0:
                    # First player uses phase centroid
                    pos = self.sample_position(side=side, phase=phase)
                else:
                    # Subsequent players sample near team but not too close
                    base = random.choice(positions)
                    pos = self.sample_position(near_position=base, spread=0.08)

                # Check spacing from other players
                too_close = False
                for other in positions:
                    dist = math.sqrt((pos[0] - other[0])**2 + (pos[1] - other[1])**2)
                    if dist < min_spacing:
                        too_close = True
                        break

                if not too_close:
                    positions.append(pos)
                    break
            else:
                # Couldn't find spaced position, add anyway
                positions.append(pos)

        return positions


# Convenience functions
_samplers = {}

def get_sampler(map_name: str) -> PositionSampler:
    """Get or create a position sampler for a map."""
    if map_name not in _samplers:
        _samplers[map_name] = PositionSampler(map_name)
    return _samplers[map_name]

def sample_position(map_name: str, side: str = 'attack', phase: str = 'execute') -> Tuple[float, float]:
    """Quick function to sample a position."""
    return get_sampler(map_name).sample_position(side=side, phase=phase)
