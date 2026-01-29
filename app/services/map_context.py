"""
Map Context Service - Single source of truth for map data.

Loads v4 walkable masks and provides:
- Walkability checks
- Navigation grids for pathfinding
- Obstacle detection
- Position validation
- Line-of-sight checks (using dilated masks for better LOS)
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from functools import lru_cache

try:
    from scipy import ndimage
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

DATA_DIR = Path(__file__).parent.parent / "data"
MASK_FILE = DATA_DIR / "figma_masks" / "walkable_masks_v4.json"

# LOS mask dilation iterations - allows sightlines through corridors
LOS_DILATION_ITERATIONS = 8


class MapContext:
    """Provides map context from v4 walkable masks."""

    _instance: Optional['MapContext'] = None
    _masks_cache: Dict = {}
    _los_masks_cache: Dict = {}  # Dilated masks for LOS checks
    _walkable_cells_cache: Dict = {}  # Cache for walkable cell coordinates

    def __new__(cls):
        """Singleton pattern - one instance shared across simulation."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._load_masks()
        self._initialized = True

    def _load_masks(self):
        """Load all v4 masks from JSON and create dilated LOS masks."""
        if MASK_FILE.exists():
            with open(MASK_FILE, 'r') as f:
                data = json.load(f)
                self._masks_cache = data.get('maps', {})

            # Create dilated masks for LOS checks
            self._create_los_masks()

            print(f"MapContext: Loaded {len(self._masks_cache)} map masks")
        else:
            print(f"MapContext: Warning - mask file not found at {MASK_FILE}")
            self._masks_cache = {}

    def _create_los_masks(self):
        """Create dilated masks for line-of-sight checks.

        Map-derived masks are geometrically accurate but too restrictive for LOS.
        Dilating allows sightlines through corridors while keeping positioning accurate.
        """
        if not HAS_SCIPY:
            print("MapContext: scipy not available, LOS will use original masks")
            return

        struct = ndimage.generate_binary_structure(2, 2)  # 8-connectivity

        for map_name, data in self._masks_cache.items():
            walkable = np.array(data['walkable_mask'], dtype=np.uint8)
            # Dilate walkable area for LOS (invert obstacle check)
            dilated = ndimage.binary_dilation(walkable, struct, iterations=LOS_DILATION_ITERATIONS)
            # Store as obstacle mask (inverted)
            self._los_masks_cache[map_name] = 1 - dilated.astype(np.uint8)

    def get_map_data(self, map_name: str) -> Optional[Dict]:
        """Get raw mask data for a map."""
        return self._masks_cache.get(map_name.lower())

    def get_walkable_mask(self, map_name: str, grid_size: int = 150) -> np.ndarray:
        """Get walkable mask as numpy array, resized to requested grid size.

        Args:
            map_name: Map name (e.g., 'haven', 'ascent')
            grid_size: Target grid size (default 150, pathfinding uses 128)

        Returns:
            2D numpy array where 1=walkable, 0=not walkable
        """
        data = self.get_map_data(map_name)
        if data is None:
            # Return empty walkable grid if no data
            return np.ones((grid_size, grid_size), dtype=np.uint8)

        mask = np.array(data['walkable_mask'], dtype=np.uint8)

        # Resize if needed
        if mask.shape[0] != grid_size:
            from PIL import Image
            img = Image.fromarray(mask.astype(np.uint8) * 255)
            img = img.resize((grid_size, grid_size), Image.NEAREST)
            mask = (np.array(img) > 128).astype(np.uint8)

        return mask

    def get_obstacle_mask(self, map_name: str, grid_size: int = 150) -> np.ndarray:
        """Get obstacle mask as numpy array.

        Returns:
            2D numpy array where 1=obstacle, 0=not obstacle
        """
        data = self.get_map_data(map_name)
        if data is None:
            return np.zeros((grid_size, grid_size), dtype=np.uint8)

        mask = np.array(data['obstacle_mask'], dtype=np.uint8)

        if mask.shape[0] != grid_size:
            from PIL import Image
            img = Image.fromarray(mask.astype(np.uint8) * 255)
            img = img.resize((grid_size, grid_size), Image.NEAREST)
            mask = (np.array(img) > 128).astype(np.uint8)

        return mask

    def get_nav_grid(self, map_name: str, grid_size: int = 128) -> np.ndarray:
        """Get navigation grid for pathfinding.

        Returns:
            2D numpy array where 0=walkable, 1=blocked (inverted from walkable_mask)
        """
        walkable = self.get_walkable_mask(map_name, grid_size)
        # Invert: pathfinding expects 0=walkable, 1=blocked
        return (1 - walkable).astype(np.uint8)

    def is_walkable(self, map_name: str, x: float, y: float) -> bool:
        """Check if a normalized position (0-1) is walkable.

        Args:
            map_name: Map name
            x: X coordinate (0-1)
            y: Y coordinate (0-1)

        Returns:
            True if position is walkable
        """
        data = self.get_map_data(map_name)
        if data is None:
            return True  # Default to walkable if no data

        mask = np.array(data['walkable_mask'])
        grid_size = mask.shape[0]

        # Convert to grid coordinates
        gx = min(int(x * grid_size), grid_size - 1)
        gy = min(int(y * grid_size), grid_size - 1)

        return mask[gy, gx] == 1

    def is_obstacle(self, map_name: str, x: float, y: float) -> bool:
        """Check if a normalized position (0-1) is an obstacle."""
        data = self.get_map_data(map_name)
        if data is None:
            return False

        mask = np.array(data['obstacle_mask'])
        grid_size = mask.shape[0]

        gx = min(int(x * grid_size), grid_size - 1)
        gy = min(int(y * grid_size), grid_size - 1)

        return mask[gy, gx] == 1

    def get_walkable_cells(self, map_name: str) -> List[Tuple[float, float]]:
        """Get list of walkable cell centers (normalized 0-1 coordinates)."""
        map_key = map_name.lower()

        # Return cached if available
        if map_key in self._walkable_cells_cache:
            return self._walkable_cells_cache[map_key]

        data = self.get_map_data(map_name)
        if data is None:
            return []

        # walkable_cells in JSON is just a count, generate from mask
        mask = np.array(data['walkable_mask'])
        grid_size = mask.shape[0]

        cells = []
        for gy in range(grid_size):
            for gx in range(grid_size):
                if mask[gy, gx] == 1:
                    # Convert to normalized coordinates (cell center)
                    x = (gx + 0.5) / grid_size
                    y = (gy + 0.5) / grid_size
                    cells.append((x, y))

        # Cache for future calls
        self._walkable_cells_cache[map_key] = cells
        return cells

    def get_random_walkable_position(self, map_name: str) -> Tuple[float, float]:
        """Get a random walkable position on the map."""
        cells = self.get_walkable_cells(map_name)
        if not cells:
            return (0.5, 0.5)  # Center fallback

        import random
        cell = random.choice(cells)
        return (cell[0], cell[1])

    def get_nearest_walkable(self, map_name: str, x: float, y: float) -> Tuple[float, float]:
        """Find nearest walkable position to given coordinates."""
        if self.is_walkable(map_name, x, y):
            return (x, y)

        cells = self.get_walkable_cells(map_name)
        if not cells:
            return (x, y)

        # Find nearest
        min_dist = float('inf')
        nearest = (x, y)

        for cx, cy in cells:
            dist = (cx - x) ** 2 + (cy - y) ** 2
            if dist < min_dist:
                min_dist = dist
                nearest = (cx, cy)

        return nearest

    def has_line_of_sight(self, map_name: str, x1: float, y1: float,
                          x2: float, y2: float) -> bool:
        """Check if there's line of sight between two positions.

        Uses Bresenham's algorithm to trace the line and check for obstacles.
        Uses dilated LOS mask for more permissive sightlines through corridors.
        """
        map_key = map_name.lower()

        # Use dilated LOS mask if available
        if map_key in self._los_masks_cache:
            obstacle_mask = self._los_masks_cache[map_key]
        else:
            data = self.get_map_data(map_name)
            if data is None:
                return True
            obstacle_mask = np.array(data['obstacle_mask'])

        grid_size = obstacle_mask.shape[0]

        # Convert to grid coordinates
        gx1 = min(int(x1 * grid_size), grid_size - 1)
        gy1 = min(int(y1 * grid_size), grid_size - 1)
        gx2 = min(int(x2 * grid_size), grid_size - 1)
        gy2 = min(int(y2 * grid_size), grid_size - 1)

        # Bresenham's line algorithm
        dx = abs(gx2 - gx1)
        dy = abs(gy2 - gy1)
        sx = 1 if gx1 < gx2 else -1
        sy = 1 if gy1 < gy2 else -1
        err = dx - dy

        cx, cy = gx1, gy1

        while True:
            # Check if current cell is obstacle
            if obstacle_mask[cy, cx] == 1:
                return False

            if cx == gx2 and cy == gy2:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                cx += sx
            if e2 < dx:
                err += dx
                cy += sy

        return True

    def get_stats(self, map_name: str) -> Dict:
        """Get map statistics."""
        data = self.get_map_data(map_name)
        if data is None:
            return {}
        return data.get('stats', {})


# Global instance for easy access
_map_context: Optional[MapContext] = None


def get_map_context() -> MapContext:
    """Get the global MapContext instance."""
    global _map_context
    if _map_context is None:
        _map_context = MapContext()
    return _map_context
