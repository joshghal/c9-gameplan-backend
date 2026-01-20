"""A* Pathfinding for VALORANT maps.

Uses 128x128 grid-based navigation with 8-directional movement.
Supports obstacle avoidance, visibility checks, and line-of-sight calculations.
"""

import heapq
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Set, Any
import numpy as np


@dataclass
class Node:
    """A* search node."""
    x: int
    y: int
    g: float = 0  # Cost from start
    h: float = 0  # Heuristic to goal
    parent: Optional['Node'] = None

    @property
    def f(self) -> float:
        return self.g + self.h

    def __lt__(self, other: 'Node') -> bool:
        return self.f < other.f

    def __hash__(self) -> int:
        return hash((self.x, self.y))

    def __eq__(self, other) -> bool:
        return self.x == other.x and self.y == other.y


@dataclass
class PathResult:
    """Result of pathfinding query."""
    path: List[Tuple[float, float]]  # Normalized coordinates (0-1)
    distance: float
    waypoints: List[Tuple[int, int]]  # Grid coordinates
    success: bool
    error: Optional[str] = None


class AStarPathfinder:
    """A* pathfinding for VALORANT maps.

    Uses a 128x128 grid where:
    - 0 = walkable
    - 1 = obstacle (wall, object)
    - 2 = restricted (one-way, elevated)
    """

    GRID_SIZE = 128
    DIRECTIONS = [
        (0, 1), (1, 0), (0, -1), (-1, 0),  # Cardinal
        (1, 1), (1, -1), (-1, 1), (-1, -1)  # Diagonal
    ]
    DIAGONAL_COST = 1.414  # sqrt(2)
    CARDINAL_COST = 1.0

    def __init__(self, nav_grid: Optional[np.ndarray] = None):
        """Initialize pathfinder with optional navigation grid.

        Args:
            nav_grid: 128x128 numpy array of walkability (0=walkable, 1=blocked)
        """
        if nav_grid is not None:
            self.nav_grid = nav_grid
        else:
            # Default empty grid (all walkable)
            self.nav_grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.uint8)

        # Visibility cache for line-of-sight checks
        self._visibility_cache: Dict[Tuple[int, int, int, int], bool] = {}

    # Map geometry data - walls and obstacles for each map
    # Coordinates are in grid units (0-127)
    # Each entry is (x1, y1, x2, y2) defining a rectangular wall
    MAP_GEOMETRY = {
        'ascent': [
            # Mid barriers
            (55, 40, 73, 55),   # Mid market building
            (55, 55, 60, 70),   # Mid bottom building
            (68, 55, 73, 70),   # Mid tiles building
            # A site structures
            (15, 10, 40, 20),   # A site back wall
            (15, 20, 20, 35),   # A site left wall
            (35, 20, 40, 35),   # A site right wall (tree)
            (20, 25, 30, 30),   # A site generator
            # B site structures
            (88, 10, 113, 20),  # B site back wall
            (88, 20, 93, 35),   # B site left wall
            (108, 20, 113, 35), # B site right wall
            (95, 25, 105, 30),  # B site default box
            # A main/lobby
            (25, 50, 35, 65),   # A lobby pillar
            (10, 60, 20, 75),   # A main wall
            # B main
            (95, 50, 105, 65),  # B lobby pillar
            (108, 60, 118, 75), # B main wall
            # Spawn barriers
            (45, 95, 55, 105),  # Attack spawn left
            (73, 95, 83, 105),  # Attack spawn right
        ],
        'bind': [
            # A site structures
            (10, 15, 35, 25),   # A site back
            (10, 25, 15, 40),   # A lamps
            (30, 25, 35, 40),   # A short corner
            (15, 30, 25, 35),   # A site box
            # B site structures
            (93, 15, 118, 25),  # B site back
            (93, 25, 98, 40),   # B long corner
            (113, 25, 118, 40), # B elbow
            (100, 30, 110, 35), # B site box
            # Hookah/shower
            (85, 45, 95, 60),   # Hookah building
            # A short tunnel
            (25, 45, 35, 55),   # A short building
            # Teleporters (marked as walls for nav but special)
            (20, 55, 25, 60),   # A teleporter
            (103, 55, 108, 60), # B teleporter
            # Mid
            (55, 35, 73, 50),   # Mid cubby area
        ],
        'haven': [
            # A site
            (5, 15, 25, 25),    # A site back
            (5, 25, 10, 40),    # A long wall
            (20, 25, 25, 40),   # A short wall
            # B site (middle)
            (55, 8, 73, 18),    # B site back
            (55, 18, 60, 30),   # B site left
            (68, 18, 73, 30),   # B site right
            # C site
            (103, 15, 123, 25), # C site back
            (103, 25, 108, 40), # C long wall
            (118, 25, 123, 40), # C short wall
            # Mid window/doors
            (40, 40, 50, 55),   # A link building
            (78, 40, 88, 55),   # C link building
            # Garage
            (55, 55, 73, 70),   # Garage/mid
            # Spawn areas
            (55, 100, 73, 115), # Attack spawn structure
        ],
        'split': [
            # A site
            (10, 10, 35, 22),   # A site back/heaven
            (10, 22, 15, 38),   # A ramps
            (30, 22, 35, 38),   # A screens
            (17, 28, 27, 33),   # A default
            # B site
            (93, 10, 118, 22),  # B site back/heaven
            (93, 22, 98, 38),   # B main wall
            (113, 22, 118, 38), # B garage
            (100, 28, 110, 33), # B default
            # Mid
            (50, 35, 78, 50),   # Mid vents/mail
            (55, 50, 60, 65),   # Mid bottom left
            (68, 50, 73, 65),   # Mid bottom right
            # Sewers
            (40, 70, 50, 85),   # A sewer
            (78, 70, 88, 85),   # B sewer
            # Ropes area (impassable)
            (60, 25, 68, 35),   # Ropes
        ],
        'icebox': [
            # A site
            (15, 12, 40, 25),   # A site rafters
            (15, 25, 20, 42),   # A pipes
            (35, 25, 40, 42),   # A screens
            (22, 30, 32, 37),   # A site box
            # B site (larger)
            (80, 15, 110, 30),  # B site back
            (80, 30, 85, 48),   # B orange
            (105, 30, 110, 48), # B green
            (88, 35, 100, 42),  # B yellow
            # Mid
            (50, 35, 78, 52),   # Mid boiler
            (55, 52, 60, 68),   # Mid tube left
            (68, 52, 73, 68),   # Mid tube right
            # Kitchen/attacker side
            (45, 75, 55, 90),   # Kitchen
            (73, 75, 83, 90),   # B entry
        ],
        'breeze': [
            # A site (pyramid area)
            (8, 15, 30, 28),    # A back/cave
            (8, 28, 13, 45),    # A shop
            (25, 28, 30, 45),   # A switch
            (14, 33, 22, 40),   # A pyramid
            # B site
            (98, 18, 120, 30),  # B back wall
            (98, 30, 103, 48),  # B tunnel
            (115, 30, 120, 48), # B wall
            (105, 35, 113, 42), # B box
            # Mid
            (48, 28, 80, 42),   # Mid pillar/nest
            (55, 42, 60, 58),   # Mid cannon
            (68, 42, 73, 58),   # Mid arches
            # Hall
            (35, 55, 45, 75),   # A hall
            (83, 55, 93, 75),   # B hall
        ],
        'fracture': [
            # A site (unique - attackers can come from both sides)
            (10, 40, 30, 55),   # A site main
            (15, 55, 25, 65),   # A drop
            # B site
            (98, 40, 118, 55),  # B site main
            (103, 55, 113, 65), # B arcade
            # Mid (defender spawn area)
            (50, 35, 78, 55),   # Mid dish
            (55, 55, 60, 70),   # Mid door
            (68, 55, 73, 70),   # Mid tree
            # Attacker tunnels (top and bottom)
            (40, 5, 88, 15),    # Top attacker path
            (40, 105, 88, 115), # Bottom attacker path
            # Connector buildings
            (35, 60, 45, 80),   # A connector
            (83, 60, 93, 80),   # B connector
        ],
        'pearl': [
            # A site
            (12, 15, 35, 28),   # A site back
            (12, 28, 17, 45),   # A main wall
            (30, 28, 35, 45),   # A link wall
            (18, 33, 28, 40),   # A default
            # B site
            (93, 18, 116, 30),  # B site back
            (93, 30, 98, 48),   # B main wall
            (111, 30, 116, 48), # B hall wall
            (100, 35, 108, 42), # B default
            # Mid
            (50, 32, 78, 48),   # Mid plaza
            (55, 48, 60, 65),   # Mid art
            (68, 48, 73, 65),   # Mid shops
            # Connectors
            (38, 55, 48, 72),   # A link
            (80, 55, 90, 72),   # B link
        ],
        'lotus': [
            # A site
            (8, 18, 28, 32),    # A site back
            (8, 32, 13, 48),    # A main
            (23, 32, 28, 48),   # A rubble
            (13, 38, 21, 44),   # A default
            # B site (middle)
            (52, 10, 76, 25),   # B site back
            (52, 25, 57, 40),   # B main
            (71, 25, 76, 40),   # B upper
            (59, 30, 68, 37),   # B default
            # C site
            (100, 18, 120, 32), # C site back
            (100, 32, 105, 48), # C main
            (115, 32, 120, 48), # C mound
            (107, 38, 115, 44), # C default
            # Rotating doors (special - marked as open)
            # Mid
            (55, 55, 73, 72),   # Mid building
        ],
        'sunset': [
            # A site
            (12, 12, 35, 25),   # A site back
            (12, 25, 17, 42),   # A main
            (30, 25, 35, 42),   # A elbow
            (18, 30, 28, 37),   # A default
            # B site
            (93, 15, 116, 28),  # B site back
            (93, 28, 98, 45),   # B main
            (111, 28, 116, 45), # B market
            (100, 33, 108, 40), # B default
            # Mid
            (50, 35, 78, 52),   # Mid courtyard
            (55, 52, 60, 68),   # Mid tiles
            (68, 52, 73, 68),   # Mid boba
            # Spawn barriers
            (45, 80, 55, 95),   # A lobby
            (73, 80, 83, 95),   # B lobby
        ],
        'abyss': [
            # A site (note: abyss has fall-off areas)
            (15, 15, 38, 28),   # A site back
            (15, 28, 20, 45),   # A main
            (33, 28, 38, 45),   # A tower
            (21, 33, 31, 40),   # A default
            # B site
            (90, 18, 113, 30),  # B site back
            (90, 30, 95, 48),   # B main
            (108, 30, 113, 48), # B ramp
            (97, 35, 105, 42),  # B default
            # Mid (bridge areas - narrow)
            (52, 38, 76, 52),   # Mid bridge structure
            (55, 52, 60, 68),   # Mid left
            (68, 52, 73, 68),   # Mid right
            # Danger zones (fall areas) - wider walls to prevent pathing
            (0, 50, 10, 80),    # Left abyss
            (118, 50, 127, 80), # Right abyss
        ],
    }

    # Cover positions for each map - locations that provide partial cover
    # These affect combat but allow movement
    MAP_COVER_POSITIONS = {
        'ascent': [
            (0.25, 0.35), (0.35, 0.25), (0.45, 0.45),  # A side cover
            (0.75, 0.35), (0.65, 0.25), (0.55, 0.45),  # B side cover
            (0.50, 0.60), (0.45, 0.70), (0.55, 0.70),  # Mid cover
        ],
        # Add more maps as needed...
    }

    def load_nav_grid_from_v4(self, map_name: str) -> bool:
        """Load navigation grid from v4 walkable masks (recommended).

        Args:
            map_name: Name of the map (e.g., 'ascent', 'bind')

        Returns:
            True if loaded successfully
        """
        try:
            try:
                from app.services.map_context import get_map_context
            except ImportError:
                from backend.app.services.map_context import get_map_context
            ctx = get_map_context()
            self.nav_grid = ctx.get_nav_grid(map_name, self.GRID_SIZE)
            self._visibility_cache.clear()
            return True
        except Exception as e:
            print(f"Warning: Could not load v4 mask for {map_name}: {e}")
            # Fallback to hardcoded geometry
            return self.load_nav_grid(map_name)

    def load_nav_grid(self, map_name: str) -> bool:
        """Load navigation grid for a specific map (fallback to hardcoded geometry).

        Args:
            map_name: Name of the map (e.g., 'ascent', 'bind')

        Returns:
            True if loaded successfully
        """
        self.nav_grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.uint8)

        # Add basic map boundaries
        self.nav_grid[0, :] = 1  # Top wall
        self.nav_grid[-1, :] = 1  # Bottom wall
        self.nav_grid[:, 0] = 1  # Left wall
        self.nav_grid[:, -1] = 1  # Right wall

        # Add map-specific geometry
        geometry = self.MAP_GEOMETRY.get(map_name.lower(), [])
        for x1, y1, x2, y2 in geometry:
            # Clamp to grid bounds
            x1 = max(0, min(x1, self.GRID_SIZE - 1))
            y1 = max(0, min(y1, self.GRID_SIZE - 1))
            x2 = max(0, min(x2, self.GRID_SIZE - 1))
            y2 = max(0, min(y2, self.GRID_SIZE - 1))
            # Fill rectangular area
            self.nav_grid[y1:y2+1, x1:x2+1] = 1

        self._visibility_cache.clear()
        return True

    def get_cover_positions(self, map_name: str) -> List[Tuple[float, float]]:
        """Get cover positions for a map in normalized coordinates."""
        return self.MAP_COVER_POSITIONS.get(map_name.lower(), [])

    def normalized_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert normalized (0-1) coordinates to grid coordinates."""
        gx = min(max(int(x * self.GRID_SIZE), 0), self.GRID_SIZE - 1)
        gy = min(max(int(y * self.GRID_SIZE), 0), self.GRID_SIZE - 1)
        return (gx, gy)

    def grid_to_normalized(self, gx: int, gy: int) -> Tuple[float, float]:
        """Convert grid coordinates to normalized (0-1) coordinates."""
        return ((gx + 0.5) / self.GRID_SIZE, (gy + 0.5) / self.GRID_SIZE)

    def is_walkable(self, x: int, y: int) -> bool:
        """Check if a grid cell is walkable."""
        if 0 <= x < self.GRID_SIZE and 0 <= y < self.GRID_SIZE:
            return self.nav_grid[y, x] == 0
        return False

    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Octile distance heuristic (better for 8-directional movement)."""
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        return max(dx, dy) + (self.DIAGONAL_COST - 1) * min(dx, dy)

    def find_path(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
        avoid_positions: Optional[List[Tuple[float, float]]] = None
    ) -> PathResult:
        """Find path from start to goal using A*.

        Args:
            start: Starting position (normalized 0-1)
            goal: Goal position (normalized 0-1)
            avoid_positions: Optional positions to avoid (enemy positions)

        Returns:
            PathResult with path and metadata
        """
        start_grid = self.normalized_to_grid(start[0], start[1])
        goal_grid = self.normalized_to_grid(goal[0], goal[1])

        # Check if start/goal are walkable
        if not self.is_walkable(start_grid[0], start_grid[1]):
            return PathResult([], 0, [], False, "Start position is blocked")
        if not self.is_walkable(goal_grid[0], goal_grid[1]):
            return PathResult([], 0, [], False, "Goal position is blocked")

        # Build avoidance set (positions to avoid)
        avoid_set: Set[Tuple[int, int]] = set()
        if avoid_positions:
            for pos in avoid_positions:
                gx, gy = self.normalized_to_grid(pos[0], pos[1])
                # Avoid 3x3 area around each position
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        avoid_set.add((gx + dx, gy + dy))

        # A* algorithm
        open_set: List[Node] = []
        closed_set: Set[Tuple[int, int]] = set()
        start_node = Node(start_grid[0], start_grid[1], 0, self.heuristic(start_grid, goal_grid))
        heapq.heappush(open_set, start_node)
        node_map: Dict[Tuple[int, int], Node] = {(start_node.x, start_node.y): start_node}

        iterations = 0
        max_iterations = self.GRID_SIZE * self.GRID_SIZE

        while open_set and iterations < max_iterations:
            iterations += 1
            current = heapq.heappop(open_set)

            if (current.x, current.y) == goal_grid:
                # Reconstruct path
                path = []
                waypoints = []
                node = current
                while node:
                    waypoints.append((node.x, node.y))
                    path.append(self.grid_to_normalized(node.x, node.y))
                    node = node.parent
                path.reverse()
                waypoints.reverse()

                # Simplify path by removing collinear points
                simplified_path = self._simplify_path(path)

                return PathResult(
                    path=simplified_path,
                    distance=current.g,
                    waypoints=waypoints,
                    success=True
                )

            closed_set.add((current.x, current.y))

            # Explore neighbors
            for i, (dx, dy) in enumerate(self.DIRECTIONS):
                nx, ny = current.x + dx, current.y + dy

                if not self.is_walkable(nx, ny):
                    continue
                if (nx, ny) in closed_set:
                    continue

                # Add penalty for positions to avoid
                avoid_penalty = 10.0 if (nx, ny) in avoid_set else 0.0

                # Movement cost
                move_cost = self.DIAGONAL_COST if i >= 4 else self.CARDINAL_COST
                new_g = current.g + move_cost + avoid_penalty

                existing = node_map.get((nx, ny))
                if existing is None or new_g < existing.g:
                    neighbor = Node(
                        nx, ny,
                        new_g,
                        self.heuristic((nx, ny), goal_grid),
                        current
                    )
                    node_map[(nx, ny)] = neighbor
                    heapq.heappush(open_set, neighbor)

        return PathResult([], 0, [], False, "No path found")

    def _simplify_path(self, path: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Remove unnecessary waypoints from path using line-of-sight."""
        if len(path) <= 2:
            return path

        simplified = [path[0]]
        i = 0

        while i < len(path) - 1:
            # Find furthest visible point
            furthest = i + 1
            for j in range(i + 2, len(path)):
                if self.has_line_of_sight(path[i], path[j]):
                    furthest = j
            simplified.append(path[furthest])
            i = furthest

        return simplified

    def has_line_of_sight(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float]
    ) -> bool:
        """Check if there's clear line of sight between two points.

        Uses Bresenham's line algorithm to check for obstacles.
        """
        start_grid = self.normalized_to_grid(start[0], start[1])
        end_grid = self.normalized_to_grid(end[0], end[1])

        # Check cache
        cache_key = (start_grid[0], start_grid[1], end_grid[0], end_grid[1])
        if cache_key in self._visibility_cache:
            return self._visibility_cache[cache_key]

        # Bresenham's line algorithm
        x0, y0 = start_grid
        x1, y1 = end_grid

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            if not self.is_walkable(x0, y0):
                self._visibility_cache[cache_key] = False
                return False

            if x0 == x1 and y0 == y1:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

        self._visibility_cache[cache_key] = True
        return True

    def _line_intersects_circle(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        center: Tuple[float, float],
        radius: float
    ) -> bool:
        """Check if a line segment intersects a circle (for smoke collision).

        Uses geometric projection to find closest point on line to circle center.

        Args:
            start: Line segment start point (normalized)
            end: Line segment end point (normalized)
            center: Circle center (ability position)
            radius: Circle radius (ability effect radius)

        Returns:
            True if line passes through or touches the circle
        """
        # Vector from start to end
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        line_length_sq = dx * dx + dy * dy

        if line_length_sq == 0:
            # Degenerate case: start == end
            dist_sq = (start[0] - center[0])**2 + (start[1] - center[1])**2
            return dist_sq <= radius * radius

        # Project circle center onto line
        t = max(0, min(1, (
            (center[0] - start[0]) * dx +
            (center[1] - start[1]) * dy
        ) / line_length_sq))

        # Closest point on line segment
        closest_x = start[0] + t * dx
        closest_y = start[1] + t * dy

        # Distance from closest point to circle center
        dist_sq = (closest_x - center[0])**2 + (closest_y - center[1])**2

        return dist_sq <= radius * radius

    def has_line_of_sight_with_abilities(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        active_abilities: List[Any]
    ) -> bool:
        """Check LOS accounting for active smoke abilities blocking vision.

        First checks wall-based LOS, then checks if any active smokes
        with vision_block=True intersect the line of sight.

        Args:
            start: Start position (normalized 0-1)
            end: End position (normalized 0-1)
            active_abilities: List of ActiveAbility objects with:
                - ability.stats.vision_block: bool
                - position: Tuple[float, float]
                - ability.stats.radius: float (ability effect radius)

        Returns:
            True if clear LOS (no walls or smokes blocking)
        """
        # First check wall-based LOS
        if not self.has_line_of_sight(start, end):
            return False

        # Then check active smokes
        for active_ability in active_abilities:
            # Check if this ability blocks vision
            ability = getattr(active_ability, 'ability', None)
            if ability is None:
                continue

            vision_block = getattr(ability, 'vision_block', False)
            if not vision_block:
                continue

            # Get ability position and radius
            ability_pos = getattr(active_ability, 'position', None)
            if ability_pos is None:
                continue

            # Get radius - default smoke radius ~0.03 normalized (~300 units)
            ability_radius = getattr(ability, 'radius', 0.03)

            # Check if line passes through smoke
            if self._line_intersects_circle(start, end, ability_pos, ability_radius):
                return False

        return True

    def get_visible_positions(
        self,
        position: Tuple[float, float],
        max_distance: float = 0.3,
        num_rays: int = 36
    ) -> List[Tuple[float, float]]:
        """Get all positions visible from a given position.

        Args:
            position: Observer position (normalized 0-1)
            max_distance: Maximum visibility distance (normalized)
            num_rays: Number of rays to cast (more = more accurate but slower)

        Returns:
            List of visible positions
        """
        visible = []
        angle_step = 2 * math.pi / num_rays

        for i in range(num_rays):
            angle = i * angle_step
            # Cast ray
            for d in range(1, int(max_distance * self.GRID_SIZE)):
                norm_d = d / self.GRID_SIZE
                test_x = position[0] + norm_d * math.cos(angle)
                test_y = position[1] + norm_d * math.sin(angle)

                if not (0 <= test_x <= 1 and 0 <= test_y <= 1):
                    break

                grid_pos = self.normalized_to_grid(test_x, test_y)
                if not self.is_walkable(grid_pos[0], grid_pos[1]):
                    break

                if self.has_line_of_sight(position, (test_x, test_y)):
                    visible.append((test_x, test_y))

        return visible

    def calculate_exposure(
        self,
        position: Tuple[float, float],
        enemy_positions: List[Tuple[float, float]]
    ) -> float:
        """Calculate how exposed a position is to enemy sight lines.

        Args:
            position: Position to check (normalized 0-1)
            enemy_positions: List of enemy positions

        Returns:
            Exposure score (0 = safe, 1 = fully exposed)
        """
        if not enemy_positions:
            return 0.0

        visible_count = sum(
            1 for enemy in enemy_positions
            if self.has_line_of_sight(position, enemy)
        )

        return visible_count / len(enemy_positions)
