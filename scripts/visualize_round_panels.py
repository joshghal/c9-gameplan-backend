#!/usr/bin/env python3
"""
Visualize a simulation round with side-by-side panels for each 5-second snapshot.
Uses MAP_DATA from SimulationEngine for correct positions.
"""

import sys
import os
import random
import math
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Add paths
backend_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, backend_path)
project_path = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, project_path)

from backend.app.services.map_context import get_map_context
from backend.app.services.pathfinding import AStarPathfinder

# MAP_DATA with spawn positions from VCT heatmaps (vct_map_data/vct_positions_*.png)
# Site positions from cloud9-webapp (valorant-api.com)
# Coordinate system: (0,0) = top-left, (1,1) = bottom-right
ENGINE_MAP_DATA = {
    'ascent': {
        # Sites from cloud9-webapp: A (0.350, 0.142), B (0.285, 0.737)
        # VCT spawns: Attackers RIGHT (0.85, 0.58), Defenders LEFT (0.15, 0.42)
        'sites': {
            'A': {'center': (0.350, 0.142), 'radius': 0.07},
            'B': {'center': (0.285, 0.737), 'radius': 0.07},
        },
        'spawns': {
            'attack': [(0.85, 0.58), (0.82, 0.55), (0.82, 0.62), (0.80, 0.50), (0.80, 0.65)],
            'defense': [(0.15, 0.42), (0.20, 0.35), (0.20, 0.50), (0.35, 0.20), (0.30, 0.70)],
        },
    },
    'bind': {
        # Sites: A (0.734, 0.333), B (0.292, 0.312)
        # VCT spawns: Attackers BOTTOM (0.55, 0.90), Defenders TOP (0.55, 0.10)
        'sites': {
            'A': {'center': (0.734, 0.333), 'radius': 0.07},
            'B': {'center': (0.292, 0.312), 'radius': 0.07},
        },
        'spawns': {
            'attack': [(0.55, 0.88), (0.50, 0.85), (0.60, 0.85), (0.48, 0.82), (0.62, 0.82)],
            'defense': [(0.55, 0.12), (0.70, 0.25), (0.30, 0.25), (0.734, 0.333), (0.292, 0.312)],
        },
    },
    'split': {
        # Sites: A (0.315, 0.184), B (0.354, 0.867)
        # VCT spawns: Attackers RIGHT (0.85, 0.55), Defenders LEFT (0.12, 0.55)
        'sites': {
            'A': {'center': (0.315, 0.184), 'radius': 0.07},
            'B': {'center': (0.348, 0.855), 'radius': 0.07},
        },
        'spawns': {
            'attack': [(0.85, 0.55), (0.82, 0.50), (0.82, 0.60), (0.80, 0.45), (0.80, 0.65)],
            'defense': [(0.12, 0.55), (0.18, 0.40), (0.18, 0.70), (0.315, 0.25), (0.348, 0.80)],
        },
    },
    'icebox': {
        # Sites: A (0.691, 0.765), B (0.646, 0.180)
        # VCT spawns: Attackers LEFT (0.10, 0.58), Defenders RIGHT (0.92, 0.55)
        'sites': {
            'A': {'center': (0.691, 0.765), 'radius': 0.07},
            'B': {'center': (0.646, 0.180), 'radius': 0.07},
        },
        'spawns': {
            'attack': [(0.10, 0.58), (0.12, 0.52), (0.12, 0.64), (0.15, 0.48), (0.15, 0.68)],
            'defense': [(0.90, 0.55), (0.85, 0.45), (0.85, 0.65), (0.70, 0.25), (0.70, 0.75)],
        },
    },
    'breeze': {
        # Sites: A (0.908, 0.495), B (0.070, 0.382)
        'sites': {
            'A': {'center': (0.908, 0.495), 'radius': 0.09},
            'B': {'center': (0.070, 0.382), 'radius': 0.08},
        },
        'spawns': {
            'attack': [(0.50, 0.85), (0.45, 0.82), (0.55, 0.82), (0.42, 0.80), (0.58, 0.80)],
            'defense': [(0.50, 0.15), (0.85, 0.45), (0.12, 0.38), (0.908, 0.495), (0.070, 0.382)],
        },
    },
    'haven': {
        # Sites: A (0.402, 0.170), B (0.401, 0.501), C (0.418, 0.821)
        # VCT spawns: Attackers RIGHT (0.88, 0.55), Defenders LEFT (0.12, 0.40)
        'sites': {
            'A': {'center': (0.402, 0.170), 'radius': 0.07},
            'B': {'center': (0.401, 0.501), 'radius': 0.07},
            'C': {'center': (0.418, 0.821), 'radius': 0.07},
        },
        'spawns': {
            'attack': [(0.88, 0.55), (0.85, 0.48), (0.85, 0.62), (0.82, 0.42), (0.82, 0.68)],
            'defense': [(0.12, 0.40), (0.20, 0.25), (0.20, 0.55), (0.25, 0.75), (0.40, 0.50)],
        },
    },
    'lotus': {
        # Sites: A (0.855, 0.361), B (0.503, 0.459), C (0.148, 0.437)
        # VCT spawns: Attackers BOTTOM (0.50, 0.88), Defenders TOP (0.62, 0.18)
        'sites': {
            'A': {'center': (0.840, 0.379), 'radius': 0.07},
            'B': {'center': (0.503, 0.459), 'radius': 0.07},
            'C': {'center': (0.148, 0.437), 'radius': 0.07},
        },
        'spawns': {
            'attack': [(0.50, 0.85), (0.45, 0.82), (0.55, 0.82), (0.42, 0.80), (0.58, 0.80)],
            'defense': [(0.62, 0.18), (0.50, 0.25), (0.75, 0.35), (0.25, 0.42), (0.50, 0.45)],
        },
    },
}
DEFAULT_MAP_DATA = {
    'sites': {
        'A': {'center': (0.3, 0.25), 'radius': 0.08},
        'B': {'center': (0.7, 0.25), 'radius': 0.08},
    },
    'spawns': {
        'attack': [(0.85, 0.55), (0.82, 0.50), (0.82, 0.60), (0.80, 0.45), (0.80, 0.65)],
        'defense': [(0.15, 0.55), (0.18, 0.45), (0.18, 0.65), (0.30, 0.25), (0.30, 0.75)],
    },
}


TICK_MS = 100
ROUND_TIME_MS = 100000
SPIKE_TIME_MS = 45000
SNAPSHOT_INTERVAL_MS = 5000


@dataclass
class SimPlayer:
    id: str
    side: str
    x: float
    y: float
    is_alive: bool = True
    has_spike: bool = False
    target: Optional[Tuple[float, float]] = None
    path: List[Tuple[float, float]] = field(default_factory=list)
    path_idx: int = 0


@dataclass
class SimEvent:
    time_ms: int
    event_type: str
    x: float
    y: float
    player_id: str
    target_id: Optional[str] = None


class SimpleSimulation:
    """Uses MAP_DATA from SimulationEngine for correct positions."""

    def __init__(self, map_name: str = 'ascent'):
        self.map_name = map_name.lower()

        # Get MAP_DATA from ENGINE_MAP_DATA (copied from SimulationEngine)
        engine_data = ENGINE_MAP_DATA.get(self.map_name, DEFAULT_MAP_DATA)

        # Convert engine format to our format
        # Engine: {'sites': {'A': {'center': (x,y), 'radius': r}}, 'spawns': {'attack': [...], 'defense': [...]}}
        # Ours: {'sites': {'A': (x,y)}, 'attack_spawn': (x,y), 'defense_spawn': (x,y), 'defense_holds': [...]}
        self.map_data = {
            'sites': {
                site_name: site_info['center']
                for site_name, site_info in engine_data.get('sites', {}).items()
            },
            'attack_spawn': engine_data['spawns']['attack'][0],  # First spawn point
            'defense_spawn': engine_data['spawns']['defense'][0],  # First spawn point
            # Defense holds: all defender spawn points (they spread out to hold positions)
            'defense_holds': engine_data['spawns']['defense'][:5],
        }
        self.players: Dict[str, SimPlayer] = {}
        self.events: List[SimEvent] = []
        self.snapshots: List[Dict] = []
        self.current_time_ms = 0
        self.spike_planted = False
        self.spike_site = None
        self.spike_plant_time = 0
        self.target_site = None

        self.map_ctx = get_map_context()
        self.pathfinder = AStarPathfinder()
        self.pathfinder.load_nav_grid_from_v4(self.map_name)

        # Cache walkable mask for position validation
        self.walkable_mask = self.map_ctx.get_walkable_mask(self.map_name, grid_size=128)

    def _snap_to_walkable(self, x: float, y: float) -> Tuple[float, float]:
        """Snap a position to the nearest walkable cell using v4 mask."""
        grid_size = self.walkable_mask.shape[0]
        gx = int(x * grid_size)
        gy = int(y * grid_size)

        # Check if already walkable
        gx = max(0, min(grid_size - 1, gx))
        gy = max(0, min(grid_size - 1, gy))
        if self.walkable_mask[gy, gx] == 1:
            return (x, y)

        # Spiral search for nearest walkable
        for radius in range(1, 30):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if abs(dx) != radius and abs(dy) != radius:
                        continue
                    nx, ny = gx + dx, gy + dy
                    if 0 <= nx < grid_size and 0 <= ny < grid_size:
                        if self.walkable_mask[ny, nx] == 1:
                            return ((nx + 0.5) / grid_size, (ny + 0.5) / grid_size)

        return (x, y)  # Fallback

    def initialize(self):
        # Snap spawn positions to walkable areas using v4 mask
        attack_spawn = self._snap_to_walkable(*self.map_data['attack_spawn'])
        defense_spawn = self._snap_to_walkable(*self.map_data['defense_spawn'])

        # Snap defense hold positions
        defense_holds = [self._snap_to_walkable(*pos) for pos in self.map_data.get('defense_holds', [])]
        self.map_data['defense_holds'] = defense_holds

        for i in range(5):
            # Small cluster around spawn point
            offset_x = (i % 3 - 1) * 0.015
            offset_y = (i // 3) * 0.015
            pos = self._snap_to_walkable(attack_spawn[0] + offset_x, attack_spawn[1] + offset_y)
            self.players[f'atk_{i}'] = SimPlayer(
                id=f'atk_{i}', side='attack',
                x=pos[0], y=pos[1],
                has_spike=(i == 0)
            )

        for i in range(5):
            # Small cluster around spawn point
            offset_x = (i % 3 - 1) * 0.015
            offset_y = (i // 3) * 0.015
            pos = self._snap_to_walkable(defense_spawn[0] + offset_x, defense_spawn[1] + offset_y)
            self.players[f'def_{i}'] = SimPlayer(
                id=f'def_{i}', side='defense',
                x=pos[0], y=pos[1]
            )

        self.target_site = random.choice(list(self.map_data['sites'].keys()))

        # Also snap target site to walkable
        site_pos = self.map_data['sites'][self.target_site]
        self.map_data['sites'][self.target_site] = self._snap_to_walkable(*site_pos)

        self._take_snapshot()

    def _take_snapshot(self):
        # Get events that happened since last snapshot
        last_time = self.snapshots[-1]['time_ms'] if self.snapshots else 0
        recent_events = [e for e in self.events if e.time_ms > last_time]

        self.snapshots.append({
            'time_ms': self.current_time_ms,
            'positions': {
                pid: {'x': p.x, 'y': p.y, 'alive': p.is_alive, 'side': p.side, 'has_spike': p.has_spike}
                for pid, p in self.players.items()
            },
            'spike_planted': self.spike_planted,
            'spike_site': self.spike_site,
            'events': recent_events,
            'attack_alive': sum(1 for p in self.players.values() if p.side == 'attack' and p.is_alive),
            'defense_alive': sum(1 for p in self.players.values() if p.side == 'defense' and p.is_alive),
        })

    def _move_player(self, player: SimPlayer, target: Tuple[float, float], speed: float = 0.012):
        if not player.is_alive:
            return

        if not player.path or player.path_idx >= len(player.path) or player.target != target:
            player.target = target
            result = self.pathfinder.find_path((player.x, player.y), target)
            if result.success and result.path:
                # Use 'path' (normalized 0-1), not 'waypoints' (grid 0-127)
                player.path = result.path
                player.path_idx = 0
            else:
                player.path = [target]
                player.path_idx = 0

        if player.path_idx < len(player.path):
            next_point = player.path[player.path_idx]
            dx = next_point[0] - player.x
            dy = next_point[1] - player.y
            dist = math.sqrt(dx*dx + dy*dy)

            if dist < speed:
                player.x, player.y = next_point
                player.path_idx += 1
            else:
                player.x += (dx / dist) * speed
                player.y += (dy / dist) * speed

        # Snap to walkable area
        player.x, player.y = self._snap_to_walkable(player.x, player.y)

    def _check_combat(self):
        attackers = [p for p in self.players.values() if p.side == 'attack' and p.is_alive]
        defenders = [p for p in self.players.values() if p.side == 'defense' and p.is_alive]

        for atk in attackers:
            for dfn in defenders:
                dist = math.sqrt((atk.x - dfn.x)**2 + (atk.y - dfn.y)**2)

                # Combat ranges: close (high chance), medium (lower chance)
                if dist < 0.08:
                    kill_chance = 0.04  # 4% per tick at close range
                elif dist < 0.15:
                    kill_chance = 0.015  # 1.5% per tick at medium range
                else:
                    continue

                has_los = self.map_ctx.has_line_of_sight(self.map_name, atk.x, atk.y, dfn.x, dfn.y)
                if has_los and random.random() < kill_chance:
                    # Defender has slight advantage (holding angle)
                    if random.random() < 0.45:
                        dfn.is_alive = False
                        self.events.append(SimEvent(self.current_time_ms, 'kill', dfn.x, dfn.y, atk.id, dfn.id))
                    else:
                        atk.is_alive = False
                        self.events.append(SimEvent(self.current_time_ms, 'kill', atk.x, atk.y, dfn.id, atk.id))
                    return

    def _check_plant(self):
        if self.spike_planted:
            return
        carrier = next((p for p in self.players.values() if p.has_spike and p.is_alive), None)
        if not carrier:
            return
        site_pos = self.map_data['sites'].get(self.target_site)
        if site_pos:
            dist = math.sqrt((carrier.x - site_pos[0])**2 + (carrier.y - site_pos[1])**2)
            if dist < 0.08 and random.random() < 0.02:
                self.spike_planted = True
                self.spike_site = self.target_site
                self.spike_plant_time = self.current_time_ms
                self.events.append(SimEvent(self.current_time_ms, 'plant', carrier.x, carrier.y, carrier.id))

    def run(self) -> Dict:
        self.initialize()
        target_pos = self.map_data['sites'][self.target_site]

        while self.current_time_ms < ROUND_TIME_MS:
            self.current_time_ms += TICK_MS

            alive_attack = sum(1 for p in self.players.values() if p.side == 'attack' and p.is_alive)
            alive_defense = sum(1 for p in self.players.values() if p.side == 'defense' and p.is_alive)

            if alive_attack == 0 or alive_defense == 0:
                break
            if self.spike_planted and self.current_time_ms - self.spike_plant_time > SPIKE_TIME_MS:
                break

            for pid, player in self.players.items():
                if player.side == 'attack' and player.is_alive:
                    # Only add jitter occasionally to avoid constant path recalculation
                    if random.random() < 0.05:  # 5% chance per tick
                        jitter = (random.gauss(0, 0.02), random.gauss(0, 0.02))
                        target = (target_pos[0] + jitter[0], target_pos[1] + jitter[1])
                    else:
                        target = target_pos
                    self._move_player(player, target, speed=0.008)

                elif player.side == 'defense' and player.is_alive:
                    player_idx = int(pid.split('_')[1])
                    defense_holds = self.map_data.get('defense_holds', [])
                    hold_pos = defense_holds[player_idx] if player_idx < len(defense_holds) else self.map_data['defense_spawn']

                    # Get all alive attackers
                    alive_attackers = [p for p in self.players.values() if p.side == 'attack' and p.is_alive]

                    # Find nearest attacker
                    nearest_atk = None
                    nearest_dist = float('inf')
                    for atk in alive_attackers:
                        dist = math.sqrt((atk.x - player.x)**2 + (atk.y - player.y)**2)
                        if dist < nearest_dist:
                            nearest_dist = dist
                            nearest_atk = atk

                    if self.spike_planted and self.spike_site:
                        # Phase 3: After plant - all rotate to site for retake
                        site_pos = self.map_data['sites'][self.spike_site]
                        self._move_player(player, site_pos, speed=0.008)

                    elif self.current_time_ms < 12000:
                        # Phase 1: Setup (0-12s) - move from spawn to hold positions
                        self._move_player(player, hold_pos, speed=0.007)

                    elif nearest_atk and nearest_dist < 0.15:
                        # Close contact - engage aggressively
                        self._move_player(player, (nearest_atk.x, nearest_atk.y), speed=0.006)

                    elif nearest_atk and nearest_dist < 0.35:
                        # Medium range - move to intercept
                        # Move toward a point between current position and attacker
                        intercept_x = player.x + (nearest_atk.x - player.x) * 0.5
                        intercept_y = player.y + (nearest_atk.y - player.y) * 0.5
                        self._move_player(player, (intercept_x, intercept_y), speed=0.005)

                    else:
                        # No immediate threat - rotate/roam based on role
                        if player_idx in [0, 1]:
                            # Site anchors - hold position but micro-adjust
                            jitter = (random.gauss(0, 0.02), random.gauss(0, 0.02))
                            adjusted_hold = (hold_pos[0] + jitter[0], hold_pos[1] + jitter[1])
                            self._move_player(player, adjusted_hold, speed=0.003)

                        elif player_idx == 2:
                            # Mid player - patrol between sites
                            if random.random() < 0.03:
                                sites = list(self.map_data['sites'].values())
                                patrol_target = random.choice(sites)
                                # Move toward site but not all the way
                                mid_x = (player.x + patrol_target[0]) / 2
                                mid_y = (player.y + patrol_target[1]) / 2
                                self._move_player(player, (mid_x, mid_y), speed=0.004)
                            else:
                                self._move_player(player, hold_pos, speed=0.003)

                        else:
                            # Flex/Roamer - push for info periodically
                            if random.random() < 0.05:
                                # Push toward attackers' likely path
                                push_target = (
                                    hold_pos[0] + random.gauss(0, 0.08),
                                    hold_pos[1] + random.gauss(0, 0.08)
                                )
                                self._move_player(player, push_target, speed=0.005)
                            elif random.random() < 0.02:
                                # Rotate toward target site to help
                                self._move_player(player, target_pos, speed=0.004)
                            else:
                                self._move_player(player, hold_pos, speed=0.003)

            self._check_combat()
            self._check_plant()

            if self.current_time_ms % SNAPSHOT_INTERVAL_MS < TICK_MS:
                self._take_snapshot()

        if self.snapshots[-1]['time_ms'] != self.current_time_ms:
            self._take_snapshot()

        alive_attack = sum(1 for p in self.players.values() if p.side == 'attack' and p.is_alive)
        alive_defense = sum(1 for p in self.players.values() if p.side == 'defense' and p.is_alive)

        if self.spike_planted and alive_attack > 0:
            winner = 'attack'
        elif alive_attack == 0:
            winner = 'defense'
        elif alive_defense == 0:
            winner = 'attack'
        else:
            winner = 'defense'

        return {
            'winner': winner,
            'duration_ms': self.current_time_ms,
            'snapshots': self.snapshots,
            'events': self.events,
            'spike_planted': self.spike_planted,
            'spike_site': self.spike_site,
            'target_site': self.target_site,
        }


def draw_panel(bg_img, snapshot, panel_size, font, font_small, all_events_so_far):
    """Draw a single panel for one snapshot."""
    panel = bg_img.copy()
    draw = ImageDraw.Draw(panel)

    time_sec = snapshot['time_ms'] / 1000

    # Draw all events that happened up to this point
    for event in all_events_so_far:
        if event.time_ms <= snapshot['time_ms']:
            px, py = int(event.x * panel_size), int(event.y * panel_size)
            if event.event_type == 'kill':
                size = 6
                draw.line([(px-size, py-size), (px+size, py+size)], fill=(255, 255, 0), width=3)
                draw.line([(px-size, py+size), (px+size, py-size)], fill=(255, 255, 0), width=3)
            elif event.event_type == 'plant':
                size = 8
                draw.rectangle([px-size, py-size, px+size, py+size], outline=(0, 255, 0), width=2)

    # Draw players
    for pid, pos in snapshot['positions'].items():
        px, py = int(pos['x'] * panel_size), int(pos['y'] * panel_size)

        if pos['alive']:
            if pos['side'] == 'attack':
                color = (255, 80, 80)
                outline = (200, 40, 40)
            else:
                color = (80, 150, 255)
                outline = (40, 100, 200)

            radius = 6
            draw.ellipse([px-radius, py-radius, px+radius, py+radius], fill=color, outline=outline, width=2)

            # Mark spike carrier
            if pos.get('has_spike') and pos['side'] == 'attack':
                draw.rectangle([px-3, py-3, px+3, py+3], fill=(255, 255, 0))
        else:
            # Dead player - small gray X
            draw.line([(px-4, py-4), (px+4, py+4)], fill=(100, 100, 100), width=2)
            draw.line([(px-4, py+4), (px+4, py-4)], fill=(100, 100, 100), width=2)

    # Add time label
    label = f"{time_sec:.0f}s"
    draw.rectangle([5, 5, 50, 25], fill=(0, 0, 0, 180))
    draw.text((10, 8), label, fill=(255, 255, 255), font=font)

    # Add alive count
    atk_alive = snapshot['attack_alive']
    def_alive = snapshot['defense_alive']
    count_label = f"{atk_alive}v{def_alive}"
    draw.rectangle([panel_size-50, 5, panel_size-5, 25], fill=(0, 0, 0, 180))
    draw.text((panel_size-45, 8), count_label, fill=(255, 255, 255), font=font)

    # Mark if spike planted
    if snapshot['spike_planted']:
        draw.rectangle([5, panel_size-25, 80, panel_size-5], fill=(0, 100, 0, 200))
        draw.text((10, panel_size-22), f"PLANTED", fill=(0, 255, 0), font=font_small)

    return panel


def visualize_round_panels(map_name: str = 'ascent', output_path: str = None):
    """Run simulation and create panel visualization."""

    print(f"Running simulation on {map_name}...")
    sim = SimpleSimulation(map_name)
    result = sim.run()

    print(f"  Target: {result['target_site']}, Duration: {result['duration_ms']/1000:.1f}s")
    print(f"  Winner: {result['winner']}, Kills: {len([e for e in result['events'] if e.event_type == 'kill'])}")
    print(f"  Snapshots: {len(result['snapshots'])}")

    # Load map image
    map_image_path = Path(__file__).parent.parent.parent / 'map_images' / f'{map_name}.png'
    if map_image_path.exists():
        bg_img = Image.open(map_image_path).convert('RGBA')
    else:
        bg_img = Image.new('RGBA', (400, 400), (40, 40, 40, 255))

    # Panel size
    panel_size = 250
    bg_img = bg_img.resize((panel_size, panel_size), Image.LANCZOS)

    # Make background slightly darker for contrast
    darkened = Image.new('RGBA', (panel_size, panel_size), (0, 0, 0, 60))
    bg_img = Image.alpha_composite(bg_img, darkened)

    # Load fonts
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 10)
        font_title = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except:
        font = ImageFont.load_default()
        font_small = font
        font_title = font

    snapshots = result['snapshots']
    all_events = result['events']

    # Calculate grid layout
    n_panels = len(snapshots)
    cols = min(6, n_panels)  # Max 6 columns
    rows = math.ceil(n_panels / cols)

    # Create panels
    panels = []
    for snap in snapshots:
        panel = draw_panel(bg_img.copy(), snap, panel_size, font, font_small, all_events)
        panels.append(panel)

    # Create final image
    padding = 5
    header_height = 50
    legend_height = 40

    total_width = cols * panel_size + (cols + 1) * padding
    total_height = header_height + rows * panel_size + (rows + 1) * padding + legend_height

    final_img = Image.new('RGBA', (total_width, total_height), (25, 25, 25, 255))
    draw = ImageDraw.Draw(final_img)

    # Header
    header_text = f"{map_name.upper()} - Target: {result['target_site']} - Duration: {result['duration_ms']/1000:.1f}s - Winner: {result['winner'].upper()}"
    draw.text((padding, 15), header_text, fill=(255, 255, 255), font=font_title)

    # Place panels
    for i, panel in enumerate(panels):
        row = i // cols
        col = i % cols
        x = padding + col * (panel_size + padding)
        y = header_height + padding + row * (panel_size + padding)
        final_img.paste(panel, (x, y))

    # Legend
    legend_y = header_height + rows * (panel_size + padding) + padding + 5
    draw.ellipse([padding, legend_y, padding+12, legend_y+12], fill=(255, 80, 80))
    draw.text((padding+18, legend_y-2), "Attacker", fill=(255, 255, 255), font=font_small)

    draw.ellipse([padding+90, legend_y, padding+102, legend_y+12], fill=(80, 150, 255))
    draw.text((padding+108, legend_y-2), "Defender", fill=(255, 255, 255), font=font_small)

    draw.line([(padding+190, legend_y), (padding+202, legend_y+12)], fill=(255, 255, 0), width=2)
    draw.line([(padding+190, legend_y+12), (padding+202, legend_y)], fill=(255, 255, 0), width=2)
    draw.text((padding+208, legend_y-2), "Kill", fill=(255, 255, 255), font=font_small)

    draw.rectangle([padding+260, legend_y, padding+272, legend_y+12], outline=(0, 255, 0), width=2)
    draw.text((padding+278, legend_y-2), "Plant", fill=(255, 255, 255), font=font_small)

    draw.rectangle([padding+340, legend_y+2, padding+348, legend_y+10], fill=(255, 255, 0))
    draw.text((padding+354, legend_y-2), "Spike Carrier", fill=(255, 255, 255), font=font_small)

    # Save
    if output_path is None:
        output_path = Path(__file__).parent.parent.parent / f'round_panels_{map_name}.png'

    final_img.save(output_path)
    print(f"  Saved: {output_path}")

    return str(output_path)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Visualize simulation round with panels')
    parser.add_argument('--map', '-m', default='ascent', help='Map name')
    parser.add_argument('--all', '-a', action='store_true', help='Generate for all maps')
    args = parser.parse_args()

    if args.all:
        maps = ['ascent', 'haven', 'bind', 'split', 'icebox', 'lotus']
        for map_name in maps:
            print(f"\n{'='*60}")
            visualize_round_panels(map_name)
    else:
        visualize_round_panels(args.map)


if __name__ == '__main__':
    main()
