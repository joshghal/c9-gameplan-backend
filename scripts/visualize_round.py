#!/usr/bin/env python3
"""
Visualize a simulation round with player positions plotted every 5 seconds.

Creates an image showing:
- Map image as background
- Player positions at 5-second intervals with time labels
- Movement trails
- Events (kills, plants, etc.) marked on the map
"""

import sys
import os
import json
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


# Constants
TICK_MS = 100  # ms per tick (slower for visualization)
ROUND_TIME_MS = 100000  # 1:40 round time
SPIKE_TIME_MS = 45000  # 45 second spike timer
SNAPSHOT_INTERVAL_MS = 5000  # 5 seconds between snapshots


@dataclass
class SimPlayer:
    """Simplified player for visualization."""
    id: str
    side: str  # 'attack' or 'defense'
    x: float
    y: float
    is_alive: bool = True
    health: int = 100
    has_spike: bool = False
    target: Optional[Tuple[float, float]] = None
    path: List[Tuple[float, float]] = field(default_factory=list)
    path_idx: int = 0


@dataclass
class SimEvent:
    """Simulation event."""
    time_ms: int
    event_type: str  # 'kill', 'death', 'plant', 'defuse'
    x: float
    y: float
    player_id: str
    target_id: Optional[str] = None


class SimpleSimulation:
    """Simplified simulation for visualization purposes.

    Coordinates:
    - Origin (0,0) = top-left
    - (1,1) = bottom-right
    - Y increases downward
    """

    # Map-specific data based on VCT player heatmaps
    # Sites are where VCT shows highest player density (plant/defend positions)
    MAP_DATA = {
        'ascent': {
            # VCT heatmap shows A site hotspot at left (~0.10, 0.40)
            # B site hotspot at right (~0.88, 0.55)
            'sites': {'A': (0.12, 0.38), 'B': (0.85, 0.55)},
            # Attackers spawn at bottom, defenders near sites
            'attack_spawn': (0.50, 0.88),
            'defense_spawn': (0.50, 0.45),
            'attack_spread': [(0.45, 0.90), (0.55, 0.90), (0.40, 0.88), (0.60, 0.88), (0.50, 0.92)],
            'defense_spread': [(0.15, 0.40), (0.20, 0.45), (0.80, 0.52), (0.85, 0.58), (0.50, 0.35)],
        },
        'haven': {
            # 3-site map: A left, B mid, C right
            'sites': {'A': (0.15, 0.35), 'B': (0.50, 0.25), 'C': (0.85, 0.35)},
            'attack_spawn': (0.50, 0.88),
            'defense_spawn': (0.50, 0.45),
            'attack_spread': [(0.45, 0.90), (0.55, 0.90), (0.40, 0.88), (0.60, 0.88), (0.50, 0.92)],
            'defense_spread': [(0.18, 0.38), (0.50, 0.28), (0.82, 0.38), (0.35, 0.45), (0.65, 0.45)],
        },
        'bind': {
            'sites': {'A': (0.20, 0.35), 'B': (0.80, 0.40)},
            'attack_spawn': (0.50, 0.88),
            'defense_spawn': (0.50, 0.50),
            'attack_spread': [(0.45, 0.90), (0.55, 0.90), (0.40, 0.88), (0.60, 0.88), (0.50, 0.92)],
            'defense_spread': [(0.22, 0.38), (0.25, 0.45), (0.78, 0.42), (0.75, 0.50), (0.50, 0.45)],
        },
        'split': {
            'sites': {'A': (0.20, 0.30), 'B': (0.80, 0.35)},
            'attack_spawn': (0.50, 0.90),
            'defense_spawn': (0.50, 0.50),
            'attack_spread': [(0.45, 0.92), (0.55, 0.92), (0.40, 0.90), (0.60, 0.90), (0.50, 0.94)],
            'defense_spread': [(0.22, 0.32), (0.25, 0.40), (0.78, 0.38), (0.75, 0.45), (0.50, 0.35)],
        },
        'icebox': {
            'sites': {'A': (0.25, 0.30), 'B': (0.75, 0.40)},
            'attack_spawn': (0.50, 0.85),
            'defense_spawn': (0.50, 0.50),
            'attack_spread': [(0.45, 0.87), (0.55, 0.87), (0.40, 0.85), (0.60, 0.85), (0.50, 0.89)],
            'defense_spread': [(0.27, 0.32), (0.30, 0.40), (0.73, 0.42), (0.70, 0.50), (0.50, 0.38)],
        },
        'lotus': {
            # 3-site map
            'sites': {'A': (0.18, 0.35), 'B': (0.50, 0.28), 'C': (0.82, 0.35)},
            'attack_spawn': (0.50, 0.85),
            'defense_spawn': (0.50, 0.50),
            'attack_spread': [(0.45, 0.87), (0.55, 0.87), (0.40, 0.85), (0.60, 0.85), (0.50, 0.89)],
            'defense_spread': [(0.20, 0.38), (0.50, 0.30), (0.80, 0.38), (0.35, 0.48), (0.65, 0.48)],
        },
    }

    def __init__(self, map_name: str = 'ascent'):
        self.map_name = map_name.lower()
        self.map_data = self.MAP_DATA.get(self.map_name, self.MAP_DATA['ascent'])

        self.players: Dict[str, SimPlayer] = {}
        self.events: List[SimEvent] = []
        self.snapshots: List[Dict] = []

        self.current_time_ms = 0
        self.spike_planted = False
        self.spike_site = None
        self.spike_plant_time = 0
        self.target_site = None

        # Load map context and pathfinder
        self.map_ctx = get_map_context()
        self.pathfinder = AStarPathfinder()
        self.pathfinder.load_nav_grid_from_v4(self.map_name)

    def initialize(self):
        """Initialize players at spawn positions."""
        attack_spread = self.map_data.get('attack_spread', [self.map_data['attack_spawn']] * 5)
        defense_spread = self.map_data.get('defense_spread', [self.map_data['defense_spawn']] * 5)

        # Create 5 attackers
        for i in range(5):
            spawn = attack_spread[i % len(attack_spread)]
            self.players[f'atk_{i}'] = SimPlayer(
                id=f'atk_{i}',
                side='attack',
                x=spawn[0],
                y=spawn[1],
                has_spike=(i == 0)
            )

        # Create 5 defenders
        for i in range(5):
            spawn = defense_spread[i % len(defense_spread)]
            self.players[f'def_{i}'] = SimPlayer(
                id=f'def_{i}',
                side='defense',
                x=spawn[0],
                y=spawn[1],
            )

        # Pick target site for attackers
        self.target_site = random.choice(list(self.map_data['sites'].keys()))

        # Take initial snapshot
        self._take_snapshot()

    def _take_snapshot(self):
        """Record current positions."""
        snapshot = {
            'time_ms': self.current_time_ms,
            'positions': {
                pid: {'x': p.x, 'y': p.y, 'alive': p.is_alive, 'side': p.side}
                for pid, p in self.players.items()
            },
            'spike_planted': self.spike_planted,
            'spike_site': self.spike_site,
        }
        self.snapshots.append(snapshot)

    def _move_player(self, player: SimPlayer, target: Tuple[float, float], speed: float = 0.012):
        """Move player toward target using pathfinding."""
        if not player.is_alive:
            return

        # If no path or target changed, calculate new path
        if not player.path or player.path_idx >= len(player.path) or player.target != target:
            player.target = target
            result = self.pathfinder.find_path((player.x, player.y), target)
            if result.success and result.waypoints:
                player.path = result.waypoints
                player.path_idx = 0
            else:
                # Fallback: move directly (will be clamped)
                player.path = [target]
                player.path_idx = 0

        # Move along path
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

        # Validate position
        player.x = max(0.05, min(0.95, player.x))
        player.y = max(0.05, min(0.95, player.y))

    def _check_combat(self):
        """Simple combat resolution."""
        attackers = [p for p in self.players.values() if p.side == 'attack' and p.is_alive]
        defenders = [p for p in self.players.values() if p.side == 'defense' and p.is_alive]

        for atk in attackers:
            for dfn in defenders:
                dist = math.sqrt((atk.x - dfn.x)**2 + (atk.y - dfn.y)**2)

                if dist < 0.12:  # Combat range
                    has_los = self.map_ctx.has_line_of_sight(
                        self.map_name, atk.x, atk.y, dfn.x, dfn.y
                    )

                    if has_los and random.random() < 0.005:  # 0.5% per tick
                        if random.random() < 0.5:
                            dfn.is_alive = False
                            self.events.append(SimEvent(
                                time_ms=self.current_time_ms,
                                event_type='kill',
                                x=dfn.x, y=dfn.y,
                                player_id=atk.id,
                                target_id=dfn.id
                            ))
                        else:
                            atk.is_alive = False
                            self.events.append(SimEvent(
                                time_ms=self.current_time_ms,
                                event_type='kill',
                                x=atk.x, y=atk.y,
                                player_id=dfn.id,
                                target_id=atk.id
                            ))
                        return

    def _check_plant(self):
        """Check if spike can be planted."""
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
                self.events.append(SimEvent(
                    time_ms=self.current_time_ms,
                    event_type='plant',
                    x=carrier.x, y=carrier.y,
                    player_id=carrier.id
                ))

    def run(self) -> Dict:
        """Run the full simulation."""
        self.initialize()

        target_pos = self.map_data['sites'][self.target_site]

        while self.current_time_ms < ROUND_TIME_MS:
            self.current_time_ms += TICK_MS

            # Check round end
            alive_attack = sum(1 for p in self.players.values() if p.side == 'attack' and p.is_alive)
            alive_defense = sum(1 for p in self.players.values() if p.side == 'defense' and p.is_alive)

            if alive_attack == 0 or alive_defense == 0:
                break

            if self.spike_planted and self.current_time_ms - self.spike_plant_time > SPIKE_TIME_MS:
                break

            # Move attackers toward site
            for pid, player in self.players.items():
                if player.side == 'attack' and player.is_alive:
                    # Add slight variation
                    jitter = (random.gauss(0, 0.01), random.gauss(0, 0.01))
                    target = (target_pos[0] + jitter[0], target_pos[1] + jitter[1])
                    self._move_player(player, target, speed=0.008)

                elif player.side == 'defense' and player.is_alive:
                    # Defenders hold or rotate to site if spike planted
                    if self.spike_planted and self.spike_site:
                        site_pos = self.map_data['sites'][self.spike_site]
                        self._move_player(player, site_pos, speed=0.006)

            self._check_combat()
            self._check_plant()

            # Snapshot every 5 seconds
            if self.current_time_ms % SNAPSHOT_INTERVAL_MS < TICK_MS:
                self._take_snapshot()

        # Final snapshot
        if self.snapshots[-1]['time_ms'] != self.current_time_ms:
            self._take_snapshot()

        # Determine winner
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


def visualize_round(map_name: str = 'ascent', output_path: str = None):
    """Run simulation and create visualization."""

    print(f"Running simulation on {map_name}...")
    sim = SimpleSimulation(map_name)
    result = sim.run()

    print(f"  Target site: {result['target_site']}")
    print(f"  Duration: {result['duration_ms']}ms ({result['duration_ms']/1000:.1f}s)")
    print(f"  Winner: {result['winner']}")
    print(f"  Spike planted: {result['spike_planted']} at {result['spike_site']}")
    print(f"  Events: {len(result['events'])}")
    print(f"  Snapshots: {len(result['snapshots'])}")

    # Load map image
    map_image_path = Path(__file__).parent.parent.parent / 'map_images' / f'{map_name}.png'
    if map_image_path.exists():
        bg_img = Image.open(map_image_path).convert('RGBA')
    else:
        bg_img = Image.new('RGBA', (800, 800), (40, 40, 40, 255))

    img_size = 800
    bg_img = bg_img.resize((img_size, img_size), Image.LANCZOS)

    # Create overlay
    overlay = Image.new('RGBA', (img_size, img_size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Load font
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 11)
    except:
        font = ImageFont.load_default()
        font_small = font

    snapshots = result['snapshots']

    # Draw trails first (so markers are on top)
    trails = {}
    for snap in snapshots:
        for pid, pos in snap['positions'].items():
            if pid not in trails:
                trails[pid] = []
            if pos['alive']:
                trails[pid].append((pos['x'], pos['y'], pos['side']))

    for pid, positions in trails.items():
        if len(positions) < 2:
            continue
        side = positions[0][2]
        trail_color = (255, 100, 100, 80) if side == 'attack' else (100, 150, 255, 80)

        for i in range(len(positions) - 1):
            x1, y1, _ = positions[i]
            x2, y2, _ = positions[i + 1]
            px1, py1 = int(x1 * img_size), int(y1 * img_size)
            px2, py2 = int(x2 * img_size), int(y2 * img_size)
            draw.line([(px1, py1), (px2, py2)], fill=trail_color, width=3)

    # Draw markers - time labels every 10s, dots for others
    for snap_idx, snap in enumerate(snapshots):
        time_sec = int(snap['time_ms'] / 1000)
        show_label = (time_sec % 10 == 0) or snap_idx == 0 or snap_idx == len(snapshots) - 1

        attackers = [(pid, pos) for pid, pos in snap['positions'].items()
                     if pos['side'] == 'attack' and pos['alive']]
        defenders = [(pid, pos) for pid, pos in snap['positions'].items()
                     if pos['side'] == 'defense' and pos['alive']]

        # Attackers
        for i, (pid, pos) in enumerate(attackers):
            px, py = int(pos['x'] * img_size), int(pos['y'] * img_size)

            if i == 0 and show_label:
                label = f"{time_sec}s"
                bbox = draw.textbbox((px, py), label, font=font)
                pad = 3
                draw.rectangle([bbox[0]-pad, bbox[1]-pad, bbox[2]+pad, bbox[3]+pad],
                              fill=(150, 30, 30, 220), outline=(255, 100, 100))
                draw.text((px, py), label, fill=(255, 255, 255), font=font, anchor="mm")
            else:
                draw.ellipse([px-5, py-5, px+5, py+5], fill=(255, 100, 100, 200), outline=(200, 50, 50))

        # Defenders
        for i, (pid, pos) in enumerate(defenders):
            px, py = int(pos['x'] * img_size), int(pos['y'] * img_size)

            if i == 0 and show_label:
                label = f"{time_sec}s"
                bbox = draw.textbbox((px, py), label, font=font)
                pad = 3
                draw.rectangle([bbox[0]-pad, bbox[1]-pad, bbox[2]+pad, bbox[3]+pad],
                              fill=(30, 50, 150, 220), outline=(100, 150, 255))
                draw.text((px, py), label, fill=(255, 255, 255), font=font, anchor="mm")
            else:
                draw.ellipse([px-5, py-5, px+5, py+5], fill=(100, 150, 255, 200), outline=(50, 100, 200))

    # Draw events
    for event in result['events']:
        px, py = int(event.x * img_size), int(event.y * img_size)

        if event.event_type == 'kill':
            size = 10
            draw.line([(px-size, py-size), (px+size, py+size)], fill=(255, 255, 0), width=4)
            draw.line([(px-size, py+size), (px+size, py-size)], fill=(255, 255, 0), width=4)
        elif event.event_type == 'plant':
            size = 12
            draw.rectangle([px-size, py-size, px+size, py+size], outline=(0, 255, 0), width=3)

    # Composite
    result_img = Image.alpha_composite(bg_img, overlay)

    # Legend
    legend_height = 80
    legend = Image.new('RGBA', (img_size, legend_height), (30, 30, 30, 255))
    legend_draw = ImageDraw.Draw(legend)

    # Info
    legend_draw.text((10, 8), f"Map: {map_name.upper()}", fill=(255, 255, 255), font=font)
    legend_draw.text((180, 8), f"Duration: {result['duration_ms']/1000:.1f}s", fill=(255, 255, 255), font=font)
    legend_draw.text((350, 8), f"Winner: {result['winner'].upper()}", fill=(255, 255, 255), font=font)
    legend_draw.text((520, 8), f"Target: {result['target_site']}", fill=(255, 255, 255), font=font)

    # Legend items
    y = 40
    legend_draw.ellipse([10, y, 22, y+12], fill=(255, 100, 100))
    legend_draw.text((28, y-2), "Attackers", fill=(255, 255, 255), font=font_small)

    legend_draw.ellipse([120, y, 132, y+12], fill=(100, 150, 255))
    legend_draw.text((138, y-2), "Defenders", fill=(255, 255, 255), font=font_small)

    legend_draw.line([(230, y), (242, y+12)], fill=(255, 255, 0), width=2)
    legend_draw.line([(230, y+12), (242, y)], fill=(255, 255, 0), width=2)
    legend_draw.text((248, y-2), "Kill", fill=(255, 255, 255), font=font_small)

    legend_draw.rectangle([310, y, 322, y+12], outline=(0, 255, 0), width=2)
    legend_draw.text((328, y-2), "Plant", fill=(255, 255, 255), font=font_small)

    y = 58
    legend_draw.text((10, y), f"Kills: {len([e for e in result['events'] if e.event_type == 'kill'])}",
                    fill=(180, 180, 180), font=font_small)
    if result['spike_planted']:
        legend_draw.text((80, y), f"Spike planted at {result['spike_site']}", fill=(0, 255, 0), font=font_small)

    # Combine
    final_img = Image.new('RGBA', (img_size, img_size + legend_height), (30, 30, 30, 255))
    final_img.paste(result_img, (0, 0))
    final_img.paste(legend, (0, img_size))

    # Save
    if output_path is None:
        output_path = Path(__file__).parent.parent.parent / f'round_{map_name}.png'

    final_img.save(output_path)
    print(f"\nSaved: {output_path}")

    return str(output_path)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Visualize simulation round')
    parser.add_argument('--map', '-m', default='ascent', help='Map name')
    parser.add_argument('--all', '-a', action='store_true', help='Generate for all maps')
    parser.add_argument('--output', '-o', help='Output path')
    args = parser.parse_args()

    if args.all:
        for map_name in ['ascent', 'haven', 'bind', 'split', 'icebox', 'lotus']:
            print(f"\n{'='*50}")
            visualize_round(map_name)
    else:
        visualize_round(args.map, args.output)


if __name__ == '__main__':
    main()
