#!/usr/bin/env python3
"""
Visualization Script - Renders simulation JSON as images.

Reads JSON snapshots from simulate.py and creates various visualizations.

Usage:
    cd backend
    source venv/bin/activate

    # Visualize simulation output
    python scripts/visualize.py output/sim_lotus_attack.json

    # Specify format
    python scripts/visualize.py output/sim.json --format panels
    python scripts/visualize.py output/sim.json --format timeline
    python scripts/visualize.py output/sim.json --format stats

    # Custom output
    python scripts/visualize.py output/sim.json -o output/viz.png
"""

import sys
import os
import json
import math
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("Warning: PIL not installed. Install with: pip install Pillow")

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Map images directory
MAP_IMAGES_DIR = Path(__file__).parent.parent.parent / "map_images"


def load_simulation(json_path: str) -> Dict[str, Any]:
    """Load simulation JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def get_map_image(map_name: str, size: int = 250) -> 'Image.Image':
    """Load map background image."""
    map_path = MAP_IMAGES_DIR / f"{map_name}.png"
    if map_path.exists():
        img = Image.open(map_path).convert('RGBA')
    else:
        # Create fallback gray background
        img = Image.new('RGBA', (400, 400), (40, 40, 40, 255))

    # Resize
    img = img.resize((size, size), Image.LANCZOS)

    # Darken slightly for better visibility
    dark = Image.new('RGBA', (size, size), (0, 0, 0, 60))
    img = Image.alpha_composite(img, dark)

    return img


def get_fonts():
    """Load fonts with fallback."""
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 10)
        font_title = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except:
        font = ImageFont.load_default()
        font_small = font
        font_title = font
    return font, font_small, font_title


def draw_panel(
    bg_img: 'Image.Image',
    snapshot: Dict[str, Any],
    panel_size: int,
    font: 'ImageFont',
    font_small: 'ImageFont',
    events_up_to_now: List[Dict],
    c9_side: str
) -> 'Image.Image':
    """Draw a single snapshot panel."""
    panel = bg_img.copy()
    draw = ImageDraw.Draw(panel)

    time_sec = snapshot['time_ms'] / 1000

    # VCT FIX: Track spike plant position for persistent marker
    spike_plant_pos = None

    # Draw events (kills, plants)
    for event in events_up_to_now:
        event_time = event.get('timestamp_ms', 0)
        if event_time <= snapshot['time_ms']:
            # Try to get position from event details or directly
            details = event.get('details', {})
            ex = event.get('position_x') or details.get('x') or details.get('position_x')
            ey = event.get('position_y') or details.get('y') or details.get('position_y')

            if ex is not None and ey is not None:
                px, py = int(ex * panel_size), int(ey * panel_size)

                if event.get('event_type') == 'kill':
                    # Yellow X for kills
                    size = 5
                    draw.line([(px-size, py-size), (px+size, py+size)], fill=(255, 255, 0), width=2)
                    draw.line([(px-size, py+size), (px+size, py-size)], fill=(255, 255, 0), width=2)
                elif event.get('event_type') == 'spike_plant':
                    # Store spike position for persistent marker
                    spike_plant_pos = (px, py)

    # VCT FIX: Draw spike plant location as prominent DIAMOND marker
    # This shows EXACT plant position on the map (not just site)
    if spike_plant_pos or snapshot.get('spike_planted'):
        # Use stored position or try to get from snapshot
        if spike_plant_pos:
            sx, sy = spike_plant_pos
        else:
            # Fallback: get from last plant event
            for event in reversed(events_up_to_now):
                if event.get('event_type') == 'spike_plant':
                    ex = event.get('position_x') or event.get('details', {}).get('x')
                    ey = event.get('position_y') or event.get('details', {}).get('y')
                    if ex is not None and ey is not None:
                        sx, sy = int(ex * panel_size), int(ey * panel_size)
                        break
            else:
                sx, sy = None, None

        if sx is not None and sy is not None:
            # Draw pulsing diamond for spike location
            size = 8
            # Yellow/orange diamond with glow effect
            # Outer glow
            draw.polygon([
                (sx, sy - size - 2), (sx + size + 2, sy),
                (sx, sy + size + 2), (sx - size - 2, sy)
            ], fill=(255, 200, 0, 100))
            # Inner diamond
            draw.polygon([
                (sx, sy - size), (sx + size, sy),
                (sx, sy + size), (sx - size, sy)
            ], fill=(255, 180, 0), outline=(255, 255, 100), width=1)
            # Center dot
            draw.ellipse([sx-2, sy-2, sx+2, sy+2], fill=(255, 255, 200))

    # Create transparent overlay for vision cones (proper alpha compositing)
    cone_layer = Image.new('RGBA', (panel_size, panel_size), (0, 0, 0, 0))
    cone_draw = ImageDraw.Draw(cone_layer)

    # First pass: Draw all vision cones on transparent layer
    for pos in snapshot['positions']:
        if not pos['is_alive']:
            continue

        px = int(pos['x'] * panel_size)
        py = int(pos['y'] * panel_size)
        px = max(0, min(panel_size - 1, px))
        py = max(0, min(panel_size - 1, py))

        is_c9 = pos['side'] == c9_side
        facing_angle = pos.get('facing_angle')

        if facing_angle is not None:
            # Draw ~90-degree FOV cone, 25 pixels long
            cone_length = 25
            fov_half = math.radians(45)  # 45 degrees each side = 90 degree FOV

            # Calculate cone endpoints
            left_angle = facing_angle - fov_half
            right_angle = facing_angle + fov_half
            left_x = px + int(math.cos(left_angle) * cone_length)
            left_y = py + int(math.sin(left_angle) * cone_length)
            right_x = px + int(math.cos(right_angle) * cone_length)
            right_y = py + int(math.sin(right_angle) * cone_length)

            # Cone color: More visible but still transparent
            # C9 = light red tint, opponent = light blue tint
            if is_c9:
                cone_color = (255, 180, 180, 70)  # Light red, semi-transparent
                outline_color = (255, 100, 100, 90)
            else:
                cone_color = (150, 200, 255, 60)  # Light blue, semi-transparent
                outline_color = (80, 150, 255, 80)

            # Draw filled cone with subtle outline
            cone_draw.polygon([(px, py), (left_x, left_y), (right_x, right_y)],
                              fill=cone_color, outline=outline_color)

    # Composite cone layer onto panel (proper transparency)
    panel = Image.alpha_composite(panel, cone_layer)
    draw = ImageDraw.Draw(panel)

    # Second pass: Draw players on top
    for pos in snapshot['positions']:
        px = int(pos['x'] * panel_size)
        py = int(pos['y'] * panel_size)

        # VCT FIX: Clamp positions to valid range and detect off-map
        is_off_map = pos['x'] < 0 or pos['x'] > 1 or pos['y'] < 0 or pos['y'] > 1
        px = max(0, min(panel_size - 1, px))
        py = max(0, min(panel_size - 1, py))

        if pos['is_alive']:
            # Determine color by side
            if pos['side'] == 'attack':
                color = (255, 80, 80)  # Red for attack
                outline = (200, 40, 40)
                name_color = (255, 150, 150)
            else:
                color = (80, 150, 255)  # Blue for defense
                outline = (40, 100, 200)
                name_color = (150, 200, 255)

            # Check if C9 player
            is_c9 = pos['side'] == c9_side
            # VCT FIX: Smaller dots - radius 7→4 for C9, 6→3 for opponents
            radius = 4 if is_c9 else 3

            # C9 players get white ring (thinner now)
            if is_c9:
                draw.ellipse(
                    [px-radius-1, py-radius-1, px+radius+1, py+radius+1],
                    outline=(255, 255, 255),
                    width=1  # VCT FIX: Ring width 2→1
                )

            # VCT FIX: Magenta border for off-map players
            if is_off_map:
                draw.ellipse(
                    [px-radius-2, py-radius-2, px+radius+2, py+radius+2],
                    outline=(255, 0, 255),
                    width=2
                )

            # Draw player dot
            draw.ellipse(
                [px-radius, py-radius, px+radius, py+radius],
                fill=color,
                outline=outline,
                width=1
            )

            # Spike carrier indicator
            if pos.get('has_spike'):
                draw.rectangle([px-2, py-2, px+2, py+2], fill=(255, 255, 0))

            # VCT FIX: Draw player name label
            player_name = pos.get('player_id', '')[:6]  # First 6 chars
            if player_name:
                # Offset name to the right of dot
                draw.text((px + radius + 2, py - 5), player_name, fill=name_color, font=font_small)

        else:
            # Dead player X
            draw.line([(px-3, py-3), (px+3, py+3)], fill=(100, 100, 100), width=1)
            draw.line([(px-3, py+3), (px+3, py-3)], fill=(100, 100, 100), width=1)

    # Time label
    label = f"{time_sec:.0f}s"
    draw.rectangle([5, 5, 50, 25], fill=(0, 0, 0, 180))
    draw.text((10, 8), label, fill=(255, 255, 255), font=font)

    # Alive count
    atk_alive = snapshot.get('attack_alive', 0)
    def_alive = snapshot.get('defense_alive', 0)
    count_label = f"{atk_alive}v{def_alive}"
    draw.rectangle([panel_size-50, 5, panel_size-5, 25], fill=(0, 0, 0, 180))
    draw.text((panel_size-45, 8), count_label, fill=(255, 255, 255), font=font)

    # Spike planted indicator
    if snapshot.get('spike_planted'):
        draw.rectangle([5, panel_size-25, 80, panel_size-5], fill=(0, 100, 0, 200))
        draw.text((10, panel_size-22), "PLANTED", fill=(0, 255, 0), font=font_small)

    return panel


def render_panels(data: Dict[str, Any], output_path: Path) -> Path:
    """Render simulation as panel grid."""
    if not HAS_PIL:
        raise ImportError("PIL required for visualization")

    metadata = data.get('metadata', {})
    result = data.get('result', {})
    snapshots = data.get('snapshots', [])
    events = data.get('events', [])

    map_name = metadata.get('map_name', 'unknown')
    c9_side = metadata.get('c9_side', 'attack')

    # Setup
    panel_size = 250
    bg_img = get_map_image(map_name, panel_size)
    font, font_small, font_title = get_fonts()

    # Create panels
    panels = []
    for snap in snapshots:
        # Get events up to this snapshot
        events_so_far = [e for e in events if e.get('timestamp_ms', 0) <= snap['time_ms']]
        panel = draw_panel(bg_img.copy(), snap, panel_size, font, font_small, events_so_far, c9_side)
        panels.append(panel)

    # Layout
    n_panels = len(panels)
    if n_panels == 0:
        print("No snapshots to visualize")
        return None

    cols = min(6, n_panels)
    rows = math.ceil(n_panels / cols)

    padding = 5
    header_height = 50
    legend_height = 70  # VCT FIX: Taller legend for new elements

    total_width = cols * panel_size + (cols + 1) * padding
    total_height = header_height + rows * panel_size + (rows + 1) * padding + legend_height

    # Create final image
    final_img = Image.new('RGBA', (total_width, total_height), (25, 25, 25, 255))
    draw = ImageDraw.Draw(final_img)

    # Header
    c9_won = result.get('c9_won', False)
    winner = result.get('winner', 'unknown')
    win_cond = result.get('win_condition', 'unknown')
    duration_ms = result.get('duration_ms', 0)
    kills = result.get('total_kills', 0)

    c9_result = "C9 WIN!" if c9_won else f"C9 Loss ({winner})"
    header = f"{map_name.upper()} - C9 {c9_side.upper()} - {duration_ms/1000:.1f}s - {kills} kills - {c9_result}"
    draw.text((padding, 15), header, fill=(255, 255, 255), font=font_title)

    # Badge
    draw.rectangle([total_width - 150, 10, total_width - 10, 35], fill=(0, 120, 180))
    draw.text((total_width - 145, 13), "C9 TACTICAL", fill=(255, 255, 255), font=font)

    # Place panels
    for i, panel in enumerate(panels):
        row = i // cols
        col = i % cols
        x = padding + col * (panel_size + padding)
        y = header_height + padding + row * (panel_size + padding)
        final_img.paste(panel, (x, y))

    # VCT FIX: Updated legend with new visual elements
    legend_y = header_height + rows * (panel_size + padding) + padding + 5
    legend_y2 = legend_y + 18  # Second row of legend

    # Row 1: Player indicators
    # C9 indicator (smaller to match new dot size)
    draw.ellipse([padding + 2, legend_y + 3, padding + 10, legend_y + 11],
                 fill=(255, 80, 80), outline=(255, 255, 255), width=1)
    draw.text((padding + 15, legend_y), "C9", fill=(255, 255, 255), font=font_small)

    # Opponent indicator
    draw.ellipse([padding + 52, legend_y + 3, padding + 58, legend_y + 9],
                 fill=(80, 150, 255), outline=(40, 100, 200))
    draw.text((padding + 65, legend_y), "Opp", fill=(255, 255, 255), font=font_small)

    # Kill indicator
    draw.line([(padding + 112, legend_y + 2), (padding + 118, legend_y + 10)],
              fill=(255, 255, 0), width=2)
    draw.line([(padding + 118, legend_y + 2), (padding + 112, legend_y + 10)],
              fill=(255, 255, 0), width=2)
    draw.text((padding + 125, legend_y), "Kill", fill=(255, 255, 255), font=font_small)

    # Spike diamond indicator (new)
    sx, sy = padding + 180, legend_y + 6
    draw.polygon([(sx, sy - 5), (sx + 5, sy), (sx, sy + 5), (sx - 5, sy)],
                 fill=(255, 180, 0), outline=(255, 255, 100))
    draw.text((padding + 192, legend_y), "Spike", fill=(255, 255, 255), font=font_small)

    # Row 2: Status indicators
    # Vision cone indicator
    draw.polygon([(padding + 5, legend_y2 + 6), (padding + 20, legend_y2 + 2),
                  (padding + 20, legend_y2 + 10)], fill=(255, 255, 255, 80))
    draw.text((padding + 25, legend_y2), "FOV", fill=(200, 200, 200), font=font_small)

    # Off-map indicator
    draw.ellipse([padding + 67, legend_y2 + 2, padding + 77, legend_y2 + 12],
                 outline=(255, 0, 255), width=2)
    draw.text((padding + 82, legend_y2), "Off-map", fill=(200, 200, 200), font=font_small)

    # Win condition
    win_text = f"Win: {win_cond}"
    draw.text((total_width - 150, legend_y), win_text, fill=(200, 200, 200), font=font_small)

    # Save
    final_img.save(output_path)
    return output_path


def render_timeline(data: Dict[str, Any], output_path: Path) -> Path:
    """Render simulation as timeline."""
    if not HAS_PIL:
        raise ImportError("PIL required for visualization")

    metadata = data.get('metadata', {})
    result = data.get('result', {})
    events = data.get('events', [])

    map_name = metadata.get('map_name', 'unknown')
    c9_side = metadata.get('c9_side', 'attack')
    duration_ms = result.get('duration_ms', 100000)

    font, font_small, font_title = get_fonts()

    # Timeline dimensions
    width = 1200
    height = 400
    margin = 60
    timeline_y = 200

    img = Image.new('RGBA', (width, height), (25, 25, 25, 255))
    draw = ImageDraw.Draw(img)

    # Header
    header = f"TIMELINE: {map_name.upper()} - C9 {c9_side.upper()}"
    draw.text((margin, 20), header, fill=(255, 255, 255), font=font_title)

    # Timeline axis
    draw.line([(margin, timeline_y), (width - margin, timeline_y)], fill=(100, 100, 100), width=2)

    # Time markers every 10 seconds
    for t in range(0, int(duration_ms/1000) + 10, 10):
        x = margin + (t * 1000 / duration_ms) * (width - 2 * margin)
        if x > width - margin:
            break
        draw.line([(x, timeline_y - 5), (x, timeline_y + 5)], fill=(150, 150, 150), width=1)
        draw.text((x - 10, timeline_y + 10), f"{t}s", fill=(150, 150, 150), font=font_small)

    # Plot events
    kill_events = [e for e in events if e.get('event_type') == 'kill']
    plant_event = next((e for e in events if e.get('event_type') == 'spike_plant'), None)

    for i, event in enumerate(kill_events):
        t = event.get('timestamp_ms', 0)
        x = margin + (t / duration_ms) * (width - 2 * margin)

        details = event.get('details', {})
        killer_side = details.get('killer_side', 'attack')

        # Color by who got the kill
        if killer_side == c9_side:
            color = (100, 255, 100)  # Green for C9 kill
        else:
            color = (255, 100, 100)  # Red for enemy kill

        # Draw kill marker
        y_offset = -30 - (i % 3) * 25  # Stagger to avoid overlap
        draw.ellipse([x-5, timeline_y + y_offset - 5, x+5, timeline_y + y_offset + 5], fill=color)

        # Label
        victim = details.get('victim_name', 'Player')[:8]
        draw.text((x - 15, timeline_y + y_offset - 20), victim, fill=color, font=font_small)

    # Plant marker
    if plant_event:
        t = plant_event.get('timestamp_ms', 0)
        x = margin + (t / duration_ms) * (width - 2 * margin)
        draw.rectangle([x-8, timeline_y + 30, x+8, timeline_y + 46], fill=(255, 255, 0))
        draw.text((x - 20, timeline_y + 50), "PLANT", fill=(255, 255, 0), font=font_small)

    # Legend
    legend_y = height - 50
    draw.ellipse([margin, legend_y, margin + 10, legend_y + 10], fill=(100, 255, 100))
    draw.text((margin + 15, legend_y - 2), "C9 Kill", fill=(255, 255, 255), font=font_small)

    draw.ellipse([margin + 100, legend_y, margin + 110, legend_y + 10], fill=(255, 100, 100))
    draw.text((margin + 115, legend_y - 2), "Enemy Kill", fill=(255, 255, 255), font=font_small)

    draw.rectangle([margin + 220, legend_y, margin + 236, legend_y + 10], fill=(255, 255, 0))
    draw.text((margin + 240, legend_y - 2), "Spike Plant", fill=(255, 255, 255), font=font_small)

    # Result
    c9_won = result.get('c9_won', False)
    win_cond = result.get('win_condition', 'unknown')
    result_text = f"{'C9 WIN' if c9_won else 'C9 LOSS'} ({win_cond})"
    result_color = (100, 255, 100) if c9_won else (255, 100, 100)
    draw.text((width - margin - 150, 20), result_text, fill=result_color, font=font_title)

    img.save(output_path)
    return output_path


def render_stats(data: Dict[str, Any], output_path: Path) -> Path:
    """Render simulation statistics as text."""
    metadata = data.get('metadata', {})
    result = data.get('result', {})
    events = data.get('events', [])

    lines = []
    lines.append("=" * 60)
    lines.append("SIMULATION STATISTICS")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Map: {metadata.get('map_name', 'unknown').upper()}")
    lines.append(f"C9 Side: {metadata.get('c9_side', 'attack').upper()}")
    lines.append(f"Round Type: {metadata.get('round_type', 'unknown')}")
    lines.append(f"Timestamp: {metadata.get('timestamp', 'unknown')}")
    lines.append("")
    lines.append("-" * 40)
    lines.append("RESULT")
    lines.append("-" * 40)
    lines.append(f"Winner: {result.get('winner', 'unknown')}")
    lines.append(f"Win Condition: {result.get('win_condition', 'unknown')}")
    lines.append(f"C9 Won: {result.get('c9_won', False)}")
    lines.append(f"Duration: {result.get('duration_ms', 0) / 1000:.1f}s")
    lines.append(f"Total Kills: {result.get('total_kills', 0)}")
    lines.append(f"Attack Kills: {result.get('attack_kills', 0)}")
    lines.append(f"Defense Kills: {result.get('defense_kills', 0)}")
    lines.append(f"Spike Planted: {result.get('spike_planted', False)}")
    lines.append(f"Spike Site: {result.get('spike_site', 'N/A')}")
    lines.append("")
    lines.append("-" * 40)
    lines.append("KILL LOG")
    lines.append("-" * 40)

    kills = [e for e in events if e.get('event_type') == 'kill']
    for kill in kills:
        t = kill.get('timestamp_ms', 0) / 1000
        details = kill.get('details', {})
        killer = details.get('killer_name', 'Unknown')
        victim = details.get('victim_name', 'Unknown')
        weapon = details.get('weapon', 'unknown')
        lines.append(f"  {t:5.1f}s: {killer} -> {victim} ({weapon})")

    if not kills:
        lines.append("  No kills")

    # Save as text
    output_txt = output_path.with_suffix('.txt')
    with open(output_txt, 'w') as f:
        f.write('\n'.join(lines))

    print('\n'.join(lines))
    return output_txt


def main():
    parser = argparse.ArgumentParser(
        description="Visualize simulation output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/visualize.py output/sim_lotus_attack.json
  python scripts/visualize.py output/sim.json --format timeline
  python scripts/visualize.py output/sim.json --format stats -o report.txt
        """
    )

    parser.add_argument('input', help='Input JSON file from simulate.py')
    parser.add_argument('--format', '-f', choices=['panels', 'timeline', 'stats'],
                        default='panels', help='Visualization format (default: panels)')
    parser.add_argument('--output', '-o', help='Output file path')

    args = parser.parse_args()

    # Load simulation data
    if not os.path.exists(args.input):
        print(f"Error: File not found: {args.input}")
        sys.exit(1)

    print(f"Loading: {args.input}")
    data = load_simulation(args.input)

    # Check if it's a batch result
    if 'rounds' in data:
        print("Batch simulation detected. Visualizing first round.")
        data = data['rounds'][0]

    metadata = data.get('metadata', {})
    map_name = metadata.get('map_name', 'unknown')
    c9_side = metadata.get('c9_side', 'attack')

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        ext = '.txt' if args.format == 'stats' else '.png'
        output_path = OUTPUT_DIR / f"viz_{map_name}_{c9_side}_{args.format}_{timestamp}{ext}"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Render
    print(f"Rendering {args.format}...")

    if args.format == 'panels':
        result_path = render_panels(data, output_path)
    elif args.format == 'timeline':
        result_path = render_timeline(data, output_path)
    elif args.format == 'stats':
        result_path = render_stats(data, output_path)
    else:
        print(f"Unknown format: {args.format}")
        sys.exit(1)

    if result_path:
        print(f"Saved: {result_path}")


if __name__ == '__main__':
    main()
