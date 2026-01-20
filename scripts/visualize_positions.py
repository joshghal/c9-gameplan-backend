"""Visualize player positions on Ascent map with alive/dead status."""
import sys
import asyncio
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'app'))
sys.path.insert(0, str(Path(__file__).parent))

from realistic_round_sim import (
    RealisticRoundSimulator, game_to_minimap, get_ascent_pos, ASCENT_COORDS,
    ENGAGEMENT_ZONES
)


def plot_round_on_ascent():
    """Plot a simulation round on the Ascent map showing ACTUAL player positions."""
    # Load Ascent map
    map_path = Path(__file__).parent.parent.parent / 'ascent_map.png'
    if not map_path.exists():
        print(f"Map not found at {map_path}")
        return

    ascent_img = mpimg.imread(str(map_path))

    # Run a simulation round and capture position history
    async def run_sim():
        sim = RealisticRoundSimulator()
        result = await sim.run_round()
        # Extract death times from events
        death_times = {}
        for event in result.get('events', []):
            if event.event_type == 'kill':
                death_times[event.target] = event.time_ms
        return sim, result, death_times

    sim, result, death_times = asyncio.run(run_sim())

    # Get player lists
    attackers = sim.attack_players
    defenders = sim.defense_players

    # Get position history from simulation
    position_history = result.get('position_history', {})

    # Create figure with 6 time snapshots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f"Round on ASCENT: {result['winner']} wins by {result['win_condition']} | {result['duration']/1000:.0f}s",
                 fontsize=14, color='white')

    # Time snapshots to show
    times = [0, 15000, 40000, 45000, 50000, 55000]
    phases = ['setup', 'control', 'execute', 'post_plant', 'post_plant', 'clutch']

    for idx, (ax, t, phase) in enumerate(zip(axes.flat, times, phases)):
        # Webapp uses: x=0 left, x=1 right, y=0 TOP, y=1 bottom
        # extent=[left, right, bottom, top] with origin='upper'
        ax.imshow(ascent_img, extent=[0, 1, 1, 0], origin='upper')

        # Plot engagement zones (faint boundaries for debugging)
        for zone_name, zone_data in ENGAGEMENT_ZONES.items():
            x_min, y_min, x_max, y_max = zone_data['bounds']
            # Draw zone rectangle (very faint)
            rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                  fill=False, edgecolor='white', alpha=0.1, linewidth=0.5)
            ax.add_patch(rect)

        # Plot site markers
        a_site = get_ascent_pos('a_site')
        b_site = get_ascent_pos('b_site')
        ax.plot(a_site[0], a_site[1], '*', color='lime', markersize=12)
        ax.plot(b_site[0], b_site[1], '*', color='orange', markersize=12)
        ax.text(a_site[0], a_site[1] + 0.04, 'A', color='lime', fontsize=9, ha='center', fontweight='bold')
        ax.text(b_site[0], b_site[1] + 0.04, 'B', color='orange', fontsize=9, ha='center', fontweight='bold')

        # Get ACTUAL positions from position history
        positions_at_t = position_history.get(t, {})

        # Count alive players at this time
        attack_alive = 0
        defense_alive = 0

        # Plot attackers (cyan=alive, gray=dead)
        for i, name in enumerate(attackers):
            pid = f'atk_{name}'
            if pid in positions_at_t:
                x, y, is_alive = positions_at_t[pid]
            else:
                # Fallback to death check
                is_alive = death_times.get(name, float('inf')) > t
                x, y = 0.85, 0.55  # Default spawn

            if is_alive:
                attack_alive += 1

            color = 'cyan' if is_alive else '#555555'
            marker = 's' if i == 0 else 'o'  # Square for spike carrier
            alpha = 1.0 if is_alive else 0.5
            size = 16 if is_alive else 10

            ax.plot(x, y, marker, color=color, markersize=size,
                    markeredgecolor='black' if is_alive else '#333333',
                    markeredgewidth=2, zorder=10, alpha=alpha)

            # Label with player name (shortened)
            label = name[:4] if is_alive else "X"
            text_color = 'white' if is_alive else '#666666'
            ax.text(x + 0.025, y, label, color=text_color, fontsize=7,
                    fontweight='bold', zorder=11, alpha=alpha)

        # Plot defenders (red=alive, gray=dead)
        for i, name in enumerate(defenders):
            pid = f'def_{name}'
            if pid in positions_at_t:
                x, y, is_alive = positions_at_t[pid]
            else:
                # Fallback to death check
                is_alive = death_times.get(name, float('inf')) > t
                x, y = 0.08, 0.40  # Default spawn

            if is_alive:
                defense_alive += 1

            color = 'red' if is_alive else '#555555'
            alpha = 1.0 if is_alive else 0.5
            size = 16 if is_alive else 10

            ax.plot(x, y, 'o', color=color, markersize=size,
                    markeredgecolor='black' if is_alive else '#333333',
                    markeredgewidth=2, zorder=10, alpha=alpha)

            # Label with player name (shortened)
            label = name[:4] if is_alive else "X"
            text_color = 'white' if is_alive else '#666666'
            ax.text(x + 0.025, y, label, color=text_color, fontsize=7,
                    fontweight='bold', zorder=11, alpha=alpha)

        # Title with player count
        planted_str = " [PLANTED]" if t >= 45000 and result['spike_planted'] else ""
        ax.set_title(f"t={t//1000}s ({phase}) | {attack_alive}v{defense_alive}{planted_str}",
                     color='white', fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='cyan', markersize=10, label='Attack (alive)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Defense (alive)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#555555', markersize=8, label='Dead'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='cyan', markersize=10, label='Spike carrier'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=9,
               facecolor='#2a2a2a', edgecolor='white', labelcolor='white')

    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    output_path = Path(__file__).parent.parent.parent / 'round_ascent_new.png'
    plt.savefig(str(output_path), dpi=120, bbox_inches='tight', facecolor='#1a1a1a')
    print(f"Saved to {output_path}")

    # Print death summary
    print(f"\nDeath timeline:")
    for name, time_ms in sorted(death_times.items(), key=lambda x: x[1]):
        side = 'ATK' if name in attackers else 'DEF'
        print(f"  {time_ms/1000:.1f}s - {name} ({side})")

    plt.close()


if __name__ == '__main__':
    plot_round_on_ascent()
