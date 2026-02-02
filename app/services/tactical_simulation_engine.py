"""
Tactical Simulation Engine — phase-by-phase wrapper around SimulationEngine.

Supports the Tactical Planner's iterative workflow:
1. User plans waypoints for one phase
2. Engine executes that phase (user team follows waypoints, opponent uses full AI)
3. Returns events + checkpoint for the next phase
4. Repeat until round ends or all 4 phases complete
"""

import copy
import math
import random
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

from .simulation_engine import SimulationEngine
from .vct_round_service import VCTRoundService
from .pathfinding import AStarPathfinder


# Phase time boundaries (ms) — maps tactical planner phases to simulation time
TACTICAL_PHASE_RANGES = {
    "setup": (0, 15000),
    "mid_round": (15000, 50000),
    "execute": (50000, 75000),
    "post_plant": (75000, 100000),
}


@dataclass
class TacticalWaypoint:
    """A user-placed waypoint for a player."""
    tick: int  # Simulation tick (tick * 128 = ms)
    x: float
    y: float
    facing: float  # Radians


@dataclass
class PlayerCheckpoint:
    """Serialized player state for checkpoint."""
    player_id: str
    x: float
    y: float
    side: str
    is_alive: bool
    health: int
    shield: int
    has_spike: bool
    agent: str
    name: str
    kills: int
    deaths: int
    facing_angle: float
    is_running: bool


@dataclass
class PhaseCheckpoint:
    """Full engine state snapshot between phases."""
    phase_name: str
    time_ms: int
    players: List[PlayerCheckpoint]
    spike_planted: bool
    spike_site: Optional[str]
    spike_plant_time_ms: int
    site_execute_active: bool
    target_site: Optional[str]


@dataclass
class PhaseResult:
    """Result of executing one tactical phase."""
    phase_name: str
    winner: Optional[str]  # 'attack' or 'defense' if round ended
    round_ended: bool
    events: List[Dict[str, Any]]
    snapshots: List[Dict[str, Any]]  # Key moment snapshots for narration
    checkpoint: PhaseCheckpoint
    end_positions: List[Dict[str, Any]]  # Final player positions for map display


class TacticalSimulationEngine:
    """Phase-by-phase SimulationEngine wrapper for the Tactical Planner."""

    TICK_MS = 128  # SimulationEngine tick duration

    def __init__(self):
        self.engine = SimulationEngine(db=None)
        self.vct_service = VCTRoundService.get_instance()
        self._initialized = False
        self._round_id = None
        self._user_side = None
        self._map_name = None
        self._guided_waypoints: Dict[str, List[TacticalWaypoint]] = {}

    async def initialize_round(
        self,
        round_id: str,
        user_side: str,
        map_name: str,
        attack_team_id: str = "cloud9",
        defense_team_id: str = "g2",
        user_team_players: Optional[List[Dict[str, str]]] = None,
    ):
        """Initialize the simulation engine for a tactical round.

        Sets up both teams using the standard SimulationEngine initialization.
        The user's team will later have their movement overridden by waypoints.

        Args:
            user_team_players: List of dicts with 'player_id', 'name', 'agent'
                from VCT round data. Used to remap engine player IDs to match
                the frontend's player IDs (e.g. 't1', 't2' instead of 'c9_1').
        """
        self._round_id = round_id
        self._user_side = user_side
        self._map_name = map_name

        # Create a mock session for SimulationEngine
        session = _MockSession(
            id=str(uuid.uuid4()),
            attack_team_id=attack_team_id,
            defense_team_id=defense_team_id,
            map_name=map_name,
            round_type="full",
        )

        # Initialize the engine (loads teams, spawns, weapons, etc.)
        await self.engine.initialize(session, round_type="full")
        self._initialized = True
        self._session = session

        # Remap user-side player IDs to match VCT round data (frontend sends these IDs)
        if user_team_players:
            self._remap_user_players(user_team_players)

    async def execute_phase(
        self,
        phase_name: str,
        user_waypoints: Dict[str, List[TacticalWaypoint]],
        checkpoint: Optional[PhaseCheckpoint] = None,
    ) -> PhaseResult:
        """Execute a single phase of the tactical plan.

        Args:
            phase_name: 'setup', 'mid_round', 'execute', or 'post_plant'
            user_waypoints: {player_id: [TacticalWaypoint, ...]} for user team
            checkpoint: State from previous phase (None for first phase)

        Returns:
            PhaseResult with events, snapshots, and checkpoint for next phase
        """
        if not self._initialized:
            raise RuntimeError("Call initialize_round() first")

        # Restore from checkpoint if provided
        if checkpoint:
            self._restore_checkpoint(checkpoint)

        # Store waypoints for guided movement
        self._guided_waypoints = user_waypoints

        # Mark ALL user team players as guided — those without waypoints hold position
        # Give guided players extra survivability to compensate for straight-line movement
        # (real players would peek, jiggle, use cover — waypoints don't model this)
        for player in self.engine.players.values():
            if player.side == self._user_side:
                player.is_guided = True
                # Boost effective HP: guided players get full shield as damage buffer
                if not getattr(player, '_tactical_buffed', False):
                    player.shield = 50  # Heavy shield
                    player._tactical_buffed = True
            else:
                player.is_guided = False
                # Reduce opponent aggression so defenders hold angles instead of pushing
                if hasattr(player, 'tendencies') and player.tendencies:
                    player.tendencies.base_aggression = min(
                        player.tendencies.base_aggression, 0.3
                    )

        # Determine phase time boundaries
        phase_start, phase_end = TACTICAL_PHASE_RANGES.get(
            phase_name, (0, 100000)
        )

        # For execute/post-plant phases on attack, pre-activate site execute
        # so the spike plant probability is boosted when guided players reach a site
        if (self._user_side == 'attack' and phase_name in ('execute', 'post_plant')
                and not self.engine.spike_planted):
            if not self.engine.site_execute_active:
                self.engine.site_execute_active = True
                # Pick nearest site to spike carrier for target
                self._auto_select_target_site(phase_name)

        # If restoring from checkpoint, start from checkpoint time
        start_time = checkpoint.time_ms if checkpoint else phase_start
        current_time = start_time

        # Track events that happen during this phase
        events_before = len(self.engine.events)
        snapshots = []

        # Run simulation tick by tick through this phase
        # Note: engine.advance(ticks=1) internally does current_time += TICK_MS,
        # so we set session time to current_time (not incremented) and let advance
        # handle the tick progression. We track time separately.
        while current_time < phase_end:
            # Set session time so advance() starts from here and adds one tick
            self._session.current_time_ms = current_time

            # Override user team positions from waypoints BEFORE engine tick
            self._apply_user_waypoints(current_time + self.TICK_MS, phase_start, phase_end)

            # Auto-trigger spike plant for guided players on site
            self._check_guided_spike_plant(current_time)

            # Phase-specific combat distances (default engine is 0.40):
            # - Setup: close range only
            # - Mid-round/execute/post-plant: moderate range
            original_eng_dist = self.engine.ENGAGEMENT_DISTANCE
            if phase_name == 'setup':
                self.engine.ENGAGEMENT_DISTANCE = 0.15
            else:
                self.engine.ENGAGEMENT_DISTANCE = 0.30

            # Throttle combat resolution: only check every ~1 second (8 ticks)
            # The per-tick engagement probabilities compound too fast otherwise,
            # causing instant wipes that prevent multi-phase gameplay.
            # Skip combat on most ticks by temporarily disabling it.
            tick_in_phase = (current_time - start_time) // self.TICK_MS
            run_combat = (tick_in_phase % 8 == 0)
            if not run_combat:
                saved_eng_dist = self.engine.ENGAGEMENT_DISTANCE
                self.engine.ENGAGEMENT_DISTANCE = 0.0  # Suppress combat this tick

            # Run one engine tick (opponent AI, combat, abilities, etc.)
            state = await self.engine.advance(self._session, ticks=1)

            if not run_combat:
                self.engine.ENGAGEMENT_DISTANCE = saved_eng_dist

            # Restore engagement distance
            self.engine.ENGAGEMENT_DISTANCE = original_eng_dist

            # Advance our time tracker to match what the engine did
            current_time += self.TICK_MS

            # Re-apply user positions AFTER engine tick (engine may have moved them)
            self._apply_user_waypoints(current_time, phase_start, phase_end)

            # Snapshot every 250ms for smooth playback (~8 per second)
            if (current_time - start_time) % 250 < self.TICK_MS:
                snapshots.append(self._build_snapshot(current_time, phase_name))

            # Check if round ended (team wipe, spike defuse, time out)
            if self._check_round_ended(state):
                break

        # Collect events from this phase, enriching kill events with player names
        phase_events = []
        for evt in self.engine.events[events_before:]:
            details = dict(evt.details) if evt.details else {}
            if evt.event_type == 'kill':
                # Add killer/victim names for the frontend kill feed
                killer_id = details.get('killer_id', '')
                victim_id = evt.player_id  # victim is stored as player_id
                killer_p = self.engine.players.get(killer_id)
                victim_p = self.engine.players.get(victim_id)
                details['killer_name'] = getattr(killer_p, 'name', killer_id) if killer_p else killer_id
                details['victim_name'] = getattr(victim_p, 'name', victim_id) if victim_p else victim_id
            phase_events.append({
                "time_ms": evt.timestamp_ms,
                "event_type": evt.event_type,
                "player_id": evt.player_id,
                "target_id": evt.target_id,
                "position_x": evt.position_x,
                "position_y": evt.position_y,
                "details": details,
            })

        # Build checkpoint for next phase
        ckpt = self._build_checkpoint(phase_name, current_time)

        # Determine winner if round ended
        winner = None
        round_ended = False
        attack_alive = sum(1 for p in self.engine.players.values()
                          if p.side == 'attack' and p.is_alive)
        defense_alive = sum(1 for p in self.engine.players.values()
                           if p.side == 'defense' and p.is_alive)

        if attack_alive == 0:
            winner = "defense"
            round_ended = True
        elif defense_alive == 0:
            winner = "attack"
            round_ended = True
        elif current_time >= 100000:
            winner = "attack" if self.engine.spike_planted else "defense"
            round_ended = True
        elif getattr(self.engine, 'spike_defused', False):
            winner = "defense"
            round_ended = True

        # End positions for map display
        end_positions = []
        for p in self.engine.players.values():
            end_positions.append({
                "player_id": p.player_id,
                "name": getattr(p, 'name', p.player_id),
                "x": p.x,
                "y": p.y,
                "side": p.side,
                "is_alive": p.is_alive,
                "health": p.health,
                "agent": getattr(p, 'agent', 'unknown'),
                "has_spike": getattr(p, 'has_spike', False),
            })

        # Clear guided state and start positions
        for player in self.engine.players.values():
            player.is_guided = False
            if hasattr(player, '_wp_start_pos'):
                del player._wp_start_pos
        self._guided_waypoints = {}

        return PhaseResult(
            phase_name=phase_name,
            winner=winner,
            round_ended=round_ended,
            events=phase_events,
            snapshots=snapshots,
            checkpoint=ckpt,
            end_positions=end_positions,
        )

    def _apply_user_waypoints(self, current_time_ms: int, phase_start: int, phase_end: int):
        """Interpolate and apply user waypoints to guided players.

        Uses player's current position as implicit start point so movement
        interpolates smoothly from checkpoint/spawn to first waypoint.
        """
        for player_id, waypoints in self._guided_waypoints.items():
            if player_id not in self.engine.players:
                continue
            player = self.engine.players[player_id]
            if not player.is_alive:
                continue

            if not waypoints:
                continue

            # Don't move player while planting — freeze at plant position
            if getattr(player, 'is_planting', False):
                continue

            # Build a position timeline: start position + waypoints
            # Start position is captured once at phase start and reused
            if not hasattr(player, '_wp_start_pos'):
                player._wp_start_pos = (player.x, player.y)
            start_x, start_y = player._wp_start_pos

            # Build full timeline: (time_ms, x, y), sorted by time
            timeline = [(phase_start, start_x, start_y)]
            for wp in waypoints:
                t_ms = wp.tick * self.TICK_MS
                timeline.append((t_ms, wp.x, wp.y))
            timeline.sort(key=lambda p: p[0])

            # Find where current_time_ms falls in timeline
            prev_pt = None
            next_pt = None
            for i, (t, x, y) in enumerate(timeline):
                if t <= current_time_ms:
                    prev_pt = (t, x, y)
                else:
                    next_pt = (t, x, y)
                    break

            if prev_pt and next_pt:
                # Interpolate between two points
                t0, x0, y0 = prev_pt
                t1, x1, y1 = next_pt
                dt = t1 - t0
                if dt > 0:
                    frac = min(1.0, (current_time_ms - t0) / dt)
                    player.x = x0 + (x1 - x0) * frac
                    player.y = y0 + (y1 - y0) * frac
                # Auto-face: movement direction
                dx = next_pt[1] - prev_pt[1]
                dy = next_pt[2] - prev_pt[2]
                if dx != 0 or dy != 0:
                    player.facing_angle = math.atan2(dy, dx)
            elif prev_pt:
                # Past last waypoint — hold at last position
                player.x = prev_pt[1]
                player.y = prev_pt[2]
                # Auto-face: nearest enemy threat
                self._auto_face_threats(player)

    def _auto_select_target_site(self, phase_name: str):
        """Pick target site based on where the spike carrier's waypoints lead."""
        # Find spike carrier
        carrier = None
        for p in self.engine.players.values():
            if p.side == self._user_side and p.is_alive and getattr(p, 'has_spike', False):
                carrier = p
                break

        # Get map sites
        sites = self.engine._get_map_sites() if hasattr(self.engine, '_get_map_sites') else {}

        if not sites:
            # Fallback: pick first available site
            self.engine.target_site = 'A'
            return

        # Find nearest site to carrier's last waypoint (or current position)
        best_site = None
        best_dist = float('inf')
        target_x, target_y = carrier.x if carrier else 0.5, carrier.y if carrier else 0.5

        # Use last waypoint as destination if available
        if carrier and carrier.player_id in self._guided_waypoints:
            wps = self._guided_waypoints[carrier.player_id]
            if wps:
                target_x, target_y = wps[-1].x, wps[-1].y

        for site_name, site_data in sites.items():
            sx, sy = site_data['center']
            d = math.sqrt((target_x - sx) ** 2 + (target_y - sy) ** 2)
            if d < best_dist:
                best_dist = d
                best_site = site_name

        self.engine.target_site = best_site or 'A'

    def _check_guided_spike_plant(self, current_time_ms: int):
        """Force spike plant when guided carrier reaches a site.

        The normal engine plant logic is probability-based and very slow.
        For tactical planner, when the user paths the spike carrier to a site,
        they clearly intend to plant — so we force it.
        """
        if self.engine.spike_planted:
            return
        if self._user_side != 'attack':
            return

        # Find guided spike carrier
        carrier = None
        for p in self.engine.players.values():
            if (p.side == self._user_side and p.is_alive
                    and getattr(p, 'has_spike', False)
                    and getattr(p, 'is_guided', False)):
                carrier = p
                break

        if not carrier or getattr(carrier, 'is_planting', False):
            return

        # Check if carrier is on a site
        sites = self.engine._get_map_sites() if hasattr(self.engine, '_get_map_sites') else {}
        if not sites:
            return

        for site_name, site_data in sites.items():
            sx, sy = site_data['center']
            site_radius = site_data.get('radius', 0.1)
            dist = math.sqrt((carrier.x - sx) ** 2 + (carrier.y - sy) ** 2)
            if dist < site_radius * 1.3:
                # Carrier is on site — force plant start
                carrier.is_planting = True
                carrier.action_start_ms = current_time_ms
                self.engine.site_execute_active = True
                self.engine.target_site = site_name
                self.engine.site_control_achieved = True  # Assume control for plant
                return

    def _auto_face_threats(self, player):
        """Auto-face nearest alive enemy. Falls back to keeping current facing."""
        nearest_dist = float('inf')
        nearest_pos = None
        for other in self.engine.players.values():
            if other.side == player.side or not other.is_alive:
                continue
            d = math.sqrt((other.x - player.x) ** 2 + (other.y - player.y) ** 2)
            if d < nearest_dist:
                nearest_dist = d
                nearest_pos = (other.x, other.y)
        if nearest_pos and nearest_dist < 0.5:
            player.facing_angle = math.atan2(
                nearest_pos[1] - player.y,
                nearest_pos[0] - player.x,
            )
        # else: keep existing facing_angle (last movement direction)

    def _check_round_ended(self, state) -> bool:
        """Check if the round has ended."""
        attack_alive = sum(1 for p in self.engine.players.values()
                          if p.side == 'attack' and p.is_alive)
        defense_alive = sum(1 for p in self.engine.players.values()
                           if p.side == 'defense' and p.is_alive)

        if attack_alive == 0 or defense_alive == 0:
            return True
        if state.current_time_ms >= 100000:
            return True
        if getattr(state, 'round_over', False):
            return True
        return False

    def _build_snapshot(self, time_ms: int, phase: str) -> Dict[str, Any]:
        """Build a snapshot of current state for narration."""
        players = []
        for p in self.engine.players.values():
            players.append({
                "player_id": p.player_id,
                "name": getattr(p, 'name', p.player_id),
                "x": p.x,
                "y": p.y,
                "side": p.side,
                "is_alive": p.is_alive,
                "health": p.health,
                "agent": getattr(p, 'agent', 'unknown'),
                "has_spike": getattr(p, 'has_spike', False),
                "facing_angle": getattr(p, 'facing_angle', 0),
            })
        return {
            "time_ms": time_ms,
            "phase": phase,
            "players": players,
            "spike_planted": self.engine.spike_planted,
            "spike_site": getattr(self.engine, 'spike_site', None),
            "dropped_spike_x": getattr(self.engine, 'dropped_spike_position', None) and getattr(self.engine, 'dropped_spike_position')[0],
            "dropped_spike_y": getattr(self.engine, 'dropped_spike_position', None) and getattr(self.engine, 'dropped_spike_position')[1],
        }

    def _build_checkpoint(self, phase_name: str, time_ms: int) -> PhaseCheckpoint:
        """Build a serializable checkpoint of the full engine state."""
        players = []
        for p in self.engine.players.values():
            players.append(PlayerCheckpoint(
                player_id=p.player_id,
                x=p.x,
                y=p.y,
                side=p.side,
                is_alive=p.is_alive,
                health=p.health,
                shield=getattr(p, 'shield', 0),
                has_spike=getattr(p, 'has_spike', False),
                agent=getattr(p, 'agent', 'unknown'),
                name=getattr(p, 'name', p.player_id),
                kills=getattr(p, 'kills', 0),
                deaths=getattr(p, 'deaths', 0),
                facing_angle=getattr(p, 'facing_angle', 0),
                is_running=getattr(p, 'is_running', False),
            ))

        return PhaseCheckpoint(
            phase_name=phase_name,
            time_ms=time_ms,
            players=players,
            spike_planted=self.engine.spike_planted,
            spike_site=getattr(self.engine, 'spike_site', None),
            spike_plant_time_ms=getattr(self.engine, 'spike_plant_time_ms', 0),
            site_execute_active=getattr(self.engine, 'site_execute_active', False),
            target_site=getattr(self.engine, 'target_site', None),
        )

    def _restore_checkpoint(self, checkpoint: PhaseCheckpoint):
        """Restore engine state from a checkpoint."""
        # Restore player states
        for pck in checkpoint.players:
            if pck.player_id in self.engine.players:
                p = self.engine.players[pck.player_id]
                p.x = pck.x
                p.y = pck.y
                p.is_alive = pck.is_alive
                p.health = pck.health
                p.shield = pck.shield
                p.has_spike = pck.has_spike
                p.kills = pck.kills
                p.deaths = pck.deaths
                p.facing_angle = pck.facing_angle
                p.is_running = pck.is_running

        # Restore spike state
        self.engine.spike_planted = checkpoint.spike_planted
        self.engine.spike_site = checkpoint.spike_site
        self.engine.spike_plant_time_ms = checkpoint.spike_plant_time_ms
        self.engine.site_execute_active = checkpoint.site_execute_active
        self.engine.target_site = checkpoint.target_site

        # Set session time
        self._session.current_time_ms = checkpoint.time_ms

    def _remap_user_players(self, user_team_players: List[Dict[str, str]]):
        """Remap engine player IDs/names to match VCT round data.

        The SimulationEngine creates players with IDs like 'c9_1', 'c9_2' from
        DEFAULT_TEAMS, but the frontend uses VCT IDs like 't1', 't2'. This method
        replaces the engine's player dict keys and player attributes so waypoints
        from the frontend map correctly to engine players.
        """
        # Collect user-side players in order
        user_players = [
            p for p in self.engine.players.values()
            if p.side == self._user_side
        ]
        # Sort by original ID to ensure consistent ordering
        user_players.sort(key=lambda p: p.player_id)

        # Build new players dict, remapping user-side players
        new_players = {}
        remapped = 0
        for pid, player in self.engine.players.items():
            if player.side == self._user_side and remapped < len(user_team_players):
                vct = user_team_players[remapped]
                new_id = vct["player_id"]
                player.player_id = new_id
                player.name = vct.get("name", player.name)
                if vct.get("agent"):
                    player.agent = vct["agent"]
                new_players[new_id] = player
                # Also remap in ability system
                if hasattr(self.engine, 'ability_system') and pid in self.engine.ability_system.player_states:
                    self.engine.ability_system.player_states[new_id] = (
                        self.engine.ability_system.player_states.pop(pid)
                    )
                remapped += 1
            else:
                new_players[pid] = player

        self.engine.players = new_players


@dataclass
class _MockSession:
    """Mock session for SimulationEngine compatibility."""
    id: str
    attack_team_id: str
    defense_team_id: str
    map_name: str
    round_type: str
    status: str = "created"
    current_time_ms: int = 0
    phase: str = "opening"
