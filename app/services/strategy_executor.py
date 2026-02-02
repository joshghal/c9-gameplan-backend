"""
Strategy Executor — runs a tactical plan against pro-anchored AI opponents.

Takes user waypoints (4 phases) + opponent VCT trajectory anchors,
interpolates both to per-second positions, resolves combat encounters,
produces snapshots + events + winner.
"""

import math
import random
import uuid
from typing import Dict, List, Optional, Tuple

from .vct_round_service import VCTRoundService, SIDE_MAP, MAP_SPAWN_POSITIONS
from .pathfinding import AStarPathfinder
from .combat_model import resolve_combat
from .weapon_system import meters_from_normalized
from .player_profiles import profile_manager
from .map_context import get_map_context

# Combat constants
ENGAGEMENT_RANGE = 0.12  # normalized distance for combat
ENGAGEMENT_COOLDOWN = 2  # seconds between re-duels for same pair
TICK_MS = 1000  # 1-second resolution for strategy sim
DEFAULT_PRO_SKILL = 0.8  # VCT pro baseline skill (0-1)
MAX_DIST_PER_SEC = 0.06  # ~6% of map per second — speed cap (pro run speed ~6m/s on 100m map)


class StrategyExecutor:
    """Execute a strategy plan against pro VCT opponents."""

    def __init__(self):
        self.service = VCTRoundService.get_instance()
        self._map_ctx = None

    def _get_map_ctx(self):
        if self._map_ctx is None:
            self._map_ctx = get_map_context()
        return self._map_ctx

    def _validate_position(
        self, map_name: str, x: float, y: float, prev_x: float, prev_y: float
    ) -> Tuple[float, float]:
        """Validate and snap position to walkable ground, matching simulation_engine."""
        x = max(0.02, min(0.98, x))
        y = max(0.02, min(0.98, y))
        ctx = self._get_map_ctx()
        if ctx.is_walkable(map_name, x, y):
            return (x, y)
        nearest = ctx.get_nearest_walkable(map_name, x, y)
        if nearest:
            dist = ((nearest[0] - x) ** 2 + (nearest[1] - y) ** 2) ** 0.5
            if dist < 0.1:
                return nearest
        return (max(0.02, min(0.98, prev_x)), max(0.02, min(0.98, prev_y)))

    def execute(
        self,
        round_id: str,
        user_side: str,
        plans: Dict[str, List[Dict]],  # phase → [{player_id, waypoints}]
        teammates: List[Dict],  # from StrategyRound
    ) -> Dict:
        """Run the strategy execution.

        Returns: {session_id, winner, events, snapshots, reveal}
        """
        # Load opponent trajectories
        opponent_trajs = self.service.get_opponent_trajectories(round_id, user_side)
        if not opponent_trajs:
            raise ValueError(f"Could not load opponent data for round {round_id}")

        round_data = self.service.get_round_by_id(round_id)
        if not round_data:
            raise ValueError(f"Round {round_id} not found")

        map_name = round_data.get("_map", "")
        opponent_side = "defense" if user_side == "attack" else "attack"
        round_duration = round_data.get("round_duration_s", 100)

        # Compute phase ranges from duration (same proportions as vct_round_service)
        phase_ranges = {
            "setup": (0, round(round_duration * 0.15)),
            "mid_round": (round(round_duration * 0.15), round(round_duration * 0.40)),
            "execute": (round(round_duration * 0.40), round(round_duration * 0.70)),
            "post_plant": (round(round_duration * 0.70), round_duration),
        }

        # Initialize pathfinder for walkable movement
        pathfinder = AStarPathfinder()
        pathfinder.load_nav_grid_from_v4(map_name) if map_name else pathfinder.load_nav_grid(map_name)

        # Build user team position timeline
        user_positions = self._build_user_timeline(plans, teammates, phase_ranges, round_duration, pathfinder, map_name)

        # Build opponent position timeline from VCT anchors
        opponent_positions = self._build_opponent_timeline(
            opponent_trajs, opponent_side, round_duration, map_name, pathfinder
        )

        # Run combat simulation
        events, snapshots, winner = self._simulate_combat(
            user_positions, opponent_positions, user_side, opponent_side, map_name,
            teammates, list(opponent_trajs.keys()), round_duration, phase_ranges,
        )

        # Build reveal info
        opponent_team = "Unknown"
        user_team = "Unknown"
        for pname, ptraj in round_data.get("player_trajectories", {}).items():
            if not ptraj:
                continue
            team = ptraj[0].get("team", "Unknown")
            if pname in opponent_trajs:
                opponent_team = team
            else:
                user_team = team
            if opponent_team != "Unknown" and user_team != "Unknown":
                break

        round_num = round_data.get("round_num", "?")
        round_dur = round_data.get("round_duration_s", 0)
        game_id = round_data.get("game_id", "")
        opponent_names_list = list(opponent_trajs.keys())

        # Match metadata from GRID data
        match_meta = self.service.get_match_metadata(game_id) or {}
        tournament = match_meta.get("tournament", "")
        match_date = match_meta.get("date", "")

        # Kill summary
        kills = [e for e in events if e["event_type"] == "kill"]
        user_kills = sum(1 for k in kills if k.get("player_id", "").startswith("t"))
        opp_kills = sum(1 for k in kills if k.get("player_id", "").startswith("opp"))

        # Sim duration (last snapshot time)
        sim_dur_s = snapshots[-1]["time_ms"] / 1000 if snapshots else 0

        return {
            "session_id": str(uuid.uuid4()),
            "winner": winner,
            "events": events,
            "snapshots": snapshots,
            "reveal": {
                "opponent_team": opponent_team,
                "user_team": user_team,
                "round_desc": f"Round {round_num} on {map_name.capitalize()}",
                "round_num": round_num,
                "map_name": map_name,
                "user_side": user_side,
                "opponent_players": opponent_names_list,
                "round_duration_s": round_dur,
                "sim_duration_s": round(sim_dur_s),
                "score_line": f"{user_kills}–{opp_kills}",
                "tournament": tournament,
                "match_date": match_date,
            },
        }

    def _build_user_timeline(
        self,
        plans: Dict[str, List[Dict]],
        teammates: List[Dict],
        phase_ranges: Dict[str, Tuple[int, int]],
        round_duration: int,
        pathfinder: Optional[AStarPathfinder] = None,
        map_name: str = "",
    ) -> Dict[str, List[Dict]]:
        """Build per-second position timeline for user team from waypoints."""

        timelines: Dict[str, List[Dict]] = {}

        for t in teammates:
            pid = t["player_id"] if isinstance(t, dict) else t.player_id
            spawn = t["spawn"] if isinstance(t, dict) else t.spawn
            positions: List[Dict] = []

            # Validate spawn position
            current_x, current_y = self._validate_position(map_name, spawn[0], spawn[1], 0.5, 0.5)
            current_facing = 0.0

            for phase_name in ["setup", "mid_round", "execute", "post_plant"]:
                start_s, end_s = phase_ranges[phase_name]
                phase_plans = plans.get(phase_name, [])

                # Find this player's waypoints in this phase
                player_wps = []
                for pp in phase_plans:
                    pp_id = pp.get("player_id", "")
                    if pp_id == pid:
                        player_wps = pp.get("waypoints", [])
                        break

                if not player_wps:
                    # Hold position for entire phase with micro-jitter
                    for t_sec in range(start_s, end_s):
                        jx = current_x + random.gauss(0, 0.002)
                        jy = current_y + random.gauss(0, 0.002)
                        vx, vy = self._validate_position(map_name, jx, jy, current_x, current_y)
                        positions.append({
                            "time_s": t_sec, "x": vx, "y": vy,
                            "facing": current_facing, "alive": True,
                        })
                else:
                    # Convert waypoints and validate positions
                    raw_wps = []
                    for wp in player_wps:
                        wp_time = wp.get("tick", 0) * 0.128
                        wp_time = max(start_s, min(end_s - 1, wp_time))
                        wx, wy = self._validate_position(map_name, wp["x"], wp["y"], current_x, current_y)
                        raw_wps.append({
                            "x": wx, "y": wy,
                            "facing": wp.get("facing", 0), "time_s": wp_time,
                        })

                    # Build all_wps: skip implicit spawn if first wp is close
                    first_wp = raw_wps[0] if raw_wps else None
                    if first_wp:
                        dist_to_first = ((first_wp["x"] - current_x) ** 2 + (first_wp["y"] - current_y) ** 2) ** 0.5
                        if dist_to_first > 0.05:
                            all_wps = [{"x": current_x, "y": current_y, "facing": current_facing, "time_s": start_s}]
                            all_wps.extend(raw_wps)
                        else:
                            # First wp is near spawn — use it directly, ensure it starts at start_s
                            raw_wps[0]["time_s"] = max(start_s, raw_wps[0]["time_s"])
                            all_wps = list(raw_wps)
                    else:
                        all_wps = [{"x": current_x, "y": current_y, "facing": current_facing, "time_s": start_s}]

                    # Sort by time
                    all_wps.sort(key=lambda w: w["time_s"])

                    # Deduplicate: ensure minimum 1s gap between consecutive waypoints
                    deduped = [all_wps[0]]
                    for wp in all_wps[1:]:
                        if wp["time_s"] - deduped[-1]["time_s"] >= 1.0:
                            deduped.append(wp)
                        else:
                            # Merge: keep later position at the existing time
                            deduped[-1]["x"] = wp["x"]
                            deduped[-1]["y"] = wp["y"]
                            deduped[-1]["facing"] = wp["facing"]
                    all_wps = deduped

                    # Ensure at least 2 waypoints for interpolation
                    if len(all_wps) < 2:
                        all_wps.append({"x": all_wps[0]["x"], "y": all_wps[0]["y"],
                                        "facing": all_wps[0]["facing"], "time_s": end_s - 1})

                    # Build A* paths between consecutive waypoints for smooth movement
                    wp_paths: List[List[Tuple[float, float]]] = []
                    for wi in range(len(all_wps) - 1):
                        a = all_wps[wi]
                        b = all_wps[wi + 1]
                        dist = ((b["x"] - a["x"]) ** 2 + (b["y"] - a["y"]) ** 2) ** 0.5
                        if dist > 0.05 and pathfinder:
                            result = pathfinder.find_path((a["x"], a["y"]), (b["x"], b["y"]))
                            if result.success and result.path and len(result.path) > 1:
                                wp_paths.append(result.path)
                            else:
                                wp_paths.append([(a["x"], a["y"]), (b["x"], b["y"])])
                        else:
                            wp_paths.append([(a["x"], a["y"]), (b["x"], b["y"])])

                    for t_sec in range(start_s, end_s):
                        # Find surrounding waypoints
                        seg_idx = len(all_wps) - 2
                        for i in range(len(all_wps) - 1):
                            if all_wps[i]["time_s"] <= t_sec <= all_wps[i + 1]["time_s"]:
                                seg_idx = i
                                break

                        before = all_wps[seg_idx]
                        after = all_wps[seg_idx + 1] if seg_idx + 1 < len(all_wps) else all_wps[-1]
                        dt = after["time_s"] - before["time_s"]
                        frac = (t_sec - before["time_s"]) / dt if dt > 0 else 1.0
                        frac = max(0, min(1, frac))

                        # Interpolate along A* path for smooth movement
                        path = wp_paths[seg_idx] if seg_idx < len(wp_paths) else [(before["x"], before["y"]), (after["x"], after["y"])]
                        pos_f = frac * (len(path) - 1)
                        idx_lo = min(int(pos_f), len(path) - 1)
                        idx_hi = min(idx_lo + 1, len(path) - 1)
                        sub_frac = pos_f - idx_lo
                        x = path[idx_lo][0] + (path[idx_hi][0] - path[idx_lo][0]) * sub_frac
                        y = path[idx_lo][1] + (path[idx_hi][1] - path[idx_lo][1]) * sub_frac
                        facing = after["facing"]

                        # Speed cap
                        if positions:
                            last = positions[-1]
                            move_dist = ((x - last["x"]) ** 2 + (y - last["y"]) ** 2) ** 0.5
                            if move_dist > MAX_DIST_PER_SEC:
                                scale = MAX_DIST_PER_SEC / move_dist
                                x = last["x"] + (x - last["x"]) * scale
                                y = last["y"] + (y - last["y"]) * scale

                        # Validate position
                        prev_x = positions[-1]["x"] if positions else current_x
                        prev_y = positions[-1]["y"] if positions else current_y
                        x, y = self._validate_position(map_name, x, y, prev_x, prev_y)

                        positions.append({
                            "time_s": t_sec, "x": x, "y": y,
                            "facing": facing, "alive": True,
                        })

                    # Update current position for next phase
                    if positions:
                        current_x = positions[-1]["x"]
                        current_y = positions[-1]["y"]
                        if player_wps:
                            current_facing = player_wps[-1].get("facing", current_facing)

            timelines[pid] = positions

        return timelines

    def _build_opponent_timeline(
        self,
        opponent_trajs: Dict[str, List[Dict]],
        opponent_side: str,
        round_duration: int,
        map_name: str = "",
        pathfinder: Optional[AStarPathfinder] = None,
    ) -> Dict[str, List[Dict]]:
        """Build per-second timeline from VCT anchor points."""
        timelines: Dict[str, List[Dict]] = {}

        # Get spawn positions for this side
        spawn_list = MAP_SPAWN_POSITIONS.get(map_name, {}).get(opponent_side, [])

        for i, (name, anchors) in enumerate(opponent_trajs.items()):
            if not anchors:
                continue

            pid = f"opp_{i + 1}"
            positions: List[Dict] = []

            # Anchors sorted by clock (descending, clock counts down from 100)
            # Convert to time_s (ascending)
            anchor_points = []
            for a in anchors:
                time_s = 100 - a["clock"]
                anchor_points.append({
                    "time_s": time_s, "x": a["x"], "y": a["y"], "alive": a["alive"],
                })
            anchor_points.sort(key=lambda a: a["time_s"])

            if not anchor_points:
                continue

            # Determine spawn position: use map spawn if available, else first anchor
            first_anchor = anchor_points[0]
            if spawn_list and i < len(spawn_list):
                spawn_x, spawn_y = spawn_list[i]
            else:
                spawn_x, spawn_y = first_anchor["x"], first_anchor["y"]

            # Validate spawn
            spawn_x, spawn_y = self._validate_position(map_name, spawn_x, spawn_y, 0.5, 0.5)

            # Build A* path from spawn to first anchor if there's a time gap
            spawn_to_anchor_path: List[Tuple[float, float]] = []
            first_anchor_time = first_anchor["time_s"]  # float, not int
            # Enforce minimum 3s walk time to prevent teleportation
            effective_walk_time = max(3.0, first_anchor_time)

            if first_anchor_time > 0 and pathfinder:
                ax, ay = self._validate_position(map_name, first_anchor["x"], first_anchor["y"], spawn_x, spawn_y)
                result = pathfinder.find_path((spawn_x, spawn_y), (ax, ay))
                if result.success and result.path:
                    spawn_to_anchor_path = result.path

            # VCT death is NOT used — simulation combat decides all deaths.
            # After trajectory data ends, hold last position with jitter.
            last_anchor_time = anchor_points[-1]["time_s"]

            for t_sec in range(0, round_duration):
                # Before first anchor: walk from spawn via A* path
                if t_sec < effective_walk_time:
                    if spawn_to_anchor_path and effective_walk_time > 0:
                        frac = t_sec / effective_walk_time
                        frac = min(1.0, frac)
                        path_idx = min(int(frac * (len(spawn_to_anchor_path) - 1)), len(spawn_to_anchor_path) - 1)
                        px, py = spawn_to_anchor_path[path_idx]
                    else:
                        # Lerp from spawn to first anchor
                        frac = t_sec / effective_walk_time if effective_walk_time > 0 else 1.0
                        frac = min(1.0, frac)
                        px = spawn_x + (first_anchor["x"] - spawn_x) * frac
                        py = spawn_y + (first_anchor["y"] - spawn_y) * frac

                    # Speed cap
                    if positions:
                        last = positions[-1]
                        move_dist = ((px - last["x"]) ** 2 + (py - last["y"]) ** 2) ** 0.5
                        if move_dist > MAX_DIST_PER_SEC:
                            scale = MAX_DIST_PER_SEC / move_dist
                            px = last["x"] + (px - last["x"]) * scale
                            py = last["y"] + (py - last["y"]) * scale

                    prev_x = positions[-1]["x"] if positions else spawn_x
                    prev_y = positions[-1]["y"] if positions else spawn_y
                    px, py = self._validate_position(map_name, px, py, prev_x, prev_y)
                    positions.append({
                        "time_s": t_sec, "x": px, "y": py,
                        "facing": 0, "alive": True, "name": name,
                    })
                    continue

                # After last anchor: hold position with jitter (continuous movement)
                if t_sec >= last_anchor_time:
                    hold_x = anchor_points[-1]["x"]
                    hold_y = anchor_points[-1]["y"]
                    if positions:
                        hold_x = positions[-1]["x"]
                        hold_y = positions[-1]["y"]
                    hold_x += random.gauss(0, 0.002)
                    hold_y += random.gauss(0, 0.002)
                    prev_x = positions[-1]["x"] if positions else hold_x
                    prev_y = positions[-1]["y"] if positions else hold_y
                    hold_x, hold_y = self._validate_position(map_name, hold_x, hold_y, prev_x, prev_y)
                    positions.append({
                        "time_s": t_sec, "x": hold_x, "y": hold_y,
                        "facing": 0, "alive": True, "name": name,
                    })
                    continue

                # Catmull-Rom spline interpolation between anchors
                x, y = self._catmull_rom_interpolate(anchor_points, t_sec)

                # Speed cap: prevent teleportation from bad data
                if positions:
                    last = positions[-1]
                    dist = ((x - last["x"]) ** 2 + (y - last["y"]) ** 2) ** 0.5
                    if dist > MAX_DIST_PER_SEC:
                        scale = MAX_DIST_PER_SEC / dist
                        x = last["x"] + (x - last["x"]) * scale
                        y = last["y"] + (y - last["y"]) * scale

                # Small jitter for natural movement
                x += random.gauss(0, 0.002)
                y += random.gauss(0, 0.002)

                # Validate position (replaces manual walkable search)
                prev_x = positions[-1]["x"] if positions else spawn_x
                prev_y = positions[-1]["y"] if positions else spawn_y
                x, y = self._validate_position(map_name, x, y, prev_x, prev_y)

                positions.append({
                    "time_s": t_sec, "x": x, "y": y,
                    "facing": 0, "alive": True, "name": name,
                })

            timelines[pid] = positions

        return timelines

    @staticmethod
    def _catmull_rom_interpolate(
        points: List[Dict], t: float, alpha: float = 0.5
    ) -> Tuple[float, float]:
        """Catmull-Rom spline interpolation for smooth curves between anchor points.

        Returns (x, y) at time t. Falls back to linear lerp when < 4 points available.
        """
        # Find the segment [p1, p2] that contains t
        seg_idx = 0
        for j in range(len(points) - 1):
            if points[j]["time_s"] <= t <= points[j + 1]["time_s"]:
                seg_idx = j
                break

        p1 = points[seg_idx]
        p2 = points[seg_idx + 1] if seg_idx + 1 < len(points) else p1

        dt = p2["time_s"] - p1["time_s"]
        frac = (t - p1["time_s"]) / dt if dt > 0 else 1.0
        frac = max(0.0, min(1.0, frac))

        # Need 4 points for Catmull-Rom: p0, p1, p2, p3
        if len(points) < 4:
            # Linear fallback
            return (
                p1["x"] + (p2["x"] - p1["x"]) * frac,
                p1["y"] + (p2["y"] - p1["y"]) * frac,
            )

        # Get surrounding control points
        p0 = points[max(0, seg_idx - 1)]
        p3 = points[min(len(points) - 1, seg_idx + 2)]

        # Catmull-Rom matrix multiplication
        t2 = frac * frac
        t3 = t2 * frac

        x = 0.5 * (
            (2 * p1["x"])
            + (-p0["x"] + p2["x"]) * frac
            + (2 * p0["x"] - 5 * p1["x"] + 4 * p2["x"] - p3["x"]) * t2
            + (-p0["x"] + 3 * p1["x"] - 3 * p2["x"] + p3["x"]) * t3
        )
        y = 0.5 * (
            (2 * p1["y"])
            + (-p0["y"] + p2["y"]) * frac
            + (2 * p0["y"] - 5 * p1["y"] + 4 * p2["y"] - p3["y"]) * t2
            + (-p0["y"] + 3 * p1["y"] - 3 * p2["y"] + p3["y"]) * t3
        )

        # Clamp to valid range
        return (max(0.0, min(1.0, x)), max(0.0, min(1.0, y)))

    _skill_cache: Dict[str, float] = {}
    _profiles_loaded: bool = False

    def _get_player_skill(self, name: str) -> float:
        """Get player skill (0-1) from VCT profile or default."""
        if not name:
            return DEFAULT_PRO_SKILL
        if name in self._skill_cache:
            return self._skill_cache[name]

        # Use already-loaded profiles only (no GRID loading — too slow)
        profile = profile_manager.profiles.get(name) if profile_manager.profiles else None
        if profile:
            hs_rate = profile.headshot_rate
            skill = min(1.0, max(0.0, (hs_rate - 0.15) / 0.20))
            self._skill_cache[name] = skill
            return skill

        self._skill_cache[name] = DEFAULT_PRO_SKILL
        return DEFAULT_PRO_SKILL

    @staticmethod
    def _is_peeking(positions: List[Dict], t_sec: int) -> bool:
        """Check if player is peeking (moved significantly in last second)."""
        if t_sec < 1 or t_sec >= len(positions):
            return False
        prev = positions[t_sec - 1]
        curr = positions[t_sec]
        dist = ((curr["x"] - prev["x"]) ** 2 + (curr["y"] - prev["y"]) ** 2) ** 0.5
        return dist > 0.01

    def _simulate_combat(
        self,
        user_positions: Dict[str, List[Dict]],
        opponent_positions: Dict[str, List[Dict]],
        user_side: str,
        opponent_side: str,
        map_name: str,
        teammates: List[Dict],
        opponent_names: List[str],
        round_duration: int = 100,
        phase_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
    ) -> Tuple[List[Dict], List[Dict], str]:
        """Simulate combat between user team and opponents."""
        if phase_ranges is None:
            phase_ranges = {
                "setup": (0, 15), "mid_round": (15, 40),
                "execute": (40, 70), "post_plant": (70, 100),
            }

        events: List[Dict] = []
        snapshots: List[Dict] = []

        # Health tracking (replaces bool alive dicts)
        user_health: Dict[str, float] = {pid: 100.0 for pid in user_positions}
        opp_health: Dict[str, float] = {pid: 100.0 for pid in opponent_positions}

        # Build name maps
        user_names: Dict[str, Dict] = {}
        user_weapons: Dict[str, str] = {}
        for t in teammates:
            pid = t["player_id"] if isinstance(t, dict) else t.player_id
            name = t["name"] if isinstance(t, dict) else t.name
            agent = t["agent"] if isinstance(t, dict) else t.agent
            weapon = t.get("weapon", "vandal") if isinstance(t, dict) else getattr(t, "weapon", "vandal")
            user_names[pid] = {"name": name, "agent": agent}
            user_weapons[pid] = weapon.lower() if weapon else "vandal"

        opp_name_map: Dict[str, Dict] = {}
        for i, (pid, positions) in enumerate(opponent_positions.items()):
            name = opponent_names[i] if i < len(opponent_names) else f"Opponent {i+1}"
            opp_name_map[pid] = {"name": name, "agent": "unknown"}

        pr = phase_ranges
        phase_at = lambda t: (
            "setup" if t < pr["mid_round"][0] else
            "mid_round" if t < pr["execute"][0] else
            "execute" if t < pr["post_plant"][0] else
            "post_plant"
        )

        spike_planted = False
        spike_site = None

        # Track death positions for freezing
        death_positions: Dict[str, Tuple[float, float]] = {}

        # Engagement cooldown: (u_pid, o_pid) -> last engagement time
        last_engagement: Dict[Tuple[str, str], int] = {}

        for t_sec in range(0, round_duration):
            phase = phase_at(t_sec)

            # Check engagements
            for u_pid in user_positions:
                if user_health[u_pid] <= 0:
                    continue
                u_pos = user_positions[u_pid][t_sec] if t_sec < len(user_positions[u_pid]) else None
                if not u_pos:
                    continue

                for o_pid in opponent_positions:
                    if opp_health[o_pid] <= 0:
                        continue
                    o_pos = opponent_positions[o_pid][t_sec] if t_sec < len(opponent_positions[o_pid]) else None
                    if not o_pos:
                        continue

                    # Engagement cooldown
                    pair_key = (u_pid, o_pid)
                    if pair_key in last_engagement and t_sec - last_engagement[pair_key] < ENGAGEMENT_COOLDOWN:
                        continue

                    dist = math.sqrt((u_pos["x"] - o_pos["x"])**2 + (u_pos["y"] - o_pos["y"])**2)

                    if dist < ENGAGEMENT_RANGE:
                        last_engagement[pair_key] = t_sec

                        # Determine peeking from movement
                        u_peeking = self._is_peeking(user_positions[u_pid], t_sec)
                        o_peeking = self._is_peeking(opponent_positions[o_pid], t_sec)

                        # Get player skills
                        u_name = user_names.get(u_pid, {}).get("name", "")
                        o_name = opp_name_map.get(o_pid, {}).get("name", "")
                        u_skill = self._get_player_skill(u_name)
                        o_skill = self._get_player_skill(o_name)

                        # Weapons
                        u_weapon = user_weapons.get(u_pid, "vandal")
                        o_weapon = "vandal"

                        # Resolve combat via realistic model
                        dist_meters = meters_from_normalized(dist)
                        try:
                            result = resolve_combat(
                                attacker_skill=u_skill,
                                attacker_weapon=u_weapon,
                                attacker_health=user_health[u_pid],
                                defender_skill=o_skill,
                                defender_weapon=o_weapon,
                                defender_health=opp_health[o_pid],
                                distance_meters=dist_meters,
                                attacker_peeking=u_peeking,
                                defender_peeking=o_peeking,
                            )
                        except Exception:
                            continue

                        # Apply damage from combat result
                        if result.winner_id == "player_a":
                            # User player won the duel
                            opp_health[o_pid] -= result.damage_dealt_winner
                            user_health[u_pid] -= result.damage_dealt_loser
                            if opp_health[o_pid] <= 0:
                                opp_health[o_pid] = 0
                                death_positions[o_pid] = (o_pos["x"], o_pos["y"])
                                events.append({
                                    "time_ms": t_sec * 1000,
                                    "event_type": "kill",
                                    "player_id": u_pid,
                                    "target_id": o_pid,
                                    "details": {
                                        "killer_name": u_name or u_pid,
                                        "victim_name": o_name or o_pid,
                                        "headshot": result.headshot_kill,
                                        "weapon": u_weapon,
                                        "ttk_ms": round(result.time_to_kill_ms),
                                        "damage_to_killer": round(result.damage_dealt_loser),
                                    },
                                })
                            if user_health[u_pid] <= 0:
                                user_health[u_pid] = 0
                                death_positions[u_pid] = (u_pos["x"], u_pos["y"])
                        else:
                            # Opponent won the duel
                            user_health[u_pid] -= result.damage_dealt_winner
                            opp_health[o_pid] -= result.damage_dealt_loser
                            if user_health[u_pid] <= 0:
                                user_health[u_pid] = 0
                                death_positions[u_pid] = (u_pos["x"], u_pos["y"])
                                events.append({
                                    "time_ms": t_sec * 1000,
                                    "event_type": "kill",
                                    "player_id": o_pid,
                                    "target_id": u_pid,
                                    "details": {
                                        "killer_name": o_name or o_pid,
                                        "victim_name": u_name or u_pid,
                                        "headshot": result.headshot_kill,
                                        "weapon": o_weapon,
                                        "ttk_ms": round(result.time_to_kill_ms),
                                        "damage_to_killer": round(result.damage_dealt_loser),
                                    },
                                })
                            if opp_health[o_pid] <= 0:
                                opp_health[o_pid] = 0
                                death_positions[o_pid] = (o_pos["x"], o_pos["y"])

            # Spike plant check (attack side, execute phase)
            if not spike_planted and phase in ("execute", "post_plant") and user_side == "attack":
                alive_attackers = sum(1 for h in user_health.values() if h > 0)
                exec_start = phase_ranges["execute"][0]
                if alive_attackers > 0 and t_sec > exec_start + 5 and random.random() < 0.08:
                    spike_planted = True
                    spike_site = random.choice(["A", "B"])
                    events.append({
                        "time_ms": t_sec * 1000,
                        "event_type": "spike_plant",
                        "player_id": None,
                        "details": {"site": spike_site},
                    })

            # Snapshot every second for smooth playback
            if t_sec % 1 == 0:
                players = []
                for pid in user_positions:
                    pos = user_positions[pid][t_sec] if t_sec < len(user_positions[pid]) else None
                    if pos:
                        info = user_names.get(pid, {})
                        is_alive = user_health[pid] > 0
                        px, py = pos["x"], pos["y"]
                        if not is_alive and pid in death_positions:
                            px, py = death_positions[pid]
                        # Compute facing angle from movement direction
                        facing = None
                        if is_alive and t_sec > 0 and t_sec - 1 < len(user_positions[pid]):
                            prev = user_positions[pid][t_sec - 1]
                            dx = px - prev["x"]
                            dy = py - prev["y"]
                            if abs(dx) > 0.0001 or abs(dy) > 0.0001:
                                facing = round(math.atan2(dy, dx), 4)
                        players.append({
                            "player_id": pid,
                            "x": round(px, 4),
                            "y": round(py, 4),
                            "side": user_side,
                            "is_alive": is_alive,
                            "agent": info.get("agent", "unknown"),
                            "name": info.get("name", pid),
                            "health": max(0, int(user_health[pid])),
                            "team_id": "user",
                            "facing_angle": facing,
                        })
                for pid in opponent_positions:
                    pos = opponent_positions[pid][t_sec] if t_sec < len(opponent_positions[pid]) else None
                    if pos:
                        info = opp_name_map.get(pid, {})
                        is_alive = opp_health[pid] > 0
                        px, py = pos["x"], pos["y"]
                        if not is_alive and pid in death_positions:
                            px, py = death_positions[pid]
                        # Compute facing angle from movement direction
                        facing = None
                        if is_alive and t_sec > 0 and t_sec - 1 < len(opponent_positions[pid]):
                            prev = opponent_positions[pid][t_sec - 1]
                            dx = px - prev["x"]
                            dy = py - prev["y"]
                            if abs(dx) > 0.0001 or abs(dy) > 0.0001:
                                facing = round(math.atan2(dy, dx), 4)
                        players.append({
                            "player_id": pid,
                            "x": round(px, 4),
                            "y": round(py, 4),
                            "side": opponent_side,
                            "is_alive": is_alive,
                            "agent": info.get("agent", "unknown"),
                            "name": info.get("name", pid),
                            "health": max(0, int(opp_health[pid])),
                            "team_id": "opponent",
                            "facing_angle": facing,
                        })

                snapshots.append({
                    "time_ms": t_sec * 1000,
                    "phase": phase,
                    "players": players,
                })

            # Check round end
            user_alive_count = sum(1 for h in user_health.values() if h > 0)
            opp_alive_count = sum(1 for h in opp_health.values() if h > 0)

            if user_alive_count == 0 or opp_alive_count == 0:
                break

        # Determine winner
        user_alive_count = sum(1 for h in user_health.values() if h > 0)
        opp_alive_count = sum(1 for h in opp_health.values() if h > 0)

        if user_alive_count > opp_alive_count:
            winner = user_side
        elif opp_alive_count > user_alive_count:
            winner = opponent_side
        else:
            # Tie: spike decides
            if spike_planted:
                winner = "attack"
            else:
                winner = "defense"

        return events, snapshots, winner
