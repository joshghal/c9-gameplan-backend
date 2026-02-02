"""
VCT Round Service — loads position_trajectories.json, indexes rounds by map+side,
serves random rounds with normalized coordinates for the strategy planner.
"""

import json
import random
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..schemas.strategy import StrategyRound, TeamMember, GhostPath, GhostPoint
from .pathfinding import AStarPathfinder

DATA_DIR = Path(__file__).parent.parent / "data"

# Per-map affine transforms: mx = scale * raw_y + offset_x, my = -scale * raw_x + offset_y
# Derived from extracted_player_positions.json (R² = 1.0 for all maps)
MAP_TRANSFORMS: Dict[str, Tuple[float, float, float]] = {
    # map_name: (scale, offset_x, offset_y)
    # mx = scale * y + offset_x
    # my = -scale * x + offset_y
    "abyss":    (0.0000810, 0.500000, 0.500000),
    "ascent":   (0.0000700, 0.813895, 0.573242),
    "bind":     (0.0000590, 0.576941, 0.967566),
    "corrode":  (0.0000700, 0.813895, 0.573242),
    "fracture": (0.0000780, 0.556952, 1.155886),
    "haven":    (0.0000750, 1.093450, 0.642728),
    "icebox":   (0.0000720, 0.460214, 0.304687),
    "lotus":    (0.0000720, 0.454789, 0.917752),
    "pearl":    (0.0000780, 0.480469, 0.916016),
    "split":    (0.0000780, 0.842188, 0.697578),
    "sunset":   (0.0000780, 0.500000, 0.515625),
}

# VCT-derived spawn barrier positions per map per side (normalized 0-1 coords)
# Source: simulation_engine.py MAP_DATA (extracted from actual pro match data)
MAP_SPAWN_POSITIONS: Dict[str, Dict[str, List[Tuple[float, float]]]] = {
    "ascent": {
        "attack": [(0.85, 0.58), (0.83, 0.55), (0.87, 0.55), (0.82, 0.60), (0.88, 0.60)],
        "defense": [(0.15, 0.42), (0.13, 0.40), (0.17, 0.40), (0.12, 0.44), (0.18, 0.44)],
    },
    "bind": {
        "attack": [(0.55, 0.90), (0.52, 0.88), (0.58, 0.88), (0.50, 0.92), (0.60, 0.92)],
        "defense": [(0.55, 0.15), (0.52, 0.13), (0.58, 0.13), (0.50, 0.17), (0.60, 0.17)],
    },
    "split": {
        "attack": [(0.82, 0.55), (0.80, 0.52), (0.84, 0.52), (0.79, 0.58), (0.85, 0.58)],
        "defense": [(0.15, 0.55), (0.13, 0.52), (0.17, 0.52), (0.12, 0.58), (0.18, 0.58)],
    },
    "icebox": {
        "attack": [(0.12, 0.58), (0.10, 0.55), (0.14, 0.55), (0.09, 0.61), (0.15, 0.61)],
        "defense": [(0.92, 0.55), (0.90, 0.52), (0.94, 0.52), (0.89, 0.58), (0.95, 0.58)],
    },
    "fracture": {
        "attack": [(0.48, 0.87), (0.45, 0.85), (0.51, 0.85), (0.43, 0.89), (0.53, 0.89)],
        "defense": [(0.60, 0.42), (0.57, 0.40), (0.63, 0.40), (0.55, 0.44), (0.65, 0.44)],
    },
    "pearl": {
        "attack": [(0.52, 0.90), (0.49, 0.88), (0.55, 0.88), (0.47, 0.92), (0.57, 0.92)],
        "defense": [(0.52, 0.10), (0.49, 0.08), (0.55, 0.08), (0.47, 0.12), (0.57, 0.12)],
    },
    "sunset": {
        "attack": [(0.55, 0.93), (0.52, 0.91), (0.58, 0.91), (0.50, 0.95), (0.60, 0.95)],
        "defense": [(0.55, 0.12), (0.52, 0.10), (0.58, 0.10), (0.50, 0.14), (0.60, 0.14)],
    },
    "abyss": {
        "attack": [(0.88, 0.42), (0.86, 0.40), (0.90, 0.40), (0.85, 0.44), (0.91, 0.44)],
        "defense": [(0.12, 0.45), (0.10, 0.43), (0.14, 0.43), (0.09, 0.47), (0.15, 0.47)],
    },
    "corrode": {
        "attack": [(0.85, 0.50), (0.83, 0.48), (0.87, 0.48), (0.82, 0.52), (0.88, 0.52)],
        "defense": [(0.10, 0.50), (0.08, 0.48), (0.12, 0.48), (0.07, 0.52), (0.13, 0.52)],
    },
    "haven": {
        "attack": [(0.87, 0.55), (0.85, 0.52), (0.89, 0.52), (0.84, 0.58), (0.90, 0.58)],
        "defense": [(0.12, 0.40), (0.10, 0.38), (0.14, 0.38), (0.09, 0.42), (0.15, 0.42)],
    },
    "lotus": {
        "attack": [(0.50, 0.85), (0.47, 0.83), (0.53, 0.83), (0.45, 0.87), (0.55, 0.87)],
        "defense": [(0.55, 0.15), (0.52, 0.13), (0.58, 0.13), (0.50, 0.17), (0.60, 0.17)],
    },
}

# Role → typical agent mapping
ROLE_AGENTS = {
    "duelist": ["jett", "raze", "reyna", "neon", "iso", "yoru"],
    "controller": ["omen", "astra", "brimstone", "viper", "harbor", "clove"],
    "initiator": ["sova", "fade", "skye", "breach", "kayo", "gekko"],
    "sentinel": ["killjoy", "cypher", "sage", "chamber", "deadlock"],
}

# Default weapons by buy type
DEFAULT_WEAPON = "vandal"

# Side normalization
SIDE_MAP = {
    "attacker": "attack",
    "defender": "defense",
    "attack": "attack",
    "defense": "defense",
}


class VCTRoundService:
    """Lazy-loaded, cached service for VCT round data."""

    _instance: Optional["VCTRoundService"] = None
    _loaded: bool = False

    def __init__(self):
        self._trajectories: Dict = {}
        self._index: Dict[str, Dict[str, List[int]]] = {}  # {map: {attack: [idx], defense: [idx]}}
        self._rounds: List[Dict] = []  # flat list of all rounds
        self._profiles: Dict[str, Dict] = {}  # player_name → profile data
        self._match_metadata: Dict[str, Dict] = {}  # game_id → {tournament, date, teams, ...}

    @classmethod
    def get_instance(cls) -> "VCTRoundService":
        if cls._instance is None:
            cls._instance = cls()
        if not cls._loaded:
            cls._instance._load()
            cls._loaded = True
        return cls._instance

    def _load(self):
        """Load and index trajectory data."""
        traj_path = DATA_DIR / "position_trajectories.json"
        data = json.loads(traj_path.read_text())
        traj_by_map = data.get("trajectories_by_map", {})

        # Load player profiles for name/role/weapon lookup
        profiles_path = DATA_DIR / "simulation_profiles.json"
        if profiles_path.exists():
            prof_data = json.loads(profiles_path.read_text())
            pb = prof_data.get("player_behaviors", {})
            for group_name, players in pb.items():
                if isinstance(players, list):
                    for p in players:
                        name = p.get("name", "")
                        if name:
                            self._profiles[name.lower()] = p

        # Load match metadata (tournament, date, teams) from GRID extraction
        meta_path = DATA_DIR / "vct_match_metadata.json"
        if meta_path.exists():
            self._match_metadata = json.loads(meta_path.read_text())

        # Build flat round list + index
        round_idx = 0
        for map_name, rounds in traj_by_map.items():
            for round_data in rounds:
                round_data["_map"] = map_name
                self._rounds.append(round_data)

                # Determine which side each team is on
                sides_found = {"attack": False, "defense": False}
                for player_name, traj in round_data.get("player_trajectories", {}).items():
                    if traj:
                        raw_side = SIDE_MAP.get(traj[0].get("side", ""), "")
                        if raw_side:
                            sides_found[raw_side] = True

                # Index this round for both sides (user can play either)
                self._index.setdefault(map_name, {"attack": [], "defense": []})
                if sides_found["attack"] and sides_found["defense"]:
                    self._index[map_name]["attack"].append(round_idx)
                    self._index[map_name]["defense"].append(round_idx)

                round_idx += 1

    def get_match_metadata(self, game_id: str) -> Optional[Dict]:
        """Get tournament/date/teams metadata for a game_id."""
        return self._match_metadata.get(game_id)

    def normalize_coords(self, map_name: str, raw_x: float, raw_y: float) -> Tuple[float, float]:
        """Convert raw game coords to normalized 0-1 minimap coords."""
        transform = MAP_TRANSFORMS.get(map_name)
        if not transform:
            # Fallback: generic transform
            transform = (0.00007, 0.5, 0.5)

        scale, offset_x, offset_y = transform
        mx = scale * raw_y + offset_x
        my = -scale * raw_x + offset_y
        return (max(0.0, min(1.0, mx)), max(0.0, min(1.0, my)))

    def get_random_round(self, map_name: str, side: str) -> Optional[StrategyRound]:
        """Pick a random VCT round and return it formatted for the strategy planner."""
        map_name = map_name.lower()
        side = side.lower()

        if map_name not in self._index:
            return None

        indices = self._index[map_name].get(side, [])
        if not indices:
            return None

        idx = random.choice(indices)
        round_data = self._rounds[idx]

        return self._format_round(round_data, map_name, side)

    def get_round_by_id(self, round_id: str) -> Optional[Dict]:
        """Retrieve a specific round by its ID. Returns raw round data + map + side."""
        # round_id format: "{map}_{round_num}_{hash8}"
        parts = round_id.rsplit("_", 2)
        if len(parts) < 3:
            return None

        target_map = parts[0]
        try:
            target_round = int(parts[1])
        except ValueError:
            return None
        target_hash = parts[2]

        # Search for matching round
        for rd in self._rounds:
            if rd.get("_map") == target_map and rd.get("round_num") == target_round:
                game_id = rd.get("game_id", "")
                h = hashlib.md5(f"{target_map}_{target_round}_{game_id}".encode()).hexdigest()[:8]
                if h == target_hash:
                    return rd
        return None

    def _format_round(self, round_data: Dict, map_name: str, user_side: str) -> StrategyRound:
        """Format a raw round into a StrategyRound for the API."""
        round_num = round_data.get("round_num", 0)
        game_id = round_data.get("game_id", "")
        round_hash = hashlib.md5(f"{map_name}_{round_num}_{game_id}".encode()).hexdigest()[:8]
        round_id = f"{map_name}_{round_num}_{round_hash}"

        duration = round_data.get("round_duration_s", 100)
        # VCT-derived phase times for ghost path segmentation (in seconds)
        phase_times = {
            "setup": [0, round(duration * 0.15)],
            "mid_round": [round(duration * 0.15), round(duration * 0.40)],
            "execute": [round(duration * 0.40), round(duration * 0.70)],
            "post_plant": [round(duration * 0.70), duration],
        }
        # Tactical phase times aligned with backend TACTICAL_PHASE_RANGES (in seconds)
        # Frontend uses these for waypoint tick calculation, must match engine simulation time
        tactical_phase_times = {
            "setup": [0, 15],
            "mid_round": [15, 50],
            "execute": [50, 75],
            "post_plant": [75, 100],
        }

        # Separate players by side, collect trajectories for ghost paths
        user_players = []
        user_trajectories: List[tuple] = []  # (player_name, agent, trajectory)
        spawn_list = MAP_SPAWN_POSITIONS.get(map_name, {}).get(user_side, [])
        for player_name, trajectory in round_data.get("player_trajectories", {}).items():
            if not trajectory:
                continue
            raw_side = SIDE_MAP.get(trajectory[0].get("side", ""), "")
            if raw_side == user_side:
                player_idx = len(user_players)
                # Use map spawn positions (authoritative), fall back to first trajectory point
                if spawn_list and player_idx < len(spawn_list):
                    spawn_x, spawn_y = spawn_list[player_idx]
                else:
                    first_pos = trajectory[0]
                    spawn_x, spawn_y = self.normalize_coords(map_name, first_pos["x"], first_pos["y"])

                profile = self._profiles.get(player_name.lower(), {})
                role = profile.get("primary_role", "flex")
                weapon = profile.get("primary_weapon", DEFAULT_WEAPON)
                agent = self._pick_agent(role, len(user_players))

                user_players.append(TeamMember(
                    player_id=f"t{player_idx + 1}",
                    name=player_name,
                    role=role,
                    agent=agent,
                    weapon=weapon,
                    spawn=[round(spawn_x, 4), round(spawn_y, 4)],
                ))
                user_trajectories.append((player_name, agent, trajectory))

        # Ensure exactly 5 teammates
        while len(user_players) > 5:
            user_players.pop()
            if len(user_trajectories) > 5:
                user_trajectories.pop()
        while len(user_players) < 5:
            idx = len(user_players)
            user_players.append(TeamMember(
                player_id=f"t{idx + 1}",
                name=f"Player {idx + 1}",
                role="flex",
                agent=self._pick_agent("flex", idx),
                weapon=DEFAULT_WEAPON,
                spawn=[0.5, 0.5],
            ))

        # Initialize pathfinder for walkable ghost paths
        pathfinder = None
        try:
            pf = AStarPathfinder()
            pf.load_nav_grid_from_v4(map_name)
            pathfinder = pf
        except Exception:
            pass

        # Build ghost paths from user-side VCT trajectories
        ghost_paths = []
        for i, player in enumerate(user_players):
            if i < len(user_trajectories):
                pname, agent, traj = user_trajectories[i]
                # Convert trajectory to normalized time_s anchor points
                raw_points = []
                for pt in traj:
                    time_s = 100 - pt["clock"]
                    nx, ny = self.normalize_coords(map_name, pt["x"], pt["y"])
                    raw_points.append({
                        "time_s": round(time_s, 1), "x": round(nx, 4), "y": round(ny, 4),
                        "alive": pt.get("alive", True),
                    })
                raw_points.sort(key=lambda p: p["time_s"])

                # Deduplicate timestamps — VCT data often has multiple positions
                # at the same clock value (spectator/death positions). Keep first.
                seen_times: set = set()
                deduped: List[Dict] = []
                for pt in raw_points:
                    if pt["time_s"] not in seen_times:
                        deduped.append(pt)
                        seen_times.add(pt["time_s"])
                raw_points = deduped

                # Truncate at death — positions after death are spectator camera
                death_time: Optional[float] = None
                truncated: List[Dict] = []
                for pt in raw_points:
                    if not pt["alive"]:
                        death_time = pt["time_s"]
                        break
                    truncated.append(pt)
                raw_points = truncated if truncated else raw_points[:1]

                # Filter out spectator camera jumps — impossible movement speed
                # Max VALORANT run speed ~5.4m/s on ~10000 unit maps = ~0.02 norm/s
                # Use generous threshold to allow for teleports/abilities
                MAX_SPEED = 0.06  # normalized units per second
                if len(raw_points) > 2:
                    filtered: List[Dict] = [raw_points[0]]
                    for j in range(1, len(raw_points)):
                        dt = raw_points[j]["time_s"] - filtered[-1]["time_s"]
                        if dt <= 0:
                            continue
                        dx = raw_points[j]["x"] - filtered[-1]["x"]
                        dy = raw_points[j]["y"] - filtered[-1]["y"]
                        speed = (dx * dx + dy * dy) ** 0.5 / dt
                        if speed <= MAX_SPEED:
                            filtered.append(raw_points[j])
                        # else: skip spectator camera jump
                    raw_points = filtered

                if not raw_points:
                    ghost_paths.append(GhostPath(
                        player_id=player.player_id, name=player.name,
                        agent=player.agent, segments={p: [] for p in phase_times},
                    ))
                    continue

                # Snap all anchor points to walkable cells
                if pathfinder:
                    for pt in raw_points:
                        pt["x"], pt["y"] = _snap_to_walkable(pathfinder, pt["x"], pt["y"])

                # Snap spawn to walkable cell
                spawn_x, spawn_y = player.spawn[0], player.spawn[1]
                if pathfinder:
                    spawn_x, spawn_y = _snap_to_walkable(pathfinder, spawn_x, spawn_y)

                # Build A* paths between all consecutive anchors
                anchor_paths: List[List[Tuple[float, float]]] = []
                for ai in range(len(raw_points) - 1):
                    a1 = raw_points[ai]
                    a2 = raw_points[ai + 1]
                    if pathfinder:
                        result = pathfinder.find_path((a1["x"], a1["y"]), (a2["x"], a2["y"]), simplify=False)
                        if result.success and result.path and len(result.path) > 1:
                            anchor_paths.append(result.path)
                        else:
                            # A* failed — re-snap endpoints with larger radius and retry
                            s1 = _snap_to_walkable(pathfinder, a1["x"], a1["y"])
                            s2 = _snap_to_walkable(pathfinder, a2["x"], a2["y"])
                            retry = pathfinder.find_path(s1, s2, simplify=False)
                            if retry.success and retry.path and len(retry.path) > 1:
                                anchor_paths.append(retry.path)
                            else:
                                anchor_paths.append([s1, s2])
                    else:
                        anchor_paths.append([(a1["x"], a1["y"]), (a2["x"], a2["y"])])

                # A* path from spawn to first anchor (always, even if first_time=0)
                spawn_path: List[Tuple[float, float]] = [(spawn_x, spawn_y)]
                first_time = raw_points[0]["time_s"]
                first_anchor = (raw_points[0]["x"], raw_points[0]["y"])
                # Only build A* if spawn != first anchor (more than trivial distance)
                spawn_dist = ((spawn_x - first_anchor[0]) ** 2 + (spawn_y - first_anchor[1]) ** 2) ** 0.5
                if spawn_dist > 0.01 and pathfinder:
                    result = pathfinder.find_path(
                        (spawn_x, spawn_y), first_anchor, simplify=False,
                    )
                    if result.success and result.path:
                        spawn_path = result.path
                    else:
                        spawn_path = [(spawn_x, spawn_y), first_anchor]

                # Time to reach first anchor: use first_time if > 0, else assume ~3s setup
                reach_time = max(first_time, 3.0)

                # Interpolate per-second positions
                interpolated: List[GhostPoint] = []
                last_alive_pos = (raw_points[-1]["x"], raw_points[-1]["y"])
                for t in range(0, duration):
                    t_f = float(t)

                    # After death: freeze at last alive position
                    if death_time is not None and t_f >= death_time:
                        interpolated.append(GhostPoint(
                            time_s=t_f, x=last_alive_pos[0], y=last_alive_pos[1],
                        ))
                        continue

                    # Before reaching first anchor: walk from spawn via A* path
                    if t_f < reach_time:
                        frac = t_f / reach_time
                        pos_f = frac * (len(spawn_path) - 1)
                        idx_lo = min(int(pos_f), len(spawn_path) - 1)
                        idx_hi = min(idx_lo + 1, len(spawn_path) - 1)
                        sub_frac = pos_f - idx_lo
                        px = spawn_path[idx_lo][0] + (spawn_path[idx_hi][0] - spawn_path[idx_lo][0]) * sub_frac
                        py = spawn_path[idx_lo][1] + (spawn_path[idx_hi][1] - spawn_path[idx_lo][1]) * sub_frac
                        last_alive_pos = (round(px, 4), round(py, 4))
                        interpolated.append(GhostPoint(time_s=t_f, x=last_alive_pos[0], y=last_alive_pos[1]))
                        continue

                    # After last anchor: hold position
                    if t_f >= raw_points[-1]["time_s"]:
                        last_alive_pos = (raw_points[-1]["x"], raw_points[-1]["y"])
                        interpolated.append(GhostPoint(
                            time_s=t_f, x=last_alive_pos[0], y=last_alive_pos[1],
                        ))
                        continue

                    # Between anchors: walk along A* path with linear sub-interpolation
                    for ai in range(len(raw_points) - 1):
                        a1 = raw_points[ai]
                        a2 = raw_points[ai + 1]
                        if a1["time_s"] <= t_f < a2["time_s"]:
                            dt = a2["time_s"] - a1["time_s"]
                            frac = (t_f - a1["time_s"]) / dt if dt > 0 else 0.0
                            path = anchor_paths[ai]
                            # Smoothly interpolate along A* path
                            pos_f = frac * (len(path) - 1)
                            idx_lo = min(int(pos_f), len(path) - 1)
                            idx_hi = min(idx_lo + 1, len(path) - 1)
                            sub_frac = pos_f - idx_lo
                            x = path[idx_lo][0] + (path[idx_hi][0] - path[idx_lo][0]) * sub_frac
                            y = path[idx_lo][1] + (path[idx_hi][1] - path[idx_lo][1]) * sub_frac
                            last_alive_pos = (round(x, 4), round(y, 4))
                            interpolated.append(GhostPoint(time_s=t_f, x=last_alive_pos[0], y=last_alive_pos[1]))
                            break

                # Split into phase segments with overlap for continuity
                segments: Dict[str, List[GhostPoint]] = {}
                phase_list = list(phase_times.keys())
                for pi, phase_name in enumerate(phase_list):
                    start, end = phase_times[phase_name]
                    seg = [p for p in interpolated if start <= p.time_s < end]
                    # Add last point from previous phase as first point for continuity
                    if pi > 0:
                        prev_phase = phase_list[pi - 1]
                        prev_seg = segments.get(prev_phase, [])
                        if prev_seg and (not seg or seg[0].time_s > prev_seg[-1].time_s):
                            seg.insert(0, prev_seg[-1])
                    segments[phase_name] = seg

                ghost_paths.append(GhostPath(
                    player_id=player.player_id,
                    name=player.name,
                    agent=player.agent,
                    segments=segments,
                ))
            else:
                # Filler player — empty ghost path
                ghost_paths.append(GhostPath(
                    player_id=player.player_id,
                    name=player.name,
                    agent=player.agent,
                    segments={p: [] for p in phase_times},
                ))

        return StrategyRound(
            round_id=round_id,
            map_name=map_name,
            user_side=user_side,
            teammates=user_players,
            phase_times=tactical_phase_times,
            round_duration_s=100,  # Match tactical engine's 100s total
            ghost_paths=ghost_paths,
        )

    def _pick_agent(self, role: str, index: int) -> str:
        """Pick an agent based on role, using index for variety."""
        agents = ROLE_AGENTS.get(role, ROLE_AGENTS["duelist"])
        return agents[index % len(agents)]

    def get_available_maps(self) -> List[str]:
        """Return list of maps that have indexed rounds."""
        return sorted(self._index.keys())

    def _detect_c9_side(self, round_data: Dict) -> str:
        """Detect which side Cloud9 played from trajectory data."""
        c9_keywords = {"cloud9", "c9"}
        meta = self.get_match_metadata(round_data.get("game_id", ""))
        teams = meta.get("teams", []) if meta else []

        # Find which team name is C9
        c9_team = None
        for t in teams:
            if any(kw in t.lower() for kw in c9_keywords):
                c9_team = t
                break

        # Check player trajectories: find a player whose team matches C9
        for player_name, traj in round_data.get("player_trajectories", {}).items():
            if not traj:
                continue
            # Trajectory points have "team" (team name) and "side" (attacker/defender)
            team_field = traj[0].get("team", "") or traj[0].get("team_id", "")
            raw_side = SIDE_MAP.get(traj[0].get("side", ""), "")
            if any(kw in team_field.lower() for kw in c9_keywords):
                return raw_side or "attack"

        return "attack"  # default fallback

    def list_rounds(self, map_name: str) -> List[Dict]:
        """List all rounds for a map with match metadata."""
        results = []
        seen_ids = set()
        # Use attack index only (rounds are indexed under both sides)
        indices = self._index.get(map_name, {}).get("attack", [])
        # Also grab defense-only rounds
        def_indices = self._index.get(map_name, {}).get("defense", [])
        all_indices = list(set(indices + def_indices))
        for idx in all_indices:
            rd = self._rounds[idx]
            round_num = rd.get("round_num", 0)
            game_id = rd.get("game_id", "")
            round_hash = hashlib.md5(
                f"{map_name}_{round_num}_{game_id}".encode()
            ).hexdigest()[:8]
            rid = f"{map_name}_{round_num}_{round_hash}"
            if rid in seen_ids:
                continue
            seen_ids.add(rid)
            meta = self.get_match_metadata(game_id)
            c9_side = self._detect_c9_side(rd)
            results.append({
                "round_id": rid,
                "round_num": round_num,
                "teams": meta.get("teams", []) if meta else [],
                "date": meta.get("date", "") if meta else "",
                "tournament": meta.get("tournament", "") if meta else "",
                "winner": rd.get("winner_team", ""),
                "side": c9_side,
                "duration_s": rd.get("round_duration_s", 0),
            })
        results.sort(key=lambda r: (r["date"], r["round_num"]), reverse=True)
        return results

    def get_opponent_trajectories(self, round_id: str, user_side: str) -> Optional[Dict[str, List[Dict]]]:
        """Get opponent player trajectories for a round (normalized, filtered).

        Applies the same filtering pipeline as ghost paths:
        1. Timestamp dedup
        2. Death truncation
        3. Speed filter (removes spectator camera jumps)
        """
        round_data = self.get_round_by_id(round_id)
        if not round_data:
            return None

        map_name = round_data.get("_map", "")
        opponent_side = "defense" if user_side == "attack" else "attack"
        opponents = {}

        MAX_SPEED = 0.06  # normalized units per second

        for player_name, trajectory in round_data.get("player_trajectories", {}).items():
            if not trajectory:
                continue
            raw_side = SIDE_MAP.get(trajectory[0].get("side", ""), "")
            if raw_side == opponent_side:
                # Normalize coords and compute time_s
                normalized_traj = []
                for pt in trajectory:
                    nx, ny = self.normalize_coords(map_name, pt["x"], pt["y"])
                    time_s = 100 - pt["clock"]
                    normalized_traj.append({
                        "clock": pt["clock"],
                        "time_s": round(time_s, 1),
                        "x": round(nx, 4),
                        "y": round(ny, 4),
                        "alive": pt["alive"],
                    })
                normalized_traj.sort(key=lambda p: p["time_s"])

                # 1. Dedup timestamps (keep first per time_s)
                seen_times: set = set()
                deduped = []
                for pt in normalized_traj:
                    if pt["time_s"] not in seen_times:
                        deduped.append(pt)
                        seen_times.add(pt["time_s"])

                # 2. Death truncation (stop at first alive=False)
                truncated = []
                for pt in deduped:
                    if not pt["alive"]:
                        break
                    truncated.append(pt)
                if not truncated:
                    truncated = deduped[:1]

                # 3. Speed filter (remove spectator camera jumps)
                if len(truncated) > 2:
                    filtered = [truncated[0]]
                    for j in range(1, len(truncated)):
                        dt = truncated[j]["time_s"] - filtered[-1]["time_s"]
                        if dt <= 0:
                            continue
                        dx = truncated[j]["x"] - filtered[-1]["x"]
                        dy = truncated[j]["y"] - filtered[-1]["y"]
                        speed = (dx ** 2 + dy ** 2) ** 0.5 / dt
                        if speed <= MAX_SPEED:
                            filtered.append(truncated[j])
                    truncated = filtered

                opponents[player_name] = truncated

        return opponents


def _snap_to_walkable(
    pathfinder: AStarPathfinder, x: float, y: float,
) -> Tuple[float, float]:
    """Snap a normalized coordinate to the nearest walkable cell."""
    gx, gy = pathfinder.normalized_to_grid(x, y)
    if pathfinder.is_walkable(gx, gy):
        return (x, y)
    for radius in range(1, 20):
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if pathfinder.is_walkable(gx + dx, gy + dy):
                    return pathfinder.grid_to_normalized(gx + dx, gy + dy)
    return (x, y)


def _catmull_rom_interpolate(
    points: List[Dict], t: float
) -> Tuple[float, float]:
    """Catmull-Rom spline interpolation for smooth curves between anchor points.

    Returns (x, y) at time t. Falls back to linear lerp when < 4 points available.
    Shared logic with strategy_executor.py.
    """
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

    if len(points) < 4:
        return (
            p1["x"] + (p2["x"] - p1["x"]) * frac,
            p1["y"] + (p2["y"] - p1["y"]) * frac,
        )

    p0 = points[max(0, seg_idx - 1)]
    p3 = points[min(len(points) - 1, seg_idx + 2)]

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

    return (max(0.0, min(1.0, x)), max(0.0, min(1.0, y)))
