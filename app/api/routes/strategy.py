from fastapi import APIRouter, HTTPException, Query

import math
import uuid

from ...services.vct_round_service import VCTRoundService, SIDE_MAP
from ...services.strategy_executor import StrategyExecutor
from ...services.tactical_simulation_engine import (
    TacticalSimulationEngine, TacticalWaypoint, PhaseCheckpoint, PlayerCheckpoint,
)
from ...schemas.strategy import (
    StrategyRound, StrategyExecuteRequest, StrategyReplayRequest, StrategyResult,
    PhaseExecuteRequest, PhaseExecuteResponse, PhaseResultSchema,
    PhaseCheckpointSchema, PlayerCheckpointSchema,
    StrategyEvent, StrategySnapshot,
)

router = APIRouter()


def _get_service() -> VCTRoundService:
    return VCTRoundService.get_instance()


@router.get("/rounds", response_model=StrategyRound)
async def get_strategy_round(
    map_name: str = Query(..., description="Map name (e.g. haven, bind, ascent)"),
    side: str = Query(..., description="User side: attack or defense"),
):
    """Get a random VCT round setup for the strategy planner."""
    service = _get_service()

    if side.lower() not in ("attack", "defense"):
        raise HTTPException(status_code=400, detail="side must be 'attack' or 'defense'")

    available = service.get_available_maps()
    if map_name.lower() not in available:
        raise HTTPException(
            status_code=404,
            detail=f"Map '{map_name}' not found. Available: {available}",
        )

    result = service.get_random_round(map_name, side)
    if not result:
        raise HTTPException(status_code=404, detail="No rounds found for this map/side")

    return result


@router.get("/rounds/list")
async def list_rounds(
    map_name: str = Query(..., description="Map name"),
):
    """List all available rounds for a map with match metadata."""
    service = _get_service()
    available = service.get_available_maps()
    if map_name.lower() not in available:
        raise HTTPException(status_code=404, detail=f"Map '{map_name}' not found")
    rounds = service.list_rounds(map_name.lower())
    return {"rounds": rounds, "map_name": map_name}


@router.get("/maps")
async def list_strategy_maps():
    """List maps available for strategy planning."""
    service = _get_service()
    return {"maps": service.get_available_maps()}


@router.post("/execute", response_model=StrategyResult)
async def execute_strategy(request: StrategyExecuteRequest):
    """Execute a strategy plan against pro-anchored AI opponents."""
    service = _get_service()

    side = request.side.lower()
    if side not in ("attack", "defense"):
        raise HTTPException(status_code=400, detail="side must be 'attack' or 'defense'")

    round_data = service.get_round_by_id(request.round_id)
    if not round_data:
        raise HTTPException(status_code=404, detail=f"Round '{request.round_id}' not found")

    map_name = round_data.get("_map", "")
    formatted = service._format_round(round_data, map_name, side)

    try:
        executor = StrategyExecutor()
        plans_dict = {}
        for phase, player_plans in request.plans.items():
            plans_dict[phase] = [pp.model_dump() for pp in player_plans]

        result = executor.execute(
            round_id=request.round_id,
            user_side=side,
            plans=plans_dict,
            teammates=[t.model_dump() for t in formatted.teammates],
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/replay", response_model=StrategyResult)
async def replay_vct_round(request: StrategyReplayRequest):
    """Replay a real VCT round — actual pro player positions and deaths."""
    service = _get_service()

    round_data = service.get_round_by_id(request.round_id)
    if not round_data:
        raise HTTPException(status_code=404, detail=f"Round '{request.round_id}' not found")

    map_name = round_data.get("_map", "")
    round_duration = round_data.get("round_duration_s", 100)
    round_num = round_data.get("round_num", "?")
    game_id = round_data.get("game_id", "")
    trajs = round_data.get("player_trajectories", {})

    # Parse all players: normalize coords, convert clock to time_s, detect deaths
    players = {}  # name → {side, team, points: [{time_s, x, y, alive}]}
    for pname, raw_traj in trajs.items():
        if not raw_traj:
            continue
        side = SIDE_MAP.get(raw_traj[0].get("side", ""), "attack")
        team = raw_traj[0].get("team", "Unknown")

        points = []
        for pt in raw_traj:
            time_s = 100 - pt["clock"]
            nx, ny = service.normalize_coords(map_name, pt["x"], pt["y"])
            points.append({"time_s": time_s, "x": nx, "y": ny, "alive": pt["alive"]})
        points.sort(key=lambda p: p["time_s"])

        # Truncate at death (first alive=false)
        death_time = None
        death_x, death_y = None, None
        filtered = []
        for p in points:
            if not p["alive"] and death_time is None:
                death_time = p["time_s"]
                # Use last alive position as death location (death coords are often extreme/off-map)
                if filtered:
                    death_x, death_y = filtered[-1]["x"], filtered[-1]["y"]
                else:
                    death_x, death_y = p["x"], p["y"]
                break
            filtered.append(p)

        players[pname] = {
            "side": side, "team": team, "points": filtered,
            "death_time": death_time, "death_x": death_x, "death_y": death_y,
        }

    # Cross-reference real kill events to fill in missing deaths.
    # Many players' trajectories end with alive=True because the position
    # data didn't sample at the exact moment of death.
    raw_kills = round_data.get("kills", [])
    for k in raw_kills:
        victim = k.get("victim", "")
        if victim in players and players[victim]["death_time"] is None:
            kill_time = 100 - k["clock"]
            # Use last known position as death position
            pts = players[victim]["points"]
            if pts:
                last_pt = pts[-1]
                players[victim]["death_time"] = kill_time
                players[victim]["death_x"] = last_pt["x"]
                players[victim]["death_y"] = last_pt["y"]

    # Interpolated position lookup: linearly interpolate between known data points
    def lerp_position(pts, t):
        if not pts:
            return 0.5, 0.5, True
        if t <= pts[0]["time_s"]:
            return pts[0]["x"], pts[0]["y"], pts[0]["alive"]
        if t >= pts[-1]["time_s"]:
            return pts[-1]["x"], pts[-1]["y"], pts[-1]["alive"]
        # Find bracketing points
        for i in range(len(pts) - 1):
            if pts[i]["time_s"] <= t <= pts[i + 1]["time_s"]:
                dt = pts[i + 1]["time_s"] - pts[i]["time_s"]
                if dt <= 0:
                    return pts[i]["x"], pts[i]["y"], pts[i]["alive"]
                frac = (t - pts[i]["time_s"]) / dt
                x = pts[i]["x"] + (pts[i + 1]["x"] - pts[i]["x"]) * frac
                y = pts[i]["y"] + (pts[i + 1]["y"] - pts[i]["y"]) * frac
                return x, y, pts[i]["alive"]
        return pts[-1]["x"], pts[-1]["y"], pts[-1]["alive"]

    # Build kill events from real GRID data (with fallback to alive→dead inference)
    raw_kills = round_data.get("kills", [])
    events = []
    if raw_kills:
        for k in raw_kills:
            time_s = 100 - k["clock"]
            events.append({
                "time_ms": int(time_s * 1000),
                "event_type": "kill",
                "player_id": k.get("killer", "unknown"),
                "target_id": k.get("victim", "unknown"),
                "details": {
                    "killer_name": k.get("killer", "Unknown"),
                    "victim_name": k.get("victim", "Unknown"),
                    "headshot": k.get("headshot", False),
                    "weapon": k.get("weapon", "unknown"),
                    "ttk_ms": 0,
                },
            })
    else:
        # Fallback: infer from alive→dead transitions
        for pname, pdata in players.items():
            if pdata["death_time"] is None:
                continue
            dt = pdata["death_time"]
            dx, dy = pdata["death_x"], pdata["death_y"]
            victim_side = pdata["side"]
            best_killer = None
            best_dist = float("inf")
            for ename, edata in players.items():
                if edata["side"] == victim_side:
                    continue
                if edata["death_time"] is not None and edata["death_time"] <= dt:
                    continue
                ex, ey, ealive = lerp_position(edata["points"], dt)
                if not ealive:
                    continue
                dist = math.sqrt((ex - dx) ** 2 + (ey - dy) ** 2)
                if dist < best_dist:
                    best_dist = dist
                    best_killer = ename
            events.append({
                "time_ms": int(dt * 1000),
                "event_type": "kill",
                "player_id": best_killer or "unknown",
                "target_id": pname,
                "details": {
                    "killer_name": best_killer or "Unknown",
                    "victim_name": pname,
                    "headshot": False,
                    "weapon": "unknown",
                    "ttk_ms": 0,
                },
            })
    events.sort(key=lambda e: e["time_ms"])

    # Ensure ALL kill event victims are marked dead in the player data.
    # The cross-reference above may miss victims due to name case mismatches
    # or missing trajectory data. Use the events we just built as ground truth.
    for evt in events:
        if evt["event_type"] != "kill":
            continue
        victim_name = evt["details"].get("victim_name", "")
        kill_time_s = evt["time_ms"] / 1000.0
        # Try exact match first, then case-insensitive
        matched = None
        if victim_name in players:
            matched = victim_name
        else:
            for pname in players:
                if pname.lower() == victim_name.lower():
                    matched = pname
                    break
        if matched and players[matched]["death_time"] is None:
            pts = players[matched]["points"]
            if pts:
                players[matched]["death_time"] = kill_time_s
                players[matched]["death_x"] = pts[-1]["x"]
                players[matched]["death_y"] = pts[-1]["y"]
            else:
                players[matched]["death_time"] = kill_time_s
                players[matched]["death_x"] = 0.5
                players[matched]["death_y"] = 0.5

    # Build snapshots with interpolated positions at regular intervals for smooth playback.
    # Include real data timestamps (kills, trajectory points) plus regular 2s samples.
    all_times = set()
    for pdata in players.values():
        for pt in pdata["points"]:
            all_times.add(pt["time_s"])
        if pdata["death_time"] is not None:
            all_times.add(pdata["death_time"])
    for evt in events:
        all_times.add(evt["time_ms"] / 1000.0)
    # Add regular 2-second interval samples for smooth interpolated movement
    if all_times:
        t_min, t_max = min(all_times), max(all_times)
        t = t_min
        while t <= t_max:
            all_times.add(round(t, 2))
            t += 2.0
    all_times = sorted(all_times)
    if not all_times:
        all_times = [0]

    snapshots = []
    phase_pcts = [("setup", 0.15), ("mid_round", 0.40), ("execute", 0.70), ("post_plant", 1.0)]
    for t in all_times:
        frac = t / round_duration if round_duration > 0 else 0
        phase = "post_plant"
        for pname_phase, pct in phase_pcts:
            if frac < pct:
                phase = pname_phase
                break

        snap_players = []
        for i, (pname, pdata) in enumerate(players.items()):
            x, y, alive = lerp_position(pdata["points"], t)
            if pdata["death_time"] is not None and t >= pdata["death_time"]:
                x, y, alive = pdata["death_x"], pdata["death_y"], False

            snap_players.append({
                "player_id": f"p{i+1}",
                "x": round(x, 4),
                "y": round(y, 4),
                "side": pdata["side"],
                "is_alive": alive,
                "agent": "unknown",
                "name": pname,
                "health": 100 if alive else 0,
                "team_id": pdata["team"],
            })

        snapshots.append({
            "time_ms": int(t * 1000),
            "phase": phase,
            "players": snap_players,
        })

    # Teams
    teams_by_side = {"attack": set(), "defense": set()}
    for pdata in players.values():
        teams_by_side[pdata["side"]].add(pdata["team"])
    atk_team = next(iter(teams_by_side["attack"]), "Unknown")
    def_team = next(iter(teams_by_side["defense"]), "Unknown")

    # Determine winner from GRID data (team-won-round event)
    winner_team_name = round_data.get("winner_team", "")
    if winner_team_name:
        if winner_team_name == atk_team:
            winner = "attack"
        elif winner_team_name == def_team:
            winner = "defense"
        else:
            # Fallback: partial match (e.g. "MIBR" vs "MIBR (1)")
            winner = "attack" if winner_team_name in atk_team else "defense"
    else:
        # Fallback: count kills
        atk_deaths = sum(1 for k in raw_kills if players.get(k.get("victim", ""), {}).get("side") == "attack")
        def_deaths = sum(1 for k in raw_kills if players.get(k.get("victim", ""), {}).get("side") == "defense")
        winner = "attack" if def_deaths > atk_deaths else "defense" if atk_deaths > def_deaths else "defense"

    # Kill counts
    atk_deaths_count = sum(1 for k in raw_kills if players.get(k.get("victim", ""), {}).get("side") == "attack")
    def_deaths_count = sum(1 for k in raw_kills if players.get(k.get("victim", ""), {}).get("side") == "defense")
    atk_kills = def_deaths_count  # attack killed defenders
    def_kills = atk_deaths_count  # defense killed attackers

    # Match metadata
    match_meta = service.get_match_metadata(game_id) or {}

    return {
        "session_id": str(uuid.uuid4()),
        "winner": winner,
        "events": events,
        "snapshots": snapshots,
        "reveal": {
            "opponent_team": def_team,
            "user_team": atk_team,
            "atk_team": atk_team,
            "def_team": def_team,
            "round_desc": f"Round {round_num} on {map_name.capitalize()}",
            "round_num": round_num,
            "map_name": map_name,
            "user_side": "replay",
            "opponent_players": [n for n, p in players.items() if p["side"] == "defense"],
            "round_duration_s": round_duration,
            "sim_duration_s": round_duration,
            "score_line": f"{atk_kills}–{def_kills}",
            "tournament": match_meta.get("tournament", ""),
            "match_date": match_meta.get("date", ""),
        },
    }


# --- Phase-by-phase tactical execution ---

# Cache active tactical engines per round
_tactical_engines: dict = {}


@router.post("/execute-phase", response_model=PhaseExecuteResponse)
async def execute_tactical_phase(request: PhaseExecuteRequest):
    """Execute a single phase of the tactical plan using full SimulationEngine.

    First phase: no checkpoint needed, starts from spawn.
    Subsequent phases: pass checkpoint from previous phase result.
    """
    side = request.side.lower()
    if side not in ("attack", "defense"):
        raise HTTPException(status_code=400, detail="side must be 'attack' or 'defense'")

    valid_phases = ("setup", "mid_round", "execute", "post_plant")
    if request.phase not in valid_phases:
        raise HTTPException(status_code=400, detail=f"phase must be one of {valid_phases}")

    service = _get_service()
    round_data = service.get_round_by_id(request.round_id)
    if not round_data:
        raise HTTPException(status_code=404, detail=f"Round '{request.round_id}' not found")

    map_name = round_data.get("_map", "")

    # Extract user team players from VCT round data for ID remapping
    formatted_round = service._format_round(round_data, map_name, side)
    user_team_players = [
        {"player_id": t.player_id, "name": t.name, "agent": t.agent}
        for t in formatted_round.teammates
    ]

    try:
        # Create or reuse engine
        engine_key = f"{request.round_id}_{side}"

        # First phase or no checkpoint: create fresh engine
        if request.checkpoint is None:
            engine = TacticalSimulationEngine()
            await engine.initialize_round(
                round_id=request.round_id,
                user_side=side,
                map_name=map_name,
                user_team_players=user_team_players,
            )
            _tactical_engines[engine_key] = engine
        else:
            # Reuse existing engine or create new one
            engine = _tactical_engines.get(engine_key)
            if not engine:
                engine = TacticalSimulationEngine()
                await engine.initialize_round(
                    round_id=request.round_id,
                    user_side=side,
                    map_name=map_name,
                    user_team_players=user_team_players,
                )
                _tactical_engines[engine_key] = engine

        # Convert waypoints from schema to dataclass
        waypoints = {}
        for player_id, wp_list in request.waypoints.items():
            waypoints[player_id] = [
                TacticalWaypoint(tick=wp.tick, x=wp.x, y=wp.y, facing=wp.facing)
                for wp in wp_list
            ]

        # Convert checkpoint schema to dataclass if provided
        checkpoint = None
        if request.checkpoint:
            cp = request.checkpoint
            checkpoint = PhaseCheckpoint(
                phase_name=cp.phase_name,
                time_ms=cp.time_ms,
                players=[
                    PlayerCheckpoint(
                        player_id=p.player_id, x=p.x, y=p.y, side=p.side,
                        is_alive=p.is_alive, health=p.health, shield=p.shield,
                        has_spike=p.has_spike, agent=p.agent, name=p.name,
                        kills=p.kills, deaths=p.deaths,
                        facing_angle=p.facing_angle, is_running=p.is_running,
                    )
                    for p in cp.players
                ],
                spike_planted=cp.spike_planted,
                spike_site=cp.spike_site,
                spike_plant_time_ms=cp.spike_plant_time_ms,
                site_execute_active=cp.site_execute_active,
                target_site=cp.target_site,
            )

        # Execute the phase
        result = await engine.execute_phase(
            phase_name=request.phase,
            user_waypoints=waypoints,
            checkpoint=checkpoint,
        )

        # Convert dataclass result to schema
        phase_result = PhaseResultSchema(
            phase_name=result.phase_name,
            winner=result.winner,
            round_ended=result.round_ended,
            events=[
                StrategyEvent(
                    time_ms=e["time_ms"],
                    event_type=e["event_type"],
                    player_id=e.get("player_id"),
                    target_id=e.get("target_id"),
                    details=e.get("details"),
                )
                for e in result.events
            ],
            snapshots=[
                StrategySnapshot(
                    time_ms=s["time_ms"],
                    phase=s["phase"],
                    players=s["players"],
                )
                for s in result.snapshots
            ],
            checkpoint=PhaseCheckpointSchema(
                phase_name=result.checkpoint.phase_name,
                time_ms=result.checkpoint.time_ms,
                players=[
                    PlayerCheckpointSchema(
                        player_id=p.player_id, x=p.x, y=p.y, side=p.side,
                        is_alive=p.is_alive, health=p.health, shield=p.shield,
                        has_spike=p.has_spike, agent=p.agent, name=p.name,
                        kills=p.kills, deaths=p.deaths,
                        facing_angle=p.facing_angle, is_running=p.is_running,
                    )
                    for p in result.checkpoint.players
                ],
                spike_planted=result.checkpoint.spike_planted,
                spike_site=result.checkpoint.spike_site,
                spike_plant_time_ms=result.checkpoint.spike_plant_time_ms,
                site_execute_active=result.checkpoint.site_execute_active,
                target_site=result.checkpoint.target_site,
            ),
            end_positions=result.end_positions,
        )

        return PhaseExecuteResponse(phase_result=phase_result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
