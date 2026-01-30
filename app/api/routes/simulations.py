from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List, Optional, Dict, Any
from uuid import UUID, uuid4
from dataclasses import dataclass
import copy
import random
import asyncio

from ...database import get_db
from ...models import SimulationSession, Team
from ...schemas.simulations import (
    SimulationCreate, SimulationResponse, SimulationState,
    PlayerPosition, WhatIfScenario
)
from ...services.simulation_engine import SimulationEngine

router = APIRouter()

# In-memory storage for demo mode (when DB unavailable)
_demo_sessions: Dict[str, Dict[str, Any]] = {}
# Store SimulationEngine instances for each session
_demo_engines: Dict[str, SimulationEngine] = {}


@dataclass
class DemoSession:
    """Mock session object for demo mode."""
    id: str
    attack_team_id: str
    defense_team_id: str
    map_name: str
    round_type: str
    status: str = "created"
    current_time_ms: int = 0
    phase: str = "opening"


def _normalize_round_type(round_type: str) -> str:
    """Normalize round_type to match BuyType enum values."""
    mapping = {
        "full_buy": "full",
        "full-buy": "full",
        "fullbuy": "full",
        "half_buy": "half",
        "half-buy": "half",
        "halfbuy": "half",
        "force_buy": "force",
        "force-buy": "force",
        "forcebuy": "force",
    }
    return mapping.get(round_type.lower(), round_type.lower())


@router.post("/")
async def create_simulation(config: SimulationCreate):
    """Create a new simulation session."""
    try:
        from ...database import AsyncSessionLocal
        async with AsyncSessionLocal() as db:
            # Verify teams exist
            for team_id in [config.attack_team_id, config.defense_team_id]:
                result = await db.execute(select(Team).where(Team.id == team_id))
                if not result.scalar_one_or_none():
                    raise HTTPException(status_code=404, detail=f"Team {team_id} not found")

            # Create session
            session = SimulationSession(
                attack_team_id=config.attack_team_id,
                defense_team_id=config.defense_team_id,
                map_name=config.map_name,
                round_type=config.round_type,
                status="created",
            )
            db.add(session)
            await db.commit()
            await db.refresh(session)
            return {"id": str(session.id), "status": session.status}
    except Exception:
        # Demo mode - create in-memory session using actual SimulationEngine
        session_id = str(uuid4())
        _demo_sessions[session_id] = {
            "id": session_id,
            "attack_team_id": config.attack_team_id,
            "defense_team_id": config.defense_team_id,
            "map_name": config.map_name,
            "round_type": config.round_type,
            "status": "created",
            "current_time_ms": 0,
            "phase": "opening",
            "positions": [],
            "events": [],
            "spike_planted": False,
            "spike_site": None,
        }
        # Create engine without database (will use built-in defaults)
        _demo_engines[session_id] = SimulationEngine(db=None)
        return {"id": session_id, "status": "created"}


@router.get("/{session_id}")
async def get_simulation(session_id: str):
    """Get a simulation session by ID."""
    # Check demo sessions first
    if session_id in _demo_sessions:
        return _demo_sessions[session_id]

    try:
        from ...database import AsyncSessionLocal
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(SimulationSession).where(SimulationSession.id == UUID(session_id))
            )
            session = result.scalar_one_or_none()
            if not session:
                raise HTTPException(status_code=404, detail="Simulation not found")
            return session
    except Exception:
        raise HTTPException(status_code=404, detail="Simulation not found")


@router.post("/{session_id}/start")
async def start_simulation(
    session_id: str,
    round_type: str = Query("full", description="Round type: pistol, eco, force, half, full"),
):
    """Start or resume a simulation."""
    # Check demo sessions first (uses actual SimulationEngine without DB)
    if session_id in _demo_sessions:
        session_data = _demo_sessions[session_id]
        engine = _demo_engines.get(session_id)
        if not engine:
            engine = SimulationEngine(db=None)
            _demo_engines[session_id] = engine

        # Normalize round_type to match BuyType enum
        normalized_round_type = _normalize_round_type(round_type)

        # Create a mock session object for the engine
        demo_session = DemoSession(
            id=session_id,
            attack_team_id=session_data["attack_team_id"],
            defense_team_id=session_data["defense_team_id"],
            map_name=session_data["map_name"],
            round_type=normalized_round_type,
        )

        # Use the actual engine to initialize
        state = await engine.initialize(demo_session, round_type=normalized_round_type)

        # Update session data
        session_data["status"] = "running"
        session_data["current_time_ms"] = state.current_time_ms
        session_data["phase"] = state.phase

        return {
            "session_id": session_id,
            "current_time_ms": state.current_time_ms,
            "phase": state.phase,
            "status": "running",
            "positions": [p.model_dump() if hasattr(p, 'model_dump') else p.__dict__ for p in state.positions],
            "events": [e.model_dump() if hasattr(e, 'model_dump') else e.__dict__ for e in state.events] if state.events else [],
            "spike_planted": state.spike_planted,
            "spike_site": state.spike_site,
        }

    try:
        from ...database import AsyncSessionLocal
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(SimulationSession).where(SimulationSession.id == UUID(session_id))
            )
            session = result.scalar_one_or_none()
            if not session:
                raise HTTPException(status_code=404, detail="Simulation not found")

            engine = SimulationEngine(db)
            state = await engine.initialize(session, round_type=round_type)
            session.status = "running"
            session.round_type = round_type
            await db.commit()
            return state
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=404, detail="Simulation not found")


@router.post("/{session_id}/tick")
async def tick_simulation(
    session_id: str,
    ticks: int = Query(1, ge=1, le=100),
):
    """Advance simulation by specified ticks."""
    # Check demo sessions first (uses actual SimulationEngine)
    if session_id in _demo_sessions:
        session_data = _demo_sessions[session_id]
        if session_data["status"] != "running":
            raise HTTPException(status_code=400, detail="Simulation is not running")

        engine = _demo_engines.get(session_id)
        if not engine:
            raise HTTPException(status_code=400, detail="Simulation engine not initialized")

        # Create mock session object
        demo_session = DemoSession(
            id=session_id,
            attack_team_id=session_data["attack_team_id"],
            defense_team_id=session_data["defense_team_id"],
            map_name=session_data["map_name"],
            round_type=session_data["round_type"],
            status=session_data["status"],
            current_time_ms=session_data["current_time_ms"],
        )

        # Track snapshot count before advance
        prev_snapshot_count = len(engine.snapshots)

        # Use the actual engine to advance
        state = await engine.advance(demo_session, ticks)

        # Update session data
        session_data["current_time_ms"] = state.current_time_ms
        session_data["phase"] = state.phase
        session_data["spike_planted"] = state.spike_planted
        session_data["spike_site"] = state.spike_site

        # Check for round end
        attack_alive = sum(1 for p in state.positions if p.side == "attack" and p.is_alive)
        defense_alive = sum(1 for p in state.positions if p.side == "defense" and p.is_alive)

        if attack_alive == 0 or defense_alive == 0 or state.current_time_ms >= 100000:
            session_data["status"] = "completed"

        # Collect new snapshots created during this tick
        new_snapshots = []
        for snap in engine.snapshots[prev_snapshot_count:]:
            attack_count = sum(1 for p in snap.players if p.get('side') == 'attack' and p.get('is_alive'))
            defense_count = sum(1 for p in snap.players if p.get('side') == 'defense' and p.get('is_alive'))
            new_snapshots.append({
                "id": snap.id,
                "time_ms": snap.time_ms,
                "phase": snap.phase,
                "label": snap.label,
                "spike_planted": snap.spike_planted,
                "spike_site": snap.spike_site,
                "player_count": {"attack": attack_count, "defense": defense_count},
                "players": snap.players,
            })

        result = {
            "session_id": session_id,
            "current_time_ms": state.current_time_ms,
            "phase": state.phase,
            "status": session_data["status"],
            "positions": [p.model_dump() if hasattr(p, 'model_dump') else p.__dict__ for p in state.positions],
            "events": [e.model_dump() if hasattr(e, 'model_dump') else e.__dict__ for e in state.events] if state.events else [],
            "spike_planted": state.spike_planted,
            "spike_site": state.spike_site,
        }
        if new_snapshots:
            result["snapshots"] = new_snapshots

        return result

    try:
        from ...database import AsyncSessionLocal
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(SimulationSession).where(SimulationSession.id == UUID(session_id))
            )
            session = result.scalar_one_or_none()
            if not session:
                raise HTTPException(status_code=404, detail="Simulation not found")
            if session.status != "running":
                raise HTTPException(status_code=400, detail="Simulation is not running")

            engine = SimulationEngine(db)
            state = await engine.advance(session, ticks)
            session.current_time_ms = state.current_time_ms
            session.phase = state.phase
            await db.commit()
            return state
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=404, detail="Simulation not found")


@router.post("/{session_id}/pause")
async def pause_simulation(session_id: str):
    """Pause a running simulation."""
    if session_id in _demo_sessions:
        _demo_sessions[session_id]["status"] = "paused"
        return {"id": session_id, "status": "paused"}

    try:
        from ...database import AsyncSessionLocal
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(SimulationSession).where(SimulationSession.id == UUID(session_id))
            )
            session = result.scalar_one_or_none()
            if not session:
                raise HTTPException(status_code=404, detail="Simulation not found")
            session.status = "paused"
            await db.commit()
            return {"id": str(session.id), "status": "paused"}
    except Exception:
        raise HTTPException(status_code=404, detail="Simulation not found")


@router.post("/{session_id}/snapshot")
async def create_snapshot(session_id: str):
    """Create a snapshot of current simulation state for what-if scenarios."""
    if session_id in _demo_sessions:
        return {
            "snapshot_id": str(uuid4()),
            "time_ms": _demo_sessions[session_id].get("current_time_ms", 0)
        }

    try:
        from ...database import AsyncSessionLocal
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(SimulationSession).where(SimulationSession.id == UUID(session_id))
            )
            session = result.scalar_one_or_none()
            if not session:
                raise HTTPException(status_code=404, detail="Simulation not found")
            engine = SimulationEngine(db)
            snapshot_id = await engine.create_snapshot(session)
            return {"snapshot_id": snapshot_id, "time_ms": session.current_time_ms}
    except Exception:
        raise HTTPException(status_code=404, detail="Simulation not found")


@router.post("/{session_id}/what-if")
async def run_what_if(session_id: str, scenario: WhatIfScenario):
    """Run a what-if scenario from a snapshot."""
    if session_id in _demo_sessions:
        session_data = _demo_sessions[session_id]
        engine = _demo_engines.get(session_id)
        if engine:
            # Create mock session for what-if
            demo_session = DemoSession(
                id=session_id,
                attack_team_id=session_data["attack_team_id"],
                defense_team_id=session_data["defense_team_id"],
                map_name=session_data["map_name"],
                round_type=session_data["round_type"],
                current_time_ms=session_data["current_time_ms"],
            )
            try:
                state = await engine.run_what_if(demo_session, scenario)
                return {
                    "session_id": session_id,
                    "current_time_ms": state.current_time_ms,
                    "phase": state.phase,
                    "status": "running",
                    "positions": [p.model_dump() if hasattr(p, 'model_dump') else p.__dict__ for p in state.positions],
                    "events": [e.model_dump() if hasattr(e, 'model_dump') else e.__dict__ for e in state.events] if state.events else [],
                    "spike_planted": state.spike_planted,
                    "spike_site": state.spike_site,
                }
            except Exception:
                pass
        # Fallback
        return {
            "session_id": session_id,
            "current_time_ms": session_data.get("current_time_ms", 0),
            "phase": session_data.get("phase", "opening"),
            "status": "running",
            "positions": [],
            "events": [],
            "spike_planted": False,
            "spike_site": None,
        }

    try:
        from ...database import AsyncSessionLocal
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(SimulationSession).where(SimulationSession.id == UUID(session_id))
            )
            session = result.scalar_one_or_none()
            if not session:
                raise HTTPException(status_code=404, detail="Simulation not found")
            engine = SimulationEngine(db)
            state = await engine.run_what_if(session, scenario)
            return state
    except Exception:
        raise HTTPException(status_code=404, detail="Simulation not found")


@router.get("/{session_id}/analysis")
async def get_analysis(session_id: str):
    """Get analysis and improvement suggestions for a simulation."""
    if session_id in _demo_sessions:
        session_data = _demo_sessions[session_id]
        engine = _demo_engines.get(session_id)

        # Get player states from engine if available
        if engine and hasattr(engine, 'players'):
            attack_alive = sum(1 for p in engine.players.values() if p.side == "attack" and p.is_alive)
            defense_alive = sum(1 for p in engine.players.values() if p.side == "defense" and p.is_alive)
            events = [e.__dict__ if hasattr(e, '__dict__') else e for e in engine.events] if hasattr(engine, 'events') else []
        else:
            attack_alive = 0
            defense_alive = 0
            events = []

        # Determine winner
        if session_data.get("status") == "completed":
            if defense_alive == 0 and attack_alive > 0:
                winner = "attack"
            elif attack_alive == 0 and defense_alive > 0:
                winner = "defense"
            elif session_data.get("spike_planted", False):
                winner = "attack"  # Spike detonated
            else:
                winner = "defense"  # Time ran out
        else:
            winner = "ongoing"

        kills = [e for e in events if isinstance(e, dict) and e.get("event_type") == "kill"]

        return {
            "winner": winner,
            "total_duration_ms": session_data.get("current_time_ms", 0),
            "kills": kills,
            "spike_planted": session_data.get("spike_planted", False),
            "spike_site": session_data.get("spike_site"),
            "attack_alive": attack_alive,
            "defense_alive": defense_alive,
            "improvement_suggestions": [
                "Consider using more utility before executing",
                "Trade timings could be improved",
                "Map control priority should match team composition"
            ],
            "key_moments": events[:5] if events else [],
        }

    try:
        from ...database import AsyncSessionLocal
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(SimulationSession).where(SimulationSession.id == UUID(session_id))
            )
            session = result.scalar_one_or_none()
            if not session:
                raise HTTPException(status_code=404, detail="Simulation not found")
            engine = SimulationEngine(db)
            analysis = await engine.analyze(session)
            return analysis
    except Exception:
        raise HTTPException(status_code=404, detail="Simulation not found")



@router.get("/{session_id}/snapshots")
async def get_snapshots(session_id: str):
    """Get all auto-snapshots for a simulation session."""
    if session_id not in _demo_sessions:
        raise HTTPException(status_code=404, detail="Simulation not found")

    engine = _demo_engines.get(session_id)
    if not engine or not hasattr(engine, 'snapshots'):
        return {"snapshots": []}

    snapshots = []
    for snap in engine.snapshots:
        snapshots.append({
            "id": snap.id,
            "time_ms": snap.time_ms,
            "phase": snap.phase,
            "label": snap.label,
            "spike_planted": snap.spike_planted,
            "spike_site": snap.spike_site,
            "players": snap.players,
            "events": snap.events,
            "round_state": snap.round_state_data,
            "player_knowledge": snap.player_knowledge_data,
            "decisions": snap.decisions_data,
        })

    return {"snapshots": snapshots}


@router.post("/{session_id}/run-to-completion")
async def run_to_completion(session_id: str):
    """Run simulation to completion and return final state with auto-snapshots."""
    if session_id in _demo_sessions:
        session_data = _demo_sessions[session_id]
        engine = _demo_engines.get(session_id)
        if not engine:
            raise HTTPException(status_code=400, detail="Simulation engine not initialized")

        if session_data["status"] != "running":
            raise HTTPException(status_code=400, detail="Simulation is not running")

        demo_session = DemoSession(
            id=session_id,
            attack_team_id=session_data["attack_team_id"],
            defense_team_id=session_data["defense_team_id"],
            map_name=session_data["map_name"],
            round_type=session_data["round_type"],
            status=session_data["status"],
            current_time_ms=session_data["current_time_ms"],
        )

        # Run until completion (max 1000 ticks, 5 ticks per advance for speed)
        state = None
        for _ in range(1000):
            state = await engine.advance(demo_session, 5)
            demo_session.current_time_ms = state.current_time_ms
            demo_session.phase = state.phase
            session_data["current_time_ms"] = state.current_time_ms
            session_data["phase"] = state.phase
            session_data["spike_planted"] = state.spike_planted
            session_data["spike_site"] = state.spike_site

            attack_alive = sum(1 for p in state.positions if p.side == "attack" and p.is_alive)
            defense_alive = sum(1 for p in state.positions if p.side == "defense" and p.is_alive)

            if attack_alive == 0 or defense_alive == 0 or state.current_time_ms >= 100000:
                session_data["status"] = "completed"
                break

        if state is None:
            raise HTTPException(status_code=400, detail="No simulation state")

        # Collect snapshots
        snapshots = []
        if hasattr(engine, 'snapshots'):
            for snap in engine.snapshots:
                snapshots.append({
                    "id": snap.id,
                    "time_ms": snap.time_ms,
                    "phase": snap.phase,
                    "label": snap.label,
                    "spike_planted": snap.spike_planted,
                    "spike_site": snap.spike_site,
                    "player_count": {
                        "attack": sum(1 for p in snap.players if p.get("side") == "attack" and p.get("is_alive", True)),
                        "defense": sum(1 for p in snap.players if p.get("side") == "defense" and p.get("is_alive", True)),
                    },
                    "round_state": snap.round_state_data,
                    "player_knowledge": snap.player_knowledge_data,
                    "decisions": snap.decisions_data,
                })

        return {
            "session_id": session_id,
            "status": session_data["status"],
            "current_time_ms": state.current_time_ms,
            "phase": state.phase,
            "positions": [p.model_dump() if hasattr(p, 'model_dump') else p.__dict__ for p in state.positions],
            "events": [e.model_dump() if hasattr(e, 'model_dump') else e.__dict__ for e in state.events] if state.events else [],
            "spike_planted": state.spike_planted,
            "spike_site": state.spike_site,
            "snapshots": snapshots,
        }

    raise HTTPException(status_code=404, detail="Simulation not found")


@router.post("/{session_id}/compare")
async def compare_scenarios(
    session_id: str,
    scenario: WhatIfScenario,
):
    """Run original vs what-if scenario and return side-by-side comparison."""
    if session_id not in _demo_sessions:
        raise HTTPException(status_code=404, detail="Simulation not found")

    session_data = _demo_sessions[session_id]
    engine = _demo_engines.get(session_id)
    if not engine:
        raise HTTPException(status_code=400, detail="Simulation engine not initialized")

    # Save original engine state before what-if mutates it
    saved_players = copy.deepcopy(engine.players)
    saved_events = copy.deepcopy(engine.events)
    saved_spike_planted = engine.spike_planted
    saved_spike_site = engine.spike_site
    saved_snapshots = copy.deepcopy(engine.snapshots) if hasattr(engine, 'snapshots') else []
    saved_round_state = copy.deepcopy(engine.round_state) if hasattr(engine, 'round_state') else None

    # Capture original final state
    original_positions = []
    if hasattr(engine, 'players'):
        for p in engine.players.values():
            original_positions.append({
                "player_id": p.player_id,
                "side": p.side,
                "x": p.x, "y": p.y,
                "is_alive": p.is_alive,
                "health": p.health,
                "agent": p.agent,
            })

    original_attack_alive = sum(1 for p in original_positions if p["side"] == "attack" and p["is_alive"])
    original_defense_alive = sum(1 for p in original_positions if p["side"] == "defense" and p["is_alive"])
    original_kills = len([e for e in engine.events if hasattr(e, 'event_type') and e.event_type == 'kill']) if hasattr(engine, 'events') else 0

    # Run what-if on a separate engine to avoid mutation
    whatif_engine = SimulationEngine(db=None)
    whatif_engine.players = copy.deepcopy(saved_players)
    whatif_engine.events = copy.deepcopy(saved_events)
    whatif_engine.spike_planted = saved_spike_planted
    whatif_engine.spike_site = saved_spike_site
    whatif_engine.snapshots = copy.deepcopy(saved_snapshots)
    if saved_round_state:
        whatif_engine.round_state = copy.deepcopy(saved_round_state)
    # Copy other critical state
    for attr in ['map_name', 'map_data', 'info_manager', 'ai_behavior', 'use_ai_decisions',
                 'combat_model', 'ability_system', 'round_state', 'TICK_DURATION_MS',
                 'PHASE_TIMINGS', 'PLANT_DURATION_MS']:
        if hasattr(engine, attr):
            try:
                setattr(whatif_engine, attr, copy.deepcopy(getattr(engine, attr)))
            except Exception:
                setattr(whatif_engine, attr, getattr(engine, attr))

    demo_session = DemoSession(
        id=session_id,
        attack_team_id=session_data["attack_team_id"],
        defense_team_id=session_data["defense_team_id"],
        map_name=session_data["map_name"],
        round_type=session_data["round_type"],
        current_time_ms=session_data["current_time_ms"],
    )

    try:
        whatif_state = await whatif_engine.run_what_if(demo_session, scenario)
        whatif_positions = [p.model_dump() if hasattr(p, 'model_dump') else p.__dict__ for p in whatif_state.positions]
        whatif_attack_alive = sum(1 for p in whatif_state.positions if p.side == "attack" and p.is_alive)
        whatif_defense_alive = sum(1 for p in whatif_state.positions if p.side == "defense" and p.is_alive)
        whatif_kills = len([e for e in whatif_engine.events if hasattr(e, 'event_type') and e.event_type == 'kill'])
        whatif_spike = whatif_state.spike_planted
    except Exception:
        whatif_positions = []
        whatif_attack_alive = 0
        whatif_defense_alive = 0
        whatif_kills = 0
        whatif_spike = False

    # Restore original engine state
    engine.players = saved_players
    engine.events = saved_events
    engine.spike_planted = saved_spike_planted
    engine.spike_site = saved_spike_site
    engine.snapshots = saved_snapshots
    if saved_round_state:
        engine.round_state = saved_round_state

    # Determine winners
    def get_winner(atk, dfn, spike):
        if atk == 0:
            return "defense"
        if dfn == 0:
            return "attack"
        if spike:
            return "attack"
        return "defense"

    original_winner = get_winner(original_attack_alive, original_defense_alive, session_data.get("spike_planted", False))
    whatif_winner = get_winner(whatif_attack_alive, whatif_defense_alive, whatif_spike)

    # Compute key differences
    key_differences = []
    if original_winner != whatif_winner:
        key_differences.append(f"Winner changed from {original_winner} to {whatif_winner}")
    alive_diff_atk = whatif_attack_alive - original_attack_alive
    alive_diff_def = whatif_defense_alive - original_defense_alive
    if alive_diff_atk != 0:
        key_differences.append(f"Attack survivors: {original_attack_alive} → {whatif_attack_alive} ({'+' if alive_diff_atk > 0 else ''}{alive_diff_atk})")
    if alive_diff_def != 0:
        key_differences.append(f"Defense survivors: {original_defense_alive} → {whatif_defense_alive} ({'+' if alive_diff_def > 0 else ''}{alive_diff_def})")
    kill_diff = whatif_kills - original_kills
    if kill_diff != 0:
        key_differences.append(f"Total kills: {original_kills} → {whatif_kills} ({'+' if kill_diff > 0 else ''}{kill_diff})")
    if session_data.get("spike_planted", False) != whatif_spike:
        key_differences.append(f"Spike planted: {session_data.get('spike_planted', False)} → {whatif_spike}")

    return {
        "original": {
            "winner": original_winner,
            "attack_alive": original_attack_alive,
            "defense_alive": original_defense_alive,
            "positions": original_positions,
            "spike_planted": session_data.get("spike_planted", False),
            "total_kills": original_kills,
        },
        "what_if": {
            "winner": whatif_winner,
            "attack_alive": whatif_attack_alive,
            "defense_alive": whatif_defense_alive,
            "positions": whatif_positions,
            "spike_planted": whatif_spike,
            "total_kills": whatif_kills,
        },
        "key_differences": key_differences,
        "modifications": scenario.modifications,
        "snapshot_time_ms": scenario.snapshot_time_ms,
    }


@router.post("/{session_id}/monte-carlo")
async def run_monte_carlo(
    session_id: str,
    iterations: int = Query(20, ge=5, le=100),
    scenario: Optional[WhatIfScenario] = None,
):
    """Run Monte Carlo simulation for statistical analysis."""
    if session_id not in _demo_sessions:
        raise HTTPException(status_code=404, detail="Simulation not found")

    session_data = _demo_sessions[session_id]
    sem = asyncio.Semaphore(10)

    async def run_single(i: int) -> Dict[str, Any]:
        async with sem:
            engine = SimulationEngine(db=None)
            from uuid import uuid4 as _uuid4
            demo_session = DemoSession(
                id=str(_uuid4()),
                attack_team_id=session_data["attack_team_id"],
                defense_team_id=session_data["defense_team_id"],
                map_name=session_data["map_name"],
                round_type=session_data["round_type"],
            )

            normalized_round_type = _normalize_round_type(session_data["round_type"])
            state = await engine.initialize(demo_session, round_type=normalized_round_type)

            # Apply scenario modifications if provided
            if scenario and hasattr(engine, 'players'):
                for player_id, mods in scenario.modifications.items():
                    if player_id in engine.players:
                        player = engine.players[player_id]
                        if 'x' in mods:
                            player.x = max(0.02, min(0.98, mods['x']))
                        if 'y' in mods:
                            player.y = max(0.02, min(0.98, mods['y']))
                        if 'is_alive' in mods:
                            player.is_alive = mods['is_alive']

            # Run to completion
            winner = "defense"
            final_time = 0
            spike_planted = False
            for _ in range(1000):
                state = await engine.advance(demo_session, 5)
                demo_session.current_time_ms = state.current_time_ms
                attack_alive = sum(1 for p in state.positions if p.side == "attack" and p.is_alive)
                defense_alive = sum(1 for p in state.positions if p.side == "defense" and p.is_alive)
                final_time = state.current_time_ms
                spike_planted = state.spike_planted

                if attack_alive == 0 or defense_alive == 0 or state.current_time_ms >= 100000:
                    if defense_alive == 0:
                        winner = "attack"
                    elif attack_alive == 0:
                        winner = "defense"
                    elif spike_planted:
                        winner = "attack"
                    break

            kills = [e for e in state.events if hasattr(e, 'event_type') and e.event_type == 'kill'] if state.events else []
            return {
                "iteration": i + 1,
                "winner": winner,
                "duration_ms": final_time,
                "spike_planted": spike_planted,
                "kills": len(kills),
            }

    iteration_results = await asyncio.gather(*[run_single(i) for i in range(iterations)])

    results = {"attack_wins": 0, "defense_wins": 0, "iterations": list(iteration_results), "total": iterations}
    for ir in iteration_results:
        if ir["winner"] == "attack":
            results["attack_wins"] += 1
        else:
            results["defense_wins"] += 1

    results["attack_win_rate"] = results["attack_wins"] / iterations
    results["defense_win_rate"] = results["defense_wins"] / iterations

    return results
