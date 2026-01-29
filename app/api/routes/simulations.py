from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List, Optional, Dict, Any
from uuid import UUID, uuid4
from dataclasses import dataclass
import random

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

        return {
            "session_id": session_id,
            "current_time_ms": state.current_time_ms,
            "phase": state.phase,
            "status": session_data["status"],
            "positions": [p.model_dump() if hasattr(p, 'model_dump') else p.__dict__ for p in state.positions],
            "events": [e.model_dump() if hasattr(e, 'model_dump') else e.__dict__ for e in state.events[-10:]] if state.events else [],
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
