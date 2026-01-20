from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List, Optional
from uuid import UUID

from ...database import get_db
from ...models import SimulationSession, Team
from ...schemas.simulations import (
    SimulationCreate, SimulationResponse, SimulationState,
    PlayerPosition, WhatIfScenario
)
from ...services.simulation_engine import SimulationEngine

router = APIRouter()


@router.post("/", response_model=SimulationResponse)
async def create_simulation(
    config: SimulationCreate,
    db: AsyncSession = Depends(get_db)
):
    """Create a new simulation session."""
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
    return session


@router.get("/{session_id}", response_model=SimulationResponse)
async def get_simulation(session_id: UUID, db: AsyncSession = Depends(get_db)):
    """Get a simulation session by ID."""
    result = await db.execute(
        select(SimulationSession).where(SimulationSession.id == session_id)
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Simulation not found")
    return session


@router.post("/{session_id}/start", response_model=SimulationState)
async def start_simulation(
    session_id: UUID,
    round_type: str = Query("full", description="Round type: pistol, eco, force, half, full"),
    db: AsyncSession = Depends(get_db)
):
    """Start or resume a simulation.

    Args:
        session_id: Simulation session ID
        round_type: Economy round type affecting loadouts
            - pistol: 800 credits, sidearms only
            - eco: 2000 credits, saving round
            - force: 3000 credits, buying suboptimal
            - half: 4000 credits, partial buy
            - full: 5000 credits, full rifles + heavy armor
    """
    result = await db.execute(
        select(SimulationSession).where(SimulationSession.id == session_id)
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Simulation not found")

    # Validate round_type
    valid_types = ["pistol", "eco", "force", "half", "full"]
    if round_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid round_type. Must be one of: {valid_types}"
        )

    # Initialize simulation engine with round type
    engine = SimulationEngine(db)
    state = await engine.initialize(session, round_type=round_type)

    # Update session
    session.status = "running"
    session.round_type = round_type
    await db.commit()

    return state


@router.post("/{session_id}/tick", response_model=SimulationState)
async def tick_simulation(
    session_id: UUID,
    ticks: int = Query(1, ge=1, le=100),
    db: AsyncSession = Depends(get_db)
):
    """Advance simulation by specified ticks."""
    result = await db.execute(
        select(SimulationSession).where(SimulationSession.id == session_id)
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Simulation not found")
    if session.status != "running":
        raise HTTPException(status_code=400, detail="Simulation is not running")

    engine = SimulationEngine(db)
    state = await engine.advance(session, ticks)

    # Update session
    session.current_time_ms = state.current_time_ms
    session.phase = state.phase
    await db.commit()

    return state


@router.post("/{session_id}/pause", response_model=SimulationResponse)
async def pause_simulation(session_id: UUID, db: AsyncSession = Depends(get_db)):
    """Pause a running simulation."""
    result = await db.execute(
        select(SimulationSession).where(SimulationSession.id == session_id)
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Simulation not found")

    session.status = "paused"
    await db.commit()
    return session


@router.post("/{session_id}/snapshot")
async def create_snapshot(session_id: UUID, db: AsyncSession = Depends(get_db)):
    """Create a snapshot of current simulation state for what-if scenarios."""
    result = await db.execute(
        select(SimulationSession).where(SimulationSession.id == session_id)
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Simulation not found")

    engine = SimulationEngine(db)
    snapshot_id = await engine.create_snapshot(session)

    return {"snapshot_id": snapshot_id, "time_ms": session.current_time_ms}


@router.post("/{session_id}/what-if", response_model=SimulationState)
async def run_what_if(
    session_id: UUID,
    scenario: WhatIfScenario,
    db: AsyncSession = Depends(get_db)
):
    """Run a what-if scenario from a snapshot."""
    result = await db.execute(
        select(SimulationSession).where(SimulationSession.id == session_id)
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Simulation not found")

    engine = SimulationEngine(db)
    state = await engine.run_what_if(session, scenario)

    return state


@router.get("/{session_id}/analysis")
async def get_analysis(session_id: UUID, db: AsyncSession = Depends(get_db)):
    """Get analysis and improvement suggestions for a simulation."""
    result = await db.execute(
        select(SimulationSession).where(SimulationSession.id == session_id)
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Simulation not found")

    engine = SimulationEngine(db)
    analysis = await engine.analyze(session)

    return analysis
