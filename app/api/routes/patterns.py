from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_
from typing import List, Optional

from ...database import get_db
from ...models import MovementPattern, PlayerTendency, TeamStrategy
from ...schemas.patterns import (
    PatternCreate, PatternResponse, PatternQuery,
    TendencyResponse, StrategyResponse
)

router = APIRouter()


@router.get("/", response_model=List[PatternResponse])
async def list_patterns(
    team_id: Optional[str] = Query(None),
    map_name: Optional[str] = Query(None),
    side: Optional[str] = Query(None),
    phase: Optional[str] = Query(None),
    limit: int = Query(20, le=100),
    db: AsyncSession = Depends(get_db)
):
    """List movement patterns with optional filters."""
    query = select(MovementPattern).where(MovementPattern.is_active == True)

    if team_id:
        # Include team-specific and global patterns
        query = query.where(or_(MovementPattern.team_id == team_id, MovementPattern.team_id.is_(None)))
    if map_name:
        query = query.where(MovementPattern.map_name == map_name)
    if side:
        query = query.where(MovementPattern.side == side)
    if phase:
        query = query.where(MovementPattern.phase == phase)

    query = query.order_by(MovementPattern.frequency.desc()).limit(limit)

    result = await db.execute(query)
    patterns = result.scalars().all()
    return patterns


@router.post("/query", response_model=List[PatternResponse])
async def query_patterns(query: PatternQuery, db: AsyncSession = Depends(get_db)):
    """Query patterns for simulation - returns best matching patterns."""
    stmt = select(MovementPattern).where(
        and_(
            MovementPattern.is_active == True,
            MovementPattern.map_name == query.map_name,
            MovementPattern.side == query.side,
            MovementPattern.phase == query.phase,
            or_(MovementPattern.team_id == query.team_id, MovementPattern.team_id.is_(None))
        )
    ).order_by(
        # Prefer team-specific patterns
        MovementPattern.team_id.is_(None),
        MovementPattern.frequency.desc()
    ).limit(query.limit)

    result = await db.execute(stmt)
    patterns = result.scalars().all()
    return patterns


@router.get("/{pattern_id}", response_model=PatternResponse)
async def get_pattern(pattern_id: int, db: AsyncSession = Depends(get_db)):
    """Get a specific pattern by ID."""
    result = await db.execute(
        select(MovementPattern).where(MovementPattern.id == pattern_id)
    )
    pattern = result.scalar_one_or_none()
    if not pattern:
        raise HTTPException(status_code=404, detail="Pattern not found")
    return pattern


@router.post("/", response_model=PatternResponse)
async def create_pattern(pattern: PatternCreate, db: AsyncSession = Depends(get_db)):
    """Create a new movement pattern."""
    db_pattern = MovementPattern(
        team_id=pattern.team_id,
        map_name=pattern.map_name,
        side=pattern.side,
        phase=pattern.phase,
        pattern_name=pattern.pattern_name,
        waypoints=[w.model_dump() for w in pattern.waypoints],
        frequency=pattern.frequency,
        success_rate=pattern.success_rate,
        sample_count=pattern.sample_count,
    )
    db.add(db_pattern)
    await db.commit()
    await db.refresh(db_pattern)
    return db_pattern


# Player Tendencies
@router.get("/tendencies/{player_id}", response_model=List[TendencyResponse])
async def get_player_tendencies(
    player_id: str,
    map_name: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db)
):
    """Get tendencies for a player."""
    query = select(PlayerTendency).where(PlayerTendency.player_id == player_id)
    if map_name:
        query = query.where(PlayerTendency.map_name == map_name)

    result = await db.execute(query)
    tendencies = result.scalars().all()
    return tendencies


# Team Strategies
@router.get("/strategies/{team_id}", response_model=List[StrategyResponse])
async def get_team_strategies(
    team_id: str,
    map_name: Optional[str] = Query(None),
    side: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db)
):
    """Get strategies for a team."""
    query = select(TeamStrategy).where(
        and_(TeamStrategy.team_id == team_id, TeamStrategy.is_active == True)
    )
    if map_name:
        query = query.where(TeamStrategy.map_name == map_name)
    if side:
        query = query.where(TeamStrategy.side == side)

    query = query.order_by(TeamStrategy.frequency.desc())

    result = await db.execute(query)
    strategies = result.scalars().all()
    return strategies
