from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from typing import List, Optional

from ...database import get_db
from ...models import Match, Round
from ...schemas.matches import MatchCreate, MatchResponse, MatchWithRoundsResponse, RoundResponse

router = APIRouter()


@router.get("/", response_model=List[MatchResponse])
async def list_matches(
    team_id: Optional[str] = Query(None, description="Filter by team"),
    map_name: Optional[str] = Query(None, description="Filter by map"),
    limit: int = Query(50, le=100),
    db: AsyncSession = Depends(get_db)
):
    """List matches with optional filters."""
    query = select(Match)

    if team_id:
        query = query.where((Match.team1_id == team_id) | (Match.team2_id == team_id))
    if map_name:
        query = query.where(Match.map_name == map_name)

    query = query.order_by(Match.match_date.desc()).limit(limit)

    result = await db.execute(query)
    matches = result.scalars().all()
    return matches


@router.get("/{match_id}", response_model=MatchWithRoundsResponse)
async def get_match(match_id: str, db: AsyncSession = Depends(get_db)):
    """Get a match by ID with its rounds."""
    result = await db.execute(
        select(Match)
        .options(selectinload(Match.rounds))
        .where(Match.id == match_id)
    )
    match = result.scalar_one_or_none()
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")
    return match


@router.post("/", response_model=MatchResponse)
async def create_match(match: MatchCreate, db: AsyncSession = Depends(get_db)):
    """Create a new match."""
    db_match = Match(**match.model_dump())
    db.add(db_match)
    await db.commit()
    await db.refresh(db_match)
    return db_match


@router.get("/{match_id}/rounds", response_model=List[RoundResponse])
async def get_match_rounds(match_id: str, db: AsyncSession = Depends(get_db)):
    """Get all rounds for a match."""
    result = await db.execute(
        select(Round).where(Round.match_id == match_id).order_by(Round.round_number)
    )
    rounds = result.scalars().all()
    return rounds
