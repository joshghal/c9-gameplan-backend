from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from typing import List

from ...database import get_db
from ...models import Team, Player
from ...schemas.teams import TeamCreate, TeamResponse, TeamWithPlayersResponse, PlayerCreate, PlayerResponse

router = APIRouter()


@router.get("/", response_model=List[TeamResponse])
async def list_teams(db: AsyncSession = Depends(get_db)):
    """List all teams."""
    result = await db.execute(select(Team).order_by(Team.name))
    teams = result.scalars().all()
    return teams


@router.get("/{team_id}", response_model=TeamWithPlayersResponse)
async def get_team(team_id: str, db: AsyncSession = Depends(get_db)):
    """Get a team by ID with its players."""
    result = await db.execute(
        select(Team)
        .options(selectinload(Team.players))
        .where(Team.id == team_id)
    )
    team = result.scalar_one_or_none()
    if not team:
        raise HTTPException(status_code=404, detail="Team not found")
    return team


@router.post("/", response_model=TeamResponse)
async def create_team(team: TeamCreate, db: AsyncSession = Depends(get_db)):
    """Create a new team."""
    db_team = Team(**team.model_dump())
    db.add(db_team)
    await db.commit()
    await db.refresh(db_team)
    return db_team


@router.get("/{team_id}/players", response_model=List[PlayerResponse])
async def get_team_players(team_id: str, db: AsyncSession = Depends(get_db)):
    """Get all players for a team."""
    result = await db.execute(
        select(Player).where(Player.team_id == team_id).order_by(Player.name)
    )
    players = result.scalars().all()
    return players


@router.post("/{team_id}/players", response_model=PlayerResponse)
async def add_player_to_team(team_id: str, player: PlayerCreate, db: AsyncSession = Depends(get_db)):
    """Add a player to a team."""
    # Verify team exists
    result = await db.execute(select(Team).where(Team.id == team_id))
    team = result.scalar_one_or_none()
    if not team:
        raise HTTPException(status_code=404, detail="Team not found")

    db_player = Player(**player.model_dump(), team_id=team_id)
    db.add(db_player)
    await db.commit()
    await db.refresh(db_player)
    return db_player
