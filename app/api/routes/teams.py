from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from typing import List

from ...database import get_db
from ...models import Team, Player
from ...schemas.teams import TeamCreate, TeamResponse, TeamWithPlayersResponse, PlayerCreate, PlayerResponse

router = APIRouter()

# Fallback mock data when database is unavailable
MOCK_TEAMS = [
    {"id": "cloud9", "name": "Cloud9", "short_name": "C9", "region": "NA", "logo_url": None},
    {"id": "sentinels", "name": "Sentinels", "short_name": "SEN", "region": "NA", "logo_url": None},
    {"id": "fnatic", "name": "Fnatic", "short_name": "FNC", "region": "EMEA", "logo_url": None},
    {"id": "loud", "name": "LOUD", "short_name": "LOUD", "region": "BR", "logo_url": None},
    {"id": "drx", "name": "DRX", "short_name": "DRX", "region": "KR", "logo_url": None},
    {"id": "nrg", "name": "NRG Esports", "short_name": "NRG", "region": "NA", "logo_url": None},
    {"id": "g2", "name": "G2 Esports", "short_name": "G2", "region": "EMEA", "logo_url": None},
    {"id": "100t", "name": "100 Thieves", "short_name": "100T", "region": "NA", "logo_url": None},
]


@router.get("/")
async def list_teams():
    """List all teams."""
    try:
        from ...database import AsyncSessionLocal
        async with AsyncSessionLocal() as db:
            result = await db.execute(select(Team).order_by(Team.name))
            teams = result.scalars().all()
            if teams:
                return teams
            return MOCK_TEAMS
    except Exception:
        # Return mock data if database unavailable
        return MOCK_TEAMS


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
