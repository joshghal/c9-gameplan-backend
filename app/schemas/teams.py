from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime


class TeamBase(BaseModel):
    name: str
    short_name: Optional[str] = None
    region: Optional[str] = None
    logo_url: Optional[str] = None


class TeamCreate(TeamBase):
    id: str


class TeamResponse(TeamBase):
    id: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class PlayerBase(BaseModel):
    name: str
    real_name: Optional[str] = None
    country: Optional[str] = None
    role: Optional[str] = None  # 'duelist', 'initiator', 'controller', 'sentinel'


class PlayerCreate(PlayerBase):
    id: str
    team_id: Optional[str] = None


class PlayerResponse(PlayerBase):
    id: str
    team_id: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class TeamWithPlayersResponse(TeamResponse):
    players: List[PlayerResponse] = []
