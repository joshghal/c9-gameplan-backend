from pydantic import BaseModel
from typing import Optional, List
from datetime import date, datetime


class RoundBase(BaseModel):
    round_number: int
    attacking_team: Optional[str] = None
    defending_team: Optional[str] = None
    winner_id: Optional[str] = None
    round_type: Optional[str] = None  # 'pistol', 'eco', 'force', 'full_buy'
    spike_planted: bool = False
    spike_site: Optional[str] = None
    duration_ms: Optional[int] = None
    attack_score: Optional[int] = None
    defense_score: Optional[int] = None


class RoundResponse(RoundBase):
    id: int
    match_id: str

    class Config:
        from_attributes = True


class MatchBase(BaseModel):
    map_name: str
    tournament: Optional[str] = None
    match_date: Optional[date] = None
    final_score: Optional[str] = None
    data_source: str = "grid"


class MatchCreate(MatchBase):
    id: str
    team1_id: Optional[str] = None
    team2_id: Optional[str] = None
    winner_id: Optional[str] = None
    grid_series_id: Optional[str] = None


class MatchResponse(MatchBase):
    id: str
    team1_id: Optional[str] = None
    team2_id: Optional[str] = None
    winner_id: Optional[str] = None
    grid_series_id: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


class MatchWithRoundsResponse(MatchResponse):
    rounds: List[RoundResponse] = []
