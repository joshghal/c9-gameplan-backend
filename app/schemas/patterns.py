from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime


class Waypoint(BaseModel):
    timestamp_ms: int
    x: float
    y: float
    variance_x: Optional[float] = None
    variance_y: Optional[float] = None


class PatternBase(BaseModel):
    map_name: str
    side: str  # 'attack' or 'defense'
    phase: str  # 'opening', 'mid_round', 'post_plant', 'retake'
    pattern_name: str
    waypoints: List[Waypoint]


class PatternCreate(PatternBase):
    team_id: Optional[str] = None
    frequency: float = 0.0
    success_rate: Optional[float] = None
    sample_count: int = 0


class PatternResponse(PatternBase):
    id: int
    team_id: Optional[str] = None
    frequency: float
    success_rate: Optional[float] = None
    sample_count: int
    is_active: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class PatternQuery(BaseModel):
    team_id: Optional[str] = None
    map_name: str
    side: str
    phase: str
    limit: int = 10


class CommonPosition(BaseModel):
    x: float
    y: float
    frequency: float


class TendencyResponse(BaseModel):
    id: int
    player_id: str
    map_name: Optional[str] = None
    side: Optional[str] = None
    avg_movement_speed: Optional[float] = None
    aggression_index: Optional[float] = None
    positioning_tendency: Optional[str] = None
    avg_first_contact_time: Optional[float] = None
    common_positions: Optional[List[CommonPosition]] = None
    kd_ratio: Optional[float] = None
    headshot_rate: Optional[float] = None
    first_blood_rate: Optional[float] = None
    clutch_rate: Optional[float] = None
    avg_damage_per_round: Optional[float] = None
    sample_count: int
    updated_at: datetime

    class Config:
        from_attributes = True


class StrategyResponse(BaseModel):
    id: int
    team_id: str
    map_name: str
    side: str
    strategy_name: str
    description: Optional[str] = None
    player_roles: Optional[Dict[str, str]] = None
    pattern_ids: Optional[List[int]] = None
    frequency: float
    success_rate: Optional[float] = None
    sample_count: int
    round_types: Optional[List[str]] = None
    economy_min: Optional[int] = None
    economy_max: Optional[int] = None
    is_active: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
