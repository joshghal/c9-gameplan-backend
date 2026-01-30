from pydantic import BaseModel
from typing import List, Optional, Dict


class TeamMember(BaseModel):
    player_id: str
    name: str
    role: str  # 'entry', 'support', 'lurk', 'anchor', 'flex'
    agent: str
    weapon: str
    spawn: List[float]  # [x, y] normalized 0-1


class GhostPoint(BaseModel):
    time_s: float
    x: float
    y: float


class GhostPath(BaseModel):
    player_id: str
    name: str
    agent: str
    segments: Dict[str, List[GhostPoint]]  # phase → points


class StrategyRound(BaseModel):
    round_id: str
    map_name: str
    user_side: str  # 'attack' or 'defense'
    teammates: List[TeamMember]
    phase_times: Dict[str, List[int]]  # e.g. {"setup": [0, 12], ...}
    round_duration_s: int  # actual round duration in seconds
    ghost_paths: List[GhostPath]  # pro player paths for user's side


class Waypoint(BaseModel):
    tick: int
    x: float
    y: float
    facing: float  # radians


class PlayerPlan(BaseModel):
    player_id: str
    waypoints: List[Waypoint]


class StrategyExecuteRequest(BaseModel):
    round_id: str
    side: str  # 'attack' or 'defense' — which side the user picked
    plans: Dict[str, List[PlayerPlan]]  # phase_name → player plans


class StrategyReplayRequest(BaseModel):
    round_id: str


class StrategyEvent(BaseModel):
    time_ms: int
    event_type: str
    player_id: Optional[str] = None
    target_id: Optional[str] = None
    details: Optional[Dict] = None


class StrategySnapshot(BaseModel):
    time_ms: int
    phase: str
    players: List[Dict]  # [{player_id, x, y, side, is_alive, ...}]


class StrategyReveal(BaseModel):
    opponent_team: str
    user_team: str = "Cloud9"
    atk_team: Optional[str] = None
    def_team: Optional[str] = None
    round_desc: str
    round_num: Optional[int] = None
    map_name: Optional[str] = None
    user_side: Optional[str] = None
    opponent_players: Optional[List[str]] = None
    round_duration_s: Optional[int] = None
    sim_duration_s: Optional[int] = None
    score_line: Optional[str] = None
    tournament: Optional[str] = None
    match_date: Optional[str] = None


class StrategyResult(BaseModel):
    session_id: str
    winner: str  # 'attack' or 'defense'
    events: List[StrategyEvent]
    snapshots: List[StrategySnapshot]
    reveal: StrategyReveal
