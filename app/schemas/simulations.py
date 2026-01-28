from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime
from uuid import UUID


class PlayerPosition(BaseModel):
    player_id: str
    team_id: str
    x: float
    y: float
    is_alive: bool = True
    health: int = 100
    agent: Optional[str] = None
    side: str  # 'attack' or 'defense'

    # NEW: Weapon & Economy fields
    shield: int = 0
    weapon_name: Optional[str] = None
    armor_name: Optional[str] = None
    loadout_value: int = 0

    # NEW: Status effects
    is_flashed: bool = False
    is_slowed: bool = False
    is_revealed: bool = False

    # NEW: Role assignment
    role: Optional[str] = None  # 'entry', 'support', 'lurk', 'anchor', 'flex'

    # VCT-derived: Facing direction and spike status
    facing_angle: Optional[float] = None  # Radians (0 = right, pi/2 = down)
    has_spike: bool = False


class SimulationEvent(BaseModel):
    timestamp_ms: int
    event_type: str  # 'kill', 'death', 'spike_plant', 'spike_defuse', 'ability'
    player_id: Optional[str] = None
    target_id: Optional[str] = None
    position_x: Optional[float] = None
    position_y: Optional[float] = None
    details: Optional[Dict[str, Any]] = None


class SimulationTick(BaseModel):
    timestamp_ms: int
    phase: str
    positions: List[PlayerPosition]
    events: List[SimulationEvent] = []
    spike_planted: bool = False
    spike_site: Optional[str] = None
    attack_alive: int = 5
    defense_alive: int = 5


class SimulationCreate(BaseModel):
    attack_team_id: str
    defense_team_id: str
    map_name: str
    round_type: str = "full_buy"


class SimulationState(BaseModel):
    session_id: UUID
    current_time_ms: int
    phase: str
    status: str
    positions: List[PlayerPosition]
    events: List[SimulationEvent]
    spike_planted: bool = False
    spike_site: Optional[str] = None

    # NEW: Win probability tracking
    win_probability: Optional[Dict[str, float]] = None  # {'attack': 0.55, 'defense': 0.45}

    # NEW: Round state tracking
    first_blood_team: Optional[str] = None
    first_blood_time_ms: Optional[int] = None
    attack_alive: int = 5
    defense_alive: int = 5

    # NEW: Economy state
    attack_buy_type: Optional[str] = None  # 'pistol', 'eco', 'force', 'half', 'full'
    defense_buy_type: Optional[str] = None
    attack_loadout_value: int = 0
    defense_loadout_value: int = 0

    # NEW: Strategy info
    attack_strategy: Optional[str] = None
    defense_strategy: Optional[str] = None


class SimulationResponse(BaseModel):
    id: UUID
    attack_team_id: Optional[str] = None
    defense_team_id: Optional[str] = None
    map_name: str
    round_type: str
    current_time_ms: int
    phase: str
    status: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class WhatIfScenario(BaseModel):
    """Request to run a what-if scenario from a snapshot."""
    snapshot_time_ms: int
    modifications: Dict[str, Any]  # e.g., {"player_1": {"x": 100, "y": 200}}


class SimulationAnalysis(BaseModel):
    """Analysis results for a completed simulation."""
    winner: str  # 'attack' or 'defense'
    total_duration_ms: int
    kills: List[SimulationEvent]
    key_moments: List[Dict[str, Any]]
    improvement_suggestions: List[str]

    # NEW: Detailed round outcome
    spike_planted: bool = False
    spike_site: Optional[str] = None
    attack_alive: int = 0
    defense_alive: int = 0

    # NEW: Win probability at end of round
    win_probability: Optional[Dict[str, float]] = None

    # NEW: First blood analysis
    first_blood_team: Optional[str] = None
    first_blood_time_ms: Optional[int] = None

    # NEW: Economy analysis
    attack_loadout_value: int = 0
    defense_loadout_value: int = 0
    economy_advantage: Optional[str] = None  # 'attack', 'defense', 'even'

    # NEW: Trade analysis
    total_trades: int = 0
    trade_rate: float = 0.0

    # NEW: Ability usage summary
    abilities_used: int = 0
    smokes_deployed: int = 0
    flashes_thrown: int = 0
