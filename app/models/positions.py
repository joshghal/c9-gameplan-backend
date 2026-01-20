from sqlalchemy import Column, String, Integer, Float, Boolean, Text
from sqlalchemy.dialects.postgresql import TIMESTAMP, JSONB
from sqlalchemy.sql import func
from ..database import Base


class RawPosition(Base):
    """Time-series position data - stored in TimescaleDB hypertable."""
    __tablename__ = "raw_positions"

    time = Column(TIMESTAMP(timezone=True), primary_key=True, nullable=False)
    match_id = Column(String(64), primary_key=True, nullable=False)
    player_id = Column(String(64), primary_key=True, nullable=False)
    round_number = Column(Integer, nullable=False)
    timestamp_ms = Column(Integer, nullable=False)  # Milliseconds from round start
    team_id = Column(String(64), nullable=False)
    map_name = Column(String(32), nullable=False)
    side = Column(String(10), nullable=False)  # 'attack' or 'defense'
    x = Column(Float, nullable=False)  # Game coordinate X
    y = Column(Float, nullable=False)  # Game coordinate Y
    source = Column(String(10), nullable=False)  # 'cv' or 'grid'
    confidence = Column(Float, default=1.0)


class GridEvent(Base):
    """Events from GRID API - kills, deaths, spike plants, etc."""
    __tablename__ = "grid_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    match_id = Column(String(64), nullable=False)
    round_number = Column(Integer, nullable=False)
    timestamp_ms = Column(Integer, nullable=False)  # Milliseconds from round start

    event_type = Column(String(20), nullable=False)  # 'kill', 'death', 'spike_plant', 'spike_defuse', 'ability'

    # Position data
    position_x = Column(Float)
    position_y = Column(Float)

    # Kill/Death specific
    player_id = Column(String(64))
    player_team_id = Column(String(64))
    killer_id = Column(String(64))
    killer_team_id = Column(String(64))
    weapon = Column(String(32))
    headshot = Column(Boolean)
    damage = Column(Integer)

    # Ability specific
    ability_name = Column(String(32))
    agent = Column(String(32))

    # Raw data backup
    raw_data = Column(JSONB)

    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
