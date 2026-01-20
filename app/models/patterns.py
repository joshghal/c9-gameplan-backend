from sqlalchemy import Column, String, Integer, Float, Boolean, ForeignKey, ARRAY
from sqlalchemy.dialects.postgresql import TIMESTAMP, JSONB
from sqlalchemy.sql import func
from ..database import Base


class MovementPattern(Base):
    """Learned movement patterns - waypoints and trajectories."""
    __tablename__ = "movement_patterns"

    id = Column(Integer, primary_key=True, autoincrement=True)
    team_id = Column(String(64), ForeignKey("teams.id"))  # NULL for global patterns
    map_name = Column(String(32), nullable=False)
    side = Column(String(10), nullable=False)  # 'attack' or 'defense'
    phase = Column(String(20), nullable=False)  # 'opening', 'mid_round', 'post_plant', 'retake'
    pattern_name = Column(String(64), nullable=False)

    # Pattern data: array of {timestamp_ms, x, y, variance_x, variance_y}
    waypoints = Column(JSONB, nullable=False)

    # Statistics
    frequency = Column(Float, nullable=False, default=0)  # How often this pattern occurs (0-1)
    success_rate = Column(Float)  # Win rate when using this pattern
    sample_count = Column(Integer, nullable=False, default=0)

    # Metadata
    is_active = Column(Boolean, default=True)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())


class PlayerTendency(Base):
    """Player-specific behavioral statistics."""
    __tablename__ = "player_tendencies"

    id = Column(Integer, primary_key=True, autoincrement=True)
    player_id = Column(String(64), ForeignKey("players.id"))
    map_name = Column(String(32))  # NULL for global stats
    side = Column(String(10))  # NULL for global stats

    # Movement tendencies
    avg_movement_speed = Column(Float)  # Units per second
    aggression_index = Column(Float)  # 0 (passive) to 1 (aggressive)
    positioning_tendency = Column(String(20))  # 'entry', 'lurk', 'anchor', 'support'
    avg_first_contact_time = Column(Float)  # Seconds until first engagement

    # Position heat map (simplified)
    common_positions = Column(JSONB)  # Array of {x, y, frequency}

    # Combat stats (from GRID)
    kd_ratio = Column(Float)
    headshot_rate = Column(Float)
    first_blood_rate = Column(Float)
    clutch_rate = Column(Float)
    avg_damage_per_round = Column(Float)

    sample_count = Column(Integer, nullable=False, default=0)
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())


class TeamStrategy(Base):
    """Team-level strategic patterns."""
    __tablename__ = "team_strategies"

    id = Column(Integer, primary_key=True, autoincrement=True)
    team_id = Column(String(64), ForeignKey("teams.id"), nullable=False)
    map_name = Column(String(32), nullable=False)
    side = Column(String(10), nullable=False)  # 'attack' or 'defense'

    strategy_name = Column(String(64), nullable=False)  # 'default', 'a_split', 'b_rush', etc.
    description = Column(String)

    # Strategy composition
    player_roles = Column(JSONB)  # {player_id: role} mapping
    pattern_ids = Column(ARRAY(Integer))  # References to movement_patterns

    # Usage stats
    frequency = Column(Float, nullable=False, default=0)
    success_rate = Column(Float)
    sample_count = Column(Integer, nullable=False, default=0)

    # Conditions when used
    round_types = Column(ARRAY(String(20)))  # When is this strategy used?
    economy_min = Column(Integer)
    economy_max = Column(Integer)

    is_active = Column(Boolean, default=True)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())
