from sqlalchemy import Column, String, Integer, ForeignKey
from sqlalchemy.dialects.postgresql import TIMESTAMP, JSONB, UUID
from sqlalchemy.sql import func
import uuid
from ..database import Base


class SimulationSession(Base):
    """User simulation sessions - tracking state and events."""
    __tablename__ = "simulation_sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Configuration
    attack_team_id = Column(String(64), ForeignKey("teams.id"))
    defense_team_id = Column(String(64), ForeignKey("teams.id"))
    map_name = Column(String(32), nullable=False)
    round_type = Column(String(20), default="full_buy")

    # State
    current_time_ms = Column(Integer, default=0)
    phase = Column(String(20), default="opening")
    status = Column(String(20), default="created")  # 'created', 'running', 'paused', 'completed'

    # Snapshots for what-if scenarios
    snapshots = Column(JSONB, default=list)

    # Results
    events_log = Column(JSONB, default=list)

    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())
