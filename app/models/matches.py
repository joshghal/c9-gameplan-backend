from sqlalchemy import Column, String, Integer, Boolean, Date, ForeignKey
from sqlalchemy.dialects.postgresql import TIMESTAMP
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from ..database import Base


class Match(Base):
    __tablename__ = "matches"

    id = Column(String(64), primary_key=True)
    team1_id = Column(String(64), ForeignKey("teams.id"))
    team2_id = Column(String(64), ForeignKey("teams.id"))
    map_name = Column(String(32), nullable=False)
    tournament = Column(String(128))
    match_date = Column(Date)
    final_score = Column(String(10))
    winner_id = Column(String(64), ForeignKey("teams.id"))
    data_source = Column(String(20), default="grid")  # 'grid', 'cv', 'both'
    grid_series_id = Column(String(64))
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    # Relationships
    team1 = relationship("Team", foreign_keys=[team1_id])
    team2 = relationship("Team", foreign_keys=[team2_id])
    winner = relationship("Team", foreign_keys=[winner_id])
    rounds = relationship("Round", back_populates="match", cascade="all, delete-orphan")


class Round(Base):
    __tablename__ = "rounds"

    id = Column(Integer, primary_key=True, autoincrement=True)
    match_id = Column(String(64), ForeignKey("matches.id", ondelete="CASCADE"))
    round_number = Column(Integer, nullable=False)
    attacking_team = Column(String(64), ForeignKey("teams.id"))
    defending_team = Column(String(64), ForeignKey("teams.id"))
    winner_id = Column(String(64), ForeignKey("teams.id"))
    round_type = Column(String(20))  # 'pistol', 'eco', 'force', 'full_buy'
    spike_planted = Column(Boolean, default=False)
    spike_site = Column(String(1))
    duration_ms = Column(Integer)
    attack_score = Column(Integer)
    defense_score = Column(Integer)

    # Relationships
    match = relationship("Match", back_populates="rounds")
