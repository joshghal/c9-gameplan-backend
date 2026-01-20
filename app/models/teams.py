from sqlalchemy import Column, String, ForeignKey, Text
from sqlalchemy.dialects.postgresql import TIMESTAMP
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from ..database import Base


class Team(Base):
    __tablename__ = "teams"

    id = Column(String(64), primary_key=True)
    name = Column(String(128), nullable=False)
    short_name = Column(String(16))
    region = Column(String(32))
    logo_url = Column(Text)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    players = relationship("Player", back_populates="team")


class Player(Base):
    __tablename__ = "players"

    id = Column(String(64), primary_key=True)
    team_id = Column(String(64), ForeignKey("teams.id"))
    name = Column(String(64), nullable=False)
    real_name = Column(String(128))
    country = Column(String(3))
    role = Column(String(32))  # 'duelist', 'initiator', 'controller', 'sentinel'
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    team = relationship("Team", back_populates="players")
