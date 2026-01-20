from sqlalchemy import Column, String, Integer, Float, Boolean, Text, ARRAY
from sqlalchemy.dialects.postgresql import JSONB
from ..database import Base


class MapConfig(Base):
    """Static map configuration - coordinate conversion and bounds."""
    __tablename__ = "map_config"

    id = Column(Integer, primary_key=True, autoincrement=True)
    map_name = Column(String(32), unique=True, nullable=False)
    display_name = Column(String(64), nullable=False)

    # Coordinate conversion (from valorant-api.com)
    # Note: Coordinates are SWAPPED!
    # minimapX = gameY * xMultiplier + xScalarToAdd
    # minimapY = gameX * yMultiplier + yScalarToAdd
    x_multiplier = Column(Float, nullable=False)
    y_multiplier = Column(Float, nullable=False)
    x_scalar_add = Column(Float, nullable=False)
    y_scalar_add = Column(Float, nullable=False)

    # Map bounds (GRID coordinates)
    bounds_min_x = Column(Float)
    bounds_min_y = Column(Float)
    bounds_max_x = Column(Float)
    bounds_max_y = Column(Float)

    # Assets
    minimap_url = Column(Text)
    splash_url = Column(Text)

    is_active = Column(Boolean, default=True)


class MapZone(Base):
    """Map zones - callouts, sites, chokepoints."""
    __tablename__ = "map_zones"

    id = Column(Integer, primary_key=True, autoincrement=True)
    map_name = Column(String(32), nullable=False)
    zone_name = Column(String(64), nullable=False)
    zone_type = Column(String(20), nullable=False)  # 'site', 'chokepoint', 'rotation', 'spawn', 'callout'
    super_region = Column(String(32))  # 'A', 'B', 'C', 'Mid'

    # Position
    center_x = Column(Float, nullable=False)
    center_y = Column(Float, nullable=False)

    # Polygon boundary (optional)
    boundary = Column(JSONB)  # Array of {x, y} points

    # Navigation
    connected_zones = Column(ARRAY(String(64)))  # Adjacent zone names
