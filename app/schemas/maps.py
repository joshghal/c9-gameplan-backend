from pydantic import BaseModel
from typing import Optional, List, Dict, Any


class Coordinate(BaseModel):
    x: float
    y: float


class MapConfigResponse(BaseModel):
    id: int
    map_name: str
    display_name: str
    x_multiplier: float
    y_multiplier: float
    x_scalar_add: float
    y_scalar_add: float
    bounds_min_x: Optional[float] = None
    bounds_min_y: Optional[float] = None
    bounds_max_x: Optional[float] = None
    bounds_max_y: Optional[float] = None
    minimap_url: Optional[str] = None
    splash_url: Optional[str] = None
    is_active: bool

    class Config:
        from_attributes = True


class MapZoneResponse(BaseModel):
    id: int
    map_name: str
    zone_name: str
    zone_type: str  # 'site', 'chokepoint', 'rotation', 'spawn', 'callout'
    super_region: Optional[str] = None
    center_x: float
    center_y: float
    boundary: Optional[List[Coordinate]] = None
    connected_zones: Optional[List[str]] = None

    class Config:
        from_attributes = True


class CoordinateConvertRequest(BaseModel):
    game_x: float
    game_y: float
    map_name: str


class CoordinateConvertResponse(BaseModel):
    normalized_x: float
    normalized_y: float
    game_x: float
    game_y: float
    map_name: str
