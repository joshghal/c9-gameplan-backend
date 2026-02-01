from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List, Optional

from ...database import get_db
from ...models import MapConfig, MapZone
from ...schemas.maps import MapConfigResponse, MapZoneResponse, CoordinateConvertRequest, CoordinateConvertResponse

router = APIRouter()

# Fallback mock data when database is unavailable
MOCK_MAPS = [
    {"id": 1, "map_name": "abyss", "display_name": "Abyss", "x_multiplier": 0.00007, "y_multiplier": -0.00007, "x_scalar_add": 0.5, "y_scalar_add": 0.5, "bounds_min_x": -5000, "bounds_min_y": -5000, "bounds_max_x": 5000, "bounds_max_y": 5000, "minimap_url": None, "splash_url": None, "is_active": True},
    {"id": 2, "map_name": "ascent", "display_name": "Ascent", "x_multiplier": 0.00007, "y_multiplier": -0.00007, "x_scalar_add": 0.5, "y_scalar_add": 0.5, "bounds_min_x": -5000, "bounds_min_y": -5000, "bounds_max_x": 5000, "bounds_max_y": 5000, "minimap_url": None, "splash_url": None, "is_active": True},
    {"id": 3, "map_name": "bind", "display_name": "Bind", "x_multiplier": 0.00007, "y_multiplier": -0.00007, "x_scalar_add": 0.5, "y_scalar_add": 0.5, "bounds_min_x": -5000, "bounds_min_y": -5000, "bounds_max_x": 5000, "bounds_max_y": 5000, "minimap_url": None, "splash_url": None, "is_active": True},
    {"id": 4, "map_name": "breeze", "display_name": "Breeze", "x_multiplier": 0.00007, "y_multiplier": -0.00007, "x_scalar_add": 0.5, "y_scalar_add": 0.5, "bounds_min_x": -5000, "bounds_min_y": -5000, "bounds_max_x": 5000, "bounds_max_y": 5000, "minimap_url": None, "splash_url": None, "is_active": True},
    {"id": 5, "map_name": "corrode", "display_name": "Corrode", "x_multiplier": 0.00007, "y_multiplier": -0.00007, "x_scalar_add": 0.5, "y_scalar_add": 0.5, "bounds_min_x": -5000, "bounds_min_y": -5000, "bounds_max_x": 5000, "bounds_max_y": 5000, "minimap_url": None, "splash_url": None, "is_active": True},
    {"id": 6, "map_name": "fracture", "display_name": "Fracture", "x_multiplier": 0.00007, "y_multiplier": -0.00007, "x_scalar_add": 0.5, "y_scalar_add": 0.5, "bounds_min_x": -5000, "bounds_min_y": -5000, "bounds_max_x": 5000, "bounds_max_y": 5000, "minimap_url": None, "splash_url": None, "is_active": True},
    {"id": 7, "map_name": "haven", "display_name": "Haven", "x_multiplier": 0.00007, "y_multiplier": -0.00007, "x_scalar_add": 0.5, "y_scalar_add": 0.5, "bounds_min_x": -5000, "bounds_min_y": -5000, "bounds_max_x": 5000, "bounds_max_y": 5000, "minimap_url": None, "splash_url": None, "is_active": True},
    {"id": 8, "map_name": "icebox", "display_name": "Icebox", "x_multiplier": 0.00007, "y_multiplier": -0.00007, "x_scalar_add": 0.5, "y_scalar_add": 0.5, "bounds_min_x": -5000, "bounds_min_y": -5000, "bounds_max_x": 5000, "bounds_max_y": 5000, "minimap_url": None, "splash_url": None, "is_active": True},
    {"id": 9, "map_name": "lotus", "display_name": "Lotus", "x_multiplier": 0.00007, "y_multiplier": -0.00007, "x_scalar_add": 0.5, "y_scalar_add": 0.5, "bounds_min_x": -5000, "bounds_min_y": -5000, "bounds_max_x": 5000, "bounds_max_y": 5000, "minimap_url": None, "splash_url": None, "is_active": True},
    {"id": 10, "map_name": "pearl", "display_name": "Pearl", "x_multiplier": 0.00007, "y_multiplier": -0.00007, "x_scalar_add": 0.5, "y_scalar_add": 0.5, "bounds_min_x": -5000, "bounds_min_y": -5000, "bounds_max_x": 5000, "bounds_max_y": 5000, "minimap_url": None, "splash_url": None, "is_active": True},
    {"id": 11, "map_name": "split", "display_name": "Split", "x_multiplier": 0.00007, "y_multiplier": -0.00007, "x_scalar_add": 0.5, "y_scalar_add": 0.5, "bounds_min_x": -5000, "bounds_min_y": -5000, "bounds_max_x": 5000, "bounds_max_y": 5000, "minimap_url": None, "splash_url": None, "is_active": True},
    {"id": 12, "map_name": "sunset", "display_name": "Sunset", "x_multiplier": 0.00007, "y_multiplier": -0.00007, "x_scalar_add": 0.5, "y_scalar_add": 0.5, "bounds_min_x": -5000, "bounds_min_y": -5000, "bounds_max_x": 5000, "bounds_max_y": 5000, "minimap_url": None, "splash_url": None, "is_active": True},
]


@router.get("/")
async def list_maps(active_only: bool = Query(True)):
    """List all map configurations."""
    try:
        from ...database import AsyncSessionLocal
        async with AsyncSessionLocal() as db:
            query = select(MapConfig)
            if active_only:
                query = query.where(MapConfig.is_active == True)
            query = query.order_by(MapConfig.display_name)

            result = await db.execute(query)
            maps = result.scalars().all()
            if maps:
                return maps
            return MOCK_MAPS
    except Exception:
        # Return mock data if database unavailable
        return MOCK_MAPS


@router.get("/{map_name}", response_model=MapConfigResponse)
async def get_map(map_name: str, db: AsyncSession = Depends(get_db)):
    """Get a specific map configuration."""
    result = await db.execute(
        select(MapConfig).where(MapConfig.map_name == map_name)
    )
    map_config = result.scalar_one_or_none()
    if not map_config:
        raise HTTPException(status_code=404, detail="Map not found")
    return map_config


@router.get("/{map_name}/zones", response_model=List[MapZoneResponse])
async def get_map_zones(
    map_name: str,
    zone_type: Optional[str] = Query(None),
    super_region: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db)
):
    """Get zones for a map."""
    query = select(MapZone).where(MapZone.map_name == map_name)
    if zone_type:
        query = query.where(MapZone.zone_type == zone_type)
    if super_region:
        query = query.where(MapZone.super_region == super_region)
    query = query.order_by(MapZone.zone_name)

    result = await db.execute(query)
    zones = result.scalars().all()
    return zones


@router.post("/convert-coordinates", response_model=CoordinateConvertResponse)
async def convert_coordinates(
    request: CoordinateConvertRequest,
    db: AsyncSession = Depends(get_db)
):
    """Convert game coordinates to normalized minimap coordinates.

    Note: Coordinates are SWAPPED in VALORANT!
    - minimapX = gameY * xMultiplier + xScalarToAdd
    - minimapY = gameX * yMultiplier + yScalarToAdd
    """
    result = await db.execute(
        select(MapConfig).where(MapConfig.map_name == request.map_name)
    )
    map_config = result.scalar_one_or_none()
    if not map_config:
        raise HTTPException(status_code=404, detail="Map not found")

    # Apply coordinate conversion (coordinates are swapped!)
    normalized_x = request.game_y * map_config.x_multiplier + map_config.x_scalar_add
    normalized_y = request.game_x * map_config.y_multiplier + map_config.y_scalar_add

    return CoordinateConvertResponse(
        normalized_x=normalized_x,
        normalized_y=normalized_y,
        game_x=request.game_x,
        game_y=request.game_y,
        map_name=request.map_name,
    )
