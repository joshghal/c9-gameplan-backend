from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List, Optional

from ...database import get_db
from ...models import MapConfig, MapZone
from ...schemas.maps import MapConfigResponse, MapZoneResponse, CoordinateConvertRequest, CoordinateConvertResponse

router = APIRouter()


@router.get("/", response_model=List[MapConfigResponse])
async def list_maps(
    active_only: bool = Query(True),
    db: AsyncSession = Depends(get_db)
):
    """List all map configurations."""
    query = select(MapConfig)
    if active_only:
        query = query.where(MapConfig.is_active == True)
    query = query.order_by(MapConfig.display_name)

    result = await db.execute(query)
    maps = result.scalars().all()
    return maps


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
