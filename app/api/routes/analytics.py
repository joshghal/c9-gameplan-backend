"""Analytics API routes for visualization data."""

from typing import Optional
from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel

router = APIRouter()


# Response Models
class HeatmapData(BaseModel):
    """Heatmap grid data."""
    map_name: str
    side: str
    grid: list[list[float]]
    grid_size: int
    sample_count: int
    origin: str = "top-left"  # Coordinate system origin: row[0] = top of map


class PredictionPath(BaseModel):
    """A predicted movement path."""
    player_id: str
    agent: str
    points: list[dict]  # [{x, y, timestamp_ms, confidence}]
    overall_confidence: float


class KillCluster(BaseModel):
    """A cluster of kill/death locations."""
    center_x: float
    center_y: float
    radius: float
    kill_count: int
    death_count: int
    dominant_side: str
    common_angles: list[str]


class EconomyEvent(BaseModel):
    """Economy event in timeline."""
    timestamp_ms: int
    team: str
    buy_type: str
    total_credits: int
    weapons: list[str]


class AbilityEvent(BaseModel):
    """Ability usage event."""
    timestamp_ms: int
    player_id: str
    agent: str
    ability: str
    location_x: float
    location_y: float
    result: Optional[str]


# Endpoints
@router.get("/heatmap/{map_name}", response_model=HeatmapData)
async def get_heatmap(
    map_name: str,
    side: str = Query("attack", pattern="^(attack|defense)$"),
    team_id: Optional[str] = Query(None),
    phase: Optional[str] = Query(None),
):
    """Get position heatmap data for a map.

    Returns a 20x20 grid of position frequencies normalized to 0-1.
    Only generates values for walkable areas of the map.
    """
    from pathlib import Path
    import json
    import random

    # Load walkable mask to constrain heatmap to playable area
    from app.services.map_context import get_map_context
    ctx = get_map_context()
    walkable_mask = ctx.get_walkable_mask(map_name, grid_size=20)

    # Site positions for hotspot generation (inline to avoid circular imports)
    MAP_SITES = {
        'split': {'sites': {'A': (0.30, 0.20), 'B': (0.50, 0.20)}, 'attack_spawn': (0.82, 0.55), 'defense_spawn': (0.15, 0.55)},
        'ascent': {'sites': {'A': (0.30, 0.20), 'B': (0.50, 0.20)}, 'attack_spawn': (0.85, 0.58), 'defense_spawn': (0.15, 0.42)},
        'bind': {'sites': {'A': (0.25, 0.35), 'B': (0.75, 0.35)}, 'attack_spawn': (0.55, 0.90), 'defense_spawn': (0.55, 0.15)},
        'haven': {'sites': {'A': (0.25, 0.20), 'B': (0.45, 0.40), 'C': (0.70, 0.20)}, 'attack_spawn': (0.87, 0.55), 'defense_spawn': (0.12, 0.40)},
        'icebox': {'sites': {'A': (0.55, 0.15), 'B': (0.60, 0.80)}, 'attack_spawn': (0.12, 0.58), 'defense_spawn': (0.92, 0.55)},
        'breeze': {'sites': {'A': (0.18, 0.25), 'B': (0.82, 0.28)}, 'attack_spawn': (0.50, 0.85), 'defense_spawn': (0.50, 0.20)},
        'fracture': {'sites': {'A': (0.20, 0.45), 'B': (0.80, 0.45)}, 'attack_spawn': (0.48, 0.87), 'defense_spawn': (0.60, 0.42)},
        'pearl': {'sites': {'A': (0.20, 0.35), 'B': (0.80, 0.35)}, 'attack_spawn': (0.52, 0.90), 'defense_spawn': (0.52, 0.10)},
        'lotus': {'sites': {'A': (0.20, 0.45), 'B': (0.50, 0.40), 'C': (0.80, 0.35)}, 'attack_spawn': (0.50, 0.85), 'defense_spawn': (0.55, 0.15)},
    }
    map_info = MAP_SITES.get(map_name.lower(), {'sites': {'A': (0.3, 0.3), 'B': (0.7, 0.3)}, 'attack_spawn': (0.5, 0.85), 'defense_spawn': (0.5, 0.15)})

    # Load from pattern data
    data_path = Path(__file__).parent.parent.parent.parent / "data" / "position_patterns.json"

    if data_path.exists():
        with open(data_path) as f:
            patterns = json.load(f)

        map_data = patterns.get(map_name, {})
        side_data = map_data.get(side, {})

        if side_data:
            # Extract grid from pattern data, mask to walkable areas
            raw_grid = side_data.get("heatmap", [[0] * 20 for _ in range(20)])
            grid = [
                [raw_grid[i][j] if walkable_mask[i, j] == 1 else 0 for j in range(20)]
                for i in range(20)
            ]
            sample_count = side_data.get("sample_count", 0)
        else:
            # Generate synthetic data - ONLY in walkable areas
            grid = [[0.0] * 20 for _ in range(20)]

            # Base noise in walkable areas
            for i in range(20):
                for j in range(20):
                    if walkable_mask[i, j] == 1:
                        grid[i][j] = random.random() * 0.15

            # Add hotspots near spawns and sites
            if side == "attack":
                # Heat near attack spawn
                sx, sy = map_info['attack_spawn']
                spawn_gx, spawn_gy = int(sx * 20), int(sy * 20)
                for di in range(-4, 5):
                    for dj in range(-4, 5):
                        ni, nj = spawn_gy + di, spawn_gx + dj
                        if 0 <= ni < 20 and 0 <= nj < 20 and walkable_mask[ni, nj] == 1:
                            dist = (di**2 + dj**2) ** 0.5
                            grid[ni][nj] += max(0, 0.7 - dist * 0.12)
            else:
                # Heat near sites for defenders
                for site_name, (sx, sy) in map_info['sites'].items():
                    site_gx, site_gy = int(sx * 20), int(sy * 20)
                    for di in range(-4, 5):
                        for dj in range(-4, 5):
                            ni, nj = site_gy + di, site_gx + dj
                            if 0 <= ni < 20 and 0 <= nj < 20 and walkable_mask[ni, nj] == 1:
                                dist = (di**2 + dj**2) ** 0.5
                                grid[ni][nj] += max(0, 0.8 - dist * 0.12)

            sample_count = 1000
    else:
        # Demo data - only in walkable areas
        grid = [
            [random.random() * 0.3 if walkable_mask[i, j] == 1 else 0 for j in range(20)]
            for i in range(20)
        ]
        sample_count = 500

    return HeatmapData(
        map_name=map_name,
        side=side,
        grid=grid,
        grid_size=20,
        sample_count=sample_count,
        origin="top-left",  # row[0] = top of map, col[0] = left of map
    )


@router.get("/predictions/{session_id}")
async def get_predictions(
    session_id: str,
    lookahead_ms: int = Query(5000, ge=1000, le=30000),
):
    """Get position predictions for current simulation state.

    Returns predicted paths for each player over the next lookahead_ms.
    """
    # This would integrate with the simulation engine
    # For now, return structured demo data
    predictions = [
        PredictionPath(
            player_id="player_1",
            agent="jett",
            points=[
                {"x": 0.5, "y": 0.8, "timestamp_ms": 0, "confidence": 1.0},
                {"x": 0.5, "y": 0.7, "timestamp_ms": 1000, "confidence": 0.9},
                {"x": 0.45, "y": 0.6, "timestamp_ms": 2000, "confidence": 0.7},
                {"x": 0.4, "y": 0.5, "timestamp_ms": 3000, "confidence": 0.5},
            ],
            overall_confidence=0.75,
        ),
        PredictionPath(
            player_id="player_2",
            agent="omen",
            points=[
                {"x": 0.3, "y": 0.85, "timestamp_ms": 0, "confidence": 1.0},
                {"x": 0.35, "y": 0.75, "timestamp_ms": 1500, "confidence": 0.85},
                {"x": 0.4, "y": 0.6, "timestamp_ms": 3000, "confidence": 0.6},
            ],
            overall_confidence=0.7,
        ),
    ]

    return {"session_id": session_id, "predictions": [p.model_dump() for p in predictions]}


@router.get("/kill-clusters/{map_name}")
async def get_kill_clusters(
    map_name: str,
    side: Optional[str] = Query(None),
    min_kills: int = Query(3, ge=1),
):
    """Get clustered kill/death locations for a map.

    Uses DBSCAN-style clustering to group nearby events.
    """
    from pathlib import Path
    import json

    # Load from trade patterns or match data
    data_path = Path(__file__).parent.parent.parent.parent / "data" / "trade_patterns.json"

    clusters = []

    if data_path.exists():
        with open(data_path) as f:
            trades = json.load(f)

        map_trades = trades.get(map_name, {})

        # Extract kill locations and cluster them
        # For now, use pre-defined common engagement spots
        common_spots = {
            "ascent": [
                {"x": 0.25, "y": 0.3, "name": "A Main"},
                {"x": 0.75, "y": 0.3, "name": "B Main"},
                {"x": 0.5, "y": 0.5, "name": "Mid"},
            ],
            "bind": [
                {"x": 0.2, "y": 0.35, "name": "A Short"},
                {"x": 0.8, "y": 0.35, "name": "B Long"},
                {"x": 0.5, "y": 0.6, "name": "Hookah"},
            ],
            "haven": [
                {"x": 0.15, "y": 0.3, "name": "A Long"},
                {"x": 0.5, "y": 0.3, "name": "Garage"},
                {"x": 0.85, "y": 0.3, "name": "C Long"},
            ],
        }

        for spot in common_spots.get(map_name, []):
            clusters.append(KillCluster(
                center_x=spot["x"],
                center_y=spot["y"],
                radius=0.08,
                kill_count=15,
                death_count=12,
                dominant_side="attack" if spot["y"] > 0.4 else "defense",
                common_angles=["from site", "from rotate"],
            ))
    else:
        # Demo clusters
        import random
        for i in range(3):
            clusters.append(KillCluster(
                center_x=0.2 + i * 0.3,
                center_y=0.3 + random.random() * 0.2,
                radius=0.05 + random.random() * 0.05,
                kill_count=random.randint(5, 20),
                death_count=random.randint(5, 15),
                dominant_side="attack" if random.random() > 0.5 else "defense",
                common_angles=["angle_1", "angle_2"],
            ))

    return {"map_name": map_name, "clusters": [c.model_dump() for c in clusters]}


@router.get("/economy-timeline/{session_id}")
async def get_economy_timeline(session_id: str):
    """Get economy timeline for a simulation session.

    Returns buy decisions and credit flow over rounds.
    """
    # Would integrate with simulation state
    # Demo data showing typical economy flow
    events = []
    credits = 800  # Pistol round

    for round_num in range(1, 13):
        # Simulate economy
        if round_num == 1:
            buy_type = "pistol"
            credits = 800
        elif credits < 2000:
            buy_type = "save"
        elif credits < 3900:
            buy_type = "force"
        else:
            buy_type = "full_buy"
            credits -= 4500

        events.append(EconomyEvent(
            timestamp_ms=round_num * 100000,  # Approximate round timing
            team="attack",
            buy_type=buy_type,
            total_credits=credits,
            weapons=["vandal", "phantom"] if buy_type == "full_buy" else ["sheriff"],
        ).model_dump())

        # Simulate credit gain
        credits += 1900 if round_num % 2 == 0 else 2400

    return {"session_id": session_id, "events": events}


@router.get("/ability-timeline/{session_id}")
async def get_ability_timeline(session_id: str):
    """Get ability usage timeline for a simulation.

    Returns when and where abilities were used.
    """
    # Would integrate with simulation events
    # Demo data
    events = [
        AbilityEvent(
            timestamp_ms=5000,
            player_id="player_1",
            agent="jett",
            ability="Cloudburst",
            location_x=0.5,
            location_y=0.7,
            result="blocked_sightline",
        ),
        AbilityEvent(
            timestamp_ms=8000,
            player_id="player_2",
            agent="sova",
            ability="Recon Bolt",
            location_x=0.3,
            location_y=0.4,
            result="revealed_2_enemies",
        ),
        AbilityEvent(
            timestamp_ms=12000,
            player_id="player_3",
            agent="killjoy",
            ability="Turret",
            location_x=0.7,
            location_y=0.3,
            result="placed",
        ),
    ]

    return {"session_id": session_id, "events": [e.model_dump() for e in events]}


@router.get("/engagement-stats/{map_name}")
async def get_engagement_stats(
    map_name: str,
    side: Optional[str] = Query(None),
):
    """Get aggregated engagement statistics for a map."""
    return {
        "map_name": map_name,
        "stats": {
            "average_first_blood_time_ms": 18500,
            "average_round_duration_ms": 85000,
            "site_take_rates": {"A": 0.55, "B": 0.45},
            "retake_success_rate": 0.32,
            "trade_rate": 0.68,
            "common_first_blood_locations": [
                {"x": 0.5, "y": 0.5, "frequency": 0.3},
                {"x": 0.25, "y": 0.35, "frequency": 0.25},
            ],
        },
    }
