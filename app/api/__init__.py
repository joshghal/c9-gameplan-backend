from fastapi import APIRouter
from .routes import teams, matches, patterns, simulations, maps

api_router = APIRouter()

api_router.include_router(teams.router, prefix="/teams", tags=["teams"])
api_router.include_router(matches.router, prefix="/matches", tags=["matches"])
api_router.include_router(patterns.router, prefix="/patterns", tags=["patterns"])
api_router.include_router(simulations.router, prefix="/simulations", tags=["simulations"])
api_router.include_router(maps.router, prefix="/maps", tags=["maps"])
