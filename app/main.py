"""C9 Tactical Vision - FastAPI Backend.

VALORANT tactical simulation and analysis platform.
"""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .api import api_router

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown."""
    # Startup
    print("Starting C9 Tactical Vision API...")

    # Initialize Redis connection (optional - skip if unavailable)
    try:
        import redis.asyncio as redis
        app.state.redis = redis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True
        )
        await app.state.redis.ping()
        print("Redis connection successful")
    except Exception as e:
        print(f"Redis unavailable, running without cache: {e}")
        app.state.redis = None

    # Test database connection (optional - skip if unavailable)
    try:
        from .database import async_engine
        from sqlalchemy import text
        async with async_engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        print("Database connection successful")
        app.state.db_available = True
    except Exception as e:
        print(f"Database unavailable, running without DB: {e}")
        app.state.db_available = False

    yield

    # Shutdown
    print("Shutting down C9 Tactical Vision API...")
    if app.state.redis:
        await app.state.redis.close()
    if getattr(app.state, 'db_available', False):
        from .database import async_engine
        await async_engine.dispose()


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    description="""
    C9 Tactical Vision API - VALORANT tactical simulation platform.

    Features:
    - Movement pattern analysis and prediction
    - A* pathfinding with visibility calculations
    - Real-time simulation of tactical scenarios
    - What-if scenario analysis
    - Team strategy insights
    """,
    version="1.0.0",
    lifespan=lifespan,
)

# CORS configuration - allow deployed frontend and localhost
cors_origins = [
    "http://localhost:3000",
    "http://localhost:3001",
    "http://127.0.0.1:3000",
]

# Add deployed frontend URL from env var
frontend_url = os.environ.get("FRONTEND_URL")
if frontend_url:
    cors_origins.append(frontend_url)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix=f"/api/{settings.api_version}")


@app.get("/")
async def root():
    """Root endpoint - API info."""
    return {
        "name": settings.app_name,
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


# For running with uvicorn directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        reload=True,
    )
