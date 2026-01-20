"""C9 Tactical Vision - FastAPI Backend.

VALORANT tactical simulation and analysis platform.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import redis.asyncio as redis

from .config import get_settings
from .api import api_router
from .database import async_engine

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown."""
    # Startup
    print("Starting C9 Tactical Vision API...")

    # Initialize Redis connection
    app.state.redis = redis.from_url(
        settings.redis_url,
        encoding="utf-8",
        decode_responses=True
    )

    # Test database connection
    try:
        async with async_engine.connect() as conn:
            await conn.execute("SELECT 1")
        print("Database connection successful")
    except Exception as e:
        print(f"Database connection failed: {e}")

    yield

    # Shutdown
    print("Shutting down C9 Tactical Vision API...")
    await app.state.redis.close()
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

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
    ],
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
        port=8000,
        reload=True,
    )
