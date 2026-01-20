from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Application
    app_name: str = "C9 Tactical Vision"
    debug: bool = True
    api_version: str = "v1"

    # Database
    database_url: str = "postgresql+asyncpg://c9admin:c9tactical2026@localhost:5432/valorant_kb"
    database_sync_url: str = "postgresql://c9admin:c9tactical2026@localhost:5432/valorant_kb"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # GRID API
    grid_api_url: str = "https://api-op.grid.gg/central-data/graphql"
    grid_api_key: str = "XsetiGFZvo03aZRaMRQbDf5eljk0jU6iVBHnNhTl"

    # Simulation settings
    simulation_tick_rate: int = 128  # ms per tick
    max_simulation_time: int = 120000  # 2 minutes max

    # Pattern learning
    min_pattern_samples: int = 5
    pattern_similarity_threshold: float = 0.85

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
