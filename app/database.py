import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from .config import get_settings

settings = get_settings()

# Only create engine if a real database URL is configured
_db_url = settings.database_url
_use_db = not _db_url.startswith("sqlite") and "localhost" not in _db_url or os.environ.get("DATABASE_URL")

if os.environ.get("DATABASE_URL"):
    _db_url = os.environ["DATABASE_URL"]

try:
    async_engine = create_async_engine(
        _db_url,
        echo=settings.debug,
        pool_size=10,
        max_overflow=20,
    )

    AsyncSessionLocal = async_sessionmaker(
        async_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
except Exception:
    async_engine = None
    AsyncSessionLocal = None

Base = declarative_base()


async def get_db() -> AsyncSession:
    """Dependency for getting async database sessions."""
    if AsyncSessionLocal is None:
        raise RuntimeError("Database not available")
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


async def init_db():
    """Initialize database tables."""
    if async_engine is None:
        return
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
