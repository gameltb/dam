"""Database management for DAM worlds."""

import logging
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from dam.models.core.base_class import Base  # Corrected import for Base

# Import WorldConfig for type hinting
from .config import WorldConfig

logger = logging.getLogger(__name__)

__all__ = ["Base", "DatabaseManager"]


class DatabaseManager:
    """
    Manages asynchronous database connections and sessions for a single, specific ECS World.

    An instance of DatabaseManager is typically associated with one World instance.
    """

    def __init__(self, world_config: WorldConfig):
        """
        Initialize the DatabaseManager for a specific world.

        Args:
            world_config: The configuration for the world this manager will handle.

        """
        self.world_config = world_config
        self._engine: AsyncEngine | None = None
        self._session_local: async_sessionmaker[AsyncSession] | None = None
        self._initialize_engine()

    def _initialize_engine(self) -> None:
        """Initialize the async engine and session maker for this world."""
        if not self.world_config.DATABASE_URL:
            raise ValueError(f"DATABASE_URL not set for world '{self.world_config.name}'. Cannot initialize database.")

        # Ensure the DATABASE_URL is compatible with aiosqlite if it's a sqlite URL
        # The project aims to use aiosqlite for async SQLite operations.
        self._engine = create_async_engine(
            self.world_config.DATABASE_URL,
            # echo=True # Uncomment for debugging SQL statements
        )
        self._session_local = async_sessionmaker(bind=self._engine, expire_on_commit=False)
        logger.info(
            "Initialized async database engine for world: '%s' (%s)",
            self.world_config.name,
            self.world_config.DATABASE_URL,
        )

    @property
    def engine(self) -> AsyncEngine:
        """Return the SQLAlchemy AsyncEngine for this world."""
        if self._engine is None:
            raise RuntimeError(f"Async Database engine for world '{self.world_config.name}' has not been initialized.")
        return self._engine

    @property
    def session_local(self) -> async_sessionmaker[AsyncSession]:
        """Return the AsyncSessionLocal factory for this world."""
        if self._session_local is None:
            raise RuntimeError(f"AsyncSessionLocal for world '{self.world_config.name}' has not been initialized.")
        return self._session_local

    def get_db_session(self) -> AsyncSession:
        """
        Provide a new asynchronous database session for this world.

        The caller is responsible for closing the session, typically using `async with`.
        """
        if self._session_local is None:  # Should ideally be caught by property access if it were None
            raise RuntimeError(
                f"AsyncSessionLocal for world '{self.world_config.name}' has not been initialized and cannot create a session."
            )
        return self._session_local()

    async def create_db_and_tables(self) -> None:
        """Create all database tables for this world using its async engine."""
        logger.info("Attempting to create database tables for world '%s'...", self.world_config.name)

        if self._engine is None:  # Guard against uninitialized engine
            raise RuntimeError(
                f"Async engine not initialized for world '{self.world_config.name}'. Cannot create tables."
            )

        try:
            async with self._engine.begin() as conn:
                if "sqlite" not in self.world_config.DATABASE_URL:
                    await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                await conn.run_sync(Base.metadata.create_all)
            logger.info(
                "Database tables created (or verified existing) for world '%s' (%s)",
                self.world_config.name,
                self.world_config.DATABASE_URL,
            )
        except Exception:
            logger.exception("Error creating tables for world '%s'", self.world_config.name)
            raise

    def get_world_name(self) -> str:
        """Return the name of the world this manager is associated with."""
        return self.world_config.name
