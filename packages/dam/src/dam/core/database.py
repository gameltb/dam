"""Database management for DAM worlds."""

import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from dam.models.core.base_class import Base  # Corrected import for Base

logger = logging.getLogger(__name__)

__all__ = ["Base", "DatabaseManager"]


class DatabaseManager:
    """
    Manages asynchronous database connections and sessions for a single, specific ECS World.

    An instance of DatabaseManager is typically associated with one World instance.
    """

    def __init__(self, database_url: str):
        """
        Initialize the DatabaseManager for a specific world.

        Args:
            database_url: The database connection URL for the world this manager will handle.

        """
        self.database_url = database_url
        self._engine: AsyncEngine | None = None
        self._session_local: async_sessionmaker[AsyncSession] | None = None
        self._initialize_engine()

    def _initialize_engine(self) -> None:
        """Initialize the async engine and session maker for this world."""
        if not self.database_url:
            raise ValueError("DATABASE_URL not set. Cannot initialize database.")

        # Ensure the DATABASE_URL is compatible with aiosqlite if it's a sqlite URL
        # The project aims to use aiosqlite for async SQLite operations.
        self._engine = create_async_engine(
            self.database_url,
            # echo=True # Uncomment for debugging SQL statements
        )
        self._session_local = async_sessionmaker(bind=self._engine, expire_on_commit=False)
        logger.info(
            "Initialized async database engine for (%s)",
            self.database_url,
        )

    @property
    def engine(self) -> AsyncEngine:
        """Return the SQLAlchemy AsyncEngine for this world."""
        if self._engine is None:
            raise RuntimeError("Async Database engine has not been initialized.")
        return self._engine

    @property
    def session_local(self) -> async_sessionmaker[AsyncSession]:
        """Return the AsyncSessionLocal factory for this world."""
        if self._session_local is None:
            raise RuntimeError("AsyncSessionLocal has not been initialized.")
        return self._session_local

    def get_db_session(self) -> AsyncSession:
        """
        Provide a new asynchronous database session for this world.

        The caller is responsible for closing the session, typically using `async with`.
        """
        if self._session_local is None:  # Should ideally be caught by property access if it were None
            raise RuntimeError("AsyncSessionLocal has not been initialized and cannot create a session.")
        return self._session_local()

    async def create_db_and_tables(self) -> None:
        """Create all database tables for this world using its async engine."""
        logger.info("Attempting to create database tables...")

        if self._engine is None:  # Guard against uninitialized engine
            raise RuntimeError("Async engine not initialized. Cannot create tables.")

        try:
            async with self._engine.begin() as conn:
                if "sqlite" not in self.database_url:
                    await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                await conn.run_sync(Base.metadata.create_all)
            logger.info(
                "Database tables created (or verified existing) for (%s)",
                self.database_url,
            )
        except Exception:
            logger.exception("Error creating tables.")
            raise
