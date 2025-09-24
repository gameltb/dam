import logging
from typing import Any, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from dam.models.core.base_class import Base  # Corrected import for Base

# Import WorldConfig for type hinting
from .config import WorldConfig

logger = logging.getLogger(__name__)

__all__ = ["DatabaseManager", "Base"]


class DatabaseManager:
    """
    Manages asynchronous database connections and sessions for a single, specific ECS World.
    An instance of DatabaseManager is typically associated with one World instance.
    """

    def __init__(self, world_config: WorldConfig):
        """
        Initializes the DatabaseManager for a specific world.

        Args:
            world_config: The configuration for the world this manager will handle.
        """
        self.world_config = world_config
        self._engine: Optional[AsyncEngine] = None
        self._session_local: Optional[async_sessionmaker[AsyncSession]] = None
        self._initialize_engine()

    def _initialize_engine(self) -> None:
        """Initializes the async engine and session maker for this world."""
        if not self.world_config.DATABASE_URL:
            raise ValueError(f"DATABASE_URL not set for world '{self.world_config.name}'. Cannot initialize database.")

        # Ensure the DATABASE_URL is compatible with aiosqlite if it's a sqlite URL
        # The project aims to use aiosqlite for async SQLite operations.
        if "sqlite://" in self.world_config.DATABASE_URL and not self.world_config.DATABASE_URL.startswith(
            "sqlite+aiosqlite://"
        ):
            # Automatically adjust sqlite DSNs to use aiosqlite
            # This might be too aggressive if other async sqlite drivers were intended,
            # but for this project, aiosqlite is the standard.
            logger.warning(
                f"Adjusting DATABASE_URL for world '{self.world_config.name}' to use 'sqlite+aiosqlite'. Original: '{self.world_config.DATABASE_URL}'"
            )
            self.world_config.DATABASE_URL = self.world_config.DATABASE_URL.replace("sqlite://", "sqlite+aiosqlite://")

        # connect_args for aiosqlite are generally not needed for basic operation.
        # 'check_same_thread' is not applicable.
        # If specific pragmas or extensions are needed, they can be passed via listeners or engine events.
        connect_args: dict[str, Any] = {}  # Kept empty for now, can be populated if specific needs arise.

        self._engine = create_async_engine(
            self.world_config.DATABASE_URL,
            connect_args=connect_args,  # Pass empty connect_args
            # echo=True # Uncomment for debugging SQL statements
        )
        self._session_local = async_sessionmaker(bind=self._engine, expire_on_commit=False)
        logger.info(
            f"Initialized async database engine for world: '{self.world_config.name}' ({self.world_config.DATABASE_URL})"
        )

    @property
    def engine(self) -> AsyncEngine:
        """Returns the SQLAlchemy AsyncEngine for this world."""
        if self._engine is None:
            raise RuntimeError(f"Async Database engine for world '{self.world_config.name}' has not been initialized.")
        return self._engine

    @property
    def session_local(self) -> async_sessionmaker[AsyncSession]:
        """Returns the AsyncSessionLocal factory for this world."""
        if self._session_local is None:
            raise RuntimeError(f"AsyncSessionLocal for world '{self.world_config.name}' has not been initialized.")
        return self._session_local

    def get_db_session(self) -> AsyncSession:
        """
        Provides a new asynchronous database session for this world.
        The caller is responsible for closing the session, typically using `async with`.
        """
        if self._session_local is None:  # Should ideally be caught by property access if it were None
            raise RuntimeError(
                f"AsyncSessionLocal for world '{self.world_config.name}' has not been initialized and cannot create a session."
            )
        return self._session_local()

    def get_session_maker(self) -> async_sessionmaker[AsyncSession]:
        """Returns the session maker for this world."""
        if self._session_local is None:
            raise RuntimeError(f"AsyncSessionLocal for world '{self.world_config.name}' has not been initialized.")
        return self._session_local

    async def create_db_and_tables(self) -> None:
        """
        Creates all database tables for this world using its async engine.
        """
        logger.info(f"Attempting to create database tables for world '{self.world_config.name}'...")

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
                f"Database tables created (or verified existing) for world '{self.world_config.name}' ({self.world_config.DATABASE_URL})"
            )
        except Exception as e:
            logger.error(f"Error creating tables for world '{self.world_config.name}': {e}", exc_info=True)
            raise

    def get_world_name(self) -> str:
        """Returns the name of the world this manager is associated with."""
        return self.world_config.name
