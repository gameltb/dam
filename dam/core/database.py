import logging
from typing import Optional

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from dam.models import Base

# Import WorldConfig for type hinting
from .config import WorldConfig  # Keep global_app_settings for TESTING_MODE reference

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Manages database connections and sessions for a single, specific ECS World.
    An instance of DatabaseManager is typically associated with one World instance.
    """

    def __init__(self, world_config: WorldConfig, testing_mode: bool):
        """
        Initializes the DatabaseManager for a specific world.

        Args:
            world_config: The configuration for the world this manager will handle.
            testing_mode: Global testing mode flag, affects table dropping.
        """
        self.world_config = world_config
        self.testing_mode = testing_mode  # Store global testing_mode
        self._engine: Optional[Engine] = None
        self._session_local: Optional[sessionmaker[Session]] = None
        self._initialize_engine()

    def _initialize_engine(self):
        """Initializes the engine and session maker for this world."""
        if not self.world_config.DATABASE_URL:
            # This should ideally be caught by WorldConfig validation if DATABASE_URL is mandatory
            raise ValueError(f"DATABASE_URL not set for world '{self.world_config.name}'. Cannot initialize database.")

        connect_args = {}
        if self.world_config.DATABASE_URL.startswith("sqlite"):
            connect_args["check_same_thread"] = False

        self._engine = create_engine(
            self.world_config.DATABASE_URL,
            connect_args=connect_args,
            # echo=True # Optional: for debugging SQL statements
        )
        self._session_local = sessionmaker(autocommit=False, autoflush=False, bind=self._engine)
        logger.info(
            f"Initialized database engine for world: '{self.world_config.name}' ({self.world_config.DATABASE_URL})"
        )

    @property
    def engine(self) -> Engine:
        """Returns the SQLAlchemy engine for this world."""
        if self._engine is None:
            # This should not happen if _initialize_engine is called in __init__
            raise RuntimeError(f"Database engine for world '{self.world_config.name}' has not been initialized.")
        return self._engine

    @property
    def session_local(self) -> sessionmaker[Session]:
        """Returns the SessionLocal factory for this world."""
        if self._session_local is None:
            # Should not happen
            raise RuntimeError(f"SessionLocal for world '{self.world_config.name}' has not been initialized.")
        return self._session_local

    def get_db_session(self) -> Session:
        """
        Provides a new database session for this world.
        The caller is responsible for closing the session.
        e.g., using a try/finally block or as a context manager if Session supports it.
        """
        return self.session_local()

    def create_db_and_tables(self):
        """
        Creates all database tables for this world using its engine.
        In TESTING_MODE, if the database URL suggests a test database,
        it will drop all tables before creating them.
        """
        logger.info(f"Attempting to create database tables for world '{self.world_config.name}'...")

        # WARNING: Destructive operation in testing mode.
        # Uses the testing_mode flag passed during __init__
        if self.testing_mode and (
            "pytest" in self.world_config.DATABASE_URL or "test" in self.world_config.DATABASE_URL
        ):
            try:
                Base.metadata.drop_all(bind=self.engine)
                logger.info(
                    f"Dropped all tables for world '{self.world_config.name}' ({self.world_config.DATABASE_URL}) (testing mode)"
                )
            except Exception as e:
                logger.error(
                    f"Error dropping tables for world '{self.world_config.name}' in testing mode: {e}", exc_info=True
                )
                # Depending on severity, might re-raise or just warn.
                # For now, log and continue to create_all.

        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info(
                f"Database tables created (or verified existing) for world '{self.world_config.name}' ({self.world_config.DATABASE_URL})"
            )
        except Exception as e:
            logger.error(f"Error creating tables for world '{self.world_config.name}': {e}", exc_info=True)
            raise  # Re-raise after logging, as this is a critical failure.

    def get_world_name(self) -> str:
        """Returns the name of the world this manager is associated with."""
        return self.world_config.name


# The global db_manager instance is removed.
# Each World object will create its own DatabaseManager instance.

# Convenience functions like the old global get_db_session are removed.
# Sessions should be obtained from a DatabaseManager instance,
# which itself is obtained from a World instance.

# Example usage (conceptual, would be part of a World class or CLI command logic):
#
# from dam.core.config import settings # The global app settings
#
# def some_operation_on_world(world_name: str):
#     try:
#         world_cfg = settings.get_world_config(world_name)
#         # Pass global_app_settings.TESTING_MODE to the db_manager
#         db_mgr_for_world = DatabaseManager(world_config=world_cfg, testing_mode=global_app_settings.TESTING_MODE)
#         db_session = db_mgr_for_world.get_db_session()
#         try:
#             # ... use db_session ...
#             # e.g., result = db_session.query(MyModel).all()
#             db_session.commit() # If changes were made
#         except Exception:
#             db_session.rollback()
#             raise
#         finally:
#             db_session.close()
#     except ValueError as e:
#         print(f"Error with world '{world_name}': {e}")
#
# some_operation_on_world(settings.DEFAULT_WORLD_NAME)
# some_operation_on_world("another_configured_world")
