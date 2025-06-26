from typing import Dict, Optional

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from dam.models import Base

# Import Settings for type hinting, and the global settings instance
from .config import Settings
from .config import settings as global_app_settings


class DatabaseManager:
    def __init__(self, settings_object: Settings):  # Accept a settings object
        self.settings = settings_object  # Store it
        self._engines: Dict[str, Engine] = {}
        self._session_locals: Dict[str, sessionmaker[Session]] = {}
        self._initialize_engines()

    def _initialize_engines(self):
        """Initializes engines and session makers for all configured worlds."""
        # Use self.settings instead of the global settings
        if not self.settings.worlds:
            raise RuntimeError(
                "No worlds configured. Please check your DAM_WORLDS_CONFIG environment variable or .env file."
            )

        for world_name, world_config in self.settings.worlds.items():
            connect_args = {}
            if world_config.DATABASE_URL.startswith("sqlite"):
                connect_args["check_same_thread"] = False

            engine = create_engine(
                world_config.DATABASE_URL,
                connect_args=connect_args,
                # echo=True # Optional: for debugging SQL statements
            )
            self._engines[world_name] = engine
            current_session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)
            self._session_locals[world_name] = current_session_local
            print(f"Initialized database engine for world: '{world_name}' ({world_config.DATABASE_URL})")

    def get_engine(self, world_name: Optional[str] = None) -> Engine:
        """Returns the SQLAlchemy engine for the specified world."""
        target_world_name = world_name or self.settings.DEFAULT_WORLD_NAME
        if not target_world_name or target_world_name not in self._engines:
            raise ValueError(f"Engine for world '{target_world_name}' not found or default world not set.")
        return self._engines[target_world_name]

    def get_session_local(self, world_name: Optional[str] = None) -> sessionmaker[Session]:
        """Returns the SessionLocal factory for the specified world."""
        target_world_name = world_name or self.settings.DEFAULT_WORLD_NAME
        if not target_world_name or target_world_name not in self._session_locals:
            raise ValueError(f"SessionLocal for world '{target_world_name}' not found or default world not set.")
        return self._session_locals[target_world_name]

    def get_db_session(self, world_name: Optional[str] = None) -> Session:
        """
        Provides a database session for the specified world.
        The caller is responsible for closing the session.
        """
        session_local = self.get_session_local(world_name)
        return session_local()

    def create_db_and_tables(self, world_name: Optional[str] = None):
        """
        Creates all database tables for the specified world.
        """
        target_world_name = world_name or self.settings.DEFAULT_WORLD_NAME
        if not target_world_name:
            raise ValueError("Cannot create tables: No world specified and no default world configured.")

        engine = self.get_engine(target_world_name)
        # Use self.settings here
        world_config = self.settings.get_world_config(target_world_name)

        # WARNING: Destructive operation in testing mode.
        # Use self.settings here
        if self.settings.TESTING_MODE and (
            "pytest" in world_config.DATABASE_URL or "test" in world_config.DATABASE_URL
        ):
            Base.metadata.drop_all(bind=engine)
            print(f"Dropped all tables for world '{target_world_name}' ({world_config.DATABASE_URL}) (testing mode)")

        Base.metadata.create_all(bind=engine)
        print(
            f"Database tables created for world '{target_world_name}' ({world_config.DATABASE_URL})"
            "(if they didn't exist or were dropped)"
        )

    def get_all_world_names(self) -> list[str]:
        """Returns a list of all configured world names."""
        return list(self._engines.keys())


# Global instance of DatabaseManager
db_manager = DatabaseManager(settings_object=global_app_settings)


# Convenience functions (optional, could also use db_manager directly)
def get_db_session(world_name: Optional[str] = None) -> Session:
    """
    Dependency provider style function for database sessions for a specific world.
    Ensures the session is closed after use when used with `yield`.
    If not using `yield`, caller must close.
    """
    # This version is more for direct call, not for `yield` in FastAPI style.
    # For CLI, direct call and manual close is fine.
    return db_manager.get_db_session(world_name)


# Example usage for CLI commands:
# from dam.core.database import db_manager
#
# def my_command_for_world(world_name: str):
#     db = db_manager.get_db_session(world_name)
#     try:
#         # ... use db session ...
#         db.commit() # If changes were made
#     except Exception:
#         db.rollback()
#         raise
#     finally:
#         db.close()
#
# def my_command_for_default_world():
#     db = db_manager.get_db_session() # Uses default world
#     # ...
#     db.close()
