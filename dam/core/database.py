from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from dam.models import (
    Base,
)  # To ensure Base has all models registered via their imports in dam.models.__init__

from .config import settings

# Create the SQLAlchemy engine
# For SQLite, connect_args={"check_same_thread": False} is needed for
# FastAPI/multi-threaded use, but generally good practice for SQLite usage.
connect_args = {}
if settings.DATABASE_URL.startswith("sqlite"):
    connect_args["check_same_thread"] = False

engine = create_engine(
    settings.DATABASE_URL,
    connect_args=connect_args,
    # echo=True # Optional: for debugging SQL statements
)

# Create a SessionLocal class to generate database sessions
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def create_db_and_tables():
    """
    Creates all database tables defined in Base.metadata.
    This is typically called once at application startup if tables don't exist,
    or managed by Alembic migrations.
    For CLI tools, it can be called explicitly by a setup command or checked on startup.
    """
    # This will create tables for all imported models that inherit from Base
    Base.metadata.create_all(bind=engine)
    print(f"Database tables created (if they didn't exist) for {settings.DATABASE_URL}")


def get_db_session():
    """
    Dependency provider for database sessions.
    Ensures the session is closed after use.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Example usage for CLI commands:
# from dam.core.database import SessionLocal
#
# def my_command():
#     db = SessionLocal()
#     try:
#         # ... use db session ...
#         db.commit() # If changes were made
#     except Exception:
#         db.rollback()
#         raise
#     finally:
#         db.close()
