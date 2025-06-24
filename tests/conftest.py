import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from dam.models import Base # Assuming this is your SQLAlchemy declarative base

@pytest.fixture(scope="session")
def engine():
    """
    Creates an in-memory SQLite engine for the test session.
    The engine is created once per test session.
    """
    return create_engine("sqlite:///:memory:")

@pytest.fixture(scope="function")
def db_session(engine):
    """
    Provides a transactional scope around a test function.
    Creates all tables before the test and drops them afterwards.
    A new session is provided for each test function.
    """
    # Create all tables
    Base.metadata.create_all(engine)

    # Create a session
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session: Session = SessionLocal()

    try:
        yield session
    finally:
        session.close()
        # Drop all tables
        Base.metadata.drop_all(engine)
