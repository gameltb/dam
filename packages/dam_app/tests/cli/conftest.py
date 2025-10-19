"""Fixtures for CLI tests."""

from pathlib import Path

import pytest
from dam.core.session import DBSession
from dam_test_utils.types import WorldFactory


@pytest.fixture
async def db_session(world_factory: WorldFactory) -> DBSession:
    """A database session with the dam-fs plugin loaded."""
    async with world_factory(["dam-fs"]) as world:
        async with world.get_db_session() as session:
            yield session


@pytest.fixture
def temp_image_file(tmp_path: Path) -> Path:
    """Create a temporary image file."""
    image_file = tmp_path / "test_image.jpg"
    image_file.write_bytes(b"dummy image data")
    return image_file
