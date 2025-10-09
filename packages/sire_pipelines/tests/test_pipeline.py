"""Tests for the Sire pipeline."""

import pytest
import sire


@pytest.fixture(autouse=True)
def setup_sire_for_testing():
    """Set up Sire with default pools and register the torch wrapper."""
    sire.get_resource_management().__init__()
    sire.initialize()


def test_sire_initialization():
    """Test that Sire can be initialized."""
    sire.initialize()
