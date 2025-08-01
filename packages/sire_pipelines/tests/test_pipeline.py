import pytest
import sire


@pytest.fixture(autouse=True)
def setup_sire_for_testing():
    """Sets up Sire with default pools and registers the torch wrapper."""
    sire.get_resource_management().__init__()
    sire.initialize()


def test_sire_initialization():
    sire.initialize()
