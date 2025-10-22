"""Unit tests for the TraitIdentifier."""

import pytest

from dam.traits.identifier import TraitIdentifier


def test_trait_identifier_from_string():
    """Test that a TraitIdentifier can be created from a string."""
    identifier = TraitIdentifier.from_string("asset.content.readable")
    assert identifier.parts == ("asset", "content", "readable")


def test_trait_identifier_to_string():
    """Test that a TraitIdentifier can be converted to a string."""
    identifier = TraitIdentifier(parts=("asset", "content", "readable"))
    assert str(identifier) == "asset.content.readable"


def test_trait_identifier_validation():
    """Test that a TraitIdentifier validates its parts."""
    with pytest.raises(ValueError, match="Invalid part"):
        TraitIdentifier(parts=("Invalid-Part",))
    with pytest.raises(ValueError, match="cannot be empty"):
        TraitIdentifier.from_string("")
    with pytest.raises(ValueError, match="cannot be empty"):
        TraitIdentifier(parts=())
