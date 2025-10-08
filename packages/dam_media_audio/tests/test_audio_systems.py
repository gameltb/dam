"""Tests for the audio systems."""

from pathlib import Path

import pytest
from dam.core.world import World


@pytest.mark.serial
@pytest.mark.asyncio
async def test_add_audio_components_system(test_world_alpha: World, sample_wav_file: Path) -> None:
    """Test the add_audio_components_system."""

    # TODO
