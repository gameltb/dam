"""Asset Content Readable Trait for the DAM system."""

from dataclasses import dataclass

from dam.commands.core import EntityCommand
from dam.core.types import StreamProvider
from dam.system_events.base import BaseSystemEvent
from dam.traits.traits import Trait


class AssetContentReadable(Trait):
    """A trait for components that represent asset content that can be read as a stream of bytes."""

    name = "asset.content.readable"
    description = "Provides a way to read the raw content of an asset."

    @dataclass
    class GetStream(EntityCommand[StreamProvider, BaseSystemEvent]):
        """Abstract command to get the content as a stream."""

    @dataclass
    class GetSize(EntityCommand[int, BaseSystemEvent]):
        """Abstract command to get the content size in bytes."""
