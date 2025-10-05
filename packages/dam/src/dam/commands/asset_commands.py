from dataclasses import dataclass
from typing import Any, BinaryIO, Optional, Tuple

from ..core.types import StreamProvider
from ..models.core.entity import Entity
from ..system_events.base import BaseSystemEvent
from .core import BaseCommand, EntityCommand


@dataclass
class GetOrCreateEntityFromStreamCommand(BaseCommand[Tuple[Entity, bytes], BaseSystemEvent]):
    """
    A command to get or create an entity from a stream.
    Returns a tuple of the entity and the calculated sha256 hash.
    """

    stream: BinaryIO


@dataclass
class GetAssetStreamCommand(BaseCommand[Optional[StreamProvider], BaseSystemEvent]):
    """
    A command to get a provider for a binary stream for an asset's content.

    The command's result is a `StreamProvider`, which is a callable that returns a
    fresh, readable `BinaryIO` stream positioned at the beginning. It is the
    caller's responsibility to call the provider and properly close the resulting
    stream, preferably using a `with` statement.
    """

    entity_id: int


@dataclass
class SetMimeTypeCommand(EntityCommand[None, BaseSystemEvent]):
    """
    A command to set the mime type for an asset.
    """

    mime_type: str


@dataclass
class CheckContentMimeTypeCommand(EntityCommand[bool, BaseSystemEvent]):
    """
    A command to check if an entity has a content mime type.
    """

    pass


@dataclass
class RemoveContentMimeTypeCommand(EntityCommand[None, BaseSystemEvent]):
    """
    A command to remove the content mime type from an entity.
    """

    pass


@dataclass
class SetMimeTypeFromBufferCommand(BaseCommand[None, BaseSystemEvent]):
    """
    A command to set the mime type for an asset from a buffer.
    """

    entity_id: int
    buffer: bytes


@dataclass
class GetMimeTypeCommand(EntityCommand[str | None, BaseSystemEvent]):
    """
    A command to get the mime type for an asset.
    """

    pass


@dataclass
class GetAssetMetadataCommand(EntityCommand[dict[str, Any], BaseSystemEvent]):
    """
    A command to get the metadata for an asset.
    """

    pass


@dataclass
class UpdateAssetMetadataCommand(EntityCommand[None, BaseSystemEvent]):
    """
    A command to update the metadata for an asset.
    """

    metadata: dict[str, Any]


@dataclass
class GetAssetFilenamesCommand(EntityCommand[list[str], BaseSystemEvent]):
    """
    A command to get all available filenames for an asset.
    Handlers for this command should return a list of strings,
    each representing a known filename for the asset.
    """

    pass
