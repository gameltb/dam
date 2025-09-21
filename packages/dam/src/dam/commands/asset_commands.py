from dataclasses import dataclass
from typing import Any, BinaryIO, Optional, Tuple

from dam.commands.core import BaseCommand
from dam.core.types import StreamProvider
from dam.models.core.entity import Entity
from dam.system_events import BaseSystemEvent


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
class SetMimeTypeCommand(BaseCommand[None, BaseSystemEvent]):
    """
    A command to set the mime type for an asset.
    """

    entity_id: int
    mime_type: str


@dataclass
class SetMimeTypeFromBufferCommand(BaseCommand[None, BaseSystemEvent]):
    """
    A command to set the mime type for an asset from a buffer.
    """

    entity_id: int
    buffer: bytes


@dataclass
class GetMimeTypeCommand(BaseCommand[str | None, BaseSystemEvent]):
    """
    A command to get the mime type for an asset.
    """

    entity_id: int


@dataclass
class GetAssetMetadataCommand(BaseCommand[dict[str, Any], BaseSystemEvent]):
    """
    A command to get the metadata for an asset.
    """

    entity_id: int


@dataclass
class UpdateAssetMetadataCommand(BaseCommand[None, BaseSystemEvent]):
    """
    A command to update the metadata for an asset.
    """

    entity_id: int
    metadata: dict[str, Any]


@dataclass
class GetAssetFilenamesCommand(BaseCommand[list[str], BaseSystemEvent]):
    """
    A command to get all available filenames for an asset.
    Handlers for this command should return a list of strings,
    each representing a known filename for the asset.
    """

    entity_id: int
