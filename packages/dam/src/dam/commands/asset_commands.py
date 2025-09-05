from dataclasses import dataclass
from typing import Any, BinaryIO

from dam.core.commands import BaseCommand


@dataclass
class GetAssetStreamCommand(BaseCommand[BinaryIO]):
    """
    A command to get a readable and seekable binary stream for an asset.

    The returned stream is a file-like object opened in binary mode (`typing.BinaryIO`).
    It is the responsibility of the command handler to provide a stream that can be
    read from and, if possible, seeked. The caller is responsible for closing the
    stream after use.
    """

    entity_id: int


@dataclass
class GetAssetMetadataCommand(BaseCommand[dict[str, Any]]):
    """
    A command to get the metadata for an asset.
    """

    entity_id: int


@dataclass
class UpdateAssetMetadataCommand(BaseCommand[None]):
    """
    A command to update the metadata for an asset.
    """

    entity_id: int
    metadata: dict[str, Any]


@dataclass
class GetAssetFilenamesCommand(BaseCommand[list[str]]):
    """
    A command to get all available filenames for an asset.
    Handlers for this command should return a list of strings,
    each representing a known filename for the asset.
    """

    entity_id: int
