import asyncio
from dataclasses import dataclass, field

from dam.core.commands import BaseCommand
from dam.models.core.entity import Entity


@dataclass
class ExtractAudioMetadataCommand(BaseCommand):
    entity: Entity
    result_future: asyncio.Future = field(default_factory=asyncio.Future)
