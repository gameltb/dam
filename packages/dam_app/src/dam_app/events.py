from dataclasses import dataclass
from typing import List
from dam.core.events import BaseEvent


@dataclass
class AssetsReadyForMetadataExtraction(BaseEvent):
    """
    An event that is triggered when a batch of assets is ready for metadata extraction.
    """

    entity_ids: List[int]
