from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from dam.core.commands import BaseCommand


@dataclass
class IngestPspIsosCommand(BaseCommand[None]):
    """Command to ingest PSP ISOs from a directory."""

    directory: Path
    passwords: Optional[List[str]] = None
