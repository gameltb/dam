from typing import Dict, List, Type

from .base import ArchiveHandler
from .handlers.rar import RarArchiveHandler
from .handlers.sevenzip import SevenZipArchiveHandler
from .handlers.sevenzip_cli import SevenZipCliArchiveHandler
from .handlers.zip import ZipArchiveHandler

MIME_TYPE_HANDLERS: Dict[str, List[Type[ArchiveHandler]]] = {
    "application/zip": [ZipArchiveHandler],
    "application/vnd.rar": [RarArchiveHandler],
    "application/x-7z-compressed": [
        SevenZipArchiveHandler,
        SevenZipCliArchiveHandler,
    ],
}
