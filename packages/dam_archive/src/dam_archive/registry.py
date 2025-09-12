from typing import Dict, Type

from .base import ArchiveHandler
from .handlers.rar import RarArchiveHandler
from .handlers.sevenzip import SevenZipArchiveHandler
from .handlers.zip import ZipArchiveHandler

MIME_TYPE_HANDLERS: Dict[str, Type[ArchiveHandler]] = {
    "application/zip": ZipArchiveHandler,
    "application/vnd.rar": RarArchiveHandler,
    "application/x-7z-compressed": SevenZipArchiveHandler,
}
