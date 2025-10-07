"""Systems for the archive plugin."""
# Main entry point for systems in the dam_archive package.

from .discovery import discover_archive_path_siblings_handler
from .ingestion import (
    check_archive_handler,
    clear_archive_components_handler,
    get_archive_asset_filenames_handler,
    get_archive_asset_stream_handler,
    ingest_archive_members_handler,
)
from .password import (
    check_archive_password_handler,
    remove_archive_password_handler,
    set_archive_password_handler,
)
from .split_archives import (
    bind_split_archive_handler,
    check_split_archive_binding_handler,
    create_master_archive_handler,
    unbind_split_archive_handler,
)

__all__ = [
    "bind_split_archive_handler",
    "check_archive_handler",
    "check_archive_password_handler",
    "check_split_archive_binding_handler",
    "clear_archive_components_handler",
    "create_master_archive_handler",
    "discover_archive_path_siblings_handler",
    "get_archive_asset_filenames_handler",
    "get_archive_asset_stream_handler",
    "ingest_archive_members_handler",
    "remove_archive_password_handler",
    "set_archive_password_handler",
    "unbind_split_archive_handler",
]
