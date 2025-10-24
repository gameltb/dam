"""Defines the archive plugin."""

from dam.commands.asset_commands import (
    GetAssetFilenamesCommand,
    GetAssetStreamCommand,
)
from dam.commands.discovery_commands import DiscoverPathSiblingsCommand
from dam.core.plugin import Plugin
from dam.core.world import World
from dam.models.metadata import ContentMimeTypeComponent
from dam.traits.asset_operation import AssetOperationTrait
from dam.traits.traits import TraitImplementation

from .commands.ingestion import (
    CheckArchiveCommand,
    ClearArchiveComponentsCommand,
    IngestArchiveCommand,
    ReissueArchiveMemberEventsCommand,
)
from .commands.password import (
    CheckArchivePasswordCommand,
    RemoveArchivePasswordCommand,
    SetArchivePasswordCommand,
)
from .commands.split_archives import (
    BindSplitArchiveCommand,
    CheckSplitArchiveBindingCommand,
    CreateMasterArchiveCommand,
    UnbindSplitArchiveCommand,
)
from .settings import ArchiveSettingsComponent, ArchiveSettingsModel
from .systems.discovery import discover_archive_path_siblings_handler
from .systems.ingestion import (
    check_archive_handler,
    clear_archive_components_handler,
    get_archive_asset_filenames_handler,
    get_archive_asset_stream_handler,
    ingest_archive_members_handler,
    reissue_archive_member_events_handler,
)
from .systems.password import (
    check_archive_password_handler,
    remove_archive_password_handler,
    set_archive_password_handler,
)
from .systems.split_archives import (
    bind_split_archive_handler,
    check_split_archive_binding_handler,
    create_master_archive_handler,
    unbind_split_archive_handler,
)


class ArchivePlugin(Plugin):
    """A plugin for handling archives."""

    Settings = ArchiveSettingsModel
    SettingsComponent = ArchiveSettingsComponent

    def build(self, world: World) -> None:
        """Build the archive plugin."""
        # Command Handlers
        world.register_system(create_master_archive_handler, command_type=CreateMasterArchiveCommand)
        world.register_system(unbind_split_archive_handler, command_type=UnbindSplitArchiveCommand)
        world.register_system(get_archive_asset_stream_handler, command_type=GetAssetStreamCommand)
        world.register_system(set_archive_password_handler, command_type=SetArchivePasswordCommand)
        world.register_system(get_archive_asset_filenames_handler, command_type=GetAssetFilenamesCommand)
        world.register_system(ingest_archive_members_handler, command_type=IngestArchiveCommand)
        world.register_system(clear_archive_components_handler, command_type=ClearArchiveComponentsCommand)
        world.register_system(check_archive_handler, command_type=CheckArchiveCommand)
        world.register_system(check_archive_password_handler, command_type=CheckArchivePasswordCommand)
        world.register_system(remove_archive_password_handler, command_type=RemoveArchivePasswordCommand)
        world.register_system(reissue_archive_member_events_handler, command_type=ReissueArchiveMemberEventsCommand)

        # New binding systems
        world.register_system(bind_split_archive_handler, command_type=BindSplitArchiveCommand)
        world.register_system(check_split_archive_binding_handler, command_type=CheckSplitArchiveBindingCommand)

        # Discovery system
        world.register_system(discover_archive_path_siblings_handler, command_type=DiscoverPathSiblingsCommand)

        # Register Asset Operations
        bind_split_archive_implementation = TraitImplementation(
            trait=AssetOperationTrait,
            handlers={
                AssetOperationTrait.Add: bind_split_archive_handler,
                AssetOperationTrait.Check: check_split_archive_binding_handler,
                AssetOperationTrait.Remove: unbind_split_archive_handler,
            },
            name="archive.bind-split-archive",
            description="Finds and binds all parts of a split archive into a single master entity.",
        )
        world.trait_manager.register(bind_split_archive_implementation, ContentMimeTypeComponent)

        ingest_archive_implementation = TraitImplementation(
            trait=AssetOperationTrait,
            handlers={
                AssetOperationTrait.Add: ingest_archive_members_handler,
                AssetOperationTrait.Check: check_archive_handler,
                AssetOperationTrait.Remove: clear_archive_components_handler,
                AssetOperationTrait.ReprocessDerived: reissue_archive_member_events_handler,
            },
            name="archive.ingest",
            description="Ingests members from an archive file.",
        )
        world.trait_manager.register(ingest_archive_implementation, ContentMimeTypeComponent)

        set_archive_password_implementation = TraitImplementation(
            trait=AssetOperationTrait,
            handlers={
                AssetOperationTrait.Add: set_archive_password_handler,
                AssetOperationTrait.Check: check_archive_password_handler,
                AssetOperationTrait.Remove: remove_archive_password_handler,
            },
            name="archive.set-password",
            description="Sets the password for an archive.",
        )
        world.trait_manager.register(set_archive_password_implementation, ContentMimeTypeComponent)
