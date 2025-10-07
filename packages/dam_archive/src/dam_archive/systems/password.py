"""Systems for handling archive passwords."""

import logging

from dam.core.systems import system
from dam.core.transaction import WorldTransaction

from ..commands.password import (
    CheckArchivePasswordCommand,
    RemoveArchivePasswordCommand,
    SetArchivePasswordCommand,
)
from ..models import ArchivePasswordComponent

logger = logging.getLogger(__name__)


@system(on_command=CheckArchivePasswordCommand)
async def check_archive_password_handler(
    cmd: CheckArchivePasswordCommand,
    transaction: WorldTransaction,
) -> bool:
    """Check if the ArchivePasswordComponent exists for the entity."""
    component = await transaction.get_component(cmd.entity_id, ArchivePasswordComponent)
    return component is not None


@system(on_command=RemoveArchivePasswordCommand)
async def remove_archive_password_handler(
    cmd: RemoveArchivePasswordCommand,
    transaction: WorldTransaction,
):
    """Remove the ArchivePasswordComponent from the entity."""
    component = await transaction.get_component(cmd.entity_id, ArchivePasswordComponent)
    if component:
        await transaction.remove_component(component)
        logger.info("Removed ArchivePasswordComponent from entity %s", cmd.entity_id)


@system(on_command=SetArchivePasswordCommand)
async def set_archive_password_handler(
    cmd: SetArchivePasswordCommand,
    transaction: WorldTransaction,
) -> None:
    """Handle setting the password for an archive."""
    password_comp = await transaction.get_component(cmd.entity_id, ArchivePasswordComponent)
    if password_comp:
        password_comp.password = cmd.password
    else:
        password_comp = ArchivePasswordComponent(password=cmd.password)
        await transaction.add_component_to_entity(cmd.entity_id, password_comp)
