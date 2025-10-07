
import pytest
from dam.core.transaction import WorldTransaction
from pytest_mock import MockerFixture

from dam_archive.commands import ClearArchiveComponentsCommand
from dam_archive.models import ArchiveInfoComponent, ArchiveMemberComponent
from dam_archive.systems import clear_archive_components_handler


@pytest.mark.asyncio
async def test_clear_archive_components_handler(mocker: MockerFixture):
    """
    Tests that the clear_archive_components_handler system correctly removes
    the ArchiveInfoComponent and all associated ArchiveMemberComponents.
    """
    archive_entity_id = 1
    member_entity_ids = [10, 11, 12]
    mock_info_component = ArchiveInfoComponent(comment=None)

    # Create mock member components
    member_components: list[ArchiveMemberComponent] = []
    for eid in member_entity_ids:
        comp = ArchiveMemberComponent(
            archive_entity_id=archive_entity_id, path_in_archive=f"file_{eid}.txt", modified_at=None
        )
        # Manually set entity_id for the test, as it's not part of the constructor
        comp.entity_id = eid
        member_components.append(comp)

    # Mock the transaction and its session
    mock_transaction = mocker.AsyncMock(spec=WorldTransaction)
    mock_session = mocker.AsyncMock()
    mock_result = mocker.MagicMock()
    mock_result.scalars.return_value.all.return_value = member_components
    mock_session.execute.return_value = mock_result
    mock_transaction.session = mock_session
    mock_transaction.get_component.return_value = mock_info_component

    # Create the command
    command = ClearArchiveComponentsCommand(entity_id=archive_entity_id)

    # Execute the system
    await clear_archive_components_handler(command, mock_transaction)

    # Assert that the system tried to get the ArchiveInfoComponent
    mock_transaction.get_component.assert_called_once_with(archive_entity_id, ArchiveInfoComponent)

    # Assert that ArchiveInfoComponent was removed
    mock_transaction.remove_component.assert_any_call(mock_info_component)

    # Assert that the system queried for the correct member components
    mock_session.execute.assert_called_once()

    # Assert that ArchiveMemberComponent was removed for each member entity
    assert mock_transaction.remove_component.call_count == 1 + len(member_entity_ids)
    for member_comp in member_components:
        mock_transaction.remove_component.assert_any_call(member_comp)
