import pytest
from PyQt6.QtWidgets import QApplication
from pytestqt.qt_compat import qt_api
from pytestqt.qtbot import QtBot

from dam.core.world import World
from dam.models.metadata.exiftool_metadata_component import ExiftoolMetadataComponent # Corrected case
from dam.ui.dialogs.component_viewerd_ialog import ComponentViewerDialog

# pytest-qt automatically provides a qapp fixture or handles QApplication instance.
# Removing custom qapp fixture to avoid conflicts.

@pytest.fixture
def mock_world(mocker):
    world = mocker.MagicMock(spec=World)
    world.name = "test_ui_world"

    # Mock the session and component fetching
    mock_session = mocker.MagicMock()
    world.get_db_session.return_value.__enter__.return_value = mock_session

    # Store registered components for ecs_service mocking if needed elsewhere
    # from dam.models.core.base_component import REGISTERED_COMPONENT_TYPES
    # world.REGISTERED_COMPONENT_TYPES = REGISTERED_COMPONENT_TYPES
    # For this test, direct mocking of get_components is simpler
    return world

def test_exif_metadata_display(qtbot: QtBot, mock_world, mocker):
    """
    Test that EXIF metadata is displayed in the ComponentViewerDialog.
    """
    entity_id = 1
    exif_data = {
        "Make": "CameraCorp",
        "Model": "DSLR-1000",
        "DateTimeOriginal": "2023:01:01 10:00:00"
    }

    # The ComponentViewerDialog receives a dictionary representation of components.
    # We don't need to instantiate the actual SQLAlchemy component for this test,
    # as we are testing the dialog's rendering of pre-formatted data.
    # mock_exif_component = ExiftoolMetadataComponent(entity_id=entity_id, raw_exif_json=exif_data) # This line is not needed

    # Simulate components data that ComponentViewerDialog expects
    # It's Dict[str, List[Dict[str, Any]]]
    # The dicts in the list are from component.__table__.columns
    # For simplicity, we'll mock the direct data structure the dialog receives

    components_data_for_dialog = {
        "ExiftoolMetadataComponent": [ # Corrected class name key
            {
                "entity_id": entity_id,
                "raw_exif_json": exif_data  # Corrected key to match component attribute
            }
        ]
    }

    dialog = ComponentViewerDialog(
        entity_id=entity_id,
        components_data=components_data_for_dialog,
        world_name=mock_world.name
    )
    qtbot.addWidget(dialog) # Register dialog with qtbot for cleanup

    # The dialog formats components data as JSON string in a QTextEdit
    # We need to check if the key EXIF fields are present in the text.
    dialog_text = dialog.text_edit.toPlainText()

    assert "ExiftoolMetadataComponent" in dialog_text # Corrected class name
    assert "CameraCorp" in dialog_text
    assert "DSLR-1000" in dialog_text
    assert "2023:01:01 10:00:00" in dialog_text

    # Test that the dialog can be created and doesn't crash without data
    empty_dialog = ComponentViewerDialog(entity_id=2, components_data={}, world_name="empty_world")
    qtbot.addWidget(empty_dialog)
    assert "No components found" in empty_dialog.text_edit.toPlainText()

# Placeholder for Transcoding Dialog Test
def test_transcoding_dialog_trigger(qtbot: QtBot, mock_world):
    """
    Placeholder test for triggering transcoding.
    This will require the TranscodeAssetDialog to be implemented first.
    """
    # TODO: Implement test after TranscodeAssetDialog is created
    # 1. Mock necessary services (TranscodeService, ProfileService)
    # 2. Create MainWindow or a way to trigger the dialog
    # 3. Instantiate TranscodeAssetDialog
    # 4. Simulate user input (selecting profile)
    # 5. Simulate 'Start Transcode' click
    # 6. Verify that the correct service call or event dispatch occurs
    pass

# Placeholder for Transcoding Evaluation Dialog Test
def test_transcoding_evaluation_dialog_trigger(qtbot: QtBot, mock_world):
    """
    Placeholder test for triggering transcoding evaluation.
    This will require EvaluationSetupDialog and EvaluationResultDialog.
    """
    # TODO: Implement test after evaluation dialogs are created
    # 1. Mock evaluation services/systems
    # 2. Instantiate EvaluationSetupDialog
    # 3. Simulate input (selecting assets, parameters)
    # 4. Simulate 'Start Evaluation' click
    # 5. Verify interaction with evaluation service
    # 6. (Later) Instantiate EvaluationResultDialog with mock results and verify display
    pass

# Basic check to ensure tests can run headlessly
# pytest-qt should handle this if Xvfb is available or by using offscreen platform plugin.
def test_headless_capability_check(qtbot: QtBot):
    """
    A simple test to ensure QtBot is functional, indicating headless setup is likely working.
    """
    from PyQt6.QtWidgets import QPushButton
    button = QPushButton("Test")
    qtbot.addWidget(button)
    # In 'offscreen' mode, isVisible() might be False.
    # Instead, check if the widget exists and its properties are accessible.
    assert button is not None
    assert button.text() == "Test"

# More tests will be added here as UI components are developed.
# For CI, ensure Xvfb is installed and running if needed:
# Example (GitHub Actions):
# - name: Setup Xvfb
#   run: sudo apt-get update && sudo apt-get install -y xvfb
# - name: Run tests with Xvfb
#   run: xvfb-run -a uv run pytest -x
# However, pytest-qt with `qtpy` often tries to use `PYQTWEBENGINE_CHROMIUM_FLAGS="--headless"`
# or `QT_QPA_PLATFORM=offscreen` to avoid needing Xvfb.
# The `AGENTS.md` says `uv run pytest -x`. We'll stick to that and assume
# the environment (local or CI) is set up for it.
# If `pytest-qt` is installed, it usually handles this.
# The `qapp` fixture helps ensure QApplication is set up.

# Note: The `ComponentViewerDialog` test uses the data structure it *receives*.
# A more end-to-end test would involve mocking `ecs_service.get_entity` and
# `ecs_service.get_components` if the dialog was fetching its own data.
# However, `MainWindow` currently fetches this data and passes it to the dialog.
# This test focuses on the dialog's rendering of provided data.
# The current `ComponentViewerDialog` takes `components_data` directly.
# Let's verify the `MainWindow::on_asset_double_clicked` logic that *prepares* this data
# in a separate test or as part of a main window test.

# For now, the `test_exif_metadata_display` directly tests `ComponentViewerDialog`'s rendering.
# This is a good unit test for the dialog itself.
# We will need more integrated tests later.

# Adding a test for the main window's asset loading and double-click to open viewer
# This is a more complex test and will require more mocking.

# from dam.ui.main_window import MainWindow
# from dam.models.core.entity import Entity
# from dam.models.properties.file_properties_component import FilePropertiesComponent

# def test_main_window_opens_component_viewer_with_exif(qtbot: QtBot, mock_world, mocker):
#     """Test that MainWindow loads assets and opens ComponentViewerDialog with EXIF data."""
#     # This is a more involved test and might be broken down or simplified.

#     # 1. Mock data fetching for MainWindow.load_assets()
#     entity_id = 1
#     mock_asset_data = [
#         (entity_id, "test_image_with_exif.jpg", "image/jpeg")
#     ]
#     mock_world.get_db_session.return_value.__enter__.return_value.query.return_value.all.return_value = mock_asset_data

#     # 2. Mock data fetching for MainWindow.on_asset_double_clicked() (ecs_service calls)
#     mock_entity = mocker.MagicMock(spec=Entity)
#     mock_entity.id = entity_id

#     exif_data = {"Make": "CameraCorp", "Model": "DSLR-1000"}
#     mock_exif_comp_instance = ExifToolMetadataComponent(entity_id=entity_id, metadata_=exif_data)

#     # Mock ecs_service.get_entity
#     mocker.patch("dam.services.ecs_service.get_entity", return_value=mock_entity)

#     # Mock ecs_service.get_components
#     # This needs to handle different component types.
#     # For this test, specifically return the EXIF component when asked.
#     def mock_get_components(session, e_id, comp_type_cls):
#         if comp_type_cls == ExifToolMetadataComponent:
#             return [mock_exif_comp_instance]
#         return [] # Return empty list for other component types

#     mocker.patch("dam.services.ecs_service.get_components", side_effect=mock_get_components)

#     # Need to mock REGISTERED_COMPONENT_TYPES if MainWindow iterates through it
#     # It's better if MainWindow uses a service to get all components for an entity,
#     # rather than iterating REGISTERED_COMPONENT_TYPES itself.
#     # MainWindow currently iterates REGISTERED_COMPONENT_TYPES.
#     mocker.patch.dict("dam.ui.main_window.REGISTERED_COMPONENT_TYPES", {
#         "ExifToolMetadataComponent": ExifToolMetadataComponent,
#         "FilePropertiesComponent": FilePropertiesComponent # Add other necessary ones if any
#     })


#     # 3. Create MainWindow instance
#     main_window = MainWindow(current_world=mock_world)
#     qtbot.addWidget(main_window)
#     main_window.show() # Important for some UI elements to initialize properly
#     qtbot.waitForWindowShown(main_window)

#     # 4. Simulate double-click on the first asset
#     # Ensure asset list is populated
#     assert main_window.asset_list_widget.count() > 0
#     first_item = main_window.asset_list_widget.item(0)
#     assert first_item is not None

#     # Mock the ComponentViewerDialog to spy on its creation or capture its instance
#     mock_component_viewer_dialog_instance = mocker.MagicMock(spec=ComponentViewerDialog)
#     mock_component_viewer_dialog_class = mocker.patch("dam.ui.main_window.ComponentViewerDialog", return_value=mock_component_viewer_dialog_instance)

#     # Simulate double click
#     main_window.asset_list_widget.itemDoubleClicked.emit(first_item)
#     # or directly call: main_window.on_asset_double_clicked(first_item) if emit doesn't work well in test

#     qtbot.waitSignal(main_window.statusBar().messageChanged, timeout=5000) # Wait for some async ops to settle

#     # 5. Assert ComponentViewerDialog was called with correct data
#     mock_component_viewer_dialog_class.assert_called_once()

#     # Check the arguments passed to ComponentViewerDialog constructor
#     args, _ = mock_component_viewer_dialog_class.call_args
#     called_entity_id, called_components_data, called_world_name, _ = args

#     assert called_entity_id == entity_id
#     assert called_world_name == mock_world.name
#     assert "ExifToolMetadataComponent" in called_components_data
#     exif_comp_list_in_dialog = called_components_data["ExifToolMetadataComponent"]
#     assert len(exif_comp_list_in_dialog) == 1
#     # The data passed to dialog is dict representation of component attributes
#     # This part of MainWindow's on_asset_double_clicked needs to be precise:
#     # instance_data = {c.key: getattr(comp_instance, c.key) ...}
#     # So, we expect metadata_ to be there.
#     assert exif_comp_list_in_dialog[0]["metadata_"] == exif_data

#     # This second test `test_main_window_opens_component_viewer_with_exif` is more complex
#     # and requires careful mocking. I'll comment it out for now and focus on simpler unit tests
#     # for individual dialogs first. We can add this integration test later.
#     # The key challenge is mocking the dynamic component registration and database interaction.
#     # The current ComponentViewerDialog test is more robust for now.
#     # A good strategy is to test dialogs in isolation with mocked inputs,
#     # and then test the parts of MainWindow that *prepare* those inputs.

# For now, the `test_exif_metadata_display` is a good starting point.
# We will add tests for the new dialogs (Transcode, Evaluation) once they are implemented.
# The placeholders `test_transcoding_dialog_trigger` and `test_transcoding_evaluation_dialog_trigger`
# will be filled in then.
# The `test_headless_capability_check` is a sanity check.
print("Created tests/test_ui_features.py with initial EXIF display test and placeholders.")
