from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit, QComboBox,
    QFileDialog, QMessageBox, QFormLayout, QApplication
)
from PyQt6.QtCore import Qt
from pathlib import Path
import uuid
import asyncio # For running async world events

from dam.core.world import World
from dam.core.events import FindEntityByHashQuery
from dam.services import file_operations # For calculating hash from file
from dam.ui.main_window import ComponentViewerDialog # Reuse for displaying results


class FindAssetByHashDialog(QDialog):
    def __init__(self, current_world: World, parent=None):
        super().__init__(parent)
        self.current_world = current_world
        self.setWindowTitle("Find Asset by Content Hash")
        self.setMinimumWidth(450)

        layout = QVBoxLayout(self)
        form_layout = QFormLayout()

        # Hash Value Input
        self.hash_value_input = QLineEdit()
        form_layout.addRow(QLabel("Hash Value:"), self.hash_value_input)

        # Hash Type ComboBox
        self.hash_type_combo = QComboBox()
        self.hash_type_combo.addItems(["sha256", "md5"])
        form_layout.addRow(QLabel("Hash Type:"), self.hash_type_combo)

        # Calculate from File (Optional)
        self.file_path_input = QLineEdit()
        self.file_path_input.setPlaceholderText("Optional: Select file to calculate hash from")
        self.browse_file_button = QPushButton("Browse File...")
        self.browse_file_button.clicked.connect(self.browse_for_file_to_hash)
        file_hash_layout = QHBoxLayout()
        file_hash_layout.addWidget(self.file_path_input)
        file_hash_layout.addWidget(self.browse_file_button)
        form_layout.addRow(QLabel("Calculate from File:"), file_hash_layout)

        self.calculate_and_fill_hash_button = QPushButton("Calculate & Fill Hash")
        self.calculate_and_fill_hash_button.clicked.connect(self.calculate_and_fill_hash)
        form_layout.addRow(self.calculate_and_fill_hash_button)

        layout.addLayout(form_layout)

        # Buttons
        button_layout = QHBoxLayout()
        self.find_button = QPushButton("Find Asset")
        self.find_button.clicked.connect(self.find_asset)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addStretch()
        button_layout.addWidget(self.find_button)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def browse_for_file_to_hash(self):
        path_str, _ = QFileDialog.getOpenFileName(self, "Select File to Calculate Hash")
        if path_str:
            self.file_path_input.setText(path_str)

    def calculate_and_fill_hash(self):
        filepath_str = self.file_path_input.text().strip()
        if not filepath_str:
            QMessageBox.warning(self, "Input Error", "Please select a file first to calculate its hash.")
            return

        filepath = Path(filepath_str)
        if not filepath.is_file():
            QMessageBox.warning(self, "Input Error", f"File does not exist or is not a file: {filepath_str}")
            return

        selected_hash_type = self.hash_type_combo.currentText()
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            calculated_hash = ""
            if selected_hash_type == "sha256":
                calculated_hash = file_operations.calculate_sha256_hex(filepath) # Assuming hex output needed
            elif selected_hash_type == "md5":
                calculated_hash = file_operations.calculate_md5_hex(filepath) # Assuming hex output needed

            self.hash_value_input.setText(calculated_hash)
            QMessageBox.information(self, "Hash Calculated", f"Calculated {selected_hash_type} hash: {calculated_hash}")
        except Exception as e:
            QMessageBox.critical(self, "Hash Calculation Error", f"Could not calculate hash: {e}")
        finally:
            QApplication.restoreOverrideCursor()


    def find_asset(self):
        hash_value = self.hash_value_input.text().strip()
        hash_type = self.hash_type_combo.currentText()

        if not hash_value:
            QMessageBox.warning(self, "Input Error", "Please enter a hash value.")
            return

        request_id = str(uuid.uuid4())
        query_event = FindEntityByHashQuery(
            hash_value=hash_value, # The service layer will handle hex string to bytes if needed
            hash_type=hash_type,
            world_name=self.current_world.name,
            request_id=request_id,
        )

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            # Run the async event dispatch
            async def dispatch_query_sync():
                await self.current_world.dispatch_event(query_event)

            asyncio.run(dispatch_query_sync())

            QApplication.restoreOverrideCursor() # Restore cursor before showing dialogs

            if query_event.result:
                entity_id = query_event.result.get("entity_id")
                components_data = query_event.result.get("components", {})

                # We need to structure components_data like ComponentViewerDialog expects:
                # Dict[str, List[Dict[str, Any]]]
                # The event result is Dict[str, Dict[str, Any]] for single components
                # or Dict[str, List[Dict[str,Any]]] for multi-instance components.
                # We need to ensure it's always a list of dicts for each component type.

                processed_components_data = {}
                for comp_name, data in components_data.items():
                    if isinstance(data, list):
                        processed_components_data[comp_name] = data
                    elif isinstance(data, dict): # Single instance component
                        processed_components_data[comp_name] = [data]
                    else: # Should not happen based on event structure
                        print(f"Warning: Unexpected data type for component {comp_name} in FindEntityByHashQuery result.")
                        processed_components_data[comp_name] = []


                QMessageBox.information(self, "Asset Found", f"Asset found for Entity ID: {entity_id}.")
                # Reuse ComponentViewerDialog to show the details
                # It expects dict where values are lists of component instances (as dicts)
                viewer_dialog = ComponentViewerDialog(entity_id, processed_components_data, self.current_world.name, self)
                viewer_dialog.exec()
                super().accept() # Close find dialog if asset found and viewed
            else:
                QMessageBox.information(self, "Not Found", f"No asset found for hash '{hash_value}' (type: {hash_type}).")
        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Find Asset Error", f"An error occurred: {e}")
            # import traceback; print(traceback.format_exc()) # For debugging
        finally:
            if QApplication.overrideCursor() is not None: # Ensure cursor is restored
                 QApplication.restoreOverrideCursor()


if __name__ == '__main__':
    from PyQt6.QtWidgets import QApplication
    import sys

    class MockWorld: # Same MockWorld as in add_asset_dialog.py for consistency
        def __init__(self, name="test_world"):
            self.name = name
            print(f"MockWorld '{self.name}' initialized for FindAssetByHashDialog.")

        async def dispatch_event(self, event: FindEntityByHashQuery):
            print(f"MockWorld: Event dispatched: {type(event).__name__} for hash {event.hash_value}")
            await asyncio.sleep(0.1)
            # Simulate finding an asset
            if event.hash_value == "found_hash_sha256" and event.hash_type == "sha256":
                event.result = {
                    "entity_id": 123,
                    "components": {
                        "FilePropertiesComponent": { # Single instance
                            "original_filename": "test_image.jpg",
                            "mime_type": "image/jpeg",
                            "file_size_bytes": 10240
                        },
                        "ContentHashSHA256Component": { # Single instance
                            "hash_value": "found_hash_sha256"
                        },
                        "SomeMultiInstanceComponent": [ # Example of multi-instance
                           {"value": "instance1"},
                           {"value": "instance2"}
                        ]
                    }
                }
            elif event.hash_value == "another_hash_md5" and event.hash_type == "md5":
                 event.result = {
                    "entity_id": 456,
                    "components": {
                        "FilePropertiesComponent": {
                            "original_filename": "document.pdf",
                            "mime_type": "application/pdf",
                        }
                    }
                 }
            else:
                event.result = None # Not found
            print(f"MockWorld: Event result set for request_id {event.request_id}")


    app = QApplication(sys.argv)
    # Setup a mock world
    try:
        from dam.core import config as app_config
        from dam.core.world import get_world, create_and_register_all_worlds_from_settings
        from dam.core.world_setup import register_core_systems

        worlds = create_and_register_all_worlds_from_settings(app_settings=app_config.settings)
        for w in worlds: register_core_systems(w)

        world_instance_to_use = None
        if app_config.settings.DEFAULT_WORLD_NAME: world_instance_to_use = get_world(app_config.settings.DEFAULT_WORLD_NAME)
        elif worlds: world_instance_to_use = worlds[0]

        if not world_instance_to_use: world_instance_to_use = MockWorld("default_mock_find")
        else: print(f"Using REAL world for FindAssetByHashDialog test: {world_instance_to_use.name}")

    except Exception as e:
        print(f"Error setting up world for dialog test, using MockWorld: {e}")
        world_instance_to_use = MockWorld("error_mock_find")

    dialog = FindAssetByHashDialog(current_world=world_instance_to_use)
    dialog.exec()
    sys.exit(app.exec())
