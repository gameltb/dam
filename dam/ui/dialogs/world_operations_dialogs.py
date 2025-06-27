from pathlib import Path
from typing import List as TypingList
from typing import Optional  # Added for type hinting

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,  # Added QComboBox and QFormLayout
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QVBoxLayout,
)

from dam.core.config import settings as app_settings  # To get list of all configured worlds
from dam.core.world import World
from dam.services import world_service


# --- Worker Thread for Long Operations ---
class WorldOperationWorker(QThread):
    finished = pyqtSignal(object, str)  # object will be result (e.g., None or error), str is success/error message
    progress = pyqtSignal(str)  # For status updates

    def __init__(self, world: World, operation_type: str, params: dict):
        super().__init__()
        self.world = world
        self.operation_type = operation_type
        self.params = params

    def run(self):
        try:
            if self.operation_type == "export":
                self.progress.emit(f"Exporting world '{self.world.name}' to {self.params['filepath']}...")
                world_service.export_ecs_world_to_json(self.world, self.params["filepath"])
                self.finished.emit(
                    None, f"World '{self.world.name}' exported successfully to {self.params['filepath']}."
                )
            elif self.operation_type == "import":
                self.progress.emit(f"Importing world from {self.params['filepath']} into '{self.world.name}'...")
                world_service.import_ecs_world_from_json(self.world, self.params["filepath"], self.params["merge"])
                self.finished.emit(
                    None, f"World data imported successfully into '{self.world.name}' from {self.params['filepath']}."
                )
            elif self.operation_type == "merge_db":
                source_world_name = self.params["source_world_name"]
                target_world_name = self.world.name  # Target is the dialog's current_world
                self.progress.emit(f"Merging world '{source_world_name}' into '{target_world_name}' (DB-to-DB)...")
                # We need actual World instances. The dialog should pass these.
                # For now, assuming params contains 'source_world_instance'
                source_world_instance = self.params["source_world_instance"]
                world_service.merge_ecs_worlds_db_to_db(
                    source_world=source_world_instance, target_world=self.world, strategy="add_new"
                )
                self.finished.emit(None, f"Successfully merged '{source_world_name}' into '{target_world_name}'.")
            elif self.operation_type == "split_db":
                self.progress.emit(f"Splitting world '{self.world.name}'...")
                # Parameters for split_ecs_world: source_world, target_world_selected, target_world_remaining,
                # criteria_component_name, criteria_component_attr, criteria_value, criteria_op, delete_from_source
                count_selected, count_remaining = world_service.split_ecs_world(**self.params)
                self.finished.emit(
                    None, f"Split complete: {count_selected} entities to selected, {count_remaining} to remaining."
                )
            else:
                self.finished.emit(
                    TypeError(f"Unknown operation type: {self.operation_type}"),
                    f"Error: Unknown world operation '{self.operation_type}'.",
                )
        except Exception as e:
            self.finished.emit(e, f"Error during {self.operation_type}: {e}")


# --- Export World Dialog ---
class ExportWorldDialog(QDialog):
    def __init__(self, current_world: World, parent=None):
        super().__init__(parent)
        self.current_world = current_world
        self.setWindowTitle(f"Export World: {current_world.name}")
        self.setMinimumWidth(400)

        layout = QVBoxLayout(self)

        self.path_input = QLineEdit()
        self.path_input.setPlaceholderText("Select export file path (.json)")
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self.browse_export_path)

        path_layout = QHBoxLayout()
        path_layout.addWidget(QLabel("Export to:"))
        path_layout.addWidget(self.path_input)
        path_layout.addWidget(browse_button)
        layout.addLayout(path_layout)

        button_layout = QHBoxLayout()
        self.export_button = QPushButton("Export")
        self.export_button.clicked.connect(self.start_export)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addStretch()
        button_layout.addWidget(self.export_button)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)

        self.worker: Optional[WorldOperationWorker] = None
        self.progress_dialog: Optional[QProgressDialog] = None

    def browse_export_path(self):
        path_str, _ = QFileDialog.getSaveFileName(self, "Save World Export", "", "JSON Files (*.json)")
        if path_str:
            if not path_str.endswith(".json"):
                path_str += ".json"
            self.path_input.setText(path_str)

    def start_export(self):
        filepath_str = self.path_input.text().strip()
        if not filepath_str:
            QMessageBox.warning(self, "Input Error", "Please specify an export file path.")
            return

        filepath = Path(filepath_str)
        if filepath.is_dir():
            QMessageBox.warning(self, "Input Error", "Export path cannot be a directory.")
            return
        if filepath.exists():
            reply = QMessageBox.question(
                self,
                "Confirm Overwrite",
                f"File '{filepath.name}' already exists. Overwrite?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.No:
                return

        self.export_button.setEnabled(False)
        self.cancel_button.setText("Working...")  # Indicate busy

        self.progress_dialog = QProgressDialog(
            f"Exporting world '{self.current_world.name}'...", "Cancel", 0, 0, self
        )  # 0,0 for indeterminate
        self.progress_dialog.setWindowTitle("Exporting World")
        self.progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress_dialog.canceled.connect(self.cancel_operation)

        self.worker = WorldOperationWorker(self.current_world, "export", {"filepath": filepath})
        self.worker.finished.connect(self.on_operation_finished)
        self.worker.progress.connect(lambda msg: self.progress_dialog.setLabelText(msg))  # Update progress text
        self.worker.start()
        self.progress_dialog.show()

    def on_operation_finished(self, result_exception, message):
        self.progress_dialog.close()
        self.export_button.setEnabled(True)
        self.cancel_button.setText("Cancel")

        if result_exception:
            QMessageBox.critical(self, "Export Error", message)
        else:
            QMessageBox.information(self, "Export Successful", message)
            super().accept()  # Close dialog on success

    def cancel_operation(self):
        if self.worker and self.worker.isRunning():
            # QThread.terminate() is risky. Better to have cooperative cancellation if possible.
            # For now, we'll just inform the user it might take a moment if it's mid-IO.
            # self.worker.terminate() # Avoid if possible
            self.worker.requestInterruption()  # If worker checks isInterruptionRequested()
            QMessageBox.information(
                self,
                "Export Cancelled",
                "Export operation cancelled by user. The process might take a moment to fully stop if it was in progress.",
            )
        self.progress_dialog.close()
        self.export_button.setEnabled(True)
        self.cancel_button.setText("Cancel")


# --- Import World Dialog ---
class ImportWorldDialog(QDialog):
    def __init__(self, current_world: World, parent=None):
        super().__init__(parent)
        self.current_world = current_world
        self.setWindowTitle(f"Import Data into World: {current_world.name}")
        self.setMinimumWidth(400)

        layout = QVBoxLayout(self)

        self.path_input = QLineEdit()
        self.path_input.setPlaceholderText("Select import file path (.json)")
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self.browse_import_path)

        path_layout = QHBoxLayout()
        path_layout.addWidget(QLabel("Import from:"))
        path_layout.addWidget(self.path_input)
        path_layout.addWidget(browse_button)
        layout.addLayout(path_layout)

        self.merge_checkbox = QCheckBox(
            "Merge with existing data (if unchecked, target world data might be cleared or conflict)"
        )
        self.merge_checkbox.setChecked(False)  # Default to not merging (safer initial default)
        # Note: The actual behavior of not merging (overwrite/clear) depends on service implementation.
        # For now, we assume `merge=False` means it might be destructive or error on conflict.
        layout.addWidget(self.merge_checkbox)

        button_layout = QHBoxLayout()
        self.import_button = QPushButton("Import")
        self.import_button.clicked.connect(self.start_import)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addStretch()
        button_layout.addWidget(self.import_button)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)

        self.worker: Optional[WorldOperationWorker] = None
        self.progress_dialog: Optional[QProgressDialog] = None

    def browse_import_path(self):
        path_str, _ = QFileDialog.getOpenFileName(self, "Select World Export File", "", "JSON Files (*.json)")
        if path_str:
            self.path_input.setText(path_str)

    def start_import(self):
        filepath_str = self.path_input.text().strip()
        if not filepath_str:
            QMessageBox.warning(self, "Input Error", "Please specify an import file path.")
            return

        filepath = Path(filepath_str)
        if not filepath.is_file():
            QMessageBox.warning(self, "Input Error", f"Import file not found: {filepath_str}")
            return

        merge_data = self.merge_checkbox.isChecked()

        warning_msg = f"This will import data from '{filepath.name}' into world '{self.current_world.name}'."
        if not merge_data:
            warning_msg += "\n\nWithout merging, existing data in the target world might be overwritten or cause conflicts. This operation could be destructive."
        warning_msg += "\n\nAre you sure you want to proceed?"

        reply = QMessageBox.question(
            self,
            "Confirm Import",
            warning_msg,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.No:
            return

        self.import_button.setEnabled(False)
        self.cancel_button.setText("Working...")

        self.progress_dialog = QProgressDialog(
            f"Importing data into '{self.current_world.name}'...", "Cancel", 0, 0, self
        )
        self.progress_dialog.setWindowTitle("Importing World Data")
        self.progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress_dialog.canceled.connect(self.cancel_operation)

        self.worker = WorldOperationWorker(self.current_world, "import", {"filepath": filepath, "merge": merge_data})
        self.worker.finished.connect(self.on_operation_finished)
        self.worker.progress.connect(lambda msg: self.progress_dialog.setLabelText(msg))
        self.worker.start()
        self.progress_dialog.show()

    def on_operation_finished(self, result_exception, message):
        self.progress_dialog.close()
        self.import_button.setEnabled(True)
        self.cancel_button.setText("Cancel")

        if result_exception:
            QMessageBox.critical(self, "Import Error", message)
        else:
            QMessageBox.information(self, "Import Successful", message)
            super().accept()  # Close dialog on success

    def cancel_operation(self):
        if self.worker and self.worker.isRunning():
            self.worker.requestInterruption()
            QMessageBox.information(
                self,
                "Import Cancelled",
                "Import operation cancelled by user. The process might take a moment to fully stop if it was in progress.",
            )
        self.progress_dialog.close()
        self.import_button.setEnabled(True)
        self.cancel_button.setText("Cancel")


if __name__ == "__main__":
    # Test Script
    app = QApplication([])
    # Ensure project modules are discoverable for real world loading, or use mocks.
    # This might require setting PYTHONPATH or running from project root.

    # Mock or load a real world for testing dialogs
    test_world_instance = None
    all_world_names: TypingList[str] = []
    try:
        from dam.core.world import create_and_register_all_worlds_from_settings
        from dam.core.world_setup import register_core_systems

        # Initialize worlds (important for get_world to work)
        # This ensures that the world registry is populated.
        initialized_worlds = create_and_register_all_worlds_from_settings(app_settings=app_settings)
        for w_init in initialized_worlds:
            register_core_systems(w_init)  # Register core systems for each world

        # Get all configured world names for dropdowns
        if app_settings.worlds:
            all_world_names = [w_config.name for w_config in app_settings.worlds]

        # Try to get the default world or the first one for dialogs that need a 'current_world'
        if app_settings.DEFAULT_WORLD_NAME:
            test_world_instance = get_world(app_settings.DEFAULT_WORLD_NAME)
        elif initialized_worlds:
            test_world_instance = initialized_worlds[0]

        if not test_world_instance:
            print("No default or first world found, using a basic MockWorld for some tests.")

            class MockWorld:
                def __init__(self, name="mock_world_ops"):
                    self.name = name

            test_world_instance = MockWorld()

    except Exception as e:
        print(f"Error loading real worlds for testing ({e}), using a basic MockWorld.")

        class MockWorld:  # Define here if imports fail
            def __init__(self, name="fallback_mock_ops"):
                self.name = name

        test_world_instance = MockWorld()
        all_world_names = ["mock_world1", "mock_world2", "mock_world3"]  # Dummy names for dropdowns

    print(f"--- Testing Dialogs (using world: {test_world_instance.name if test_world_instance else 'None'}) ---")

    if (
        QMessageBox.question(
            None, "Test", "Show Export Dialog?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        == QMessageBox.StandardButton.Yes
    ):
        if test_world_instance:
            export_dialog = ExportWorldDialog(current_world=test_world_instance)
            export_dialog.exec()
        else:
            QMessageBox.warning(None, "Error", "No current world to test Export Dialog.")

    if (
        QMessageBox.question(
            None, "Test", "Show Import Dialog?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        == QMessageBox.StandardButton.Yes
    ):
        if test_world_instance:
            import_dialog = ImportWorldDialog(current_world=test_world_instance)
            import_dialog.exec()
        else:
            QMessageBox.warning(None, "Error", "No current world to test Import Dialog.")

    if (
        QMessageBox.question(
            None, "Test", "Show Merge Worlds Dialog?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        == QMessageBox.StandardButton.Yes
    ):
        if test_world_instance and all_world_names:
            # The MergeWorldsDialog determines its own target from its 'current_world'
            # It needs a list of other worlds to select as source.
            merge_dialog = MergeWorldsDialog(current_world=test_world_instance, all_world_names=all_world_names)
            merge_dialog.exec()
        else:
            QMessageBox.warning(None, "Error", "Need a current world and list of all worlds for Merge Dialog.")

    if (
        QMessageBox.question(
            None, "Test", "Show Split World Dialog?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        == QMessageBox.StandardButton.Yes
    ):
        if test_world_instance and all_world_names:
            # SplitWorldDialog's 'source_world' is its 'current_world'
            split_dialog = SplitWorldDialog(source_world=test_world_instance, all_world_names=all_world_names)
            split_dialog.exec()
        else:
            QMessageBox.warning(None, "Error", "Need a source world and list of all worlds for Split Dialog.")

    sys.exit(app.exec())


# --- Merge Worlds Dialog ---
class MergeWorldsDialog(QDialog):
    def __init__(self, current_world: World, all_world_names: TypingList[str], parent=None):
        super().__init__(parent)
        self.current_world = current_world  # This will be the TARGET world
        self.all_world_names = [name for name in all_world_names if name != current_world.name]  # Exclude self

        self.setWindowTitle(f"Merge another World into: {current_world.name}")
        self.setMinimumWidth(450)
        layout = QVBoxLayout(self)
        form_layout = QFormLayout()

        self.source_world_combo = QComboBox()
        if not self.all_world_names:
            self.source_world_combo.addItem("No other worlds available to merge from.")
            self.source_world_combo.setEnabled(False)
        else:
            self.source_world_combo.addItems(self.all_world_names)
        form_layout.addRow(QLabel("Merge From (Source World):"), self.source_world_combo)
        form_layout.addRow(QLabel("Merge Into (Target World):"), QLabel(f"<b>{current_world.name}</b> (current)"))

        layout.addLayout(form_layout)
        # Strategy note - currently only 'add_new' is supported by backend
        layout.addWidget(QLabel("<i>Merge strategy: All entities from source will be added as new to target.</i>"))

        button_layout = QHBoxLayout()
        self.merge_button = QPushButton("Start Merge")
        self.merge_button.setEnabled(bool(self.all_world_names))  # Disable if no source worlds
        self.merge_button.clicked.connect(self.start_merge)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addStretch()
        button_layout.addWidget(self.merge_button)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)

        self.worker: Optional[WorldOperationWorker] = None
        self.progress_dialog: Optional[QProgressDialog] = None

    def start_merge(self):
        source_world_name = self.source_world_combo.currentText()
        if not source_world_name or source_world_name == "No other worlds available to merge from.":
            QMessageBox.warning(self, "Input Error", "Please select a valid source world.")
            return

        source_world_instance = get_world(source_world_name)
        if not source_world_instance:
            QMessageBox.critical(
                self,
                "Error",
                f"Could not get instance for source world '{source_world_name}'. It might not be initialized correctly.",
            )
            return

        if source_world_instance.name == self.current_world.name:  # Should be prevented by list population
            QMessageBox.warning(self, "Input Error", "Source and target worlds cannot be the same.")
            return

        reply = QMessageBox.question(
            self,
            "Confirm Merge",
            f"Merge all entities from '{source_world_name}' into '{self.current_world.name}'?\n"
            "This will add all source entities as new entities in the target.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.No:
            return

        self.merge_button.setEnabled(False)
        self.cancel_button.setText("Working...")
        self.progress_dialog = QProgressDialog(
            f"Merging '{source_world_name}' into '{self.current_world.name}'...", "Cancel", 0, 0, self
        )
        self.progress_dialog.setWindowTitle("Merging Worlds")
        self.progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress_dialog.canceled.connect(self.cancel_operation)

        params = {
            "source_world_name": source_world_name,  # For progress message
            "source_world_instance": source_world_instance,
            # Target world is self.current_world, passed to worker's constructor
        }
        self.worker = WorldOperationWorker(self.current_world, "merge_db", params)
        self.worker.finished.connect(self.on_operation_finished)
        self.worker.progress.connect(lambda msg: self.progress_dialog.setLabelText(msg))
        self.worker.start()
        self.progress_dialog.show()

    def on_operation_finished(self, result_exception, message):
        self.progress_dialog.close()
        self.merge_button.setEnabled(True)
        self.cancel_button.setText("Cancel")
        if result_exception:
            QMessageBox.critical(self, "Merge Error", message)
        else:
            QMessageBox.information(self, "Merge Successful", message)
            super().accept()

    def cancel_operation(self):
        if self.worker and self.worker.isRunning():
            self.worker.requestInterruption()
        self.progress_dialog.close()
        self.merge_button.setEnabled(True)
        self.cancel_button.setText("Cancel")


# --- Split World Dialog ---
class SplitWorldDialog(QDialog):
    def __init__(self, source_world: World, all_world_names: TypingList[str], parent=None):
        super().__init__(parent)
        self.source_world = source_world  # This is the world to split FROM
        # Worlds for selection as targets (must not be the source world)
        self.available_target_worlds = [name for name in all_world_names if name != source_world.name]

        self.setWindowTitle(f"Split World: {source_world.name}")
        self.setMinimumWidth(550)  # Wider for more fields
        layout = QVBoxLayout(self)
        form_layout = QFormLayout()

        form_layout.addRow(
            QLabel("<b>Source World:</b>"), QLabel(f"<b>{source_world.name}</b> (entities will be copied from here)")
        )

        self.target_selected_combo = QComboBox()
        self.target_remaining_combo = QComboBox()
        if not self.available_target_worlds or len(self.available_target_worlds) < 2:
            self.target_selected_combo.addItem("Not enough other worlds available for targets.")
            self.target_remaining_combo.addItem("Not enough other worlds available for targets.")
            self.target_selected_combo.setEnabled(False)
            self.target_remaining_combo.setEnabled(False)
        else:
            self.target_selected_combo.addItems(self.available_target_worlds)
            self.target_remaining_combo.addItems(self.available_target_worlds)
            # Try to pre-select different targets if possible
            if len(self.available_target_worlds) >= 2:
                self.target_remaining_combo.setCurrentIndex(1)

        form_layout.addRow(QLabel("Target for Selected Entities:"), self.target_selected_combo)
        form_layout.addRow(QLabel("Target for Remaining Entities:"), self.target_remaining_combo)

        form_layout.addRow(QLabel("<b>Split Criteria:</b>"))
        self.component_name_input = QLineEdit()
        self.component_name_input.setPlaceholderText("e.g., FilePropertiesComponent")
        form_layout.addRow(QLabel("Component Name:"), self.component_name_input)

        self.attribute_name_input = QLineEdit()
        self.attribute_name_input.setPlaceholderText("e.g., mime_type")
        form_layout.addRow(QLabel("Attribute Name:"), self.attribute_name_input)

        self.attribute_value_input = QLineEdit()
        self.attribute_value_input.setPlaceholderText("e.g., image/jpeg")
        form_layout.addRow(QLabel("Attribute Value:"), self.attribute_value_input)

        self.operator_combo = QComboBox()
        self.operator_combo.addItems(["eq", "ne", "contains", "startswith", "endswith", "gt", "lt", "ge", "le"])
        form_layout.addRow(QLabel("Operator:"), self.operator_combo)

        self.delete_from_source_checkbox = QCheckBox("Delete entities from source world after copying")
        form_layout.addRow(self.delete_from_source_checkbox)

        layout.addLayout(form_layout)
        button_layout = QHBoxLayout()
        self.split_button = QPushButton("Start Split")
        self.split_button.setEnabled(len(self.available_target_worlds) >= 2)
        self.split_button.clicked.connect(self.start_split)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addStretch()
        button_layout.addWidget(self.split_button)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)

        self.worker: Optional[WorldOperationWorker] = None
        self.progress_dialog: Optional[QProgressDialog] = None

    def start_split(self):
        target_selected_name = self.target_selected_combo.currentText()
        target_remaining_name = self.target_remaining_combo.currentText()

        if (
            not self.available_target_worlds or len(self.available_target_worlds) < 2
        ):  # Should be caught by button enable state
            QMessageBox.warning(self, "Setup Error", "Not enough distinct target worlds available or selected.")
            return

        if target_selected_name == target_remaining_name:
            QMessageBox.warning(
                self, "Input Error", "Target worlds for selected and remaining entities must be different."
            )
            return
        if target_selected_name == self.source_world.name or target_remaining_name == self.source_world.name:
            QMessageBox.warning(self, "Input Error", "Target worlds cannot be the same as the source world.")
            return  # Already handled by available_target_worlds logic, but good check

        # Validate criteria inputs
        criteria_comp_name = self.component_name_input.text().strip()
        criteria_attr_name = self.attribute_name_input.text().strip()
        criteria_attr_value = self.attribute_value_input.text().strip()  # Value is passed as string
        if not (criteria_comp_name and criteria_attr_name and criteria_attr_value):
            QMessageBox.warning(
                self,
                "Input Error",
                "Please fill in all criteria fields (Component Name, Attribute Name, Attribute Value).",
            )
            return

        target_selected_inst = get_world(target_selected_name)
        target_remaining_inst = get_world(target_remaining_name)

        if not target_selected_inst or not target_remaining_inst:
            QMessageBox.critical(
                self, "Error", "Could not get instances for target worlds. They might not be initialized correctly."
            )
            return

        params = {
            "source_world": self.source_world,
            "target_world_selected": target_selected_inst,
            "target_world_remaining": target_remaining_inst,
            "criteria_component_name": criteria_comp_name,
            "criteria_component_attr": criteria_attr_name,
            "criteria_value": criteria_attr_value,  # Service layer handles type conversion if needed
            "criteria_op": self.operator_combo.currentText(),
            "delete_from_source": self.delete_from_source_checkbox.isChecked(),
        }

        confirm_msg = (
            f"Split entities from '{self.source_world.name}' into:\n"
            f"  - Selected to: '{target_selected_name}'\n"
            f"  - Remaining to: '{target_remaining_name}'\n"
            f"Based on criteria: {criteria_comp_name}.{criteria_attr_name} {params['criteria_op']} '{criteria_attr_value}'\n"
        )
        if params["delete_from_source"]:
            confirm_msg += "\nWARNING: Entities will be deleted from the source world after copying!\n"
        confirm_msg += "\nAre you sure you want to proceed?"

        if (
            QMessageBox.question(
                self,
                "Confirm Split",
                confirm_msg,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            == QMessageBox.StandardButton.No
        ):
            return

        self.split_button.setEnabled(False)
        self.cancel_button.setText("Working...")
        self.progress_dialog = QProgressDialog(f"Splitting world '{self.source_world.name}'...", "Cancel", 0, 0, self)
        self.progress_dialog.setWindowTitle("Splitting World")
        self.progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress_dialog.canceled.connect(self.cancel_operation)

        # Worker receives source_world as its 'world' context, other params go into 'params'
        self.worker = WorldOperationWorker(self.source_world, "split_db", params)
        self.worker.finished.connect(self.on_operation_finished)
        self.worker.progress.connect(lambda msg: self.progress_dialog.setLabelText(msg))
        self.worker.start()
        self.progress_dialog.show()

    def on_operation_finished(self, result_exception, message):
        self.progress_dialog.close()
        self.split_button.setEnabled(True)
        self.cancel_button.setText("Cancel")
        if result_exception:
            QMessageBox.critical(self, "Split Error", message)
        else:
            QMessageBox.information(self, "Split Successful", message)
            super().accept()

    def cancel_operation(self):
        if self.worker and self.worker.isRunning():
            self.worker.requestInterruption()
        self.progress_dialog.close()
        self.split_button.setEnabled(True)  # Re-enable based on initial logic if needed
        self.cancel_button.setText("Cancel")
