import asyncio
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QLineEdit,
    QCheckBox,
    QFileDialog,
    QMessageBox,
    QFormLayout,
)

from dam.core.world import World
from dam.core.events import AssetFileIngestionRequested, AssetReferenceIngestionRequested
from dam.core.stages import SystemStage
from dam.services import file_operations


class AddAssetDialog(QDialog):
    def __init__(self, current_world: World, parent=None):
        super().__init__(parent)
        self.current_world = current_world
        self.setWindowTitle("Add New Asset(s)")
        self.setMinimumWidth(500)

        layout = QVBoxLayout(self)
        form_layout = QFormLayout()

        # File/Directory Path
        self.path_input = QLineEdit()
        self.browse_button = QPushButton("Browse...")
        self.browse_button.clicked.connect(self.browse_path)
        path_layout = QHBoxLayout()
        path_layout.addWidget(self.path_input)
        path_layout.addWidget(self.browse_button)
        form_layout.addRow(QLabel("File/Directory Path:"), path_layout)

        # Options
        self.no_copy_checkbox = QCheckBox("Add by reference (do not copy to DAM storage)")
        form_layout.addRow(self.no_copy_checkbox)

        self.recursive_checkbox = QCheckBox("Process directory recursively")
        self.recursive_checkbox.setEnabled(False) # Enabled when a directory is selected
        form_layout.addRow(self.recursive_checkbox)

        self.path_input.textChanged.connect(self.update_recursive_checkbox_state)

        layout.addLayout(form_layout)

        # Buttons
        button_layout = QHBoxLayout()
        self.add_button = QPushButton("Add Asset(s)")
        self.add_button.clicked.connect(self.accept)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addStretch()
        button_layout.addWidget(self.add_button)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def update_recursive_checkbox_state(self, text: str):
        path = Path(text)
        self.recursive_checkbox.setEnabled(path.is_dir())
        if not path.is_dir() and self.recursive_checkbox.isChecked():
            self.recursive_checkbox.setChecked(False)


    def browse_path(self):
        # Allow selecting either a file or a directory
        # Start with QFileDialog.getOpenFileName and if user cancels, try QFileDialog.getExistingDirectory
        # Or, have separate buttons, but a single smart one is often better.
        # For simplicity, let's try a generic dialog that can select both, though Qt doesn't have one directly.
        # We can use getExistingDirectory first, then getOpenFileName if user wants a file.
        # Or, more simply, let user type/paste and validate.
        # Let's try a simpler approach: offer two browse buttons or one that asks.
        # For now, we'll make it intelligent based on what the user selects.

        # Try to select a directory first. If the user cancels, try to select a file.
        # This provides a somewhat unified experience, prioritizing directories.
        dialog = QFileDialog(self, "Select Directory or File")
        dialog.setFileMode(QFileDialog.FileMode.Directory) # Prioritize directory

        # Try to make it more "any file or directory like"
        # On some platforms, getOpenFileNames can select directories if configured right,
        # but it's not standard. getExistingDirectory is explicit.

        # Let's offer a choice via dialog mode or simply try one then the other.
        # For this iteration, let's try a simpler approach that prioritizes directory selection.

        path_str = QFileDialog.getExistingDirectory(self, "Select Directory to Add Assets From")
        if path_str:
            self.path_input.setText(path_str)
            self.update_recursive_checkbox_state(path_str) # Ensure checkbox updates
        else:
            # If directory selection was cancelled, try file selection
            path_str, _ = QFileDialog.getOpenFileName(self, "Select File to Add as Asset")
            if path_str:
                self.path_input.setText(path_str)
                self.update_recursive_checkbox_state(path_str) # Ensure checkbox updates


    def get_selected_options(self):
        path_str = self.path_input.text().strip()
        if not path_str:
            QMessageBox.warning(self, "Input Error", "Please select a file or directory path.")
            return None

        path = Path(path_str)
        if not path.exists():
            QMessageBox.warning(self, "Input Error", f"Path does not exist: {path_str}")
            return None

        return {
            "path": path,
            "no_copy": self.no_copy_checkbox.isChecked(),
            "recursive": self.recursive_checkbox.isChecked() if path.is_dir() else False,
        }

    def accept(self):
        options = self.get_selected_options()
        if not options:
            return

        path: Path = options["path"]
        no_copy: bool = options["no_copy"]
        recursive: bool = options["recursive"]

        files_to_process: list[Path] = []
        if path.is_file():
            files_to_process.append(path)
        elif path.is_dir():
            if recursive:
                files_to_process.extend(p for p in path.rglob("*") if p.is_file())
            else:
                files_to_process.extend(p for p in path.glob("*") if p.is_file())

        if not files_to_process:
            QMessageBox.information(self, "No Files", f"No files found to process at '{path}'.")
            super().reject() # Close dialog if no files
            return

        # Show progress/summary later
        processed_count = 0
        error_count = 0

        # This part should ideally run in a separate thread if processing many files
        # For now, it will block the UI.
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            for filepath in files_to_process:
                try:
                    original_filename, size_bytes, mime_type = file_operations.get_file_properties(filepath)

                    event_to_dispatch: Optional[AssetFileIngestionRequested | AssetReferenceIngestionRequested] = None
                    if no_copy:
                        event_to_dispatch = AssetReferenceIngestionRequested(
                            filepath_on_disk=filepath,
                            original_filename=original_filename,
                            mime_type=mime_type,
                            size_bytes=size_bytes,
                            world_name=self.current_world.name,
                        )
                    else:
                        event_to_dispatch = AssetFileIngestionRequested(
                            filepath_on_disk=filepath,
                            original_filename=original_filename,
                            mime_type=mime_type,
                            size_bytes=size_bytes,
                            world_name=self.current_world.name,
                        )

                    async def dispatch_and_run_stages_sync(): # Helper for asyncio.run
                        if event_to_dispatch:
                            await self.current_world.dispatch_event(event_to_dispatch)
                            await self.current_world.execute_stage(SystemStage.METADATA_EXTRACTION)

                    asyncio.run(dispatch_and_run_stages_sync())
                    processed_count += 1
                except Exception as e:
                    error_count += 1
                    # Log error, maybe collect them for a summary
                    print(f"Error processing {filepath.name}: {e}") # Simple print for now

            summary_message = f"Processed {processed_count} file(s)."
            if error_count > 0:
                summary_message += f"\nEncountered {error_count} error(s)."

            QMessageBox.information(self, "Ingestion Complete", summary_message)
            super().accept() # Close dialog and signal success

        except Exception as e:
            QMessageBox.critical(self, "Ingestion Error", f"An unexpected error occurred: {e}")
            super().reject() # Close dialog on major failure
        finally:
            QApplication.restoreOverrideCursor()

if __name__ == '__main__':
    # This is for testing the dialog independently
    # In a real app, you'd need a QApplication and a World instance
    from PyQt6.QtWidgets import QApplication
    import sys

    # Mock World for testing
    class MockWorld:
        def __init__(self, name="test_world"):
            self.name = name
            print(f"MockWorld '{self.name}' initialized.")

        async def dispatch_event(self, event):
            print(f"MockWorld: Event dispatched: {type(event).__name__} for {event.original_filename}")
            # Simulate some work
            await asyncio.sleep(0.1)

        async def execute_stage(self, stage):
            print(f"MockWorld: Executing stage: {stage.name}")
            await asyncio.sleep(0.1)

        def get_db_session(self): # Required if any part tries to use it through world
            class MockSession:
                def __enter__(self): return self
                def __exit__(self, type, value, traceback): pass
                # Add other methods if needed by downstream calls during testing
            return MockSession()


    app = QApplication(sys.argv)

    # Setup a mock world (replace with actual world loading if testing with real backend)
    # This requires your project structure to be importable
    try:
        # Attempt to load a real world if configured, for more thorough testing
        from dam.core import config as app_config
        from dam.core.world import get_world, create_and_register_all_worlds_from_settings
        from dam.core.world_setup import register_core_systems

        worlds = create_and_register_all_worlds_from_settings(app_settings=app_config.settings)
        for w in worlds: register_core_systems(w)

        world_instance_to_use = None
        if app_config.settings.DEFAULT_WORLD_NAME:
            world_instance_to_use = get_world(app_config.settings.DEFAULT_WORLD_NAME)
        elif worlds:
            world_instance_to_use = worlds[0]

        if not world_instance_to_use:
            print("Using MockWorld as no real world could be loaded.")
            world_instance_to_use = MockWorld("default_mock")

    except ImportError:
        print("Project modules not found, using basic MockWorld.")
        world_instance_to_use = MockWorld("fallback_mock")
    except Exception as e:
        print(f"Error setting up real world for dialog test, using MockWorld: {e}")
        world_instance_to_use = MockWorld("error_mock")


    dialog = AddAssetDialog(current_world=world_instance_to_use)
    if dialog.exec():
        print("Add Asset Dialog accepted.")
    else:
        print("Add Asset Dialog cancelled.")
    sys.exit(app.exec())
