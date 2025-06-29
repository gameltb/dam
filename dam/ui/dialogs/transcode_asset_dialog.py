import asyncio
import uuid
from typing import Optional

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from dam.core.world import World
# Assume TranscodeProfileComponent exists and can be queried for names/IDs
# from dam.models.conceptual.transcode_profile_component import TranscodeProfileComponent
# Assume TranscodeService exists for handling the transcoding operation
# from dam.services.transcode_service import TranscodeService # Or an event-based system

# Placeholder for actual transcode profiles and service interaction
# In a real scenario, you'd fetch profiles from the DB or a service.
DUMMY_PROFILES = {
    "Profile Web (JPEG Small)": "profile_web_jpeg_small_uuid",
    "Profile Archive (JPEG XL Lossless)": "profile_archive_jxl_lossless_uuid",
    "Profile Video (MP4 H.264)": "profile_video_mp4_h264_uuid",
}


class TranscodeWorker(QThread):
    finished = pyqtSignal(bool, str)  # success (bool), message (str)
    progress = pyqtSignal(int, str) # value (int), message (str)

    def __init__(self, world: World, entity_id: int, profile_id: str, profile_name: str):
        super().__init__()
        self.world = world
        self.entity_id = entity_id
        self.profile_id = profile_id
        self.profile_name = profile_name # For messages

    def run(self):
        try:
            self.progress.emit(0, f"Starting transcode for Entity ID {self.entity_id} with profile '{self.profile_name}'...")

            # TODO: Replace with actual call to TranscodeService or event dispatch
            # For example:
            # transcode_event = TranscodeAssetRequested(
            #     entity_id=self.entity_id,
            #     profile_id=self.profile_id,
            #     request_id=str(uuid.uuid4()),
            #     world_name=self.world.name
            # )
            # asyncio.run(self.world.dispatch_event(transcode_event))
            #
            # if transcode_event.error:
            #    raise Exception(transcode_event.error)
            # elif transcode_event.result: # e.g. new entity_id of transcoded asset
            #    pass

            # Simulate work
            for i in range(101):
                if self.isInterruptionRequested():
                    self.finished.emit(False, "Transcoding cancelled.")
                    return
                QThread.msleep(50) # Simulate some processing time
                self.progress.emit(i, f"Transcoding... {i}%")

            # Simulate success
            # In a real scenario, result_message might include new asset ID or path
            result_message = f"Successfully transcoded Entity ID {self.entity_id} using profile '{self.profile_name}'."
            self.finished.emit(True, result_message)

        except Exception as e:
            error_message = f"Error transcoding Entity ID {self.entity_id}: {e}"
            self.finished.emit(False, error_message)


class TranscodeAssetDialog(QDialog):
    def __init__(self, world: World, entity_id: int, entity_filename: str, parent=None):
        super().__init__(parent)
        self.world = world
        self.entity_id = entity_id
        self.entity_filename = entity_filename

        self.setWindowTitle(f"Transcode Asset: {self.entity_filename} (ID: {self.entity_id})")
        self.setMinimumWidth(450)

        layout = QVBoxLayout(self)
        form_layout = QFormLayout()

        self.asset_label = QLabel(f"Asset: {self.entity_filename} (ID: {self.entity_id})")
        form_layout.addRow(self.asset_label)

        self.profile_combo = QComboBox()
        # TODO: Populate with actual transcode profiles from the system/world
        # For now, using dummy profiles
        for name, profile_id in DUMMY_PROFILES.items():
            self.profile_combo.addItem(name, profile_id) # Store ID as item data

        if not DUMMY_PROFILES:
            self.profile_combo.addItem("No transcode profiles available.")
            self.profile_combo.setEnabled(False)

        form_layout.addRow(QLabel("Transcode Profile:"), self.profile_combo)

        layout.addLayout(form_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False) # Show when transcoding starts
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("")
        self.status_label.setVisible(False)
        layout.addWidget(self.status_label)


        button_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Transcode")
        self.start_button.setEnabled(bool(DUMMY_PROFILES))
        self.start_button.clicked.connect(self.start_transcoding)

        self.cancel_button = QPushButton("Cancel Operation") # Becomes cancel during operation
        self.cancel_button.clicked.connect(self.cancel_or_close) # Default is close

        button_layout.addStretch()
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)

        self.worker: Optional[TranscodeWorker] = None
        self.is_transcoding = False

    def start_transcoding(self):
        if self.is_transcoding: # Should not happen if button disabled
            return

        selected_profile_name = self.profile_combo.currentText()
        selected_profile_id = self.profile_combo.currentData()

        if not selected_profile_id or selected_profile_name == "No transcode profiles available.":
            QMessageBox.warning(self, "Selection Error", "Please select a valid transcode profile.")
            return

        self.is_transcoding = True
        self.start_button.setEnabled(False)
        self.profile_combo.setEnabled(False)
        self.cancel_button.setText("Cancel Operation")
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.status_label.setText(f"Preparing to transcode with '{selected_profile_name}'...")
        self.status_label.setVisible(True)

        self.worker = TranscodeWorker(
            world=self.world,
            entity_id=self.entity_id,
            profile_id=selected_profile_id,
            profile_name=selected_profile_name
        )
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.on_transcoding_finished)
        self.worker.start()

    def update_progress(self, value: int, message: str):
        self.progress_bar.setValue(value)
        self.status_label.setText(message)

    def on_transcoding_finished(self, success: bool, message: str):
        self.is_transcoding = False
        self.start_button.setEnabled(True)
        self.profile_combo.setEnabled(True) # Re-enable profile selection
        self.cancel_button.setText("Close") # Revert to "Close"
        self.progress_bar.setVisible(False) # Hide progress bar on completion or error
        self.status_label.setText(message) # Show final status

        if success:
            QMessageBox.information(self, "Transcode Complete", message)
            # Optionally, close the dialog on success or offer to transcode with another profile
            # For now, user can close manually or start another.
            # self.accept() # To close dialog on success
        else:
            QMessageBox.critical(self, "Transcode Error", message)

        self.worker = None # Clean up worker

    def cancel_or_close(self):
        if self.is_transcoding and self.worker:
            if self.worker.isRunning():
                reply = QMessageBox.question(self, "Confirm Cancel",
                                             "Are you sure you want to cancel the ongoing transcoding operation?",
                                             QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                             QMessageBox.StandardButton.No)
                if reply == QMessageBox.StandardButton.Yes:
                    self.worker.requestInterruption()
                    self.status_label.setText("Cancelling transcoding operation...")
                    # Actual cancellation and cleanup happens in on_transcoding_finished
                    # or when the worker thread acknowledges interruption.
        else:
            self.reject() # Close the dialog if not transcoding

    def closeEvent(self, event):
        """Handle attempts to close the dialog while transcoding."""
        if self.is_transcoding and self.worker and self.worker.isRunning():
            reply = QMessageBox.question(self, "Confirm Exit",
                                         "Transcoding is in progress. Are you sure you want to exit?",
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                         QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                self.worker.requestInterruption()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

if __name__ == "__main__":
    app = QApplication([])

    # Mock World for testing dialog
    class MockWorld:
        def __init__(self, name="test_transcode_world"):
            self.name = name
            print(f"MockWorld '{self.name}' initialized for TranscodeAssetDialog.")

        async def dispatch_event(self, event): # If using event-based system
            print(f"MockWorld: Event dispatched: {type(event).__name__}")
            await asyncio.sleep(0.1)
            # Simulate event processing, set event.result or event.error
            # event.result = {"new_entity_id": 999}

    mock_world_instance = MockWorld()

    # Example entity info
    test_entity_id = 123
    test_entity_filename = "example_asset.jpg"

    dialog = TranscodeAssetDialog(
        world=mock_world_instance,
        entity_id=test_entity_id,
        entity_filename=test_entity_filename
    )
    dialog.show()
    app.exec()
