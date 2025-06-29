import asyncio
import uuid
from typing import Optional

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar, # For potential progress display during setup or short eval
    QPushButton,
    QVBoxLayout,
)

from dam.core.world import World
# from dam.services.evaluation_service import EvaluationService # Or event based
from dam.ui.dialogs.evaluation_result_dialog import EvaluationResultDialog # To show results

# Placeholder for actual evaluation logic
class EvaluationWorker(QThread):
    finished = pyqtSignal(bool, str, object)  # success (bool), message (str), result_data (object)
    progress = pyqtSignal(int, str)

    def __init__(self, world: World, entity_id_original: int, entity_id_transcoded: int):
        super().__init__()
        self.world = world
        self.entity_id_original = entity_id_original
        self.entity_id_transcoded = entity_id_transcoded

    def run(self):
        try:
            self.progress.emit(0, f"Starting evaluation between Entity ID {self.entity_id_original} and {self.entity_id_transcoded}...")

            # TODO: Replace with actual call to EvaluationService or event dispatch
            # For example:
            # eval_event = EvaluationRequested(
            #     entity_id_original=self.entity_id_original,
            #     entity_id_transcoded=self.entity_id_transcoded,
            #     request_id=str(uuid.uuid4()),
            #     world_name=self.world.name
            # )
            # asyncio.run(self.world.dispatch_event(eval_event))
            #
            # if eval_event.error:
            #    raise Exception(eval_event.error)
            # evaluation_results = eval_event.result # This would be the data for EvaluationResultDialog

            # Simulate work
            for i in range(101):
                if self.isInterruptionRequested():
                    self.finished.emit(False, "Evaluation cancelled.", None)
                    return
                QThread.msleep(30)
                self.progress.emit(i, f"Evaluating... {i}%")

            # Simulate success with some dummy result data
            dummy_result_data = {
                "original_entity_id": self.entity_id_original,
                "transcoded_entity_id": self.entity_id_transcoded,
                "metrics": {
                    "PSNR": 35.6,
                    "SSIM": 0.987,
                    "VMAF": 92.1 # If applicable
                },
                "notes": "Evaluation completed successfully."
            }
            self.finished.emit(True, "Evaluation successful.", dummy_result_data)

        except Exception as e:
            error_message = f"Error during evaluation: {e}"
            self.finished.emit(False, error_message, None)


class EvaluationSetupDialog(QDialog):
    def __init__(self, world: World, parent=None):
        super().__init__(parent)
        self.world = world
        self.setWindowTitle("Setup Transcoding Evaluation")
        self.setMinimumWidth(400)

        layout = QVBoxLayout(self)
        form_layout = QFormLayout()

        self.original_asset_id_input = QLineEdit()
        self.original_asset_id_input.setPlaceholderText("Enter Entity ID of original/reference asset")
        form_layout.addRow(QLabel("Original Asset ID:"), self.original_asset_id_input)

        self.transcoded_asset_id_input = QLineEdit()
        self.transcoded_asset_id_input.setPlaceholderText("Enter Entity ID of transcoded/comparison asset")
        form_layout.addRow(QLabel("Transcoded Asset ID:"), self.transcoded_asset_id_input)

        # TODO: Add fields for evaluation parameters if needed

        layout.addLayout(form_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("")
        self.status_label.setVisible(False)
        layout.addWidget(self.status_label)

        button_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Evaluation")
        self.start_button.clicked.connect(self.start_evaluation)

        self.cancel_button = QPushButton("Cancel Operation")
        self.cancel_button.clicked.connect(self.cancel_or_close)

        button_layout.addStretch()
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)

        self.worker: Optional[EvaluationWorker] = None
        self.is_evaluating = False

    def start_evaluation(self):
        if self.is_evaluating:
            return

        try:
            original_id = int(self.original_asset_id_input.text().strip())
            transcoded_id = int(self.transcoded_asset_id_input.text().strip())
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Entity IDs must be valid integers.")
            return

        if original_id <= 0 or transcoded_id <= 0:
            QMessageBox.warning(self, "Input Error", "Entity IDs must be positive integers.")
            return
        if original_id == transcoded_id:
            QMessageBox.warning(self, "Input Error", "Original and transcoded asset IDs cannot be the same.")
            return

        self.is_evaluating = True
        self.start_button.setEnabled(False)
        self.original_asset_id_input.setEnabled(False)
        self.transcoded_asset_id_input.setEnabled(False)
        self.cancel_button.setText("Cancel Operation")
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.status_label.setText("Preparing for evaluation...")
        self.status_label.setVisible(True)

        self.worker = EvaluationWorker(
            world=self.world,
            entity_id_original=original_id,
            entity_id_transcoded=transcoded_id
        )
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.on_evaluation_finished)
        self.worker.start()

    def update_progress(self, value: int, message: str):
        self.progress_bar.setValue(value)
        self.status_label.setText(message)

    def on_evaluation_finished(self, success: bool, message: str, result_data: Optional[dict]):
        self.is_evaluating = False
        self.start_button.setEnabled(True)
        self.original_asset_id_input.setEnabled(True)
        self.transcoded_asset_id_input.setEnabled(True)
        self.cancel_button.setText("Close")
        self.progress_bar.setVisible(False)
        self.status_label.setText(message)

        if success and result_data:
            # QMessageBox.information(self, "Evaluation Complete", message) # No longer needed, result dialog shows info
            # Open EvaluationResultDialog
            result_dialog = EvaluationResultDialog(
                world_name=self.world.name,
                evaluation_data=result_data,
                parent=self.parent() # Or self, depending on desired ownership and modality
            )
            result_dialog.exec()
            self.accept() # Close setup dialog after result dialog is shown and closed
        else:
            QMessageBox.critical(self, "Evaluation Error", message)

        self.worker = None

    def cancel_or_close(self):
        if self.is_evaluating and self.worker:
            if self.worker.isRunning():
                if QMessageBox.question(self, "Confirm Cancel",
                                     "Cancel ongoing evaluation?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                     QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes:
                    self.worker.requestInterruption()
                    self.status_label.setText("Cancelling evaluation...")
        else:
            self.reject()

    def closeEvent(self, event):
        if self.is_evaluating and self.worker and self.worker.isRunning():
            if QMessageBox.question(self, "Confirm Exit",
                                 "Evaluation in progress. Exit anyway?",
                                 QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                 QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes:
                self.worker.requestInterruption()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

if __name__ == "__main__":
    app = QApplication([])
    class MockWorld: # Same as in TranscodeAssetDialog
        def __init__(self, name="test_eval_world"): self.name = name
        async def dispatch_event(self, event):
            print(f"MockWorld: Event dispatched: {type(event).__name__}")
            await asyncio.sleep(0.1)

    mock_world_instance = MockWorld()
    dialog = EvaluationSetupDialog(world=mock_world_instance)
    dialog.show()
    app.exec()
