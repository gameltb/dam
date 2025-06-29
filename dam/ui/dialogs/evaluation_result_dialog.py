import json
from typing import Any, Dict

from PyQt6.QtWidgets import (
    QDialog,
    QScrollArea,
    QTextEdit,
    QPushButton,
    QVBoxLayout,
)

from dam.core.world import World # May not be strictly needed if all data is passed in


class EvaluationResultDialog(QDialog):
    def __init__(self, world_name: str, evaluation_data: Dict[str, Any], parent=None):
        super().__init__(parent)
        # self.world = world # Not strictly needed if world_name is sufficient for title
        self.world_name = world_name
        self.evaluation_data = evaluation_data

        self.setWindowTitle(f"Transcoding Evaluation Result (World: {self.world_name})")
        self.setGeometry(250, 250, 600, 450)

        layout = QVBoxLayout(self)

        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)

        formatted_text = f"Evaluation Results (World: {self.world_name}):\n"
        formatted_text += "=" * 30 + "\n\n"

        if not self.evaluation_data:
            formatted_text += "No evaluation data provided or evaluation failed to produce results."
        else:
            try:
                # Pretty print the evaluation data dictionary
                formatted_text += json.dumps(self.evaluation_data, indent=2, default=str)
            except Exception as e:
                formatted_text += f"Error formatting evaluation data: {e}\n"
                formatted_text += str(self.evaluation_data) # Fallback

        self.text_edit.setText(formatted_text)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.text_edit)
        layout.addWidget(scroll_area)

        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        layout.addWidget(close_button)

        self.setLayout(layout)

if __name__ == "__main__":
    from PyQt6.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)

    # Example evaluation data
    mock_eval_data = {
        "original_entity_id": 101,
        "transcoded_entity_id": 202,
        "comparison_type": "Original vs. Transcode",
        "evaluation_profile": "Generic Quality Profile v1",
        "metrics": {
            "PSNR": {
                "value": 38.5,
                "unit": "dB"
            },
            "SSIM": {
                "value": 0.991,
                "unit": ""
            },
            "VMAF": { # Video specific, but example
                "value": 95.2,
                "unit": "",
                "notes": "Calculated on first 10 seconds"
            },
            "FileSizeReduction": {
                "value": 75.5,
                "unit": "%"
            }
        },
        "status": "Completed",
        "evaluation_timestamp": "2023-10-26T14:30:00Z",
        "notes": "Transcoded asset shows excellent quality with significant size reduction."
    }

    mock_world_name = "test_eval_results_world"

    dialog = EvaluationResultDialog(world_name=mock_world_name, evaluation_data=mock_eval_data)
    dialog.show()
    sys.exit(app.exec())
