import json  # For pretty printing component data

# Actual imports for fetching data
# from dam.models.core.entity import Entity # Not directly needed if using service functions
# Added for type hinting, Optional already imported from typing
from typing import Any, Dict
from typing import List as TypingList

from PyQt6.QtWidgets import (
    QDialog,
    QScrollArea,
    QTextEdit,
)
from PyQt6.QtWidgets import (
    QPushButton as QDialogButton,  # For dialog forms
)
from PyQt6.QtWidgets import (
    QVBoxLayout as QDialogVBoxLayout,
)


class ComponentViewerDialog(QDialog):
    def __init__(
        self, entity_id: int, components_data: Dict[str, TypingList[Dict[str, Any]]], world_name: str, parent=None
    ):
        super().__init__(parent)
        self.setWindowTitle(f"Components for Entity ID: {entity_id} (World: {world_name})")
        self.setGeometry(200, 200, 700, 500)  # Adjusted size

        layout = QDialogVBoxLayout(self)  # Use alias for clarity

        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)

        formatted_text = f"Entity ID: {entity_id}\nWorld: {world_name}\n\nComponents:\n"
        formatted_text += "=" * 20 + "\n\n"

        if not components_data:
            formatted_text += "No components found for this entity."
        else:
            for comp_name, comp_list in components_data.items():
                formatted_text += f"--- {comp_name} ---\n"
                if not comp_list:
                    formatted_text += "  (No instances of this component type)\n"
                else:
                    for i, comp_data in enumerate(comp_list):
                        # Pretty print each component's data
                        try:
                            # Exclude SQLAlchemy internal state if present
                            data_to_print = {k: v for k, v in comp_data.items() if not k.startswith("_sa_")}
                            formatted_text += json.dumps(
                                data_to_print, indent=2, default=str
                            )  # default=str for non-serializable
                        except Exception as e:
                            formatted_text += f"  Error formatting component: {e}\n"
                            # Fallback to string representation if JSON fails
                            formatted_text += str(comp_data)
                        formatted_text += "\n"
                        if i < len(comp_list) - 1:
                            formatted_text += "-\n"  # Separator for multiple instances of same component type
                formatted_text += "\n"

        self.text_edit.setText(formatted_text)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.text_edit)
        layout.addWidget(scroll_area)

        close_button = QDialogButton("Close")  # Use alias
        close_button.clicked.connect(self.accept)
        layout.addWidget(close_button)

        self.setLayout(layout)
