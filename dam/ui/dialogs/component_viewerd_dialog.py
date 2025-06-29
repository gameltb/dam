# Actual imports for fetching data
# from dam.models.core.entity import Entity # Not directly needed if using service functions
# Added for type hinting, Optional already imported from typing
from typing import Any, Dict
from typing import List as TypingList

from PyQt6.QtWidgets import (
    QDialog,
    QHeaderView,  # Added QTreeWidget, QTreeWidgetItem, QHeaderView
    QTreeWidget,
    QTreeWidgetItem,
)
from PyQt6.QtWidgets import (
    QPushButton as QDialogButton,
)
from PyQt6.QtWidgets import (
    QVBoxLayout as QDialogVBoxLayout,
)

# Removed QScrollArea, QTextEdit as they are no longer used.


class ComponentViewerDialog(QDialog):
    def __init__(
        self, entity_id: int, components_data: Dict[str, TypingList[Dict[str, Any]]], world_name: str, parent=None
    ):
        super().__init__(parent)
        self.setWindowTitle(f"Components for Entity ID: {entity_id} (World: {world_name})")
        self.setGeometry(200, 200, 700, 500)

        layout = QDialogVBoxLayout(self)

        self.tree_widget = QTreeWidget()
        self.tree_widget.setColumnCount(2)
        self.tree_widget.setHeaderLabels(["Property", "Value"])

        # self._populate_tree(entity_id, components_data, world_name) # Call to populate will be in next step

        layout.addWidget(self.tree_widget)

        close_button = QDialogButton("Close")
        close_button.clicked.connect(self.accept)
        layout.addWidget(close_button)

        self.setLayout(layout)
        self._populate_tree(entity_id, components_data, world_name)  # Call to populate the tree

    def _add_attribute_to_tree(self, parent_item: QTreeWidgetItem, key: str, value: Any):
        """
        Recursively adds attributes to the tree.
        Handles nested dictionaries and lists.
        """
        if isinstance(value, dict):
            dict_item = QTreeWidgetItem(parent_item, [str(key), "(dict)"])
            for sub_key, sub_value in value.items():
                self._add_attribute_to_tree(dict_item, sub_key, sub_value)
        elif isinstance(value, list):
            list_item = QTreeWidgetItem(parent_item, [str(key), f"(list: {len(value)} items)"])
            for index, item_value in enumerate(value):
                # For list items, use index as key or a generic "item" prefix
                self._add_attribute_to_tree(list_item, f"[{index}]", item_value)
        else:
            QTreeWidgetItem(parent_item, [str(key), str(value)])

    def _populate_tree(self, entity_id: int, components_data: Dict[str, TypingList[Dict[str, Any]]], world_name: str):
        self.tree_widget.clear()  # Clear any existing items

        if not components_data:
            QTreeWidgetItem(self.tree_widget, ["No components found for this entity."])
            return

        root_item_text = f"Entity ID: {entity_id} (World: {world_name})"
        # Create a top-level item for the entity itself, spanning both columns for its main display
        entity_root_item = QTreeWidgetItem(self.tree_widget, [root_item_text, ""])
        # self.tree_widget.addTopLevelItem(entity_root_item) # Already added by parenting

        for comp_type_name, instances_list in components_data.items():
            if not instances_list:  # If there are no instances for this component type, skip it
                continue

            comp_type_item = QTreeWidgetItem(entity_root_item, [comp_type_name, f"({len(instances_list)} instance(s))"])

            # Since we checked instances_list is not empty above, this else is always taken if we reach here.
            # The 'if not instances_list:' check inside the loop is now redundant if we skip empty lists above.
            # However, the original structure was to show the type then "(No instances...)".
            # The new requirement is to not show the type at all if no instances.
            # So, the check `if not instances_list:` above handles this.

            # The following loop will only execute if instances_list is not empty.
            for idx, instance_data_dict in enumerate(instances_list):
                # If only one instance, or no 'id' in instance_data, attributes go under comp_type_item directly.
                # Otherwise, create an intermediate item for the instance.
                parent_for_attributes = comp_type_item
                instance_id = instance_data_dict.get("id")  # Assuming 'id' might be a key in component data

                if len(instances_list) > 1 or instance_id is not None:
                    instance_label = f"Instance {instance_id if instance_id is not None else idx + 1}"
                    instance_item = QTreeWidgetItem(comp_type_item, [instance_label, ""])
                    parent_for_attributes = instance_item

                for attr_key, attr_value in instance_data_dict.items():
                    self._add_attribute_to_tree(parent_for_attributes, attr_key, attr_value)

        # Expand top-level items (Entity ID, Component Types)
        # Expand the main entity item and its direct children (component types)
        if entity_root_item:  # Check if entity_root_item was created (i.e. components_data was not empty)
            entity_root_item.setExpanded(True)
            for i in range(entity_root_item.childCount()):
                entity_root_item.child(i).setExpanded(True)  # Expand component type items

        # Adjust column widths
        self.tree_widget.header().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.tree_widget.header().setStretchLastSection(True)  # Make the 'Value' column stretch
        # Alternatively, for more control if more columns are added:
        # self.tree_widget.header().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
