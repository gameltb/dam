import sys
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
    QVBoxLayout,
    QWidget,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QHBoxLayout,
    QMessageBox,
    QLineEdit,
    QComboBox,
    QDialog, # For component viewer
    QTextEdit, # To display component data
    QVBoxLayout as QDialogVBoxLayout, # Alias to avoid conflict if used elsewhere
    QScrollArea, # For scrollable component view
    QPushButton as QDialogButton, # Alias for dialog buttons
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction

# Actual imports for fetching data
from dam.core.world import World
from dam.services import ecs_service
from dam.models.properties.file_properties_component import FilePropertiesComponent
# from dam.models.core.entity import Entity # Not directly needed if using service functions

# Added for type hinting, Optional already imported from typing
from typing import Optional, Dict, Any, List as TypingList

from dam.models.core.base_component import REGISTERED_COMPONENT_TYPES
import json # For pretty printing component data

class ComponentViewerDialog(QDialog):
    def __init__(self, entity_id: int, components_data: Dict[str, TypingList[Dict[str, Any]]], world_name: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Components for Entity ID: {entity_id} (World: {world_name})")
        self.setGeometry(200, 200, 700, 500) # Adjusted size

        layout = QDialogVBoxLayout(self) # Use alias for clarity

        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)

        formatted_text = f"Entity ID: {entity_id}\nWorld: {world_name}\n\nComponents:\n"
        formatted_text += "="*20 + "\n\n"

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
                            data_to_print = {k: v for k, v in comp_data.items() if not k.startswith('_sa_')}
                            formatted_text += json.dumps(data_to_print, indent=2, default=str) # default=str for non-serializable
                        except Exception as e:
                            formatted_text += f"  Error formatting component: {e}\n"
                            # Fallback to string representation if JSON fails
                            formatted_text += str(comp_data)
                        formatted_text += "\n"
                        if i < len(comp_list) - 1:
                             formatted_text += "-\n" # Separator for multiple instances of same component type
                formatted_text += "\n"

        self.text_edit.setText(formatted_text)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.text_edit)
        layout.addWidget(scroll_area)

        close_button = QDialogButton("Close") # Use alias
        close_button.clicked.connect(self.accept)
        layout.addWidget(close_button)

        self.setLayout(layout)


class MainWindow(QMainWindow):
    def __init__(self, current_world: Optional[World] = None):
        super().__init__()

        self.current_world = current_world
        self.setWindowTitle(f"DAM UI - World: {current_world.name if current_world else 'No World Selected'}")
        self.setGeometry(100, 100, 800, 600)

        self._create_menus()
        self._create_status_bar()
        self._create_central_widget()

        # Store the world instance if needed for data fetching
        # self.current_world = None # Placeholder, needs to be set, e.g. via CLI or a selector

    def _create_menus(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("&File")
        exit_action = QAction("&Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Placeholder for other menus
        menu_bar.addMenu("&Edit")
        view_menu = menu_bar.addMenu("&View")
        refresh_action = QAction("&Refresh Assets", self)
        refresh_action.triggered.connect(self.load_assets) # Connect to asset loading
        view_menu.addAction(refresh_action)
        menu_bar.addMenu("&Help")


    def _create_status_bar(self):
        self.statusBar().showMessage("Ready")

    def _create_central_widget(self):
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)

        # Top bar for controls
        controls_layout = QHBoxLayout()

        # Search input
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search by filename...")
        self.search_input.returnPressed.connect(self.load_assets) # Trigger search on Enter
        controls_layout.addWidget(self.search_input)

        search_button = QPushButton("Search")
        search_button.clicked.connect(self.load_assets)
        controls_layout.addWidget(search_button)

        # MIME type filter
        controls_layout.addWidget(QLabel("Filter by Type:"))
        self.mime_type_filter = QComboBox()
        self.mime_type_filter.setFixedWidth(150) # Adjust width as needed
        self.mime_type_filter.addItem("All Types", "") # User data is empty string for no filter
        # self.populate_mime_type_filter() # Call this after world is confirmed
        self.mime_type_filter.currentIndexChanged.connect(self.load_assets)
        controls_layout.addWidget(self.mime_type_filter)

        self.refresh_button = QPushButton("Refresh All") # Renamed from "Clear Search" for clarity
        self.refresh_button.clicked.connect(self._clear_filters_and_refresh)
        controls_layout.addWidget(self.refresh_button)

        controls_layout.addStretch()
        main_layout.addLayout(controls_layout)

        # Asset list widget
        self.asset_list_widget = QListWidget()
        self.asset_list_widget.itemDoubleClicked.connect(self.on_asset_double_clicked)
        main_layout.addWidget(self.asset_list_widget)

        self.setCentralWidget(central_widget)
        self.populate_mime_type_filter() # Populate filter once after UI setup
        self.load_assets() # Initial load

    def _clear_filters_and_refresh(self):
        self.search_input.clear()
        self.mime_type_filter.setCurrentIndex(0) # Reset to "All Types"
        self.load_assets()

    def populate_mime_type_filter(self):
        self.mime_type_filter.blockSignals(True) # Block signals during population
        # Store current selection
        current_filter_value = self.mime_type_filter.currentData()

        self.mime_type_filter.clear()
        self.mime_type_filter.addItem("All Types", "") # Add default "All Types"

        if self.current_world:
            try:
                with self.current_world.get_db_session() as session:
                    distinct_mime_types = session.query(FilePropertiesComponent.mime_type).\
                                                filter(FilePropertiesComponent.mime_type.isnot(None)).\
                                                distinct().order_by(FilePropertiesComponent.mime_type).all()
                    for (mime_type,) in distinct_mime_types:
                        if mime_type: # Ensure not empty string if that's possible from DB
                            self.mime_type_filter.addItem(mime_type, mime_type)

                # Restore previous selection if possible
                index_to_set = self.mime_type_filter.findData(current_filter_value)
                if index_to_set != -1:
                    self.mime_type_filter.setCurrentIndex(index_to_set)
                else:
                    self.mime_type_filter.setCurrentIndex(0) # Default to "All Types"

            except Exception as e:
                print(f"Error populating MIME type filter: {e}") # Log or show error
                QMessageBox.warning(self, "Filter Error", f"Could not populate MIME type filter: {e}")
        self.mime_type_filter.blockSignals(False) # Unblock signals

    def load_assets(self):
        search_term = self.search_input.text().strip().lower()
        selected_mime_type = self.mime_type_filter.currentData() # Get data (actual MIME type string)

        status_message_parts = ["Loading assets"]
        if search_term:
            status_message_parts.append(f"searching for '{search_term}'")
        if selected_mime_type:
            status_message_parts.append(f"filtered by type '{selected_mime_type}'")

        self.statusBar().showMessage("... ".join(status_message_parts) + "...")
        self.asset_list_widget.clear()

        if not self.current_world:
            self.statusBar().showMessage("Error: No world selected.")
            QMessageBox.warning(self, "Error", "No DAM world is currently selected. Cannot load assets.")
            self.asset_list_widget.addItem("No world selected.")
            return

        try:
            with self.current_world.get_db_session() as session:
                query = session.query(FilePropertiesComponent.entity_id, FilePropertiesComponent.original_filename, FilePropertiesComponent.mime_type)

                if search_term:
                    query = query.filter(FilePropertiesComponent.original_filename.ilike(f"%{search_term}%"))

                if selected_mime_type: # Filter by MIME type if one is selected
                    query = query.filter(FilePropertiesComponent.mime_type == selected_mime_type)

                assets_found = query.all()

                if not assets_found:
                    message = "No assets found."
                    if search_term or selected_mime_type:
                        message_parts = ["No assets found"]
                        if search_term: message_parts.append(f"matching '{search_term}'")
                        if selected_mime_type: message_parts.append(f"of type '{selected_mime_type}'")
                        message = " ".join(message_parts) + "."
                    self.asset_list_widget.addItem(message)
                    self.statusBar().showMessage(message)
                    return

                for entity_id, original_filename, mime_type_val in assets_found:
                    display_text = f"ID: {entity_id} - {original_filename} ({mime_type_val or 'N/A'})"
                    item = QListWidgetItem(display_text)
                    item.setData(Qt.ItemDataRole.UserRole, entity_id)
                    self.asset_list_widget.addItem(item)

                num_found = len(assets_found)
                loaded_message_parts = [f"Loaded {num_found} asset{'s' if num_found != 1 else ''}"]
                if search_term: loaded_message_parts.append(f"matching '{search_term}'")
                if selected_mime_type: loaded_message_parts.append(f"of type '{selected_mime_type}'")
                loaded_message_parts.append(f"from world '{self.current_world.name}'.")
                self.statusBar().showMessage(" ".join(loaded_message_parts))

        except Exception as e:
            self.statusBar().showMessage(f"Error loading assets: {e}")
            QMessageBox.critical(self, "Load Assets Error", f"Could not load assets from world '{self.current_world.name}':\n{e}")
            # Consider logging the full traceback for debugging
            # import traceback
            # print(traceback.format_exc())
            self.asset_list_widget.addItem(f"Error loading assets: {e}")


    def on_asset_double_clicked(self, item: QListWidgetItem):
        asset_id = item.data(Qt.ItemDataRole.UserRole)
        if asset_id is None: # Should not happen if data is set correctly
            QMessageBox.warning(self, "Error", "No asset ID associated with this item.")
            return

        if not self.current_world:
            QMessageBox.warning(self, "Error", "No world selected. Cannot view components.")
            return

        self.statusBar().showMessage(f"Fetching components for Entity ID: {asset_id}...")

        components_data_for_dialog: Dict[str, TypingList[Dict[str, Any]]] = {}
        try:
            with self.current_world.get_db_session() as session:
                entity = ecs_service.get_entity(session, asset_id)
                if not entity:
                    QMessageBox.warning(self, "Error", f"Entity ID {asset_id} not found in world '{self.current_world.name}'.")
                    self.statusBar().showMessage(f"Entity ID {asset_id} not found.")
                    return

                for comp_type_name, comp_type_cls in REGISTERED_COMPONENT_TYPES.items():
                    components = ecs_service.get_components(session, asset_id, comp_type_cls)
                    component_instances_data = []
                    if components:
                        for comp_instance in components:
                            # Convert component instance to a dictionary
                            # This is a basic conversion; more sophisticated serialization might be needed
                            # for complex types or relationships within components.
                            instance_data = {c.key: getattr(comp_instance, c.key)
                                             for c in comp_instance.__table__.columns
                                             if not c.key.startswith('_')} # Exclude SQLAlchemy internals
                            component_instances_data.append(instance_data)
                    components_data_for_dialog[comp_type_name] = component_instances_data

            if not components_data_for_dialog:
                 self.statusBar().showMessage(f"No components found for Entity ID: {asset_id}.")
            else:
                self.statusBar().showMessage(f"Successfully fetched components for Entity ID: {asset_id}.")

            dialog = ComponentViewerDialog(asset_id, components_data_for_dialog, self.current_world.name, self)
            dialog.exec()

        except Exception as e:
            error_msg = f"Error fetching components for Entity ID {asset_id}: {e}"
            self.statusBar().showMessage(error_msg)
            QMessageBox.critical(self, "Component Fetch Error", error_msg)
            # import traceback; print(traceback.format_exc()) # For debugging


if __name__ == "__main__":
    # This part needs to be adjusted if the UI is launched via CLI
    # The CLI will handle QApplication instance.
    # For direct execution (testing main_window.py):
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)

    # Here you would ideally get the world instance if needed for the UI
    # For example, if a default world is configured:
    # from dam.core.config import settings
    # from dam.core.world_setup import register_core_systems
    #
    # if settings.DEFAULT_WORLD_NAME:
    #     world_to_use = get_world(settings.DEFAULT_WORLD_NAME)
    #     if not world_to_use: # If not initialized by CLI context
    #         # Minimal setup for standalone run - this is tricky and might not fully replicate CLI env
    #         from dam.core.world import create_and_register_all_worlds_from_settings
    #         create_and_register_all_worlds_from_settings(app_settings=settings)
    #         world_to_use = get_world(settings.DEFAULT_WORLD_NAME)
    #         if world_to_use:
    #             register_core_systems(world_to_use) # Register systems for this world
    # else:
    #     world_to_use = None
    #
    # window = MainWindow()
    # if world_to_use:
    #     window.current_world = world_to_use # Pass the world to the window
    #     print(f"Running UI with world: {world_to_use.name}")
    # else:
    #     print("Running UI without a pre-selected world (will use dummy data or require selection).")

    # For standalone testing, we might not have a world, or we might try to load a default one.
    # This part is primarily for when `python dam/ui/main_window.py` is run directly.
    # The CLI will pass the world instance.
    active_world = None
    # Example: try to load default world if running standalone for testing
    # This is a simplified setup and might need more robust error handling or configuration.
    # if not active_world: # If not passed by CLI or set above
    #     try:
    #         from dam.core import config as app_config
    #         from dam.core.world import get_world, create_and_register_all_worlds_from_settings
    #         from dam.core.world_setup import register_core_systems
    #
    #         # Ensure worlds are initialized from settings
    #         initialized_worlds = create_and_register_all_worlds_from_settings(app_settings=app_config.settings)
    #         for world_instance in initialized_worlds:
    #             register_core_systems(world_instance)
    #
    #         if app_config.settings.DEFAULT_WORLD_NAME:
    #             active_world = get_world(app_config.settings.DEFAULT_WORLD_NAME)
    #         elif initialized_worlds:
    #             active_world = initialized_worlds[0] # Fallback to the first initialized world
    #
    #         if active_world:
    #             print(f"Standalone: Using world '{active_world.name}' for UI.")
    #         else:
    #             print("Standalone: No world could be automatically determined. UI may have limited functionality.")
    #     except Exception as e:
    #         print(f"Standalone: Error initializing default world: {e}")

    window = MainWindow(current_world=active_world) # Pass active_world (could be None)
    window.show()
    sys.exit(app.exec())
