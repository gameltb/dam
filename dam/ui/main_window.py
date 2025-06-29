import sys

# from dam.models.core.entity import Entity # Not directly needed if using service functions
# Added for type hinting, Optional already imported from typing
from typing import Any, Dict, Optional
from typing import List as TypingList

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from dam.core.config import settings as app_settings  # For getting all world names

# Actual imports for fetching data
from dam.core.world import World
from dam.models.core.base_component import REGISTERED_COMPONENT_TYPES
from dam.models.properties.file_properties_component import FilePropertiesComponent
from dam.services import ecs_service
from dam.ui.dialogs.add_asset_dialog import AddAssetDialog
from dam.ui.dialogs.component_viewerd_ialog import ComponentViewerDialog
from dam.ui.dialogs.find_asset_by_hash_dialog import FindAssetByHashDialog
from dam.ui.dialogs.find_similar_images_dialog import FindSimilarImagesDialog, _pil_available
from dam.ui.dialogs.world_operations_dialogs import (
    ExportWorldDialog,
    ImportWorldDialog,
    MergeWorldsDialog,
    SplitWorldDialog,
)
from dam.ui.dialogs.transcode_asset_dialog import TranscodeAssetDialog
from dam.ui.dialogs.evaluation_setup_dialog import EvaluationSetupDialog


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

        # File Menu
        file_menu = menu_bar.addMenu("&File")
        add_asset_action = QAction("&Add Asset(s)...", self)
        add_asset_action.triggered.connect(self.open_add_asset_dialog)
        file_menu.addAction(add_asset_action)
        file_menu.addSeparator()

        find_by_hash_action = QAction("Find Asset by &Hash...", self)
        find_by_hash_action.triggered.connect(self.open_find_asset_by_hash_dialog)
        file_menu.addAction(find_by_hash_action)

        find_similar_action = QAction("Find &Similar Images...", self)
        find_similar_action.triggered.connect(self.open_find_similar_images_dialog)
        file_menu.addAction(find_similar_action)

        file_menu.addSeparator()

        exit_action = QAction("&Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Edit Menu (Placeholder)
        edit_menu = menu_bar.addMenu("&Edit")  # Keep reference if items added later

        # View Menu
        view_menu = menu_bar.addMenu("&View")
        refresh_action = QAction("&Refresh Asset List", self)
        refresh_action.triggered.connect(self._clear_filters_and_refresh)
        view_menu.addAction(refresh_action)

        # Tools Menu
        tools_menu = menu_bar.addMenu("&Tools")
        export_world_action = QAction("&Export Current World...", self)
        export_world_action.triggered.connect(self.open_export_world_dialog)
        tools_menu.addAction(export_world_action)

        import_world_action = QAction("&Import Data into Current World...", self)
        import_world_action.triggered.connect(self.open_import_world_dialog)
        tools_menu.addAction(import_world_action)
        tools_menu.addSeparator()

        merge_worlds_action = QAction("&Merge Worlds...", self)
        merge_worlds_action.triggered.connect(self.open_merge_worlds_dialog)
        tools_menu.addAction(merge_worlds_action)

        split_world_action = QAction("&Split Current World...", self)
        split_world_action.triggered.connect(self.open_split_world_dialog)
        tools_menu.addAction(split_world_action)
        tools_menu.addSeparator()

        setup_db_action = QAction("Setup &Database for Current World...", self)
        setup_db_action.triggered.connect(self.setup_current_world_db)
        tools_menu.addAction(setup_db_action)
        tools_menu.addSeparator()

        transcode_asset_action = QAction("&Transcode Selected Asset...", self)
        transcode_asset_action.triggered.connect(self.open_transcode_asset_dialog)
        tools_menu.addAction(transcode_asset_action)
        # self.transcode_asset_action = transcode_asset_action # Store if we need to enable/disable it
        tools_menu.addSeparator()

        evaluation_setup_action = QAction("Setup Transcoding &Evaluation...", self)
        evaluation_setup_action.triggered.connect(self.open_evaluation_setup_dialog)
        tools_menu.addAction(evaluation_setup_action)

        # Help Menu (Placeholder)
        help_menu = menu_bar.addMenu("&Help")

    def open_add_asset_dialog(self):
        if not self.current_world:
            QMessageBox.warning(self, "No World", "Please select or configure a DAM world first.")
            return

        dialog = AddAssetDialog(current_world=self.current_world, parent=self)
        if dialog.exec():
            self.statusBar().showMessage("Asset ingestion process completed. Refreshing asset list...")
            self.load_assets()
        else:
            self.statusBar().showMessage("Add asset operation cancelled or failed.")

    def open_find_asset_by_hash_dialog(self):
        if not self.current_world:
            QMessageBox.warning(self, "No World", "Please select or configure a DAM world first.")
            return

        dialog = FindAssetByHashDialog(current_world=self.current_world, parent=self)
        # The dialog itself handles showing results or further dialogs (like ComponentViewer)
        # so we just need to exec() it.
        dialog.exec()
        # Optionally, refresh main list if find operation could alter state, but usually not.

    def open_find_similar_images_dialog(self):
        if not self.current_world:
            QMessageBox.warning(self, "No World", "Please select or configure a DAM world first.")
            return

        if not _pil_available:  # Check if Pillow is available for image previews
            QMessageBox.warning(
                self,
                "Dependency Missing",
                "Pillow (PIL) is not installed. Image previews will not be available in the find similar images dialog. "
                "Please install 'ecs-dam-system[image]' for full functionality.",
            )

        dialog = FindSimilarImagesDialog(current_world=self.current_world, parent=self)
        dialog.exec()

    def open_export_world_dialog(self):
        if not self.current_world:
            QMessageBox.warning(self, "No World", "Please select or configure a DAM world first to export.")
            return
        dialog = ExportWorldDialog(current_world=self.current_world, parent=self)
        dialog.exec()
        # Status updates are handled by the dialog and its worker thread

    def open_import_world_dialog(self):
        if not self.current_world:
            QMessageBox.warning(self, "No World", "Please select or configure a DAM world first to import into.")
            return
        dialog = ImportWorldDialog(current_world=self.current_world, parent=self)
        if dialog.exec():  # Dialog's accept() called on success
            self.statusBar().showMessage("Import process completed. Refreshing asset list...")
            self.load_assets()  # Refresh list as data might have changed
        else:
            self.statusBar().showMessage("Import operation cancelled or failed.")

    def open_merge_worlds_dialog(self):
        if not self.current_world:
            QMessageBox.warning(self, "No World", "A current world must be active to serve as the merge target.")
            return
        all_world_names = [w.name for w in app_settings.worlds] if app_settings.worlds else []
        if len(all_world_names) < 2:
            QMessageBox.information(
                self, "Not Enough Worlds", "At least two worlds must be configured to perform a merge operation."
            )
            return

        dialog = MergeWorldsDialog(current_world=self.current_world, all_world_names=all_world_names, parent=self)
        if dialog.exec():
            self.statusBar().showMessage("Merge process completed. Refreshing asset list...")
            self.load_assets()  # Refresh list as target world data changed
        else:
            self.statusBar().showMessage("Merge operation cancelled or failed.")

    def open_split_world_dialog(self):
        if not self.current_world:
            QMessageBox.warning(self, "No World", "A current world must be active to serve as the split source.")
            return
        all_world_names = [w.name for w in app_settings.worlds] if app_settings.worlds else []
        if len(all_world_names) < 3:  # Need source + at least two unique targets
            QMessageBox.information(
                self,
                "Not Enough Worlds",
                "At least three worlds must be configured to perform a split operation (one source, two distinct targets).",
            )
            return

        dialog = SplitWorldDialog(source_world=self.current_world, all_world_names=all_world_names, parent=self)
        if dialog.exec():
            self.statusBar().showMessage("Split process completed. Refreshing asset list...")
            self.load_assets()  # Refresh list as source world data might have changed (if delete option used)
        else:
            self.statusBar().showMessage("Split operation cancelled or failed.")

    def setup_current_world_db(self):
        if not self.current_world:
            QMessageBox.warning(self, "No World", "No current world is active to set up its database.")
            return

        reply = QMessageBox.question(
            self,
            "Confirm Database Setup",
            f"This will attempt to initialize the database and create tables for world '{self.current_world.name}'.\n"
            "This is typically done once. Proceed?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.No:
            return

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        self.statusBar().showMessage(f"Setting up database for world '{self.current_world.name}'...")
        try:
            self.current_world.create_db_and_tables()
            QApplication.restoreOverrideCursor()
            QMessageBox.information(
                self, "Database Setup Successful", f"Database setup complete for world '{self.current_world.name}'."
            )
            self.statusBar().showMessage(f"Database for '{self.current_world.name}' successfully set up.")
            self.load_assets()  # Refresh asset list, in case it was empty due to no tables
        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(
                self, "Database Setup Error", f"Error during database setup for '{self.current_world.name}':\n{e}"
            )
            self.statusBar().showMessage(f"Database setup for '{self.current_world.name}' failed: {e}")
        finally:
            if QApplication.overrideCursor() is not None:  # Ensure cursor is restored
                QApplication.restoreOverrideCursor()

    def open_transcode_asset_dialog(self):
        if not self.current_world:
            QMessageBox.warning(self, "No World", "Please select or configure a DAM world first.")
            return

        selected_items = self.asset_list_widget.selectedItems()
        if not selected_items:
            QMessageBox.information(self, "No Asset Selected", "Please select an asset from the list to transcode.")
            return
        if len(selected_items) > 1:
            QMessageBox.information(self, "Multiple Assets Selected", "Please select only one asset to transcode at a time.")
            return

        list_item = selected_items[0]
        entity_id = list_item.data(Qt.ItemDataRole.UserRole) # This is the entity_id

        # We need the original filename for display in the dialog.
        # The list_item text is "ID: {entity_id} - {original_filename} ({mime_type_val or 'N/A'})"
        # We can parse it, or better, store filename as another data role if needed frequently.
        # For now, let's try to parse it. This is a bit fragile.
        # A better way would be to query FilePropertiesComponent for the filename using entity_id.
        item_text = list_item.text()
        try:
            # Example: "ID: 123 - my_image.jpg (image/jpeg)"
            filename_part = item_text.split(" - ", 1)[1]
            original_filename = filename_part.split(" (", 1)[0]
        except IndexError:
            original_filename = f"Entity {entity_id}" # Fallback

        if entity_id is None: # Should not happen if item is valid
            QMessageBox.warning(self, "Error", "Selected asset has no ID.")
            return

        dialog = TranscodeAssetDialog(
            world=self.current_world,
            entity_id=entity_id,
            entity_filename=original_filename,
            parent=self
        )
        dialog.exec()
        # After dialog closes, we might want to refresh asset list if new assets were created
        # self.load_assets() # Or only if dialog.result() == QDialog.DialogCode.Accepted

    def open_evaluation_setup_dialog(self):
        if not self.current_world:
            QMessageBox.warning(self, "No World", "Please select or configure a DAM world first.")
            return

        dialog = EvaluationSetupDialog(world=self.current_world, parent=self)
        dialog.exec()
        # Results are shown in a subsequent dialog from EvaluationSetupDialog itself.
        # May need to refresh assets if evaluation creates new components/entities.
        # self.load_assets()


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
        self.search_input.returnPressed.connect(self.load_assets)  # Trigger search on Enter
        controls_layout.addWidget(self.search_input)

        search_button = QPushButton("Search")
        search_button.clicked.connect(self.load_assets)
        controls_layout.addWidget(search_button)

        # MIME type filter
        controls_layout.addWidget(QLabel("Filter by Type:"))
        self.mime_type_filter = QComboBox()
        self.mime_type_filter.setFixedWidth(150)  # Adjust width as needed
        self.mime_type_filter.addItem("All Types", "")  # User data is empty string for no filter
        # self.populate_mime_type_filter() # Call this after world is confirmed
        self.mime_type_filter.currentIndexChanged.connect(self.load_assets)
        controls_layout.addWidget(self.mime_type_filter)

        self.refresh_button = QPushButton("Refresh All")  # Renamed from "Clear Search" for clarity
        self.refresh_button.clicked.connect(self._clear_filters_and_refresh)
        controls_layout.addWidget(self.refresh_button)

        controls_layout.addStretch()
        main_layout.addLayout(controls_layout)

        # Asset list widget
        self.asset_list_widget = QListWidget()
        self.asset_list_widget.itemDoubleClicked.connect(self.on_asset_double_clicked)
        main_layout.addWidget(self.asset_list_widget)

        self.setCentralWidget(central_widget)
        self.populate_mime_type_filter()  # Populate filter once after UI setup
        self.load_assets()  # Initial load

    def _clear_filters_and_refresh(self):
        self.search_input.clear()
        self.mime_type_filter.setCurrentIndex(0)  # Reset to "All Types"
        self.load_assets()

    def populate_mime_type_filter(self):
        self.mime_type_filter.blockSignals(True)  # Block signals during population
        # Store current selection
        current_filter_value = self.mime_type_filter.currentData()

        self.mime_type_filter.clear()
        self.mime_type_filter.addItem("All Types", "")  # Add default "All Types"

        if self.current_world:
            try:
                with self.current_world.get_db_session() as session:
                    distinct_mime_types = (
                        session.query(FilePropertiesComponent.mime_type)
                        .filter(FilePropertiesComponent.mime_type.isnot(None))
                        .distinct()
                        .order_by(FilePropertiesComponent.mime_type)
                        .all()
                    )
                    for (mime_type,) in distinct_mime_types:
                        if mime_type:  # Ensure not empty string if that's possible from DB
                            self.mime_type_filter.addItem(mime_type, mime_type)

                # Restore previous selection if possible
                index_to_set = self.mime_type_filter.findData(current_filter_value)
                if index_to_set != -1:
                    self.mime_type_filter.setCurrentIndex(index_to_set)
                else:
                    self.mime_type_filter.setCurrentIndex(0)  # Default to "All Types"

            except Exception as e:
                print(f"Error populating MIME type filter: {e}")  # Log or show error
                QMessageBox.warning(self, "Filter Error", f"Could not populate MIME type filter: {e}")
        self.mime_type_filter.blockSignals(False)  # Unblock signals

    def load_assets(self):
        search_term = self.search_input.text().strip().lower()
        selected_mime_type = self.mime_type_filter.currentData()  # Get data (actual MIME type string)

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
                query = session.query(
                    FilePropertiesComponent.entity_id,
                    FilePropertiesComponent.original_filename,
                    FilePropertiesComponent.mime_type,
                )

                if search_term:
                    query = query.filter(FilePropertiesComponent.original_filename.ilike(f"%{search_term}%"))

                if selected_mime_type:  # Filter by MIME type if one is selected
                    query = query.filter(FilePropertiesComponent.mime_type == selected_mime_type)

                assets_found = query.all()

                if not assets_found:
                    message = "No assets found."
                    if search_term or selected_mime_type:
                        message_parts = ["No assets found"]
                        if search_term:
                            message_parts.append(f"matching '{search_term}'")
                        if selected_mime_type:
                            message_parts.append(f"of type '{selected_mime_type}'")
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
                if search_term:
                    loaded_message_parts.append(f"matching '{search_term}'")
                if selected_mime_type:
                    loaded_message_parts.append(f"of type '{selected_mime_type}'")
                loaded_message_parts.append(f"from world '{self.current_world.name}'.")
                self.statusBar().showMessage(" ".join(loaded_message_parts))

        except Exception as e:
            self.statusBar().showMessage(f"Error loading assets: {e}")
            QMessageBox.critical(
                self, "Load Assets Error", f"Could not load assets from world '{self.current_world.name}':\n{e}"
            )
            # Consider logging the full traceback for debugging
            # import traceback
            # print(traceback.format_exc())
            self.asset_list_widget.addItem(f"Error loading assets: {e}")

    def on_asset_double_clicked(self, item: QListWidgetItem):
        asset_id = item.data(Qt.ItemDataRole.UserRole)
        if asset_id is None:  # Should not happen if data is set correctly
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
                    QMessageBox.warning(
                        self, "Error", f"Entity ID {asset_id} not found in world '{self.current_world.name}'."
                    )
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
                            instance_data = {
                                c.key: getattr(comp_instance, c.key)
                                for c in comp_instance.__table__.columns
                                if not c.key.startswith("_")
                            }  # Exclude SQLAlchemy internals
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

    window = MainWindow(current_world=active_world)  # Pass active_world (could be None)
    window.show()
    sys.exit(app.exec())
