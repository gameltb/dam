import sys
import asyncio
import threading # For simple threading, consider QThread for more complex Qt integration
import logging # Added import logging

# from dam.models.core.entity import Entity # Not directly needed if using service functions
# Added for type hinting, Optional already imported from typing
from typing import Any, Dict, Optional, Sequence
from typing import List as TypingList


from PyQt6.QtCore import Qt, QObject, pyqtSignal, QRunnable, QThreadPool
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


class MimeTypeFetcherSignals(QObject):
    """
    Defines signals for MimeTypeFetcher.
    - result_ready: Emitted when MIME types are successfully fetched.
    - error_occurred: Emitted when an error occurs during fetching.
    """
    result_ready = pyqtSignal(list)
    error_occurred = pyqtSignal(str)


class MimeTypeFetcher(QRunnable):
    """
    Worker runnable to fetch distinct MIME types from the database asynchronously
    in a separate thread.
    """
    # import logging # This was moved to the top of the file

    def __init__(self, world: World):
        super().__init__()
        self.world = world
        self.signals = MimeTypeFetcherSignals()
        self.logger = logging.getLogger(__name__ + ".MimeTypeFetcher")

    def run(self):
        """
        Executes the database query to fetch MIME types.
        Emits result_ready on success, error_occurred on failure.
        """
        self.logger.info(f"MimeTypeFetcher started for world: {self.world.name if self.world else 'None'}.")
        if not self.world:
            self.logger.warning("No world configured for MimeTypeFetcher.")
            self.signals.error_occurred.emit("No world configured for fetching MIME types.")
            self.logger.info("MimeTypeFetcher finished (no world).")
            return

        async def fetch_mime_types_async():
            self.logger.info("fetch_mime_types_async started.")
            async with self.world.get_db_session() as session:
                self.logger.info("Async session obtained for MIME types.")
                from sqlalchemy import select # Ensure select is imported
                stmt = (
                    select(FilePropertiesComponent.mime_type)
                    .filter(FilePropertiesComponent.mime_type.isnot(None))
                    .distinct()
                    .order_by(FilePropertiesComponent.mime_type)
                )
                self.logger.info("Executing MIME type query.")
                result = await session.execute(stmt)
                distinct_mime_types = result.scalars().all()
                self.logger.info(f"MIME type query executed, found {len(distinct_mime_types)} types.")
                return distinct_mime_types

        try:
            self.logger.info("Calling asyncio.run(fetch_mime_types_async).")
            mime_types = asyncio.run(fetch_mime_types_async())
            self.logger.info("asyncio.run(fetch_mime_types_async) completed.")
            self.signals.result_ready.emit([str(mt) for mt in mime_types if mt])
            self.logger.info("result_ready signal emitted.")
        except Exception as e:
            error_message = f"Error fetching MIME types: {e}"
            self.logger.error(error_message, exc_info=True)
            self.signals.error_occurred.emit(error_message)
        finally:
            self.logger.info("MimeTypeFetcher finished.")


class AssetLoaderSignals(QObject):
    """
    Defines signals for AssetLoader.
    - assets_ready: Emitted when assets are successfully fetched.
                    Payload is a list of tuples: (entity_id, original_filename, mime_type).
    - error_occurred: Emitted when an error occurs during fetching.
                      Payload is the error message string.
    """
    assets_ready = pyqtSignal(list)
    error_occurred = pyqtSignal(str)


class AssetLoader(QRunnable):
    """
    Worker runnable to fetch assets from the database asynchronously
    in a separate thread, applying optional search and MIME type filters.
    """
    def __init__(self, world: World, search_term: Optional[str], selected_mime_type: Optional[str]):
        super().__init__()
        self.world = world
        self.search_term = search_term
        self.selected_mime_type = selected_mime_type
        self.signals = AssetLoaderSignals()
        self.logger = logging.getLogger(__name__ + ".AssetLoader")

    def run(self):
        """
        Executes the database query to fetch assets.
        Emits assets_ready on success, error_occurred on failure.
        """
        self.logger.info(f"AssetLoader started for world: {self.world.name if self.world else 'None'}. Search: '{self.search_term}', Type: '{self.selected_mime_type}'.")
        if not self.world:
            self.logger.warning("No world configured for AssetLoader.")
            self.signals.error_occurred.emit("No world configured for fetching assets.")
            self.logger.info("AssetLoader finished (no world).")
            return

        async def fetch_assets_async():
            self.logger.info("fetch_assets_async started.")
            async with self.world.get_db_session() as session:
                self.logger.info("Async session obtained for assets.")
                from sqlalchemy import select

                query = select(
                    FilePropertiesComponent.entity_id,
                    FilePropertiesComponent.original_filename,
                    FilePropertiesComponent.mime_type,
                ).order_by(FilePropertiesComponent.original_filename)

                if self.search_term:
                    self.logger.info(f"Applying search term: {self.search_term}")
                    query = query.filter(FilePropertiesComponent.original_filename.ilike(f"%{self.search_term}%"))

                if self.selected_mime_type:
                    self.logger.info(f"Applying MIME type filter: {self.selected_mime_type}")
                    query = query.filter(FilePropertiesComponent.mime_type == self.selected_mime_type)

                self.logger.info("Executing asset query.")
                result = await session.execute(query)
                assets_found = result.all()
                self.logger.info(f"Asset query executed, found {len(assets_found)} assets.")
                return [(row.entity_id, row.original_filename, row.mime_type) for row in assets_found]

        try:
            self.logger.info("Calling asyncio.run(fetch_assets_async).")
            assets_data = asyncio.run(fetch_assets_async())
            self.logger.info("asyncio.run(fetch_assets_async) completed.")
            self.signals.assets_ready.emit(assets_data)
            self.logger.info("assets_ready signal emitted.")
        except Exception as e:
            error_message = f"Error loading assets: {e}"
            self.logger.error(error_message, exc_info=True)
            self.signals.error_occurred.emit(error_message)
        finally:
            self.logger.info("AssetLoader finished.")


class ComponentFetcherSignals(QObject):
    """
    Defines signals for ComponentFetcher.
    - components_ready: Emitted when components are successfully fetched.
                        Payload: components_data (dict), asset_id (int), world_name (str).
    - error_occurred: Emitted when an error occurs.
                      Payload: error_message (str), asset_id (int).
    """
    components_ready = pyqtSignal(dict, int, str)
    error_occurred = pyqtSignal(str, int)


class ComponentFetcher(QRunnable):
    """
    Worker runnable to fetch all components for a given asset_id from the database.
    """
    def __init__(self, world: World, asset_id: int):
        super().__init__()
        self.world = world
        self.asset_id = asset_id
        self.signals = ComponentFetcherSignals()

    def run(self):
        """
        Executes the database query to fetch components.
        Emits components_ready on success, error_occurred on failure.
        """
        if not self.world:
            self.signals.error_occurred.emit("No world configured for fetching components.", self.asset_id)
            return

        async def fetch_components_async():
            components_data_for_dialog: Dict[str, TypingList[Dict[str, Any]]] = {}
            async with self.world.get_db_session() as session:
                entity = await ecs_service.get_entity(session, self.asset_id) # Corrected: was async_get_entity
                if not entity:
                    # This case should perhaps be an error or specific signal
                    # For now, an empty components_data will be emitted.
                    # Or raise an error:
                    raise FileNotFoundError(f"Entity ID {self.asset_id} not found in world '{self.world.name}'.")


                for comp_type_name, comp_type_cls in REGISTERED_COMPONENT_TYPES.items():
                    # Assuming ecs_service.get_components can be made async or there's an async version
                    components = await ecs_service.get_components(session, self.asset_id, comp_type_cls) # Corrected: was async_get_components
                    component_instances_data = []
                    if components:
                        for comp_instance in components:
                            instance_data = {
                                c.key: getattr(comp_instance, c.key)
                                for c in comp_instance.__table__.columns
                                if not c.key.startswith("_")
                            }
                            component_instances_data.append(instance_data)
                    components_data_for_dialog[comp_type_name] = component_instances_data
            return components_data_for_dialog

        try:
            components_data = asyncio.run(fetch_components_async())
            self.signals.components_ready.emit(components_data, self.asset_id, self.world.name)
        except Exception as e:
            error_message = f"Error fetching components for Entity ID {self.asset_id}: {e}"
            # import traceback; traceback.print_exc()
            self.signals.error_occurred.emit(error_message, self.asset_id)


class DbSetupWorkerSignals(QObject):
    """Signals for database setup worker."""
    setup_complete = pyqtSignal(str) # world_name
    setup_error = pyqtSignal(str, str) # world_name, error_message


class DbSetupWorker(QRunnable):
    """Worker to run create_db_and_tables for a world."""
    def __init__(self, world: World):
        super().__init__()
        self.world = world
        self.signals = DbSetupWorkerSignals()

    def run(self):
        try:
            # world.create_db_and_tables() is an async method
            asyncio.run(self.world.create_db_and_tables())
            self.signals.setup_complete.emit(self.world.name)
        except Exception as e:
            self.signals.setup_error.emit(self.world.name, f"Error during database setup: {e}")


class MainWindow(QMainWindow):
    def __init__(self, current_world: Optional[World] = None):
        super().__init__()

        self.current_world = current_world
        self.setWindowTitle(f"DAM UI - World: {current_world.name if current_world else 'No World Selected'}")
        self.setGeometry(100, 100, 800, 600)

        self.thread_pool = QThreadPool() # For running worker tasks
        # print(f"Max QThreadPool threads: {self.thread_pool.maxThreadCount()}")


        self._create_menus()
        self._create_status_bar()
        self._create_central_widget()


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

        world_name_for_setup = self.current_world.name # Capture before potential thread issues

        reply = QMessageBox.question(
            self,
            "Confirm Database Setup",
            f"This will attempt to initialize the database and create tables for world '{world_name_for_setup}'.\n"
            "This is typically done once. Proceed?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.No:
            self.statusBar().showMessage("Database setup cancelled by user.", 3000)
            return

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        self.statusBar().showMessage(f"Starting database setup for world '{world_name_for_setup}'...")

        # Disable the button or relevant UI parts during operation
        # For simplicity, just using cursor and status messages for now.

        worker = DbSetupWorker(self.current_world)
        worker.signals.setup_complete.connect(self._on_db_setup_complete)
        worker.signals.setup_error.connect(self._on_db_setup_error)
        self.thread_pool.start(worker)

    def _on_db_setup_complete(self, world_name: str):
        QApplication.restoreOverrideCursor()
        QMessageBox.information(
            self, "Database Setup Successful", f"Database setup complete for world '{world_name}'."
        )
        self.statusBar().showMessage(f"Database for '{world_name}' successfully set up.")
        self.load_assets()  # Refresh asset list

    def _on_db_setup_error(self, world_name: str, error_message: str):
        QApplication.restoreOverrideCursor()
        QMessageBox.critical(
            self, "Database Setup Error", f"Error during database setup for '{world_name}':\n{error_message}"
        )
        self.statusBar().showMessage(f"Database setup for '{world_name}' failed: {error_message.splitlines()[0]}", 5000)
        # No finally block needed here for cursor as it's handled in each slot.


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
        self.mime_type_filter.currentIndexChanged.connect(self.load_assets)
        controls_layout.addWidget(self.mime_type_filter)

        self.refresh_button = QPushButton("Refresh All")
        self.refresh_button.clicked.connect(self._clear_filters_and_refresh)
        controls_layout.addWidget(self.refresh_button)

        controls_layout.addStretch()
        main_layout.addLayout(controls_layout)

        self.asset_list_widget = QListWidget()
        self.asset_list_widget.itemDoubleClicked.connect(self.on_asset_double_clicked)
        main_layout.addWidget(self.asset_list_widget)

        self.setCentralWidget(central_widget)
        self.populate_mime_type_filter() # Initial population will now trigger load_assets on completion/error

    def _clear_filters_and_refresh(self):
        self.search_input.clear()
        # self.mime_type_filter.setCurrentIndex(0) # This might trigger load_assets before populate is done
        self.populate_mime_type_filter() # Re-populate, then load_assets will be called by its completion or by currentIndexChanged
        self.load_assets() # Explicitly call load_assets to ensure refresh with cleared filters

    def populate_mime_type_filter(self):
        """
        Initiates the fetching of MIME types in a worker thread.
        The actual population of the QComboBox happens in _update_mime_type_filter_ui.
        """
        if not self.current_world:
            self._update_mime_type_filter_ui([], error_message="No world selected. Cannot populate MIME types.")
            return

        # Disable the filter during update to prevent user interaction / signal emission
        self.mime_type_filter.setEnabled(False)
        self.statusBar().showMessage("Populating MIME type filter...")

        fetcher = MimeTypeFetcher(world=self.current_world)
        fetcher.signals.result_ready.connect(self._on_mime_types_fetched)
        fetcher.signals.error_occurred.connect(self._on_mime_type_fetch_error)
        self.thread_pool.start(fetcher)

    def _on_mime_types_fetched(self, mime_types: TypingList[str]):
        """
        Slot to receive successfully fetched MIME types from the worker.
        """
        self._update_mime_type_filter_ui(mime_types)
        self.statusBar().showMessage("MIME type filter populated.", 3000) # Disappears after 3s
        self.mime_type_filter.setEnabled(True)
        self.load_assets() # Load assets after MIME types are fetched

    def _on_mime_type_fetch_error(self, error_message: str):
        """
        Slot to receive error messages from the MIME type fetcher worker.
        """
        self._update_mime_type_filter_ui([], error_message=error_message) # Pass empty list on error
        self.statusBar().showMessage(f"MIME filter error: {error_message}", 5000) # Disappears after 5s
        self.mime_type_filter.setEnabled(True)
        QMessageBox.warning(self, "Filter Error", f"Could not populate MIME type filter:\n{error_message}")
        self.load_assets() # Still attempt to load assets even if MIME types failed


    def _update_mime_type_filter_ui(self, mime_types: TypingList[str], error_message: Optional[str] = None):
        """
        Updates the MIME type QComboBox with fetched data or an error message.
        This method is always called on the main GUI thread.
        """
        self.mime_type_filter.blockSignals(True)
        current_filter_value = self.mime_type_filter.currentData()
        self.mime_type_filter.clear()
        self.mime_type_filter.addItem("All Types", "")

        if mime_types:
            for mime_type in sorted(list(set(mime_types))): # Ensure unique and sorted
                if mime_type:
                    self.mime_type_filter.addItem(mime_type, mime_type)

        index_to_set = self.mime_type_filter.findData(current_filter_value)
        if index_to_set != -1:
            self.mime_type_filter.setCurrentIndex(index_to_set)
        else:
            self.mime_type_filter.setCurrentIndex(0) # Default to "All Types"

        self.mime_type_filter.blockSignals(False)
        self.mime_type_filter.setEnabled(True) # Re-enable after update

        if error_message: # If there was an error, it's already logged/shown by caller
            pass

    def _set_asset_controls_enabled(self, enabled: bool):
        """Helper to enable/disable asset loading related controls."""
        self.search_input.setEnabled(enabled)
        self.mime_type_filter.setEnabled(enabled) # This might conflict with MimeTypeFetcher's own enabling/disabling
        self.refresh_button.setEnabled(enabled)
        # Consider also disabling asset_list_widget during load if it's too interactive

    def load_assets(self):
        """
        Initiates fetching of assets in a worker thread.
        Actual list update happens in _on_assets_fetched or _on_asset_fetch_error.
        """
        search_term = self.search_input.text().strip().lower()
        selected_mime_type = self.mime_type_filter.currentData()

        status_message_parts = ["Loading assets"]
        if search_term: status_message_parts.append(f"searching for '{search_term}'")
        if selected_mime_type: status_message_parts.append(f"filtered by type '{selected_mime_type}'")
        self.statusBar().showMessage("... ".join(status_message_parts) + "...")

        self.asset_list_widget.clear()
        self._set_asset_controls_enabled(False)

        if not self.current_world:
            error_msg = "No DAM world is currently selected. Cannot load assets."
            self._on_asset_fetch_error(error_msg) # Use the error handler
            # QMessageBox.warning(self, "Error", error_msg) # This is handled by _on_asset_fetch_error
            return

        asset_loader = AssetLoader(
            world=self.current_world,
            search_term=search_term,
            selected_mime_type=selected_mime_type
        )
        asset_loader.signals.assets_ready.connect(self._on_assets_fetched)
        asset_loader.signals.error_occurred.connect(self._on_asset_fetch_error)
        self.thread_pool.start(asset_loader)

    def _on_assets_fetched(self, assets_data: TypingList[tuple]):
        """
        Slot to receive successfully fetched assets from AssetLoader.
        Populates the asset list widget.
        """
        self.asset_list_widget.clear() # Clear again just in case

        search_term = self.search_input.text().strip().lower() # Get current search term for status message
        selected_mime_type = self.mime_type_filter.currentData() # Get current filter for status message

        if not assets_data:
            message = "No assets found."
            if search_term or selected_mime_type:
                message_parts = ["No assets found"]
                if search_term: message_parts.append(f"matching '{search_term}'")
                if selected_mime_type: message_parts.append(f"of type '{selected_mime_type}'")
                message = " ".join(message_parts) + "."
            self.asset_list_widget.addItem(message)
            self.statusBar().showMessage(message)
        else:
            for entity_id, original_filename, mime_type_val in assets_data:
                display_text = f"ID: {entity_id} - {original_filename} ({mime_type_val or 'N/A'})"
                item = QListWidgetItem(display_text)
                item.setData(Qt.ItemDataRole.UserRole, entity_id)
                self.asset_list_widget.addItem(item)

            num_found = len(assets_data)
            loaded_message_parts = [f"Loaded {num_found} asset{'s' if num_found != 1 else ''}"]
            if search_term: loaded_message_parts.append(f"matching '{search_term}'")
            if selected_mime_type: loaded_message_parts.append(f"of type '{selected_mime_type}'")
            if self.current_world: loaded_message_parts.append(f"from world '{self.current_world.name}'.")
            self.statusBar().showMessage(" ".join(loaded_message_parts))

        self._set_asset_controls_enabled(True)
        # Ensure mime_type_filter is specifically re-enabled if MimeTypeFetcher is also done
        # This can be tricky if both run concurrently. A simple solution is that both enable it.
        if not self.mime_type_filter.isEnabled(): # Check if MimeTypeFetcher is still running
             if self.thread_pool.activeThreadCount() == 0 : # A bit of a guess, better to have specific flags
                  self.mime_type_filter.setEnabled(True)
        # A better approach for mime_type_filter might be a separate flag or counter for disabling operations.

    def _on_asset_fetch_error(self, error_message: str):
        """
        Slot to receive error messages from the AssetLoader worker.
        """
        self.asset_list_widget.clear() # Clear previous items
        self.asset_list_widget.addItem(f"Error loading assets: {error_message.splitlines()[0]}") # Show first line of error
        self.statusBar().showMessage(f"Error loading assets: {error_message}", 5000)
        QMessageBox.critical(
            self, "Load Assets Error", f"Could not load assets:\n{error_message}"
        )
        self._set_asset_controls_enabled(True)
        # Similar logic for mime_type_filter re-enabling as in _on_assets_fetched
        if not self.mime_type_filter.isEnabled():
             if self.thread_pool.activeThreadCount() == 0 :
                  self.mime_type_filter.setEnabled(True)

    def on_asset_double_clicked(self, item: QListWidgetItem):
        asset_id = item.data(Qt.ItemDataRole.UserRole)
        if asset_id is None:
            QMessageBox.warning(self, "Error", "No asset ID associated with this item.")
            return

        if not self.current_world:
            QMessageBox.warning(self, "Error", "No world selected. Cannot view components.")
            return

        self.statusBar().showMessage(f"Fetching components for Entity ID: {asset_id}...")
        # Disable list or show busy cursor? For now, just status bar.

        fetcher = ComponentFetcher(world=self.current_world, asset_id=asset_id)
        fetcher.signals.components_ready.connect(self._on_components_fetched)
        fetcher.signals.error_occurred.connect(self._on_component_fetch_error)
        self.thread_pool.start(fetcher)

    def _on_components_fetched(self, components_data: dict, asset_id: int, world_name: str):
        """
        Slot to receive successfully fetched components from ComponentFetcher.
        Shows the ComponentViewerDialog.
        """
        if not components_data: # Or if a specific error like "not found" was signaled differently
            self.statusBar().showMessage(f"No components found for Entity ID: {asset_id}.")
            # Optionally show a QMessageBox.information here
        else:
            self.statusBar().showMessage(f"Successfully fetched components for Entity ID: {asset_id}.")

        # Ensure world_name matches current world if important, though ComponentFetcher uses its own world instance.
        # Here, world_name is mostly for display in the dialog.
        dialog = ComponentViewerDialog(asset_id, components_data, world_name, self)
        dialog.exec()
        # Clear status bar after dialog is closed or set to a default message
        self.statusBar().showMessage("Ready", 2000)


    def _on_component_fetch_error(self, error_message: str, asset_id: int):
        """
        Slot to receive error messages from the ComponentFetcher worker.
        """
        status_msg = f"Error fetching components for Entity ID {asset_id}: {error_message.splitlines()[0]}"
        self.statusBar().showMessage(status_msg, 5000)
        QMessageBox.critical(
            self, "Component Fetch Error", f"Could not fetch components for Entity ID {asset_id}:\n{error_message}"
        )

    def closeEvent(self, event):
        """
        Handle the window close event to ensure worker threads are properly managed.
        """
        # print("MainWindow closeEvent triggered.") # For debugging
        # Potentially disable further UI interactions here if needed

        # Wait for all tasks in the thread pool to complete.
        # Set a timeout (e.g., 5000ms) to prevent indefinite blocking if a task hangs.
        # waitForDone returns true if all tasks completed; false if it timed out.
        self.statusBar().showMessage("Shutting down worker threads, please wait...")
        QApplication.processEvents() # Process events to update status bar immediately

        if not self.thread_pool.waitForDone(5000):
            # print("Worker threads did not finish in time. Some tasks may be incomplete.") # For debugging
            # Optionally, show a message to the user or log this occurrence.
            # Depending on the nature of tasks, forceful termination might be considered,
            # but QThreadPool doesn't offer direct cancellation of QRunnables.
            # The runnables themselves would need to support interruption.
            # For now, we just accept the event and let Qt proceed with closing.
            QMessageBox.warning(self, "Shutdown Warning",
                                "Some background tasks did not finish quickly.\n"
                                "The application will now close. If issues persist, please restart.")
        else:
            # print("All worker threads finished.") # For debugging
            pass

        self.statusBar().showMessage("Exiting...")
        QApplication.processEvents() # Ensure this message is shown
        super().closeEvent(event) # Proceed with closing the window


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

    # Setup logging for standalone UI run
    from dam.core.logging_config import setup_logging # Moved import here
    import logging # Ensure logging is imported
    setup_logging(level=logging.INFO) # Or DEBUG for more verbosity

    window = MainWindow(current_world=active_world)  # Pass active_world (could be None)
    window.show()
    sys.exit(app.exec())
