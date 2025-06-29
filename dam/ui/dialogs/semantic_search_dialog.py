import asyncio
import logging
import uuid # For request_id
from typing import Optional, List as TypingList, Tuple, Any

from PyQt6.QtCore import Qt, QRunnable, QObject, pyqtSignal, QThreadPool
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QTextEdit,
    QPushButton, QTableWidget, QTableWidgetItem, QMessageBox, QHeaderView,
    QDialogButtonBox, QSpinBox, QApplication
)

from dam.core.world import World
from dam.core.events import SemanticSearchQuery
# Assuming services are used directly or via events for search.
# For event-based, we'd dispatch SemanticSearchQuery.
# For direct service call (less ideal for UI responsiveness without worker):
# from dam.services import semantic_service, ecs_service as dam_ecs_service
from dam.models.properties import FilePropertiesComponent # For displaying results
from dam.models.semantic import TextEmbeddingComponent # For displaying matched source

logger = logging.getLogger(__name__)

# --- Worker for Semantic Search ---
class SemanticSearcherSignals(QObject):
    # List[Tuple[Entity, float, TextEmbeddingComponent]] -> List[Tuple[int, str, float, str, str]]
    # (entity_id, filename, score, matched_source_component, matched_source_field)
    search_ready = pyqtSignal(list)
    search_error = pyqtSignal(str)

class SemanticSearcher(QRunnable):
    def __init__(self, world: World, query_text: str, top_n: int, model_name: Optional[str]):
        super().__init__()
        self.world = world
        self.query_text = query_text
        self.top_n = top_n
        self.model_name = model_name
        self.signals = SemanticSearcherSignals()

    def run(self):
        async def _search():
            request_id = str(uuid.uuid4())
            event = SemanticSearchQuery(
                query_text=self.query_text,
                world_name=self.world.name,
                request_id=request_id,
                top_n=self.top_n,
                model_name=self.model_name
            )
            event.result_future = asyncio.get_running_loop().create_future()

            await self.world.dispatch_event(event) # World needs to handle this event

            # Result is List[Tuple[Entity, float, TextEmbeddingComponent]]
            raw_results: TypingList[Tuple[Any, float, Any]] = await asyncio.wait_for(event.result_future, timeout=60.0)

            display_results = []
            async with self.world.get_db_session() as session:
                for entity_obj, score, emb_comp_obj in raw_results:
                    # entity_obj is an Entity instance, emb_comp_obj is TextEmbeddingComponent instance
                    entity_id = entity_obj.id
                    filename = "N/A"
                    fpc = await self.world.ecs_service.get_component(session, entity_id, FilePropertiesComponent)
                    if fpc:
                        filename = fpc.original_filename

                    matched_source_comp = emb_comp_obj.source_component_name if emb_comp_obj else "N/A"
                    matched_source_field = emb_comp_obj.source_field_name if emb_comp_obj else "N/A"
                    model_used = emb_comp_obj.model_name if emb_comp_obj else "N/A"

                    display_results.append((
                        entity_id, filename, score,
                        matched_source_comp, matched_source_field, model_used
                    ))
            return display_results
        try:
            results = asyncio.run(_search())
            self.signals.search_ready.emit(results)
        except Exception as e:
            logger.error(f"Error during semantic search: {e}", exc_info=True)
            self.signals.search_error.emit(str(e))


class SemanticSearchDialog(QDialog):
    # Signal to request opening component viewer for an asset_id
    view_components_requested = pyqtSignal(int)

    def __init__(self, world: World, parent=None):
        super().__init__(parent)
        self.world = world
        self.setWindowTitle("Semantic Search")
        self.setMinimumSize(700, 500)
        self.thread_pool = QThreadPool.globalInstance()

        layout = QVBoxLayout(self)

        # Search Input Area
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("Search Query:"))
        self.query_input = QLineEdit()
        self.query_input.setPlaceholderText("Enter text to search semantically...")
        self.query_input.returnPressed.connect(self._perform_search)
        input_layout.addWidget(self.query_input)

        input_layout.addWidget(QLabel("Top N:"))
        self.top_n_input = QSpinBox()
        self.top_n_input.setMinimum(1)
        self.top_n_input.setMaximum(100)
        self.top_n_input.setValue(10)
        input_layout.addWidget(self.top_n_input)

        self.search_button = QPushButton("Search")
        self.search_button.clicked.connect(self._perform_search)
        input_layout.addWidget(self.search_button)
        layout.addLayout(input_layout)

        # Results Table
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(6) # ID, Filename, Score, Source Component, Source Field, Model
        self.results_table.setHorizontalHeaderLabels([
            "Asset ID", "Filename", "Score", "Source Component", "Source Field", "Model Used"
        ])
        self.results_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.results_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.results_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch) # Filename
        self.results_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.results_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self.results_table.itemDoubleClicked.connect(self._on_result_double_clicked)
        layout.addWidget(self.results_table)

        # Dialog Buttons
        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

    def _perform_search(self):
        query_text = self.query_input.text().strip()
        if not query_text:
            QMessageBox.warning(self, "Input Error", "Search query cannot be empty.")
            return

        top_n = self.top_n_input.value()
        # model_name can be an option later, for now uses default from service via event

        self.search_button.setEnabled(False)
        self.results_table.setRowCount(0) # Clear table
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

        searcher = SemanticSearcher(self.world, query_text, top_n, model_name=None) # None means default model
        searcher.signals.search_ready.connect(self._on_search_results_fetched)
        searcher.signals.search_error.connect(self._on_search_error)
        self.thread_pool.start(searcher)

    def _on_search_results_fetched(self, results: TypingList[Tuple[int, str, float, str, str, str]]):
        QApplication.restoreOverrideCursor()
        self.results_table.setRowCount(len(results))
        for row, (entity_id, filename, score, src_comp, src_field, model_name) in enumerate(results):
            id_item = QTableWidgetItem(str(entity_id))
            id_item.setData(Qt.ItemDataRole.UserRole, entity_id)
            filename_item = QTableWidgetItem(filename)
            score_item = QTableWidgetItem(f"{score:.4f}")
            src_comp_item = QTableWidgetItem(src_comp)
            src_field_item = QTableWidgetItem(src_field)
            model_item = QTableWidgetItem(model_name)

            for item in [id_item, filename_item, score_item, src_comp_item, src_field_item, model_item]:
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled) # Make copyable

            self.results_table.setItem(row, 0, id_item)
            self.results_table.setItem(row, 1, filename_item)
            self.results_table.setItem(row, 2, score_item)
            self.results_table.setItem(row, 3, src_comp_item)
            self.results_table.setItem(row, 4, src_field_item)
            self.results_table.setItem(row, 5, model_item)

        self.search_button.setEnabled(True)
        if not results:
            QMessageBox.information(self, "No Results", "No semantic matches found for your query.")
        QApplication.processEvents()


    def _on_search_error(self, error_message: str):
        QApplication.restoreOverrideCursor()
        QMessageBox.critical(self, "Semantic Search Error", error_message)
        self.search_button.setEnabled(True)

    def _on_result_double_clicked(self, item: QTableWidgetItem):
        if not item: return
        row = item.row()
        id_item = self.results_table.item(row, 0)
        if id_item:
            asset_id = id_item.data(Qt.ItemDataRole.UserRole)
            if asset_id is not None:
                # Emit a signal to the main window or handle directly if simpler
                # For now, this dialog doesn't know about ComponentViewerDialog directly
                # We can pass a callback or use a signal.
                # Let's assume MainWindow connects to this signal.
                logger.info(f"Requesting to view components for asset ID: {asset_id}")
                self.view_components_requested.emit(asset_id)


if __name__ == '__main__':
    from dam.core.world import World, WorldConfig
    from dam.core.database import DatabaseManager
    from dam.core.world_setup import register_core_systems
    import tempfile

    logging.basicConfig(level=logging.DEBUG)

    temp_dir = tempfile.mkdtemp()
    db_path = f"{temp_dir}/test_semantic_dialog.db"
    test_config = WorldConfig(
        name="test_semantic_dialog_world",
        DATABASE_URL=f"sqlite+aiosqlite:///{db_path}",
        ASSET_STORAGE_PATH=f"{temp_dir}/assets",
        LOG_LEVEL="DEBUG"
    )
    db_manager = DatabaseManager(test_config.DATABASE_URL, is_memory_db=False)
    test_world = World(name="test_semantic_dialog_world", config=test_config, db_manager=db_manager)
    register_core_systems(test_world) # Register systems including semantic search handler

    async def setup_db_for_dialog_test(world_instance):
        await world_instance.create_db_and_tables()
        # Optionally, add some dummy data with embeddings if needed for testing UI
        # For now, just ensuring tables exist for the event system to try.

    asyncio.run(setup_db_for_dialog_test(test_world))

    app = QApplication([])
    dialog = SemanticSearchDialog(world=test_world)

    # Example of connecting the signal if MainWindow was instantiating this
    # def show_components(asset_id):
    #     QMessageBox.information(dialog, "Info", f"Would show components for asset ID: {asset_id}")
    # dialog.view_components_requested.connect(show_components)

    dialog.show()
    app.exec()

    import os
    os.unlink(db_path)
    os.rmdir(f"{temp_dir}/assets")
    os.rmdir(temp_dir)
