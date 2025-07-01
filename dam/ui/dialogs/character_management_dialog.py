import asyncio
import logging
from typing import List as TypingList
from typing import Optional, Tuple

from PyQt6.QtCore import QObject, QRunnable, Qt, QThreadPool, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
)

from dam.core.world import World
from dam.models.conceptual import CharacterConceptComponent
from dam.models.core.entity import Entity
from dam.services import character_service
from dam.services import ecs_service as dam_ecs_service

logger = logging.getLogger(__name__)


# --- Worker for fetching characters ---
class CharacterFetcherSignals(QObject):
    characters_ready = pyqtSignal(list)  # List[Tuple[int, str, str]] (id, name, description)
    error_occurred = pyqtSignal(str)


class CharacterFetcher(QRunnable):
    def __init__(self, world: World, search_term: Optional[str] = None):
        super().__init__()
        self.world = world
        self.search_term = search_term
        self.signals = CharacterFetcherSignals()

    def run(self):
        async def _fetch():
            async with self.world.get_db_session() as session:
                char_entities: TypingList[Entity] = await character_service.find_character_concepts(
                    session, query_name=self.search_term
                )
                results = []
                for entity in char_entities:
                    comp = await dam_ecs_service.get_component(session, entity.id, CharacterConceptComponent)
                    if comp:
                        results.append((entity.id, comp.concept_name, comp.concept_description or ""))
                return results

        try:
            characters = asyncio.run(_fetch())
            self.signals.characters_ready.emit(characters)
        except Exception as e:
            logger.error(f"Error fetching characters: {e}", exc_info=True)
            self.signals.error_occurred.emit(str(e))


# --- Worker for creating a character ---
class CharacterCreatorSignals(QObject):
    creation_complete = pyqtSignal(int, str)  # entity_id, name
    creation_error = pyqtSignal(str)


class CharacterCreator(QRunnable):
    def __init__(self, world: World, name: str, description: Optional[str]):
        super().__init__()
        self.world = world
        self.name = name
        self.description = description
        self.signals = CharacterCreatorSignals()

    def run(self):
        async def _create():
            async with self.world.get_db_session() as session:
                char_entity = await character_service.create_character_concept(session, self.name, self.description)
                if char_entity:
                    return char_entity.id, self.name
                else:
                    # Check if it already exists by trying to fetch it
                    try:
                        existing_char = await character_service.get_character_concept_by_name(session, self.name)
                        if existing_char:
                            raise ValueError(f"Character '{self.name}' already exists with ID {existing_char.id}.")
                    except character_service.CharacterConceptNotFoundError:
                        pass  # Should have been created or raised other error
                    raise Exception(f"Failed to create character '{self.name}'.")

        try:
            entity_id, name = asyncio.run(_create())
            self.signals.creation_complete.emit(entity_id, name)
        except Exception as e:
            logger.error(f"Error creating character: {e}", exc_info=True)
            self.signals.creation_error.emit(str(e))


# --- Worker for applying a character to an asset ---
class CharacterApplierSignals(QObject):
    apply_complete = pyqtSignal(str)  # success message
    apply_error = pyqtSignal(str)


class CharacterApplier(QRunnable):
    def __init__(self, world: World, asset_entity_id: int, character_concept_entity_id: int, role: Optional[str]):
        super().__init__()
        self.world = world
        self.asset_entity_id = asset_entity_id
        self.character_concept_entity_id = character_concept_entity_id
        self.role = role
        self.signals = CharacterApplierSignals()

    def run(self):
        async def _apply():
            async with self.world.get_db_session() as session:
                link_comp = await character_service.apply_character_to_entity(
                    session, self.asset_entity_id, self.character_concept_entity_id, self.role
                )
                if link_comp:
                    char_comp = await dam_ecs_service.get_component(
                        session, self.character_concept_entity_id, CharacterConceptComponent
                    )
                    char_name = char_comp.concept_name if char_comp else "Unknown Character"
                    return f"Character '{char_name}' applied to asset ID {self.asset_entity_id}."
                else:
                    raise Exception("Failed to apply character. Link might already exist or another error occurred.")

        try:
            message = asyncio.run(_apply())
            self.signals.apply_complete.emit(message)
        except Exception as e:
            logger.error(f"Error applying character: {e}", exc_info=True)
            self.signals.apply_error.emit(str(e))


class CharacterManagementDialog(QDialog):
    def __init__(self, world: World, current_selected_asset_id: Optional[int] = None, parent=None):
        super().__init__(parent)
        self.world = world
        self.current_selected_asset_id = current_selected_asset_id
        self.setWindowTitle("Character Management")
        self.setMinimumSize(600, 400)
        self.thread_pool = QThreadPool.globalInstance()

        layout = QVBoxLayout(self)

        # Search Characters
        search_layout = QHBoxLayout()
        search_layout.addWidget(QLabel("Search Characters:"))
        self.char_search_input = QLineEdit()
        self.char_search_input.setPlaceholderText("Enter name to search...")
        self.char_search_input.returnPressed.connect(self._load_characters)
        search_layout.addWidget(self.char_search_input)
        self.char_search_button = QPushButton("Search")
        self.char_search_button.clicked.connect(self._load_characters)
        search_layout.addWidget(self.char_search_button)
        layout.addLayout(search_layout)

        # Character Table
        self.char_table = QTableWidget()
        self.char_table.setColumnCount(3)
        self.char_table.setHorizontalHeaderLabels(["ID", "Name", "Description"])
        self.char_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.char_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.char_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.char_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.char_table)

        # Create New Character Area
        create_char_layout = QVBoxLayout()
        create_char_layout.addWidget(QLabel("Create New Character:"))
        self.new_char_name_input = QLineEdit()
        self.new_char_name_input.setPlaceholderText("Character Name")
        create_char_layout.addWidget(self.new_char_name_input)
        self.new_char_desc_input = QTextEdit()
        self.new_char_desc_input.setPlaceholderText("Character Description (optional)")
        self.new_char_desc_input.setFixedHeight(60)
        create_char_layout.addWidget(self.new_char_desc_input)
        self.create_char_button = QPushButton("Create Character")
        self.create_char_button.clicked.connect(self._create_new_character)
        create_char_layout.addWidget(self.create_char_button)
        layout.addLayout(create_char_layout)

        # Apply Character to Asset Area
        apply_layout = QHBoxLayout()
        apply_layout.addWidget(QLabel("Role for selected character in asset:"))
        self.role_input = QLineEdit()
        self.role_input.setPlaceholderText("e.g., Protagonist, Background (optional)")
        apply_layout.addWidget(self.role_input)
        self.apply_char_button = QPushButton("Apply to Selected Asset")
        self.apply_char_button.clicked.connect(self._apply_character_to_asset)
        if self.current_selected_asset_id is None:
            self.apply_char_button.setEnabled(False)
            self.apply_char_button.setToolTip("No asset selected in main window.")
        apply_layout.addWidget(self.apply_char_button)
        layout.addLayout(apply_layout)

        # Dialog Buttons
        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

        self._load_characters()

    def _load_characters(self):
        search_term = self.char_search_input.text().strip()
        self.char_search_button.setEnabled(False)
        self.char_table.setRowCount(0)  # Clear table

        fetcher = CharacterFetcher(self.world, search_term)
        fetcher.signals.characters_ready.connect(self._on_characters_fetched)
        fetcher.signals.error_occurred.connect(self._on_fetch_error)
        self.thread_pool.start(fetcher)

    def _on_characters_fetched(self, characters: TypingList[Tuple[int, str, str]]):
        self.char_table.setRowCount(len(characters))
        for row, (char_id, name, description) in enumerate(characters):
            id_item = QTableWidgetItem(str(char_id))
            id_item.setData(Qt.ItemDataRole.UserRole, char_id)  # Store ID
            name_item = QTableWidgetItem(name)
            desc_item = QTableWidgetItem(description)

            # Make items copyable
            id_item.setFlags(id_item.flags() | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
            name_item.setFlags(name_item.flags() | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
            desc_item.setFlags(desc_item.flags() | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)

            self.char_table.setItem(row, 0, id_item)
            self.char_table.setItem(row, 1, name_item)
            self.char_table.setItem(row, 2, desc_item)
        self.char_search_button.setEnabled(True)
        QApplication.processEvents()  # Update UI

    def _on_fetch_error(self, error_message: str):
        QMessageBox.critical(self, "Error Fetching Characters", error_message)
        self.char_search_button.setEnabled(True)

    def _create_new_character(self):
        name = self.new_char_name_input.text().strip()
        description = self.new_char_desc_input.toPlainText().strip()

        if not name:
            QMessageBox.warning(self, "Input Error", "Character name cannot be empty.")
            return

        self.create_char_button.setEnabled(False)
        creator = CharacterCreator(self.world, name, description)
        creator.signals.creation_complete.connect(self._on_character_created)
        creator.signals.creation_error.connect(self._on_creation_error)
        self.thread_pool.start(creator)

    def _on_character_created(self, entity_id: int, name: str):
        QMessageBox.information(self, "Success", f"Character '{name}' (ID: {entity_id}) created.")
        self.new_char_name_input.clear()
        self.new_char_desc_input.clear()
        self.create_char_button.setEnabled(True)
        self._load_characters()  # Refresh list

    def _on_creation_error(self, error_message: str):
        QMessageBox.critical(self, "Error Creating Character", error_message)
        self.create_char_button.setEnabled(True)

    def _apply_character_to_asset(self):
        if self.current_selected_asset_id is None:
            QMessageBox.warning(self, "No Asset", "No asset is currently selected in the main window.")
            return

        selected_rows = self.char_table.selectionModel().selectedRows()
        if not selected_rows:
            QMessageBox.warning(self, "No Character Selected", "Please select a character from the list to apply.")
            return

        selected_row_idx = selected_rows[0].row()
        character_id_item = self.char_table.item(selected_row_idx, 0)
        if not character_id_item:
            QMessageBox.critical(self, "Error", "Could not get character ID from table.")
            return

        character_concept_entity_id = character_id_item.data(Qt.ItemDataRole.UserRole)
        role = self.role_input.text().strip() or None  # Pass None if empty

        self.apply_char_button.setEnabled(False)
        applier = CharacterApplier(self.world, self.current_selected_asset_id, character_concept_entity_id, role)
        applier.signals.apply_complete.connect(self._on_apply_complete)
        applier.signals.apply_error.connect(self._on_apply_error)
        self.thread_pool.start(applier)

    def _on_apply_complete(self, message: str):
        QMessageBox.information(self, "Success", message)
        self.apply_char_button.setEnabled(True)
        self.role_input.clear()

    def _on_apply_error(self, error_message: str):
        QMessageBox.critical(self, "Error Applying Character", error_message)
        self.apply_char_button.setEnabled(True)


if __name__ == "__main__":
    # Example usage (requires a dummy world or proper setup)
    import tempfile

    from dam.core.database import DatabaseManager
    from dam.core.world import World, WorldConfig

    logging.basicConfig(level=logging.DEBUG)

    # Minimal world setup for testing the dialog
    # In a real app, the world comes from the main application
    temp_dir = tempfile.mkdtemp()
    db_path = f"{temp_dir}/test_char_dialog.db"
    test_config = WorldConfig(
        name="test_char_dialog_world",
        DATABASE_URL=f"sqlite+aiosqlite:///{db_path}",
        ASSET_STORAGE_PATH=f"{temp_dir}/assets",
        LOG_LEVEL="DEBUG",
    )
    db_manager = DatabaseManager(test_config.DATABASE_URL, is_memory_db=False)  # Not in memory for this test
    test_world = World(name="test_char_dialog_world", config=test_config, db_manager=db_manager)

    async def setup_db_for_dialog_test(world_instance):
        await world_instance.create_db_and_tables()  # Ensure tables exist

    asyncio.run(setup_db_for_dialog_test(test_world))

    app = QApplication([])
    # Simulate a selected asset ID
    dialog = CharacterManagementDialog(world=test_world, current_selected_asset_id=1)
    dialog.show()
    app.exec()

    # Clean up test DB
    import os

    os.unlink(db_path)
    os.rmdir(f"{temp_dir}/assets")
    os.rmdir(temp_dir)
