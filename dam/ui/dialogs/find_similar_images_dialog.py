import asyncio
import uuid
from pathlib import Path
from typing import Any, Dict  # Renamed List to TypingList
from typing import List as TypingList

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap  # For displaying query image preview
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,  # To show query image and results
    QVBoxLayout,
)

from dam.core.events import FindSimilarImagesQuery
from dam.core.world import World

# ComponentViewerDialog will be imported locally in view_result_details to avoid circular import
# from dam.ui.main_window import ComponentViewerDialog

# For image preview, ensure Pillow is available or handle gracefully
try:
    from PIL import Image as PILImage
    from PIL.ImageQt import ImageQt

    _pil_available = True
except ImportError:
    _pil_available = False


class FindSimilarImagesDialog(QDialog):
    def __init__(self, current_world: World, parent=None):
        super().__init__(parent)
        self.current_world = current_world
        self.setWindowTitle("Find Similar Images")
        self.setMinimumWidth(700)  # Increased width
        self.setMinimumHeight(500)

        main_layout = QVBoxLayout(self)

        # Top part for image selection and parameters
        form_area_layout = QFormLayout()

        # Image Path
        self.image_path_input = QLineEdit()
        self.browse_image_button = QPushButton("Browse Image...")
        self.browse_image_button.clicked.connect(self.browse_for_image)
        image_path_layout = QHBoxLayout()
        image_path_layout.addWidget(self.image_path_input)
        image_path_layout.addWidget(self.browse_image_button)
        form_area_layout.addRow(QLabel("Query Image:"), image_path_layout)

        # Thresholds
        self.phash_threshold_spin = QSpinBox()
        self.phash_threshold_spin.setRange(0, 64)
        self.phash_threshold_spin.setValue(4)
        form_area_layout.addRow(QLabel("pHash Threshold:"), self.phash_threshold_spin)

        self.ahash_threshold_spin = QSpinBox()
        self.ahash_threshold_spin.setRange(0, 64)
        self.ahash_threshold_spin.setValue(4)
        form_area_layout.addRow(QLabel("aHash Threshold:"), self.ahash_threshold_spin)

        self.dhash_threshold_spin = QSpinBox()
        self.dhash_threshold_spin.setRange(0, 64)
        self.dhash_threshold_spin.setValue(4)
        form_area_layout.addRow(QLabel("dHash Threshold:"), self.dhash_threshold_spin)

        main_layout.addLayout(form_area_layout)

        # Splitter for image preview and results
        self.splitter = QSplitter(Qt.Orientation.Horizontal)

        # Image Preview Area
        self.image_preview_label = QLabel("Image Preview")
        self.image_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_preview_label.setMinimumWidth(200)  # Min width for preview
        self.image_preview_label.setStyleSheet("border: 1px solid gray;")
        self.splitter.addWidget(self.image_preview_label)

        # Results List
        self.results_list_widget = QListWidget()
        self.results_list_widget.itemDoubleClicked.connect(self.view_result_details)
        self.splitter.addWidget(self.results_list_widget)

        self.splitter.setSizes([200, 500])  # Initial sizes for preview and list
        main_layout.addWidget(self.splitter)

        # Buttons
        button_layout = QHBoxLayout()
        self.find_button = QPushButton("Find Similar Images")
        self.find_button.clicked.connect(self.find_similar)
        self.cancel_button = QPushButton("Close")  # Changed from Cancel to Close
        self.cancel_button.clicked.connect(self.accept)  # Accept will just close it
        button_layout.addStretch()
        button_layout.addWidget(self.find_button)
        button_layout.addWidget(self.cancel_button)
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)
        self.image_path_input.textChanged.connect(self.update_image_preview)

    def browse_for_image(self):
        path_str, _ = QFileDialog.getOpenFileName(
            self, "Select Query Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.gif)"
        )
        if path_str:
            self.image_path_input.setText(path_str)
            # update_image_preview will be called by textChanged signal

    def update_image_preview(self, path_str: str):
        if not _pil_available:
            self.image_preview_label.setText("Image preview requires Pillow.")
            return

        if not path_str:
            self.image_preview_label.setText("Select an image to preview.")
            self.image_preview_label.setPixmap(QPixmap())  # Clear pixmap
            return

        path = Path(path_str)
        if path.is_file():
            try:
                pil_img = PILImage.open(path)
                # Resize for preview
                pil_img.thumbnail(
                    (self.image_preview_label.width() - 10, self.image_preview_label.height() - 10),
                    PILImage.Resampling.LANCZOS,
                )
                qt_image = ImageQt(pil_img)
                pixmap = QPixmap.fromImage(qt_image)
                self.image_preview_label.setPixmap(pixmap)
            except Exception as e:
                self.image_preview_label.setText(f"Cannot preview image:\n{e}")
                self.image_preview_label.setPixmap(QPixmap())
        else:
            self.image_preview_label.setText("Image file not found.")
            self.image_preview_label.setPixmap(QPixmap())

    def find_similar(self):
        image_path_str = self.image_path_input.text().strip()
        if not image_path_str:
            QMessageBox.warning(self, "Input Error", "Please select an image file.")
            return

        image_path = Path(image_path_str)
        if not image_path.is_file():
            QMessageBox.warning(self, "Input Error", f"Image file not found: {image_path_str}")
            return

        phash_thresh = self.phash_threshold_spin.value()
        ahash_thresh = self.ahash_threshold_spin.value()
        dhash_thresh = self.dhash_threshold_spin.value()

        request_id = str(uuid.uuid4())
        query_event = FindSimilarImagesQuery(
            image_path=image_path,
            phash_threshold=phash_thresh,
            ahash_threshold=ahash_thresh,
            dhash_threshold=dhash_thresh,
            world_name=self.current_world.name,
            request_id=request_id,
        )

        self.results_list_widget.clear()
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:

            async def dispatch_query_sync():
                await self.current_world.dispatch_event(query_event)

            asyncio.run(dispatch_query_sync())

            QApplication.restoreOverrideCursor()

            if query_event.result:
                if isinstance(query_event.result, list) and query_event.result and "error" in query_event.result[0]:
                    QMessageBox.warning(self, "Query Error", f"Error from system: {query_event.result[0]['error']}")
                    self.results_list_widget.addItem(f"Error: {query_event.result[0]['error']}")
                elif not query_event.result:
                    self.results_list_widget.addItem("No similar images found.")
                else:
                    for match in query_event.result:
                        # Ensure match is a dict, as per event definition
                        if isinstance(match, dict):
                            item_text = (
                                f"ID: {match.get('entity_id', 'N/A')} - "
                                f"{match.get('original_filename', 'N/A')} "
                                f"(Dist: {match.get('distance', '?')} by {match.get('hash_type', '?')})"
                            )
                            list_item = QListWidgetItem(item_text)
                            # Store entity_id for double-click action
                            list_item.setData(Qt.ItemDataRole.UserRole, match.get("entity_id"))
                            self.results_list_widget.addItem(list_item)
                        else:
                            self.results_list_widget.addItem(f"Unexpected result format: {match}")
            else:
                self.results_list_widget.addItem("No results or error from similarity query.")

        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Find Similar Error", f"An error occurred: {e}")
            self.results_list_widget.addItem(f"Error: {e}")
        finally:
            if QApplication.overrideCursor() is not None:
                QApplication.restoreOverrideCursor()

    def view_result_details(self, item: QListWidgetItem):
        entity_id = item.data(Qt.ItemDataRole.UserRole)
        if entity_id is None:
            return  # No ID stored

        # Fetch and display components for this entity_id using ComponentViewerDialog
        # This logic is similar to MainWindow's on_asset_double_clicked
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        components_data_for_dialog: Dict[str, TypingList[Dict[str, Any]]] = {}
        try:
            # Local import to avoid circular dependency
            from dam.models.core.base_component import REGISTERED_COMPONENT_TYPES
            from dam.services import ecs_service
            from dam.ui.main_window import ComponentViewerDialog

            with self.current_world.get_db_session() as session:
                entity = ecs_service.get_entity(session, entity_id)
                if not entity:
                    QMessageBox.warning(self, "Error", f"Entity ID {entity_id} not found.")
                    QApplication.restoreOverrideCursor()
                    return

                for comp_type_name, comp_type_cls in REGISTERED_COMPONENT_TYPES.items():
                    components = ecs_service.get_components(session, entity_id, comp_type_cls)
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

            QApplication.restoreOverrideCursor()
            viewer_dialog = ComponentViewerDialog(entity_id, components_data_for_dialog, self.current_world.name, self)
            viewer_dialog.exec()

        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(
                self, "Component Fetch Error", f"Error fetching components for Entity ID {entity_id}: {e}"
            )
        finally:
            if QApplication.overrideCursor() is not None:
                QApplication.restoreOverrideCursor()


if __name__ == "__main__":
    import sys

    from PyQt6.QtWidgets import QApplication

    class MockWorld:
        def __init__(self, name="test_world_similar"):
            self.name = name

        async def dispatch_event(self, event: FindSimilarImagesQuery):
            print(f"MockWorld: Event dispatched: {type(event).__name__} for image {event.image_path.name}")
            await asyncio.sleep(0.2)  # Simulate work
            if "error" in event.image_path.name:  # Test error case
                event.result = [{"error": "Simulated system error during similarity search."}]
            elif "empty" in event.image_path.name:  # Test no results
                event.result = []
            else:
                event.result = [
                    {"entity_id": 101, "original_filename": "similar1.jpg", "distance": 1, "hash_type": "phash"},
                    {"entity_id": 102, "original_filename": "closematch.png", "distance": 3, "hash_type": "ahash"},
                ]

        def get_db_session(self):  # For ComponentViewerDialog if used
            class MockSession:
                def __enter__(self):
                    return self

                def __exit__(self, t, v, tb):
                    pass

                def query(self, *args):
                    return self  # Dummy query

                def filter(self, *args):
                    return self  # Dummy filter

                def all(self):
                    return []  # Dummy all

            return MockSession()

    app = QApplication(sys.argv)
    world_to_use = MockWorld()

    # Example of trying to load a real world for more integrated testing
    # try:
    #     from dam.core import config as app_config
    #     from dam.core.world import get_world, create_and_register_all_worlds_from_settings
    #     from dam.core.world_setup import register_core_systems
    #     worlds = create_and_register_all_worlds_from_settings(app_settings=app_config.settings)
    #     for w in worlds: register_core_systems(w)
    #     real_world = get_world(app_config.settings.DEFAULT_WORLD_NAME or (worlds[0].name if worlds else None) )
    #     if real_world: world_to_use = real_world
    #     print(f"Using {'REAL' if real_world else 'MOCK'} world: {world_to_use.name}")
    # except Exception as e:
    #     print(f"Could not load real world for testing FindSimilarImagesDialog ({e}), using MockWorld.")

    dialog = FindSimilarImagesDialog(current_world=world_to_use)
    dialog.exec()
    sys.exit(app.exec())
