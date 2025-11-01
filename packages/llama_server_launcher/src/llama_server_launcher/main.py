"""Main module for the Llama Server Launcher GUI."""

import contextlib
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import gguf
import toml
from huggingface_hub import scan_cache_dir
from PySide6.QtCore import QProcess, Qt
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

CONFIG_FILE = Path("config.toml")


class ModelFile:
    """Stores information about a scanned model file."""

    def __init__(self, repo_id: str, path: Path):
        """Initialise the model file."""
        self.repo_id = repo_id
        self.path = path
        self.filename = path.name

    def __repr__(self) -> str:
        """Return the string representation of the model file."""
        return f"{self.repo_id} ({self.filename})"


class LlamaServerLauncher(QMainWindow):
    """Main application window."""

    def __init__(self):
        """Initialise the main window."""
        super().__init__()
        self.setWindowTitle("Llama Server Launcher (Hugging Face Cache)")
        self.setGeometry(100, 100, 1100, 750)

        self.gguf_files: list[ModelFile] = []
        self.mmproj_files: list[ModelFile] = []
        self.config: dict[str, Any] = {}
        self.server_process: QProcess | None = None

        self.load_config()
        self.init_ui()
        self.scan_cache_and_populate_ui()

    def init_ui(self):
        """Initialise all UI components."""
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setMinimumWidth(450)

        left_layout.addWidget(QLabel("1. Select Preset (or configure manually)"))
        self.preset_combo = QComboBox()
        self.preset_combo.currentIndexChanged.connect(self.apply_preset)
        left_layout.addWidget(self.preset_combo)

        left_layout.addWidget(QLabel("2. Select Model"))
        self.gguf_list = QListWidget()
        self.gguf_list.itemSelectionChanged.connect(self.on_model_selection_change)
        left_layout.addWidget(self.gguf_list)

        left_layout.addWidget(QLabel("3. (Optional) Select mmproj file (for LLaVA)"))
        self.mmproj_combo = QComboBox()
        left_layout.addWidget(self.mmproj_combo)

        left_layout.addWidget(QLabel("GGUF Model Info (read by gguf-py):"))
        self.model_info_box = QTextEdit()
        self.model_info_box.setReadOnly(True)
        self.model_info_box.setFontFamily("Courier")
        self.model_info_box.setMinimumHeight(150)
        left_layout.addWidget(self.model_info_box)

        self.rescan_button = QPushButton("Rescan Cache")
        self.rescan_button.clicked.connect(self.scan_cache_and_populate_ui)
        left_layout.addWidget(self.rescan_button)

        self.save_preset_button = QPushButton("Save Preset As...")
        self.save_preset_button.clicked.connect(self.save_preset)
        left_layout.addWidget(self.save_preset_button)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        right_layout.addWidget(QLabel("4. Check Launch Parameters"))
        self.args_input = QTextEdit()
        self.args_input.setFontFamily("Courier")
        right_layout.addWidget(self.args_input)

        button_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Server")
        self.start_button.setStyleSheet("background-color: #4CAF50; color: white;")
        self.start_button.clicked.connect(self.start_server)

        self.stop_button = QPushButton("Stop Server")
        self.stop_button.setStyleSheet("background-color: #f44336; color: white;")
        self.stop_button.clicked.connect(self.stop_server)
        self.stop_button.setEnabled(False)

        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        right_layout.addLayout(button_layout)

        right_layout.addWidget(QLabel("Server Log:"))
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setFontFamily("Courier")
        right_layout.addWidget(self.log_output)

        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([500, 600])

    def load_config(self):
        """Load and process config.toml, handling preset inheritance."""
        if not CONFIG_FILE.exists():
            # self.log_message(f"Config file not found at {CONFIG_FILE}, creating a default one.\n")
            self.config = {
                "default_preset": "default",
                "presets": {
                    "default": {"extra_args": ["--host 127.0.0.1", "--port 8080", "-ngl 32", "-c 2048"]},
                    "Llama-3-8B": {
                        "override": "default",
                        "model": {
                            "repo_id": "NousResearch/Meta-Llama-3-8B-Instruct-GGUF",
                            "filename_pattern": "*Q4_K_M.gguf",
                        },
                        "extra_args": ["-ngl -1", "-c 8192"],
                    },
                },
            }
            try:
                with CONFIG_FILE.open("w", encoding="utf-8") as f:
                    toml.dump(self.config, f)
            except Exception as e:
                self.show_error(f"Failed to create default config file: {e}")
        else:
            try:
                with CONFIG_FILE.open(encoding="utf-8") as f:
                    self.config = toml.load(f)
            except Exception as e:
                self.show_error(f"Failed to load {CONFIG_FILE}: {e}")
                self.config = {"presets": {}}

        self.resolve_presets()

    def resolve_presets(self):
        """Resolve preset inheritance."""
        presets = self.config.get("presets", {})
        resolved = {}
        for name, _ in presets.items():
            try:
                resolved[name] = self.resolve_single_preset(name, presets, set())
            except RecursionError:
                self.show_error(f"Circular dependency detected in preset '{name}'")
        self.config["presets"] = resolved

    def resolve_single_preset(self, name: str, presets: dict[str, Any], visited: set[str]) -> dict[str, Any]:
        """Recursively resolve a single preset."""
        if name in visited:
            raise RecursionError(f"Circular dependency detected: {name}")
        visited.add(name)

        preset = presets.get(name, {})
        if "override" in preset:
            base = self.resolve_single_preset(preset["override"], presets, visited)
            merged = base.copy()
            merged.update(preset)
            return merged
        return preset

    def scan_cache_and_populate_ui(self):
        """Scan the cache using huggingface-hub and populates the UI."""
        self.log_message("Scanning local cache with 'huggingface-hub'...\n")
        self.gguf_files.clear()
        self.mmproj_files.clear()
        self.gguf_list.clear()
        self.mmproj_combo.clear()
        self.model_info_box.clear()

        try:
            cache_info = scan_cache_dir()
        except Exception as e:
            self.show_error(f"Failed to scan Hugging Face cache: {e}")
            return

        all_gguf_files: list[ModelFile] = []
        for repo in cache_info.repos:  # pyright: ignore
            repo_id = repo.repo_id
            for revision in repo.revisions:
                for file in revision.files:
                    if file.file_path.suffix.lower() == ".gguf":
                        all_gguf_files.append(ModelFile(repo_id, file.file_path))  # pyright: ignore

        for model_file in all_gguf_files:
            try:
                reader = gguf.GGUFReader(str(model_file.path))  # pyright: ignore
                arch_field = reader.get_field("general.architecture")
                if arch_field and arch_field.contents() == "clip":
                    self.mmproj_files.append(model_file)
                elif "-of-" not in model_file.filename or "-00001-of-" in model_file.filename:
                    self.gguf_files.append(model_file)
            except Exception:
                # Ignore files that can't be read
                pass

        self.gguf_files.sort(key=lambda x: (x.repo_id, x.filename))
        for model_file in self.gguf_files:
            item = QListWidgetItem(str(model_file))
            item.setData(Qt.ItemDataRole.UserRole, model_file)
            self.gguf_list.addItem(item)

        self.mmproj_combo.addItem("None (Language model only)", None)
        self.mmproj_files.sort(key=lambda x: (x.repo_id, x.filename))
        for model_file in self.mmproj_files:
            self.mmproj_combo.addItem(str(model_file), model_file)

        self.log_message(
            f"Scan complete: Found {len(self.gguf_files)} language models and {len(self.mmproj_files)} projector models.\n"
        )
        self.populate_presets()

    def populate_presets(self):
        """Populate the preset dropdown from config.toml."""
        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()
        default_preset_name = self.config.get("default_preset", "default")
        self.preset_combo.addItem(f"--- Default ({default_preset_name}) ---", "default")
        for name, preset in self.config.get("presets", {}).items():
            self.preset_combo.addItem(name, preset)
        self.preset_combo.blockSignals(False)
        self.preset_combo.setCurrentIndex(0)
        self.apply_preset(0)

    def apply_preset(self, index: int):
        """Apply a selected preset."""
        self.model_info_box.clear()
        preset_data = self.preset_combo.itemData(index)

        if preset_data == "default":
            default_preset_name = self.config.get("default_preset", "default")
            preset_data = self.config.get("presets", {}).get(default_preset_name, {})

        if isinstance(preset_data, dict):
            self.args_input.setText("\n".join(preset_data.get("extra_args", [])))  # pyright: ignore
            model_info = preset_data.get("model",None)
            if isinstance(model_info, dict):
                self.select_gguf_by_repo_id(model_info.get("repo_id"))  # pyright: ignore
            else:
                self.gguf_list.clearSelection()

            mmproj_info = preset_data.get("mmproj",None)
            if isinstance(mmproj_info, dict):
                self.select_mmproj_by_repo_id(mmproj_info.get("repo_id"))  # pyright: ignore
            else:
                self.mmproj_combo.setCurrentIndex(0)

        selected_items = self.gguf_list.selectedItems()
        if selected_items:
            self.show_gguf_info(selected_items[0])

    def select_gguf_by_repo_id(self, repo_id: str | None):
        """Select a GGUF item by repo_id."""
        if not repo_id:
            self.gguf_list.clearSelection()
            return
        for i in range(self.gguf_list.count()):
            item = self.gguf_list.item(i)
            model_file: ModelFile = item.data(Qt.ItemDataRole.UserRole)  # pyright: ignore
            if model_file.repo_id == repo_id:
                item.setSelected(True)
                self.gguf_list.scrollToItem(item)
                return
        self.gguf_list.clearSelection()
        self.log_message(f"Warning: Preset requires {repo_id}, but it was not found in the cache.\n")

    def select_mmproj_by_repo_id(self, repo_id: str | None):
        """Select an mmproj item by repo_id."""
        if not repo_id:
            self.mmproj_combo.setCurrentIndex(0)
            return
        for i in range(self.mmproj_combo.count()):
            model_file: ModelFile | None = self.mmproj_combo.itemData(i)  # pyright: ignore
            if model_file and model_file.repo_id == repo_id:
                self.mmproj_combo.setCurrentIndex(i)
                return
        self.mmproj_combo.setCurrentIndex(0)
        self.log_message(f"Warning: Preset requires {repo_id} (mmproj), but it was not found in the cache.\n")

    def on_model_selection_change(self):
        """Handle manual model selection."""
        selected_items = self.gguf_list.selectedItems()
        if not selected_items:
            self.model_info_box.clear()
            return

        if self.preset_combo.currentIndex() != 0:
            self.preset_combo.setCurrentIndex(0)

        self.show_gguf_info(selected_items[0])

    def show_gguf_info(self, item: QListWidgetItem):
        """Display GGUF metadata and auto-selects mmproj file if needed."""
        model_file: ModelFile = item.data(Qt.ItemDataRole.UserRole)  # pyright: ignore
        self.model_info_box.clear()

        try:
            reader = gguf.GGUFReader(str(model_file.path))  # pyright: ignore
            info = [f"File: {model_file.filename}", f"Repo: {model_file.repo_id}\n", "--- GGUF Metadata ---"]
            fields_to_show = [
                "general.architecture",
                "general.name",
                "llama.context_length",
                "llama.embedding_length",
                "llama.block_count",
                "llama.attention.head_count",
                "llama.attention.head_count_kv",
                "vision_model.mmproj_model_file",
            ]

            mmproj_filename = None

            for key in fields_to_show:
                field = reader.get_field(key)  # pyright: ignore
                if field:
                    value = field.contents()
                    info.append(f"{key:<30}: {value}")
                    if key == "vision_model.mmproj_model_file":
                        mmproj_filename = value

            info.append(f"\\nTotal Tensors: {len(reader.tensors)}")  # pyright: ignore
            self.model_info_box.setText("\\n".join(info))

            if mmproj_filename:
                self.select_mmproj_by_filename(mmproj_filename)  # pyright: ignore
            else:
                self.mmproj_combo.setCurrentIndex(0)

        except Exception as e:
            self.model_info_box.setText(f"Could not read GGUF metadata:\\n{e}")

    def select_mmproj_by_filename(self, filename: str):
        """Select an mmproj item by filename."""
        for i in range(self.mmproj_combo.count()):
            model_file: ModelFile | None = self.mmproj_combo.itemData(i)  # pyright: ignore
            if model_file and model_file.filename == filename:
                self.mmproj_combo.setCurrentIndex(i)
                return
        self.mmproj_combo.setCurrentIndex(0)
        self.log_message(f"Warning: Model requires {filename}, but it was not found in the cache.\n")

    def save_preset(self):
        """Save the current settings as a new preset."""
        selected_items = self.gguf_list.selectedItems()
        if not selected_items:
            self.show_error("Please select a GGUF model file first!")
            return

        model_file: ModelFile = selected_items[0].data(Qt.ItemDataRole.UserRole)  # pyright: ignore
        mmproj_model: ModelFile | None = self.mmproj_combo.currentData()  # pyright: ignore
        extra_args = self.args_input.toPlainText().split("\n")

        name, ok = QInputDialog.getText(self, "Save Preset", "Preset Name:")  # pyright: ignore
        if ok and name:
            new_preset = {
                "model": {"repo_id": model_file.repo_id, "filename_pattern": model_file.filename},
                "extra_args": extra_args,
            }
            if mmproj_model:
                new_preset["mmproj"] = {"repo_id": mmproj_model.repo_id, "filename_pattern": mmproj_model.filename}

            self.config["presets"][name] = new_preset
            try:
                with CONFIG_FILE.open("w", encoding="utf-8") as f:
                    toml.dump(self.config, f)
                self.populate_presets()
                self.preset_combo.setCurrentText(name)
            except Exception as e:
                self.show_error(f"Failed to save preset: {e}")

    def start_server(self):
        """Start the llama-server process."""
        selected_items = self.gguf_list.selectedItems()
        if not selected_items:
            self.show_error("Please select a GGUF model file first!")
            return

        model_file: ModelFile = selected_items[0].data(Qt.ItemDataRole.UserRole)  # pyright: ignore
        command = "llama-server"
        args = ["-m", str(model_file.path)]

        extra_args_str = self.args_input.toPlainText()
        args.extend(shlex.split(extra_args_str))

        mmproj_model: ModelFile | None = self.mmproj_combo.currentData()  # pyright: ignore
        if mmproj_model:
            args.extend(["--mmproj", str(mmproj_model.path)])

        if self.server_process:
            self.log_message("Server is already running...\n")
            return

        self.server_process = QProcess()
        self.server_process.readyReadStandardOutput.connect(self.handle_stdout)
        self.server_process.readyReadStandardError.connect(self.handle_stderr)
        self.server_process.finished.connect(self.on_server_finished)

        full_command_str = f"{command} {' '.join(args)}"
        self.log_message(f"Starting server...\n> {full_command_str}\n\n")

        try:
            self.server_process.start(command, args)
        except Exception as e:
            self.log_message(f"Failed to start: {e}\n")
            self.log_message("Error: Is 'llama.cpp' installed and in your system's PATH?\n")
            self.server_process = None
            return

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

    def stop_server(self):
        """Stop the server process."""
        if self.server_process:
            self.log_message("\nStopping server...\n")
            self.server_process.kill()
            self.server_process = None
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def handle_stdout(self):
        """Handle stdout from the server process."""
        if self.server_process:
            data = self.server_process.readAllStandardOutput()
            self.log_message(data.data().decode("utf-8", errors="ignore"))  # pyright: ignore

    def handle_stderr(self):
        """Handle stderr from the server process."""
        if self.server_process:
            data = self.server_process.readAllStandardError()
            self.log_message(f"[STDERR] {data.data().decode('utf-8', errors='ignore')}")

    def on_server_finished(self):
        """Handle the server process finishing."""
        self.log_message("\n--- Server process stopped ---\n")
        self.server_process = None
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def log_message(self, message: str):
        """Log a message to the log output."""
        self.log_output.insertPlainText(message)

    def show_error(self, message: str):
        """Show an error message to the user."""
        self.log_message(f"[ERROR] {message}\n")
        QMessageBox.critical(self, "Error", message)

    def closeEvent(self, event: QCloseEvent):  # noqa: N802
        """Handle the close event."""
        self.stop_server()
        event.accept()


def main():
    """Run the application."""
    with contextlib.suppress(FileNotFoundError, subprocess.CalledProcessError):
        subprocess.run(["llama-server", "--version"], capture_output=True, check=True, text=True)

    app = QApplication(sys.argv)
    window = LlamaServerLauncher()
    window.show()
    sys.exit(app.exec())
