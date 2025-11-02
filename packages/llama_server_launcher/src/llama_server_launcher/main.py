"""Main module for the Llama Server Launcher GUI."""

import contextlib
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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

from .gguf_reader import GGUFMetadataReader
from .metadata_cache import (
    get_cached_gguf_metadata,
    load_metadata_cache,
    save_metadata_cache,
    update_gguf_metadata_cache,
)

CONFIG_FILE = Path("config.toml")


@dataclass
class ModelFile:
    """Stores information about a scanned model file."""

    repo_id: str
    path: Path
    relative_path: str

    @property
    def filename(self) -> str:
        """Return the filename."""
        return self.path.name

    def __repr__(self) -> str:
        """Return the string representation of the model file."""
        return f"{self.repo_id}/{self.relative_path}"


@dataclass
class ModelInfo:
    """Preset information for a model."""

    repo_id: str
    filename_pattern: str


@dataclass
class Argument:
    """A single argument for the server."""

    value: str
    enabled: bool = True


@dataclass
class Preset:
    """A preset for the server."""

    model: ModelInfo | None = None
    mmproj: ModelInfo | None = None
    extra_args: list[Argument] | None = None
    override: str | None = None


def get_default_file_manager_command() -> str:
    """Get the default file manager command based on the operating system."""
    if sys.platform == "win32":
        return "explorer"
    if sys.platform == "darwin":
        return "open"
    return "xdg-open"


@dataclass
class Config:
    """Application configuration."""

    llama_server_command: str = "llama-server"
    file_manager_command: str = field(default_factory=get_default_file_manager_command)
    default_preset: str = "default"
    presets: dict[str, Preset] = field(default_factory=lambda: {})


class LlamaServerLauncher(QMainWindow):
    """Main application window."""

    def __init__(self) -> None:
        """Initialise the main window."""
        super().__init__()
        self.setWindowTitle("Llama Server Launcher (Hugging Face Cache)")
        self.setGeometry(100, 100, 1100, 750)

        self.gguf_files: list[ModelFile] = []
        self.mmproj_files: list[ModelFile] = []
        self.config = Config()
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

        open_in_fm_layout = QHBoxLayout()
        self.open_model_dir_button = QPushButton("Open Model Folder")
        self.open_model_dir_button.clicked.connect(self.open_model_dir)
        self.open_mmproj_dir_button = QPushButton("Open mmproj Folder")
        self.open_mmproj_dir_button.clicked.connect(self.open_mmproj_dir)
        open_in_fm_layout.addWidget(self.open_model_dir_button)
        open_in_fm_layout.addWidget(self.open_mmproj_dir_button)
        left_layout.addLayout(open_in_fm_layout)

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

        self.update_preset_button = QPushButton("Update and Save Current Preset")
        self.update_preset_button.clicked.connect(self.update_preset)
        left_layout.addWidget(self.update_preset_button)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        right_layout.addWidget(QLabel("4. Check Launch Parameters"))

        self.args_list = QListWidget()
        self.args_list.itemChanged.connect(self.on_arg_item_changed)
        right_layout.addWidget(self.args_list)

        args_button_layout = QHBoxLayout()
        self.add_arg_button = QPushButton("Add")
        self.add_arg_button.clicked.connect(self.add_arg)
        self.edit_arg_button = QPushButton("Edit")
        self.edit_arg_button.clicked.connect(self.edit_arg)
        self.remove_arg_button = QPushButton("Remove")
        self.remove_arg_button.clicked.connect(self.remove_arg)
        args_button_layout.addWidget(self.add_arg_button)
        args_button_layout.addWidget(self.edit_arg_button)
        args_button_layout.addWidget(self.remove_arg_button)
        right_layout.addLayout(args_button_layout)

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

    def load_config(self) -> None:
        """Load and process config.toml, handling preset inheritance."""
        if not CONFIG_FILE.exists():
            self.create_default_config()
        else:
            try:
                with CONFIG_FILE.open(encoding="utf-8") as f:
                    config_data = toml.load(f)
                self.config = self.config_from_dict(config_data)
            except Exception as e:
                self.show_error(f"Failed to load {CONFIG_FILE}: {e}")
                self.config = Config()

        self.resolve_presets()

    def config_from_dict(self, config_data: dict[str, Any]) -> Config:
        """Create a Config object from a dictionary."""
        presets: dict[str, Preset] = {}
        presets_data = config_data.get("presets", {})
        if isinstance(presets_data, dict):
            for name, preset_data in presets_data.items():
                if isinstance(preset_data, dict):
                    model_data = preset_data.get("model")
                    model = (
                        ModelInfo(**model_data)  # type: ignore
                        if isinstance(model_data, dict)
                        else None
                    )

                    mmproj_data = preset_data.get("mmproj")
                    mmproj = (
                        ModelInfo(**mmproj_data)  # type: ignore
                        if isinstance(mmproj_data, dict)
                        else None
                    )
                    extra_args_data = preset_data.get("extra_args", [])
                    extra_args: list[Argument] = []
                    if isinstance(extra_args_data, list):
                        for arg_data in extra_args_data:
                            if isinstance(arg_data, dict):
                                extra_args.append(Argument(**arg_data))  # type: ignore
                            else:
                                extra_args.append(Argument(value=str(arg_data)))  # type: ignore
                    elif isinstance(extra_args_data, str):
                        extra_args.append(Argument(value=extra_args_data))

                    presets[name] = Preset(
                        model=model,
                        mmproj=mmproj,
                        extra_args=extra_args,
                        override=preset_data.get("override"),  # type: ignore
                    )
        return Config(
            llama_server_command=config_data.get("llama_server_command", "llama-server"),
            file_manager_command=config_data.get("file_manager_command", get_default_file_manager_command()),
            default_preset=config_data.get("default_preset", "default"),
            presets=presets,
        )

    def create_default_config(self) -> None:
        """Create and save a default configuration."""
        self.config = Config(
            presets={
                "default": Preset(
                    extra_args=[
                        Argument("--host 127.0.0.1"),
                        Argument("--port 8080"),
                        Argument("-ngl 32"),
                        Argument("-c 2048"),
                    ]
                ),
                "Llama-3-8B": Preset(
                    override="default",
                    model=ModelInfo(
                        repo_id="NousResearch/Meta-Llama-3-8B-Instruct-GGUF",
                        filename_pattern="*Q4_K_M.gguf",
                    ),
                    extra_args=[Argument("-ngl -1"), Argument("-c 8192")],
                ),
            }
        )
        self.save_config()

    def save_config(self) -> None:
        """Save the current configuration to config.toml."""
        config_data: dict[str, Any] = {
            "llama_server_command": self.config.llama_server_command,
            "file_manager_command": self.config.file_manager_command,
            "default_preset": self.config.default_preset,
            "presets": {},
        }
        for name, preset in self.config.presets.items():
            preset_data: dict[str, Any] = {}
            if preset.extra_args is not None:
                preset_data["extra_args"] = [{"value": arg.value, "enabled": arg.enabled} for arg in preset.extra_args]
            if preset.model:
                preset_data["model"] = {
                    "repo_id": preset.model.repo_id,
                    "filename_pattern": preset.model.filename_pattern,
                }
            if preset.mmproj:
                preset_data["mmproj"] = {
                    "repo_id": preset.mmproj.repo_id,
                    "filename_pattern": preset.mmproj.filename_pattern,
                }
            if preset.override:
                preset_data["override"] = preset.override
            config_data["presets"][name] = preset_data

        try:
            with CONFIG_FILE.open("w", encoding="utf-8") as f:
                toml.dump(config_data, f)
        except Exception as e:
            self.show_error(f"Failed to save config file: {e}")

    def resolve_presets(self) -> None:
        """Resolve preset inheritance."""
        resolved: dict[str, Preset] = {}
        for name in self.config.presets:
            try:
                resolved[name] = self.resolve_single_preset(name, set())
            except RecursionError:
                self.show_error(f"Circular dependency detected in preset '{name}'")
        self.config.presets = resolved

    def resolve_single_preset(self, name: str, visited: set[str]) -> Preset:
        """Recursively resolve a single preset."""
        if name in visited:
            raise RecursionError(f"Circular dependency detected: {name}")
        visited.add(name)

        preset = self.config.presets.get(name)
        if not preset:
            return Preset()
        if preset.override:
            base = self.resolve_single_preset(preset.override, visited)
            # Create a new merged preset
            merged_model = preset.model or base.model
            merged_mmproj = preset.mmproj or base.mmproj
            merged_extra_args = preset.extra_args if preset.extra_args is not None else base.extra_args
            return Preset(
                model=merged_model,
                mmproj=merged_mmproj,
                extra_args=merged_extra_args,
            )
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
        for repo in cache_info.repos:
            repo_id = repo.repo_id
            for revision in repo.revisions:
                revision_path = Path(revision.snapshot_path)
                for file in revision.files:
                    if file.file_path.suffix.lower() == ".gguf":
                        relative_path = file.file_path.relative_to(revision_path).as_posix()
                        all_gguf_files.append(ModelFile(repo_id, file.file_path, relative_path))

        metadata_cache = load_metadata_cache()
        for model_file in all_gguf_files:
            metadata = get_cached_gguf_metadata(model_file.path, metadata_cache)
            if metadata is None:
                try:
                    reader = GGUFMetadataReader(str(model_file.path))
                    metadata = {"architecture": reader.get_field("general.architecture")}
                    update_gguf_metadata_cache(model_file.path, metadata, metadata_cache)
                except Exception:
                    # Ignore files that can't be read
                    continue
            arch = metadata.get("architecture")
            if arch == "clip":
                self.mmproj_files.append(model_file)
            elif "-of-" not in model_file.filename or "-00001-of-" in model_file.filename:
                self.gguf_files.append(model_file)

        save_metadata_cache(metadata_cache)
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

    def populate_presets(self) -> None:
        """Populate the preset dropdown from config.toml."""
        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()
        self.preset_combo.addItem(f"--- Default ({self.config.default_preset}) ---", "default")
        for name, preset in self.config.presets.items():
            self.preset_combo.addItem(name, preset)
        self.preset_combo.blockSignals(False)
        self.preset_combo.setCurrentIndex(0)
        self.apply_preset(0)

    def apply_preset(self, index: int) -> None:
        """Apply a selected preset."""
        self.model_info_box.clear()
        preset_data = self.preset_combo.itemData(index)

        preset: Preset | None = None
        if preset_data == "default":
            preset = self.config.presets.get(self.config.default_preset)
        elif isinstance(preset_data, Preset):
            preset = preset_data

        self.gguf_list.blockSignals(True)
        self.args_list.clear()
        try:
            if preset:
                for arg in preset.extra_args or []:
                    item = QListWidgetItem(arg.value)
                    item.setData(Qt.ItemDataRole.UserRole, arg)
                    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    item.setCheckState(Qt.CheckState.Checked if arg.enabled else Qt.CheckState.Unchecked)
                    self.args_list.addItem(item)
                if preset.model:
                    self.select_gguf_by_model_info(preset.model)
                else:
                    self.gguf_list.clearSelection()
                if preset.mmproj:
                    self.select_mmproj_by_model_info(preset.mmproj)
                else:
                    self.mmproj_combo.setCurrentIndex(0)
            else:
                self.args_input.clear()
                self.gguf_list.clearSelection()
                self.mmproj_combo.setCurrentIndex(0)
        finally:
            self.gguf_list.blockSignals(False)

        selected_items = self.gguf_list.selectedItems()
        if selected_items:
            self.show_gguf_info(selected_items[0])

    def select_gguf_by_model_info(self, model_info: ModelInfo | None) -> None:
        """Select a GGUF item by repo_id and filename pattern."""
        if not model_info:
            self.gguf_list.clearSelection()
            return
        for i in range(self.gguf_list.count()):
            item = self.gguf_list.item(i)
            model_file: ModelFile = item.data(Qt.ItemDataRole.UserRole)
            if model_file.repo_id == model_info.repo_id and Path(model_file.path).match(model_info.filename_pattern):
                item.setSelected(True)
                self.gguf_list.scrollToItem(item)
                return
        self.gguf_list.clearSelection()
        self.log_message(
            f"Warning: Preset requires repo '{model_info.repo_id}' with file pattern '{model_info.filename_pattern}', but it was not found in the cache.\n"
        )

    def select_mmproj_by_model_info(self, model_info: ModelInfo | None) -> None:
        """Select an mmproj item by repo_id and filename pattern."""
        if not model_info:
            self.mmproj_combo.setCurrentIndex(0)
            return
        for i in range(self.mmproj_combo.count()):
            model_file: ModelFile | None = self.mmproj_combo.itemData(i)
            if (
                model_file
                and model_file.repo_id == model_info.repo_id
                and Path(model_file.path).match(model_info.filename_pattern)
            ):
                self.mmproj_combo.setCurrentIndex(i)
                return
        self.mmproj_combo.setCurrentIndex(0)
        self.log_message(
            f"Warning: Preset requires mmproj repo '{model_info.repo_id}' with file pattern '{model_info.filename_pattern}', but it was not found in the cache.\n"
        )

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
            reader = GGUFMetadataReader(str(model_file.path))
            info = [
                f"File: {model_file.relative_path}",
                f"Repo: {model_file.repo_id}\n",
                "--- GGUF Metadata ---",
            ]
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

            for key in fields_to_show:
                value = reader.get_field(key)
                if value is not None:
                    info.append(f"{key:<30}: {value}")

            info.append(f"\nTotal Tensors: {reader.get_tensor_count()}")
            self.model_info_box.setText("\n".join(info))

        except Exception as e:
            self.model_info_box.setText(f"Could not read GGUF metadata:\n{e}")

    def update_preset(self) -> None:
        """Update the current preset with the current settings."""
        current_preset_name = self.preset_combo.currentText()
        if self.preset_combo.currentIndex() == 0 or current_preset_name not in self.config.presets:
            self.show_error("Please select a valid preset to update.")
            return

        preset = self.config.presets[current_preset_name]
        preset.extra_args = [
            self.args_list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.args_list.count())
        ]

        selected_items = self.gguf_list.selectedItems()
        if selected_items:
            model_file: ModelFile = selected_items[0].data(Qt.ItemDataRole.UserRole)  # pyright: ignore
            preset.model = ModelInfo(repo_id=model_file.repo_id, filename_pattern=model_file.relative_path)

        mmproj_model: ModelFile | None = self.mmproj_combo.currentData()  # pyright: ignore
        if mmproj_model:
            preset.mmproj = ModelInfo(repo_id=mmproj_model.repo_id, filename_pattern=mmproj_model.relative_path)
        else:
            preset.mmproj = None

        self.save_config()
        self.log_message(f"Preset '{current_preset_name}' updated successfully.\n")

    def select_mmproj_by_filename(self, filename: str) -> None:
        """Select an mmproj item by filename."""
        for i in range(self.mmproj_combo.count()):
            model_file: ModelFile | None = self.mmproj_combo.itemData(i)  # pyright: ignore
            if model_file and model_file.filename == filename:
                self.mmproj_combo.setCurrentIndex(i)
                return
        self.mmproj_combo.setCurrentIndex(0)
        self.log_message(f"Warning: Model requires {filename}, but it was not found in the cache.\n")

    def save_preset(self) -> None:
        """Save the current settings as a new preset."""
        selected_items = self.gguf_list.selectedItems()
        if not selected_items:
            self.show_error("Please select a GGUF model file first!")
            return

        model_file: ModelFile = selected_items[0].data(Qt.ItemDataRole.UserRole)
        mmproj_model: ModelFile | None = self.mmproj_combo.currentData()
        extra_args = [self.args_list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.args_list.count())]

        default_name = f"{model_file.repo_id}/{model_file.relative_path}"
        if mmproj_model:
            default_name += f" + {mmproj_model.repo_id}/{mmproj_model.relative_path}"
        name, ok = QInputDialog.getText(self, "Save Preset", "Preset Name:", text=default_name)
        if ok and name:
            model_info = ModelInfo(repo_id=model_file.repo_id, filename_pattern=model_file.relative_path)
            mmproj_info = (
                ModelInfo(repo_id=mmproj_model.repo_id, filename_pattern=mmproj_model.relative_path)
                if mmproj_model
                else None
            )

            new_preset = Preset(model=model_info, mmproj=mmproj_info, extra_args=extra_args)
            self.config.presets[name] = new_preset
            self.save_config()
            self.populate_presets()
            self.preset_combo.setCurrentText(name)

    def start_server(self) -> None:
        """Start the llama-server process."""
        selected_items = self.gguf_list.selectedItems()
        if not selected_items:
            self.show_error("Please select a GGUF model file first!")
            return

        model_file: ModelFile = selected_items[0].data(Qt.ItemDataRole.UserRole)
        command_parts = shlex.split(self.config.llama_server_command)
        command = command_parts[0]
        args = command_parts[1:]

        args.extend(["-m", str(model_file.path)])

        for i in range(self.args_list.count()):
            item = self.args_list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                arg: Argument = item.data(Qt.ItemDataRole.UserRole)
                args.extend(shlex.split(arg.value))

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

    def open_model_dir(self) -> None:
        """Open the directory of the selected model file."""
        selected_items = self.gguf_list.selectedItems()
        if not selected_items:
            self.show_error("Please select a GGUF model file first!")
            return
        model_file: ModelFile = selected_items[0].data(Qt.ItemDataRole.UserRole)
        self._open_directory(model_file.path.parent)

    def open_mmproj_dir(self) -> None:
        """Open the directory of the selected mmproj file."""
        mmproj_model: ModelFile | None = self.mmproj_combo.currentData()
        if not mmproj_model:
            self.show_error("Please select an mmproj file first!")
            return
        self._open_directory(mmproj_model.path.parent)

    def _open_directory(self, path: Path) -> None:
        """Open a directory in the default file manager."""
        command = self.config.file_manager_command
        try:
            subprocess.run([command, str(path)], check=True)
        except FileNotFoundError:
            self.show_error(f"File manager command not found: '{command}'. Please configure it in config.toml.")
        except subprocess.CalledProcessError as e:
            self.show_error(f"Failed to open file manager: {e}")

    def on_arg_item_changed(self, item: QListWidgetItem) -> None:
        """Handle when an argument item is changed (e.g., checkbox)."""
        arg: Argument = item.data(Qt.ItemDataRole.UserRole)
        arg.enabled = item.checkState() == Qt.CheckState.Checked

    def add_arg(self) -> None:
        """Add a new argument."""
        text, ok = QInputDialog.getText(self, "Add Argument", "Argument:")
        if ok and text:
            arg = Argument(value=text)
            item = QListWidgetItem(arg.value)
            item.setData(Qt.ItemDataRole.UserRole, arg)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            self.args_list.addItem(item)

    def edit_arg(self) -> None:
        """Edit the selected argument."""
        selected_items = self.args_list.selectedItems()
        if not selected_items:
            return
        item = selected_items[0]
        arg: Argument = item.data(Qt.ItemDataRole.UserRole)
        new_value, ok = QInputDialog.getText(self, "Edit Argument", "Argument:", text=arg.value)
        if ok and new_value:
            arg.value = new_value
            item.setText(new_value)

    def remove_arg(self) -> None:
        """Remove the selected argument."""
        selected_items = self.args_list.selectedItems()
        if not selected_items:
            return
        for item in selected_items:
            self.args_list.takeItem(self.args_list.row(item))

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
