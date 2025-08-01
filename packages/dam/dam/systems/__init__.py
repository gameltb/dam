# This file makes the 'dam.systems' package.
# We use importlib to ensure all system modules are loaded,
# so their @system and @listens_for decorators run and register the systems.

import importlib
import logging  # Use logging for consistency

logger = logging.getLogger(__name__)

# List of system modules within this package to load
# Ensure these files exist: e.g., metadata_systems.py, asset_lifecycle_systems.py
_system_module_names = [
    # ".asset_ingestion_systems", # Removed
    ".metadata_systems",
    ".asset_lifecycle_systems",
    ".evaluation_systems",
    ".semantic_systems",  # Added semantic_systems
    ".auto_tagging_system",  # Added new auto-tagging system
    ".audio_systems",  # Added audio_systems
]

# Explicitly import and re-export specific items like markers if needed for direct import elsewhere
try:
    from .audio_systems import NeedsAudioProcessingMarker

    __all__ = ["NeedsAudioProcessingMarker"]
except ImportError:
    logger.warning(
        "Could not import NeedsAudioProcessingMarker from .audio_systems. It might not be defined yet or the module has issues."
    )
    __all__ = []


for module_name in _system_module_names:
    try:
        importlib.import_module(module_name, package=__name__)
        logger.debug(f"Successfully imported system module: {module_name} from package {__name__}")
    except ImportError as e:
        # This error is critical if a system module is expected but not found or fails to import.
        logger.error(f"Failed to import system module {module_name} from package {__name__}: {e}", exc_info=True)
        # Depending on strictness, might want to re-raise or handle gracefully.
        # For now, logging the error. If a crucial system fails to load, tests or app will likely fail later.

logger.info(f"dam.systems package initialized. System modules loaded: {_system_module_names}")
