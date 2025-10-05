# This file makes the 'dam.systems' package.
# We use importlib and pkgutil to dynamically ensure all system modules are loaded,
# so their @system decorators run and register the systems.

import importlib
import logging
import pkgutil

logger = logging.getLogger(__name__)

# Dynamically discover and import all modules in the current package
# This avoids having to manually update a list of system modules.
# This avoids having to manually update a list of system modules.
for _finder, name, _ispkg in pkgutil.iter_modules(__path__, __name__ + "."):
    try:
        importlib.import_module(name)
        logger.debug(f"Successfully imported system module: {name}")
    except ImportError as e:
        # This error is critical if a system module is expected but not found or fails to import.
        # It's particularly important to log this in case of missing optional dependencies.
        logger.warning(
            f"Could not import system module {name}. "
            f"This may be expected if optional dependencies are not installed. Error: {e}"
        )
    except Exception as e:
        # Catch other potential errors during module import
        logger.error(f"An unexpected error occurred while importing system module {name}: {e}", exc_info=True)

logger.info("dam.systems package initialized and all discoverable system modules have been loaded.")
