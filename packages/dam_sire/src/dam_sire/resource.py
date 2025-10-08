"""The Sire resource for DAM."""

import logging
from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Any

from sire.core.runtime_resource_management import AutoManageWrapper, auto_manage, get_management

logger = logging.getLogger(__name__)


class SireResource:
    """A resource that provides access to the Sire runtime."""

    def __init__(self) -> None:
        """Initialize the SireResource."""
        self.management = get_management()
        logger.info("SireResource initialized.")

    def register_model_type(self, model_class: type[Any], wrapper_class: type[Any]) -> None:
        """
        Register a wrapper class for a given model class.

        Args:
            model_class: The model class to register.
            wrapper_class: The wrapper class to associate with the model.

        """
        logger.info("Registering wrapper %s for model %s", wrapper_class.__name__, model_class.__name__)
        AutoManageWrapper.register_type_wrapper(model_class, wrapper_class)

    @property
    def auto_manage(self) -> Callable[..., AbstractContextManager[AutoManageWrapper[Any]]]:
        """Return the auto_manage context manager."""
        return auto_manage
