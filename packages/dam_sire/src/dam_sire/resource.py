import logging
from typing import Any, Callable, ContextManager, Type

from sire.core.runtime_resource_management import AutoManageWrapper, auto_manage, get_management

logger = logging.getLogger(__name__)


class SireResource:
    def __init__(self) -> None:
        self.management = get_management()
        logger.info("SireResource initialized.")

    def register_model_type(self, model_class: Type[Any], wrapper_class: Type[Any]) -> None:
        logger.info(f"Registering wrapper {wrapper_class.__name__} for model {model_class.__name__}")
        AutoManageWrapper.register_type_wrapper(model_class, wrapper_class)

    @property
    def auto_manage(self) -> Callable[..., ContextManager[AutoManageWrapper[Any]]]:
        return auto_manage
