import logging

from sire.core.runtime_resource_management import AutoManageWrapper, get_management

logger = logging.getLogger(__name__)


class SireResource:
    def __init__(self):
        self.management = get_management()
        logger.info("SireResource initialized.")

    def register_model_type(self, model_class, wrapper_class):
        logger.info(f"Registering wrapper {wrapper_class.__name__} for model {model_class.__name__}")
        AutoManageWrapper.registe_type_wrapper(model_class, wrapper_class)

    @property
    def auto_manage(self):
        from sire.core.runtime_resource_management import auto_manage

        return auto_manage
