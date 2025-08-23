from sire.core.runtime_resource_management import AutoManageWrapper, get_management
from sire.core.runtime_resource_user.pytorch_module import TorchModuleWrapper
from sentence_transformers import SentenceTransformer


class SireResource:
    def __init__(self):
        self.management = get_management()
        self._register_wrappers()

    def _register_wrappers(self):
        # Register wrappers for the model types we want to use.
        AutoManageWrapper.registe_type_wrapper(SentenceTransformer, TorchModuleWrapper)
        # Add other wrappers here, e.g., for audio and tagging models.

    def get_model(self, model_class, *args, **kwargs):
        # This is a simplified get_model. The auto_manage context manager
        # is the main way to interact with models.
        # This function can be used to get a wrapped model instance.
        model_instance = model_class(*args, **kwargs)
        return AutoManageWrapper(model_instance)

    @property
    def auto_manage(self):
        # Expose the auto_manage context manager.
        from sire.core.runtime_resource_management import auto_manage
        return auto_manage
