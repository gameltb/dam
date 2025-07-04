"""
Module to hold globally accessible singleton resources.
"""

from dam.core.model_manager import ModelExecutionManager

# Global singleton instance of ModelExecutionManager
# This instance will be initialized once when the application starts.
# For now, direct instantiation. Configuration might be loaded from environment or a global config file.
model_execution_manager = ModelExecutionManager()


def get_global_model_execution_manager() -> ModelExecutionManager:
    """
    Provides access to the global ModelExecutionManager instance.
    """
    # In a more complex setup, this might involve initialization if not already done,
    # but for now, it's instantiated at import time.
    return model_execution_manager
