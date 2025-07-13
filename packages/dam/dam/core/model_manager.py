import asyncio
import logging
import platform  # For general system info
from typing import Any, Callable, Dict, FrozenSet, Generic, Optional, Tuple, TypeVar

import psutil  # For system RAM
import torch  # For PyTorch resource detection

logger = logging.getLogger(__name__)

ModelType = TypeVar("ModelType")  # Generic type for models


class ModelExecutionManager(Generic[ModelType]):
    """
    Manages loading, caching, and basic resource awareness for ML models.
    This is a generic class; specific managers might inherit from it or
    it might be instantiated with specific model types.
    """

    def __init__(self):
        self.gpu_info: Dict[str, Any] = self._detect_gpu_info()
        self.system_ram_gb: float = self._detect_system_ram()
        self.os_info: str = platform.platform()
        self._model_cache: Dict[Tuple[str, FrozenSet[Tuple[str, Any]]], ModelType] = {}
        self._model_loaders: Dict[str, Callable[[str, Optional[Dict[str, Any]]], ModelType]] = {}
        self._batch_sizers: Dict[
            str, Callable[[ModelExecutionManager, str, float], int]
        ] = {}  # model_identifier -> batch_sizer_func

        logger.info("ModelExecutionManager initialized.")
        logger.info(f"OS: {self.os_info}")
        logger.info(f"GPU Info: {self.gpu_info if self.gpu_info['available'] else 'No CUDA GPU available/detected'}")
        logger.info(f"System RAM: {self.system_ram_gb:.2f} GB")

    def _detect_gpu_info(self) -> Dict[str, Any]:
        info = {"available": False, "device_count": 0, "devices": []}
        try:
            if torch.cuda.is_available():
                info["available"] = True
                info["device_count"] = torch.cuda.device_count()
                for i in range(torch.cuda.device_count()):
                    device_props = torch.cuda.get_device_properties(i)
                    info["devices"].append(
                        {
                            "name": device_props.name,
                            "total_memory_gb": device_props.total_memory / (1024**3),
                            # More properties can be added if needed
                        }
                    )
            elif torch.backends.mps.is_available():  # For Apple Silicon
                info["available"] = True
                info["device_count"] = 1
                info["devices"].append(
                    {
                        "name": "Apple MPS",
                        "total_memory_gb": -1,  # MPS shares system memory, difficult to quantify isolated GPU VRAM
                    }
                )

        except Exception as e:
            logger.warning(f"Error detecting GPU info: {e}", exc_info=True)
        return info

    def _detect_system_ram(self) -> float:
        try:
            virtual_mem = psutil.virtual_memory()
            return virtual_mem.total / (1024**3)  # Convert bytes to GB
        except Exception as e:
            logger.warning(f"Error detecting system RAM: {e}", exc_info=True)
            return -1.0

    def register_model_loader(
        self, model_identifier: str, loader_func: Callable[[str, Optional[Dict[str, Any]]], ModelType]
    ):
        """
        Registers a loader function for a given model identifier.
        The loader_func should take (model_name_or_path, params_dict) and return the loaded model.
        """
        if model_identifier in self._model_loaders:
            logger.warning(f"Loader for model identifier '{model_identifier}' already registered. Overwriting.")
        self._model_loaders[model_identifier] = loader_func
        logger.info(f"Registered loader for model identifier: {model_identifier}")

    def register_batch_sizer(
        self, model_identifier: str, sizer_func: Callable[["ModelExecutionManager", str, float], int]
    ):
        """
        Registers a specific batch sizing function for a model identifier.
        The sizer_func should take (manager_instance, model_name_or_path, item_size_estimate_mb) and return batch_size.
        """
        if model_identifier in self._batch_sizers:
            logger.warning(f"Batch sizer for model identifier '{model_identifier}' already registered. Overwriting.")
        self._batch_sizers[model_identifier] = sizer_func
        logger.info(f"Registered batch sizer for model identifier: {model_identifier}")

    async def get_model(
        self,
        model_identifier: str,  # An identifier for the model type (e.g., "sentence-transformer", "audio-vggish")
        model_name_or_path: str,  # Specific model name/path (e.g., "all-MiniLM-L6-v2", "/path/to/model")
        params: Optional[Dict[str, Any]] = None,  # Conceptual params for the model version or loading options
        force_reload: bool = False,
    ) -> ModelType:
        """
        Loads and caches a model using its registered loader.
        `params` are passed to the loader function.
        """
        params = params or {}
        cache_key = (model_identifier, model_name_or_path, frozenset(params.items()))

        if not force_reload and cache_key in self._model_cache:
            logger.debug(f"Returning cached model for {model_identifier} - {model_name_or_path} with params {params}")
            return self._model_cache[cache_key]  # Corrected: self._model_cache

        if model_identifier not in self._model_loaders:
            logger.error(
                f"No loader registered for model identifier '{model_identifier}'. Cannot load '{model_name_or_path}'."
            )
            raise ValueError(f"No loader registered for model identifier '{model_identifier}'")

        loader_func = self._model_loaders[model_identifier]

        logger.info(f"Loading model '{model_name_or_path}' (identifier: {model_identifier}) with params {params}...")
        try:
            # Loaders might be sync or async. For now, assume sync loaders run in executor.
            # A more robust system might differentiate or require async loaders.
            loop = asyncio.get_event_loop()
            model_instance = await loop.run_in_executor(
                None,  # Default executor
                loader_func,
                model_name_or_path,
                params,
            )
            self._model_cache[cache_key] = model_instance
            logger.info(
                f"Model '{model_name_or_path}' (identifier: {model_identifier}) loaded and cached successfully."
            )
            return model_instance
        except Exception as e:
            logger.error(
                f"Failed to load model '{model_name_or_path}' (identifier: {model_identifier}) with params {params}: {e}",
                exc_info=True,
            )
            raise

    def unload_model(
        self,
        model_identifier: str,
        model_name_or_path: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Removes a model from the cache.
        Actual GPU memory release depends on Python's GC and the model framework.
        """
        params = params or {}
        cache_key = (model_identifier, model_name_or_path, frozenset(params.items()))
        if cache_key in self._model_cache:
            logger.info(
                f"Unloading model '{model_name_or_path}' (identifier: {model_identifier}, params: {params}) from cache."
            )
            del self._model_cache[cache_key]
            # TODO: Add explicit cleanup if model objects have a .close() or .release() method
            # e.g. if hasattr(model_instance, 'to'): model_instance.to('cpu'); del model_instance; torch.cuda.empty_cache()
            # This requires storing the instance or having type-specific unloaders.
            # For now, relying on GC.
            if self.gpu_info["available"] and torch.cuda.is_available():
                torch.cuda.empty_cache()  # Try to clear PyTorch CUDA cache
            logger.info(
                f"Model '{model_name_or_path}' removed from cache. GPU cache cleared if PyTorch CUDA was active."
            )
            return True
        logger.warning(
            f"Model '{model_name_or_path}' (identifier: {model_identifier}, params: {params}) not found in cache for unloading."
        )
        return False

    def get_optimal_batch_size(
        self,
        model_identifier: str,
        model_name_or_path: str,  # To potentially have per-model heuristics
        item_size_estimate_mb: float,  # Estimated size of a single item in MB (e.g., an image tensor)
        fallback_batch_size: int = 8,
    ) -> int:
        """
        Suggests an optimal batch size.
        If a specific batch sizer is registered for the `model_identifier`, it's used.
        Otherwise, falls back to a generic heuristic based on available VRAM.
        """
        if model_identifier in self._batch_sizers:
            sizer_func = self._batch_sizers[model_identifier]
            logger.info(f"Using specific batch sizer for model identifier: {model_identifier}")
            try:
                return sizer_func(self, model_name_or_path, item_size_estimate_mb)
            except Exception as e:
                logger.error(
                    f"Error in specific batch sizer for {model_identifier} ('{model_name_or_path}'): {e}. Falling back to default.",
                    exc_info=True,
                )
                # Fall through to default heuristic if specific sizer fails

        # Default VRAM-based heuristic (as before)
        if not self.gpu_info["available"] or not self.gpu_info["devices"]:
            logger.info(
                f"Default sizer: No GPU detected or no VRAM info for '{model_name_or_path}', using fallback batch size {fallback_batch_size}."
            )
            return fallback_batch_size

        try:
            if torch.cuda.is_available():
                free_vram_bytes, _ = torch.cuda.mem_get_info(0)  # Use current device
                available_vram_gb = free_vram_bytes / (1024**3)
            elif self.gpu_info["devices"][0].get("name") == "Apple MPS":  # MPS
                # For MPS, available VRAM is essentially a portion of system RAM.
                # This heuristic can be refined by a specific MPS batch sizer if needed.
                available_vram_gb = self.system_ram_gb * 0.25  # Allow 25% of system RAM
                logger.info(
                    f"Default sizer: Apple MPS detected for '{model_name_or_path}'. Using {available_vram_gb:.2f}GB (25% of system RAM) as proxy."
                )
            elif self.gpu_info["devices"][0].get("total_memory_gb", -1) > 0:  # Other non-CUDA Torch supported GPUs
                available_vram_gb = (
                    self.gpu_info["devices"][0]["total_memory_gb"] * 0.7
                )  # Assume 70% of total is usable
                logger.info(
                    f"Default sizer: Using estimated VRAM for {self.gpu_info['devices'][0]['name']} for '{model_name_or_path}': {available_vram_gb:.2f}GB."
                )
            else:
                logger.info(
                    f"Default sizer: Could not determine VRAM for '{model_name_or_path}'. Using fallback {fallback_batch_size}."
                )
                return fallback_batch_size
        except Exception as e:
            logger.warning(
                f"Default sizer: Could not query VRAM for '{model_name_or_path}': {e}. Using total VRAM as approximation.",
                exc_info=True,
            )
            # Fallback to total memory if free cannot be obtained, with a safety factor
            if self.gpu_info["devices"]:
                available_vram_gb = (
                    self.gpu_info["devices"][0].get("total_memory_gb", 0) * 0.5
                )  # More conservative if using total
            else:  # Should not happen if gpu_info.available is true
                return fallback_batch_size

        if available_vram_gb <= 0:
            logger.warning(
                f"Default sizer: Available VRAM is 0 or unknown for '{model_name_or_path}', using fallback {fallback_batch_size}."
            )
            return fallback_batch_size
        if item_size_estimate_mb <= 0:
            logger.warning(
                f"Default sizer: Item size estimate is invalid for '{model_name_or_path}', using fallback {fallback_batch_size}."
            )
            return fallback_batch_size

        # More conservative reservation for the default heuristic
        reserved_vram_gb = min(available_vram_gb * 0.30, 1.5 if available_vram_gb > 3 else available_vram_gb * 0.15)
        usable_vram_for_batch_gb = available_vram_gb - reserved_vram_gb

        if usable_vram_for_batch_gb <= 0:
            logger.warning(
                f"Default sizer: Not enough usable VRAM for '{model_name_or_path}' after reservation, using batch size 1."
            )
            return 1

        max_items = int((usable_vram_for_batch_gb * 1024) / item_size_estimate_mb)
        optimal_batch = max(1, min(max_items, 128))  # Slightly more conservative cap for default

        logger.info(
            f"Default sizer: Estimated optimal batch for '{model_name_or_path}' (item ~{item_size_estimate_mb:.2f}MB): {optimal_batch} "
            f"(Usable VRAM: {usable_vram_for_batch_gb:.2f}GB)"
        )
        return optimal_batch

    def get_model_device_preference(self) -> str:
        """Returns 'cuda', 'mps', or 'cpu' based on availability."""
        if self.gpu_info["available"]:
            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
        return "cpu"


# Example Usage (conceptual)
async def example_model_loader(model_path: str, params: Optional[Dict[str, Any]]) -> Any:
    # In a real scenario, this would load a model from disk or download it.
    # For example, SentenceTransformer(model_path)
    logger.info(f"Example loader: Loading model from {model_path} with params {params}")
    # Simulate model loading
    await asyncio.sleep(0.5)
    return f"MockModelInstance_{model_path}_{params.get('version', 'v1') if params else 'v1'}"


async def main_example():
    manager = ModelExecutionManager()
    manager.register_model_loader("example_transformer", example_model_loader)

    try:
        model1 = await manager.get_model("example_transformer", "bert-base-uncased", params={"version": "1.0"})
        print(f"Loaded model1: {model1}")

        model2 = await manager.get_model(
            "example_transformer", "bert-base-uncased", params={"version": "1.0"}
        )  # Should be cached
        print(f"Loaded model2: {model2}")

        model3 = await manager.get_model("example_transformer", "distilbert-base-uncased", params={"precision": "fp16"})
        print(f"Loaded model3: {model3}")

        batch_size = manager.get_optimal_batch_size(
            "example_transformer", "bert-base-uncased", item_size_estimate_mb=100
        )
        print(f"Suggested batch_size: {batch_size}")

        manager.unload_model("example_transformer", "bert-base-uncased", params={"version": "1.0"})

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # To run the example:
    # asyncio.run(main_example())
    # Note: The example main won't run as part of the agent's execution directly,
    # but it's useful for testing this module in isolation.
    # For the agent, the class ModelExecutionManager itself is the deliverable.
    pass
