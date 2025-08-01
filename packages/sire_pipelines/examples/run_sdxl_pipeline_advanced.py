import logging
import os
import sys
from typing import Callable

import sire
import torch
from diffusers import DiffusionPipeline, UNet2DConditionModel

from sire import CommitABC, CommitObjectProxy, InferenceOptimizerCommit

# Global logger for the example
example_logger = logging.getLogger("SDXL_Advanced_Example")


class LoraCommit(CommitABC[UNet2DConditionModel]):
    """A commit to apply or revert a LoRA to a UNet model."""

    def __init__(self, lora_path: str, lora_kwargs: dict, load_lora_weights: Callable, unload_lora_weights: Callable):
        super().__init__()
        self.lora_path = lora_path
        self.lora_kwargs = lora_kwargs
        self._load_lora = load_lora_weights
        self._unload_lora = unload_lora_weights

    def apply(self, base_object: UNet2DConditionModel, **kwargs):
        example_logger.info(f"Applying LoRA from: {self.lora_path}")
        self._load_lora(self.lora_path, **self.lora_kwargs)

    def revert(self, base_object: UNet2DConditionModel):
        example_logger.info(f"Reverting LoRA from: {self.lora_path}")
        self._unload_lora()


def run_inference(managed_pipe, unet_proxy: CommitObjectProxy, prompt: str, num_steps: int):
    """Runs inference with a given managed pipeline and UNet proxy state."""
    unet_proxy.apply_commit_stack()
    example_logger.info(f"Running inference for UNet state: {unet_proxy.base_object_ref.state_uuid}")
    try:
        with sire.auto_manage(managed_pipe):
            pipe = managed_pipe.get_manage_object()
            with torch.no_grad():
                latents = pipe(prompt, num_inference_steps=num_steps, output_type="latent").images
            example_logger.info(f"Inference complete. Latents shape: {latents.shape if latents is not None else 'None'}")
    except Exception as e:
        example_logger.error(f"Inference error: {e}", exc_info=True)


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s"
    )
    logging.getLogger("sire").setLevel(logging.WARNING)
    example_logger.setLevel(logging.INFO)

    # Initialize Sire's resource pools and register default wrappers
    sire.initialize()

    model_path = os.getenv("CI_TEST_MODEL_PATH", "playground-v2.5-1024px-aesthetic.fp16.safetensors")
    lora_path = os.getenv("CI_TEST_LORA_PATH", "sdxl-lora.safetensors")

    if not os.path.exists(model_path):
        example_logger.warning(f"Model '{model_path}' not found. Skipping example.")
        return

    example_logger.info(f"Loading SDXL pipeline from: {model_path}")
    try:
        pipe: DiffusionPipeline = DiffusionPipeline.from_single_file(
            model_path, torch_dtype=torch.float16, local_files_only=True, use_safetensors=True, low_cpu_mem_usage=True
        )
    except Exception as e:
        example_logger.error(f"Failed to load pipeline: {e}", exc_info=True)
        sys.exit(1)

    # 1. Manage the entire pipeline with a single call.
    #    Sire will use the registered DiffusersPipelineWrapper automatically.
    managed_pipe = sire.manage(pipe)

    # 2. Create a CommitObjectProxy for the UNet to manage its state (optimizations, LoRAs, etc.)
    unet_proxy = CommitObjectProxy(pipe.unet)

    # 3. Define optimization and LoRA commits
    optimizer_commit = InferenceOptimizerCommit(
        cache_dir="example_optim_cache_sdxl_advanced",
        max_memory_gb={0: 8, "cpu": 24} if torch.cuda.is_available() and torch.cuda.device_count() > 0 else {"cpu": 24},
        force_profiling=False,
    )

    # --- Pass 1: Inference with the base, un-optimized UNet ---
    example_logger.info("\n--- Running inference on the base UNet (un-optimized) ---")
    run_inference(managed_pipe, unet_proxy, "A photorealistic cat", 3)

    # --- Pass 2: Apply memory optimization commit ---
    example_logger.info("\n--- Applying memory optimization to the UNet ---")
    unet_optimized_proxy = unet_proxy.clone_and_add_commit(optimizer_commit)
    run_inference(managed_pipe, unet_optimized_proxy, "A photorealistic dog", 3)

    # --- Pass 3: Apply LoRA on top of the optimized UNet ---
    if not os.path.exists(lora_path):
        example_logger.warning(f"\nLoRA file '{lora_path}' not found. Skipping LoRA inference test.")
    else:
        example_logger.info("\n--- Applying LoRA to the optimized UNet ---")
        lora_commit = LoraCommit(
            lora_path=lora_path,
            lora_kwargs={"adapter_name": "my_lora"},
            load_lora_weights=pipe.load_lora_weights,
            unload_lora_weights=pipe.unload_lora_weights,
        )
        unet_lora_proxy = unet_optimized_proxy.clone_and_add_commit(lora_commit)
        run_inference(managed_pipe, unet_lora_proxy, "A majestic dragon, watercolor painting", 3)

        # --- Pass 4: Revert to the just-optimized state ---
        example_logger.info("\n--- Reverting LoRA to run with only the memory optimization ---")
        run_inference(managed_pipe, unet_optimized_proxy, "A photorealistic dog", 3)

    example_logger.info("\nExample finished.")


if __name__ == "__main__":
    main()
