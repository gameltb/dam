import logging
import os
import sys

import torch
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import (
    StableDiffusionXLPipeline,
)
from sire.core.optimizer.hooks import InferenceOptimizerHook
from accelerate.hooks import add_hook_to_module


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s"
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    example_logger = logging.getLogger("SDXL_Example")
    example_logger.setLevel(logging.INFO)

    optimizer_hook = InferenceOptimizerHook(
        cache_dir="example_optim_cache_sdxl",
        max_memory_gb={0: 8, "cpu": 24} if torch.cuda.is_available() and torch.cuda.device_count() > 0 else {"cpu": 24},
        force_profiling=False,
        num_prefetch_streams=1,
    )

    model_path = os.getenv("CI_TEST_MODEL_PATH", "playground-v2.5-1024px-aesthetic.fp16.safetensors")
    if not os.path.exists(model_path):
        example_logger.warning(f"Model '{model_path}' not found. Skipping SDXL example.")
        # Create a dummy file to avoid CI errors
        with open(model_path, "w") as f:
            f.write("dummy")
    else:
        example_logger.info(f"Loading SDXL pipeline from: {model_path}")
        try:
            pipe = StableDiffusionXLPipeline.from_single_file(
                model_path, torch_dtype=torch.float16, local_files_only=True, use_safetensors=True
            )
            example_logger.info(f"Pipeline loaded. UNet dtype: {pipe.unet.dtype}")
        except Exception as e:
            example_logger.error(f"Failed to load pipeline: {e}", exc_info=True)
            sys.exit(1)

        add_hook_to_module(pipe.unet, optimizer_hook, append=False)
        example_logger.info(f"IOHook attached to UNet ({pipe.unet.__class__.__name__}).")

        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            example_logger.info("Moving VAE & TextEncoders to CPU for UNet GPU memory.")
            pipe.vae.to("cpu")
            pipe.text_encoder.to("cpu")
            pipe.text_encoder_2.to("cpu")
        else:
            example_logger.info("No CUDA GPUs. Running on CPU.")

        prompt = "A majestic dragon soaring through a vibrant sunset sky, fantasy art, highly detailed"
        num_steps = 3
        for i in range(2):
            example_logger.info(f"\n--- Inference pass {i + 1} --- (Prompt: '{prompt}', Steps: {num_steps})")
            try:
                with torch.no_grad():
                    latents = pipe(prompt, num_inference_steps=num_steps, output_type="latent").images
                example_logger.info(
                    f"Pass {i + 1} complete. Latents shape: {latents.shape if latents is not None else 'None'}"
                )
            except Exception as e:
                example_logger.error(f"Pass {i + 1} error: {e}", exc_info=True)
        example_logger.info("\nExample finished.")


if __name__ == "__main__":
    main()
