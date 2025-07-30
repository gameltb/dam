import torch
import sire
from sire_pipelines.pipelines.simple_pipeline import SimplePipeline

def main():
    print("--- Sire Pipelines Example ---")

    # Initialize Sire's resource pools. This is a crucial first step.
    sire.setup_default_pools()

    # Register the torch.nn.Module wrapper for Sire.
    # This is needed so sire.manage() knows how to handle torch models.
    from sire.core.runtime_resource_management import AutoManageWrapper
    from sire.core.runtime_resource_user.pytorch_module import TorchModuleWrapper
    if torch.nn.Module not in AutoManageWrapper.type_wrapper_map:
        AutoManageWrapper.registe_type_wrapper(torch.nn.Module, TorchModuleWrapper)

    print("\n1. Initializing the pipeline...")
    pipeline = SimplePipeline()
    print("Pipeline initialized.")
    print(f"Initial UNet device: {next(pipeline._unet.parameters()).device}")
    print(f"Initial VAE device: {next(pipeline._vae.parameters()).device}")


    # Create some dummy input data
    dummy_vae_input = torch.randn(1, 256)
    dummy_unet_input = torch.randn(1, 128)

    print("\n2. Running the pipeline...")
    # This will trigger the sire.auto_manage blocks inside the pipeline
    final_output = pipeline(dummy_vae_input, dummy_unet_input)

    print(f"\n3. Final state of models:")
    print(f"Final UNet device: {next(pipeline._unet.parameters()).device}")
    print(f"Final VAE device: {next(pipeline._vae.parameters()).device}")
    print(f"Final output shape: {final_output.shape}")

if __name__ == "__main__":
    main()
