import torch
import sire

# 1. Initialize Sire's resource pools
# This scans for available devices (CPU and CUDA) and creates a resource pool for each.
sire.setup_default_pools()

# 2. Create your PyTorch model
class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 1)

    def forward(self, x):
        # This print statement helps visualize where the model is running
        print(f"Executing forward pass on device: {x.device}")
        return self.linear(x)

def main():
    """
    A simple example demonstrating how to use Sire to manage a PyTorch model.
    """
    model = MyModel()
    print(f"Model initial device: {next(model.parameters()).device}")

    # 3. Wrap the model to make it Sire-aware
    # This returns an AutoManageWrapper that knows how to manage the model's memory.
    managed_model = sire.manage(model)
    print("Model has been wrapped by Sire.")

    # Check if a CUDA device is available to demonstrate GPU offloading
    if not torch.cuda.is_available():
        print("\nNo CUDA device found. The model will remain on the CPU.")
        # Even without a GPU, auto_manage ensures the model is "locked" during use
        # and ready for the correct device.
    else:
        print("\nCUDA device found. Demonstrating GPU offloading.")

    # 4. Use the auto_manage context manager for inference
    print("Entering auto_manage context...")
    with sire.auto_manage(managed_model) as am_wrapper:
        # Inside this block, Sire moves the model to the designated runtime device (GPU if available).
        execution_device = am_wrapper.get_execution_device()
        print(f"Inside context, model is on device: {next(model.parameters()).device}")
        assert next(model.parameters()).device.type == execution_device.type

        # The input tensor must be on the same device as the model for the operation to succeed.
        print(f"Moving input tensor to {execution_device}...")
        dummy_input = torch.randn(1, 10).to(execution_device)

        # Run inference
        output = model(dummy_input)
        print(f"Inference output: {output.cpu().item():.4f}")

    print("\nExited auto_manage context.")
    # After the 'with' block, Sire automatically offloads the model back to the CPU.
    print(f"Model is now back on device: {next(model.parameters()).device}")
    assert next(model.parameters()).device.type == "cpu"


if __name__ == "__main__":
    main()
