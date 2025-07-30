import torch
import torch.nn as nn
import os
from sire import ModelManager
import time

# 1. Define a simple PyTorch model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)

def main():
    # 2. Prepare dummy data and model file
    # Use a temporary directory to avoid clutter
    temp_dir = "temp_model_assets"
    os.makedirs(temp_dir, exist_ok=True)
    model_path = os.path.join(temp_dir, "dummy_model.pth")

    model = SimpleModel()
    torch.save(model.state_dict(), model_path)
    dummy_data = torch.randn(1, 10)

    # 3. Instantiate ModelManager
    print("Initializing ModelManager...")
    manager = ModelManager()
    print(f"Detected devices: {manager._device_manager.devices}")

    # 4. Register the model
    print(f"Registering model from {model_path}...")
    manager.register_model(
        name="simple_model",
        model_path=model_path,
        runtime="pytorch",
        model_class=SimpleModel,
    )
    print("Model 'simple_model' registered.")

    # 5. Load the model
    print("Loading model 'simple_model'...")
    manager.load_model("simple_model")
    stats = manager.get_model_stats("simple_model")
    print(f"Model loaded on device: {stats['device']}")

    # 6. Run prediction
    print(f"Running inference with data of shape: {dummy_data.shape}...")
    start_time = time.time()
    result = manager.predict("simple_model", dummy_data)
    end_time = time.time()
    print(f"Inference result: {result.item()}")
    print(f"Inference took: {end_time - start_time:.4f} seconds")


    # 7. Unload the model
    print("Unloading model 'simple_model'...")
    manager.unload_model("simple_model")
    stats = manager.get_model_stats("simple_model")
    print(f"Model loaded: {stats['loaded']}")

    # 8. Clean up
    os.remove(model_path)
    os.rmdir(temp_dir)
    print("Cleaned up dummy model file and directory.")

if __name__ == "__main__":
    main()
