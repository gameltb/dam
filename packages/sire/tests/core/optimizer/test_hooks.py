import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from accelerate.hooks import add_hook_to_module

from sire.core.optimizer.hooks import InferenceOptimizerHook
from sire.core.optimizer.plan import OptimizationPlan


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 20)
        self.layer2 = nn.Linear(20, 5)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


@pytest.fixture(autouse=True)
def mock_nvtx(mocker):
    """Mocks nvtx functions for CPU-only test environments."""
    mocker.patch("sire.core.optimizer.hooks.nvtx.range_push", return_value=None)
    mocker.patch("sire.core.optimizer.hooks.nvtx.range_pop", return_value=None)


@pytest.fixture
def temp_cache_dir():
    """Provides a temporary directory for cache files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@patch("sire.core.optimizer.hooks.torch.cuda.is_available", return_value=True)
@patch("sire.core.optimizer.hooks.torch.cuda.device_count", return_value=1)
@patch("sire.core.optimizer.hooks.get_balanced_memory", return_value={"0": 1e10, "cpu": 1e10})
def test_inference_optimizer_hook_first_run_profiling(
    mock_get_mem, mock_dev_count, mock_cuda_avail, temp_cache_dir, mocker
):
    """
    Tests the hook's behavior on the first run for a new configuration,
    triggering profiling and plan generation.
    """
    # Mock the main methods to trace their calls
    mock_profiler_cls = mocker.patch("sire.core.optimizer.hooks.Profiler")
    mock_profiler_instance = mock_profiler_cls.return_value
    mock_gen_plan = mocker.patch("sire.core.optimizer.hooks.InferenceOptimizerHook._gen_opt_plan")
    mock_setup_with_plan = mocker.patch("sire.core.optimizer.hooks.InferenceOptimizerHook._setup_module_with_plan")

    # Mock the return values
    mock_profiling_data = MagicMock()
    mock_profiler_instance.run.return_value = mock_profiling_data

    mock_plan = OptimizationPlan(optimized_device_map={"layer1": torch.device("cuda:0")})
    mock_gen_plan.return_value = mock_plan

    # Instantiate the hook
    hook = InferenceOptimizerHook(cache_dir=temp_cache_dir)
    model = SimpleModel()
    add_hook_to_module(model, hook)

    # --- First forward pass ---
    model(torch.randn(4, 10))

    # Assertions
    mock_profiler_instance.run.assert_called_once()
    mock_gen_plan.assert_called_once()
    mock_setup_with_plan.assert_called_once()
    # Check that the generated plan was passed to the setup function
    called_plan = mock_setup_with_plan.call_args[0][1]
    assert called_plan is mock_plan


@patch("sire.core.optimizer.hooks.torch.cuda.is_available", return_value=True)
@patch("sire.core.optimizer.hooks.torch.cuda.device_count", return_value=1)
@patch("sire.core.optimizer.hooks.get_balanced_memory", return_value={"0": 1e10, "cpu": 1e10})
def test_inference_optimizer_hook_second_run_cached(
    mock_get_mem, mock_dev_count, mock_cuda_avail, temp_cache_dir, mocker
):
    """
    Tests the hook's behavior on a second run, verifying it loads from cache.
    """
    # Mock the main methods
    mock_profiler_cls = mocker.patch("sire.core.optimizer.hooks.Profiler")
    mock_profiler_instance = mock_profiler_cls.return_value
    mock_gen_plan = mocker.patch("sire.core.optimizer.hooks.InferenceOptimizerHook._gen_opt_plan")
    mock_setup_with_plan = mocker.patch("sire.core.optimizer.hooks.InferenceOptimizerHook._setup_module_with_plan")

    # Mock file system interactions to simulate cached files
    mock_profiling_data = MagicMock()
    mocker.patch("sire.core.optimizer.hooks.ProfilingData.load", return_value=mock_profiling_data)

    mock_plan = OptimizationPlan(optimized_device_map={"layer1": torch.device("cuda:0")})
    mocker.patch("sire.core.optimizer.hooks.OptimizationPlan.load", return_value=mock_plan)

    hook = InferenceOptimizerHook(cache_dir=temp_cache_dir)
    model = SimpleModel()
    add_hook_to_module(model, hook)

    # --- First forward pass (will use mocked cache) ---
    model(torch.randn(4, 10))

    # Assertions
    mock_profiler_instance.run.assert_not_called()
    mock_gen_plan.assert_not_called()
    mock_setup_with_plan.assert_called_once()
    # Check that the loaded plan was used
    called_plan = mock_setup_with_plan.call_args[0][1]
    assert called_plan is mock_plan


@patch("sire.core.optimizer.hooks.torch.cuda.is_available", return_value=True)
@patch("sire.core.optimizer.hooks.torch.cuda.device_count", return_value=1)
@patch("sire.core.optimizer.hooks.get_balanced_memory", return_value={"0": 1e10, "cpu": 1e10})
def test_inference_optimizer_hook_force_profiling(
    mock_get_mem, mock_dev_count, mock_cuda_avail, temp_cache_dir, mocker
):
    """
    Tests that `force_profiling=True` re-runs profiling even if caches exist.
    """
    # Mock the main methods
    mock_profiler_cls = mocker.patch("sire.core.optimizer.hooks.Profiler")
    mock_profiler_instance = mock_profiler_cls.return_value
    mock_gen_plan = mocker.patch("sire.core.optimizer.hooks.InferenceOptimizerHook._gen_opt_plan")
    mock_setup_with_plan = mocker.patch("sire.core.optimizer.hooks.InferenceOptimizerHook._setup_module_with_plan")

    # Mock that caches exist
    mocker.patch("sire.core.optimizer.hooks.ProfilingData.load", return_value=MagicMock())
    mocker.patch("sire.core.optimizer.hooks.OptimizationPlan.load", return_value=MagicMock())

    # Instantiate hook with force_profiling=True
    hook = InferenceOptimizerHook(cache_dir=temp_cache_dir, force_profiling=True)
    model = SimpleModel()
    add_hook_to_module(model, hook)

    # --- First forward pass ---
    model(torch.randn(4, 10))

    # Assertions: Profiling should be called despite the mocked caches
    mock_profiler_instance.run.assert_called_once()
    # Plan generation will be called because profiling creates new data
    mock_gen_plan.assert_called_once()
    mock_setup_with_plan.assert_called_once()
    # Check that the internal flag is reset after one run
    assert hook._force_profiling_active is False


@pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.device_count() > 0 and os.getenv("RUN_GPU_TESTS") == "1"),
    reason="Requires GPU and RUN_GPU_TESTS=1 env var",
)
@patch("sire.core.optimizer.hooks.StableDiffusionXLPipeline.from_single_file")
def test_e2e_optimizer_hook_on_sdxl_unet(mock_from_single_file, temp_cache_dir):
    """
    An end-to-end test that runs the optimizer on a mocked SDXL UNet.
    This test requires a GPU and a specific environment variable to run.
    """
    # --- Setup a mock pipeline and model ---
    mock_pipe = MagicMock()
    mock_unet = SimpleModel().to(torch.float16)  # Use a simple model instead of full SDXL
    mock_pipe.unet = mock_unet
    mock_from_single_file.return_value = mock_pipe

    # --- Setup the hook ---
    optimizer_hook = InferenceOptimizerHook(
        cache_dir=temp_cache_dir,
        max_memory_gb={0: 8, "cpu": 24},
        force_profiling=True,  # Force profiling for a clean test run
    )
    add_hook_to_module(mock_pipe.unet, optimizer_hook)

    # --- Run inference ---
    # We don't need the full pipeline logic, just a forward pass on the UNet
    bs = 1
    num_channels = 4
    sizes = (64, 64)
    sample = torch.randn(bs, num_channels, *sizes, device="cpu").to(torch.float16)
    timestep = torch.tensor([999], device="cpu")
    encoder_hidden_states = torch.randn(bs, 77, 1024, device="cpu").to(torch.float16)

    # Move model to meta device to simulate initial state
    mock_pipe.unet.to("meta")

    # The hook will dispatch the model to CPU/GPU on the first run
    with torch.no_grad():
        # The hook will be triggered here
        noise_pred = mock_pipe.unet(sample, timestep, encoder_hidden_states=encoder_hidden_states)

    # --- Assertions ---
    # Check that the output is on the expected device (CPU in this case, as the last layer might be offloaded)
    assert noise_pred is not None

    # Check that cache files were created
    assert len(os.listdir(temp_cache_dir)) > 0

    # Second run to test caching
    with torch.no_grad():
        noise_pred_2 = mock_pipe.unet(sample, timestep, encoder_hidden_states=encoder_hidden_states)

    assert torch.allclose(noise_pred, noise_pred_2, atol=1e-3)
