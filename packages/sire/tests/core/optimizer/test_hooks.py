"""Tests for the inference optimizer hooks."""

import os
import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest
import torch
from accelerate.hooks import add_hook_to_module
from pytest_mock import MockerFixture
from torch import nn

from sire.core.optimizer.hooks import InferenceOptimizerHook
from sire.core.optimizer.plan import OptimizationPlan


class SimpleModel(nn.Module):
    """A simple model for testing."""

    def __init__(self) -> None:
        """Initialize the model."""
        super().__init__()  # type: ignore
        self.layer1 = nn.Linear(10, 20)
        self.layer2 = nn.Linear(20, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass."""
        x = self.layer1(x)
        return self.layer2(x)


@pytest.fixture(autouse=True)
def mock_nvtx(mocker: MockerFixture):
    """Mock nvtx functions for CPU-only test environments."""
    mocker.patch("sire.core.optimizer.hooks.nvtx.range_push", return_value=None)
    mocker.patch("sire.core.optimizer.hooks.nvtx.range_pop", return_value=None)


@pytest.fixture
def temp_cache_dir() -> Generator[str, None, None]:
    """Provide a temporary directory for cache files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_cuda_avail(mocker: MockerFixture) -> None:
    """Mock CUDA availability."""
    mocker.patch("torch.cuda.is_available", return_value=True)


@pytest.fixture
def mock_dev_count(mocker: MockerFixture) -> None:
    """Mock CUDA device count."""
    mocker.patch("torch.cuda.device_count", return_value=2)


@pytest.mark.usefixtures("mock_cuda_avail", "mock_dev_count")
def test_inference_optimizer_hook_first_run_profiling(temp_cache_dir: str, mocker: MockerFixture) -> None:
    """
    Test the hook's behavior on the first run for a new configuration.

    This should trigger profiling and plan generation.
    """
    mocker.patch("sire.core.optimizer.hooks.get_balanced_memory", return_value={"0": 1e10, "cpu": 1e10})
    mock_profiler_cls = mocker.patch("sire.core.optimizer.hooks.Profiler")
    mock_profiler_instance = mock_profiler_cls.return_value
    mock_gen_plan = mocker.patch("sire.core.optimizer.hooks.InferenceOptimizerHook._gen_opt_plan")
    mock_setup_with_plan = mocker.patch("sire.core.optimizer.hooks.InferenceOptimizerHook._setup_module_with_plan")

    mock_profiling_data = mocker.MagicMock()
    mock_profiler_instance.run.return_value = mock_profiling_data
    mock_plan = OptimizationPlan(optimized_device_map={"layer1": torch.device("cuda:0")})
    mock_gen_plan.return_value = mock_plan

    hook = InferenceOptimizerHook(cache_dir=temp_cache_dir)
    model = SimpleModel()
    add_hook_to_module(model, hook)

    model(torch.randn(4, 10))

    mock_profiler_instance.run.assert_called_once()
    mock_gen_plan.assert_called_once()
    mock_setup_with_plan.assert_called_once()
    called_plan = mock_setup_with_plan.call_args[0][1]
    assert called_plan is mock_plan


@pytest.mark.usefixtures("mock_cuda_avail", "mock_dev_count")
def test_inference_optimizer_hook_second_run_cached(temp_cache_dir: str, mocker: MockerFixture) -> None:
    """Test the hook's behavior on a second run, verifying it loads from cache."""
    mocker.patch("sire.core.optimizer.hooks.get_balanced_memory", return_value={"0": 1e10, "cpu": 1e10})
    mock_profiler_cls = mocker.patch("sire.core.optimizer.hooks.Profiler")
    mock_profiler_instance = mock_profiler_cls.return_value
    mock_gen_plan = mocker.patch("sire.core.optimizer.hooks.InferenceOptimizerHook._gen_opt_plan")
    mock_setup_with_plan = mocker.patch("sire.core.optimizer.hooks.InferenceOptimizerHook._setup_module_with_plan")

    mock_profiling_data = mocker.MagicMock()
    mocker.patch("sire.core.optimizer.hooks.ProfilingData.load", return_value=mock_profiling_data)
    mock_plan = OptimizationPlan(optimized_device_map={"layer1": torch.device("cuda:0")})
    mocker.patch("sire.core.optimizer.hooks.OptimizationPlan.load", return_value=mock_plan)

    hook = InferenceOptimizerHook(cache_dir=temp_cache_dir)
    model = SimpleModel()
    add_hook_to_module(model, hook)

    model(torch.randn(4, 10))

    mock_profiler_instance.run.assert_not_called()
    mock_gen_plan.assert_not_called()
    mock_setup_with_plan.assert_called_once()
    called_plan = mock_setup_with_plan.call_args[0][1]
    assert called_plan is mock_plan


@pytest.mark.usefixtures("mock_cuda_avail", "mock_dev_count")
def test_inference_optimizer_hook_force_profiling(temp_cache_dir: str, mocker: MockerFixture) -> None:
    """Test that `force_profiling=True` re-runs profiling even if caches exist."""
    mocker.patch("sire.core.optimizer.hooks.get_balanced_memory", return_value={"0": 1e10, "cpu": 1e10})
    mock_profiler_cls = mocker.patch("sire.core.optimizer.hooks.Profiler")
    mock_profiler_instance = mock_profiler_cls.return_value
    mock_gen_plan = mocker.patch("sire.core.optimizer.hooks.InferenceOptimizerHook._gen_opt_plan")
    mock_setup_with_plan = mocker.patch("sire.core.optimizer.hooks.InferenceOptimizerHook._setup_module_with_plan")

    mocker.patch("sire.core.optimizer.hooks.ProfilingData.load", return_value=mocker.MagicMock())
    mocker.patch("sire.core.optimizer.hooks.OptimizationPlan.load", return_value=mocker.MagicMock())

    hook = InferenceOptimizerHook(cache_dir=temp_cache_dir, force_profiling=True)
    model = SimpleModel()
    add_hook_to_module(model, hook)

    model(torch.randn(4, 10))

    mock_profiler_instance.run.assert_called_once()
    mock_gen_plan.assert_called_once()
    mock_setup_with_plan.assert_called_once()
    assert not hook._force_profiling_active  # pyright: ignore[reportPrivateUsage]


@pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.device_count() > 0 and os.getenv("RUN_GPU_TESTS") == "1"),
    reason="Requires GPU and RUN_GPU_TESTS=1 env var",
)
def test_e2e_optimizer_hook_on_sdxl_unet(temp_cache_dir: str, mocker: MockerFixture) -> None:
    """
    Run an end-to-end test of the optimizer on a mocked SDXL UNet.

    This test requires a GPU and a specific environment variable to run.
    """
    mock_from_single_file = mocker.patch("sire.core.optimizer.hooks.StableDiffusionXLPipeline.from_single_file")
    mock_pipe = mocker.MagicMock()
    mock_unet = SimpleModel().to(torch.float16)  # Use a simple model instead of full SDXL
    mock_pipe.unet = mock_unet
    mock_from_single_file.return_value = mock_pipe

    optimizer_hook = InferenceOptimizerHook(
        cache_dir=temp_cache_dir,
        max_memory_gb={"0": 8, "cpu": 24},  # type: ignore
        force_profiling=True,
    )
    add_hook_to_module(mock_pipe.unet, optimizer_hook)

    bs = 1
    num_channels = 4
    sizes = (64, 64)
    sample = torch.randn(bs, num_channels, *sizes, device="cpu").to(torch.float16)
    timestep = torch.tensor([999], device="cpu")
    encoder_hidden_states = torch.randn(bs, 77, 1024, device="cpu").to(torch.float16)

    mock_pipe.unet.to("meta")

    with torch.no_grad():
        noise_pred = mock_pipe.unet(sample, timestep, encoder_hidden_states=encoder_hidden_states)

    assert noise_pred is not None
    assert len(list(Path(temp_cache_dir).iterdir())) > 0

    with torch.no_grad():
        noise_pred_2 = mock_pipe.unet(sample, timestep, encoder_hidden_states=encoder_hidden_states)

    assert torch.allclose(noise_pred, noise_pred_2, atol=1e-3)
