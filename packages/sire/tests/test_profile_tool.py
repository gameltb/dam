import tempfile

import pytest
import torch
import torch.nn as nn
from accelerate.hooks import add_hook_to_module

from sire.core.profile_tool import (
    ProfilerHook,
    ProfilingData,
    _profile_run_context,
    get_module_size,
)


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 20)
        self.layer2 = nn.Linear(20, 5)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


def test_get_module_size():
    model = SimpleModel()
    size = get_module_size(model)
    assert size > 0
    # layer1: 10*20*4 + 20*4 = 880
    # layer2: 20*5*4 + 5*4 = 420
    # Total = 1300
    assert size == (10 * 20 * 4 + 20 * 4) + (20 * 5 * 4 + 5 * 4)


def test_profiling_data_save_load():
    data = ProfilingData()
    data.record_execution("test_module", 0.1, 1024)
    data.record_weight_size("test_module", 2048)

    with tempfile.NamedTemporaryFile(mode="w+", delete=True, suffix=".json") as tmp:
        data.save(tmp.name)
        loaded_data = ProfilingData.load(tmp.name)

    assert loaded_data is not None
    assert loaded_data.execution_order == ["test_module"]
    assert loaded_data.module_stats["test_module"].exec_times == [0.1]
    assert loaded_data.module_stats["test_module"].peak_vram_usages == [1024]
    assert loaded_data.module_stats["test_module"].weight_size == 2048


def test_profiler_hook_cpu():
    model = SimpleModel()
    prof_data = ProfilingData()

    for name, module in model.named_modules():
        if name:  # Skip root module
            add_hook_to_module(module, ProfilerHook(name))

    with _profile_run_context(prof_data):
        model(torch.randn(1, 10))

    assert "layer1" in prof_data.execution_order
    assert "layer2" in prof_data.execution_order
    # On CPU, exec_times and peak_vram_usages should be empty as they are CUDA-specific
    assert prof_data.module_stats["layer1"].exec_times == []
    assert prof_data.module_stats["layer1"].peak_vram_usages == []
    # Weight size should be recorded
    assert prof_data.module_stats["layer1"].weight_size > 0
    assert prof_data.module_stats["layer2"].weight_size > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_profiler_hook_gpu():
    model = SimpleModel().to("cuda")
    prof_data = ProfilingData()

    for name, module in model.named_modules():
        if name:
            add_hook_to_module(module, ProfilerHook(name))

    with _profile_run_context(prof_data):
        model(torch.randn(1, 10).to("cuda"))

    assert "layer1" in prof_data.execution_order
    assert "layer2" in prof_data.execution_order

    # On GPU, these should be populated
    assert len(prof_data.module_stats["layer1"].exec_times) == 1
    assert prof_data.module_stats["layer1"].exec_times[0] > 0
    assert len(prof_data.module_stats["layer1"].peak_vram_usages) == 1
    # VRAM usage can be 0 in some cases for small models, so just check it's recorded
    assert prof_data.module_stats["layer1"].peak_vram_usages[0] >= 0

    assert prof_data.module_stats["layer1"].weight_size > 0
    assert prof_data.module_stats["layer2"].weight_size > 0
