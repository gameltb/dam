import os
import tempfile

import torch

from sire.core.optimizer.plan import OptimizationPlan, PrefetchInstruction


def test_optimization_plan_save_load():
    """Tests saving and loading an OptimizationPlan."""
    plan = OptimizationPlan(
        optimized_device_map={
            "module1": torch.device("cuda:0"),
            "module2": torch.device("cpu"),
        },
        prefetch_schedule=[
            PrefetchInstruction(
                module_to_prefetch="module2",
                target_device=torch.device("cuda:0"),
                trigger_module="module1",
            )
        ],
    )

    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".json") as tmp:
        filepath = tmp.name

    try:
        plan.save(filepath)
        assert os.path.exists(filepath)

        loaded_plan = OptimizationPlan.load(filepath)

        assert loaded_plan is not None
        assert loaded_plan.optimized_device_map == plan.optimized_device_map
        assert loaded_plan.prefetch_schedule == plan.prefetch_schedule
        # Check that the trigger_index is correctly rebuilt
        assert "module1" in loaded_plan.trigger_index
        assert len(loaded_plan.trigger_index["module1"]) == 1
        assert loaded_plan.trigger_index["module1"][0].module_to_prefetch == "module2"

    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


def test_optimization_plan_empty():
    """Tests creating and saving/loading an empty plan."""
    plan = OptimizationPlan()

    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".json") as tmp:
        filepath = tmp.name

    try:
        plan.save(filepath)
        loaded_plan = OptimizationPlan.load(filepath)
        assert loaded_plan is not None
        assert not loaded_plan.optimized_device_map
        assert not loaded_plan.prefetch_schedule
        assert not loaded_plan.trigger_index
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)
