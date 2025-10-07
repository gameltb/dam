"""Data classes for storing and managing optimization plans."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import torch

from ...utils.json_helpers import load_from_json_file, save_to_json_file


@dataclass
class PrefetchInstruction:
    """Represents a single prefetching action."""

    module_to_prefetch: str
    target_device: torch.device
    trigger_module: str


@dataclass
class OptimizationPlan:
    """
    Stores the complete optimization strategy for a model.

    This includes device placements and the prefetching schedule.
    """

    optimized_device_map: dict[str, torch.device] = field(default_factory=dict)  # type: ignore
    prefetch_schedule: list[PrefetchInstruction] = field(default_factory=list)  # type: ignore
    module_footprints: dict[str, int] = field(default_factory=dict)  # type: ignore
    trigger_index: dict[str, list[PrefetchInstruction]] = field(
        default_factory=lambda: defaultdict(list), repr=False, compare=False
    )

    def get_total_runtime_vram(self) -> int:
        """
        Calculate the total VRAM required by this plan.

        This sums the footprints of all modules assigned to a CUDA device.
        """
        total_vram = 0
        for module_name, device in self.optimized_device_map.items():
            if device.type == "cuda":
                total_vram += self.module_footprints.get(module_name, 0)
        return total_vram

    def __post_init__(self):
        """Initialize the trigger index after the object is created."""
        self._build_trigger_index()

    def _build_trigger_index(self):
        """Build an index for quick lookup of prefetch instructions by trigger module."""
        self.trigger_index = defaultdict(list)
        for instr in self.prefetch_schedule:
            self.trigger_index[instr.trigger_module].append(instr)

    def save(self, filepath: str):
        """Save the optimization plan to a JSON file."""
        save_to_json_file(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> OptimizationPlan | None:
        """Load an optimization plan from a JSON file."""
        instance = load_from_json_file(filepath, cls)
        if isinstance(instance, cls):
            instance.__post_init__()
            return instance
        return None
