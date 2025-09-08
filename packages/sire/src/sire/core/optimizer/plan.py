from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from ...utils.json_helpers import load_from_json_file, save_to_json_file


@dataclass
class PrefetchInstruction:
    """
    Represents a single prefetching action.
    """

    module_to_prefetch: str
    target_device: torch.device
    trigger_module: str


@dataclass
class OptimizationPlan:
    """
    Stores the complete optimization strategy for a model, including device placements
    and the prefetching schedule.
    """

    optimized_device_map: Dict[str, torch.device] = field(default_factory=dict)  # type: ignore
    prefetch_schedule: List[PrefetchInstruction] = field(default_factory=list)  # type: ignore
    module_footprints: Dict[str, int] = field(default_factory=dict)  # type: ignore
    trigger_index: Dict[str, List[PrefetchInstruction]] = field(
        default_factory=lambda: defaultdict(list), repr=False, compare=False
    )

    def get_total_runtime_vram(self) -> int:
        """
        Calculates the total VRAM required by this plan, summing the footprints
        of all modules assigned to a CUDA device.
        """
        total_vram = 0
        for module_name, device in self.optimized_device_map.items():
            if device.type == "cuda":
                total_vram += self.module_footprints.get(module_name, 0)
        return total_vram

    def __post_init__(self):
        self._build_trigger_index()

    def _build_trigger_index(self):
        """Builds an index for quick lookup of prefetch instructions by trigger module."""
        self.trigger_index = defaultdict(list)
        for instr in self.prefetch_schedule:
            self.trigger_index[instr.trigger_module].append(instr)

    def save(self, filepath: str):
        """Saves the optimization plan to a JSON file."""
        save_to_json_file(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> Optional[OptimizationPlan]:
        """Loads an optimization plan from a JSON file."""
        instance = load_from_json_file(filepath, cls)
        if isinstance(instance, cls):
            instance.__post_init__()
            return instance
        return None
