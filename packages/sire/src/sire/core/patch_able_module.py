"""A module that allows for control-flow patching."""

import logging
import typing
from collections import OrderedDict
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Self, TypeVar

from torch import nn

_logger = logging.getLogger(__name__)


class PatchType(IntEnum):
    """Enum for the different types of patches."""

    EXCLUSIVE = 0
    RANDOM_ORDER = 1


@dataclass
class PatchDefine:
    """Defines the type of a patch."""

    patch_type: PatchType = PatchType.EXCLUSIVE

    @staticmethod
    def from_type_hint(patch_type_hint: Any) -> "PatchDefine":
        """Create a PatchDefine from a type hint."""
        if typing.get_origin(patch_type_hint) is list:
            patch_define = PatchDefine(PatchType.RANDOM_ORDER)
        else:
            patch_define = PatchDefine()

        return patch_define


class ControlFlowPatchModuleMixin(nn.Module):
    """A mixin for modules that can be used as control-flow patches."""

    def __init__(self) -> None:
        """Initialize the mixin."""
        super().__init__()  # type: ignore

    def get_patch_module_name(self) -> str:
        """Get the name of the patch module."""
        raise NotImplementedError()

    def get_patch_module_version(self) -> str:
        """Get the version of the patch module."""
        raise NotImplementedError()

    def get_patch_module_uid(self) -> str:
        """Get the unique ID of the patch module."""
        return self.get_patch_module_name() + self.get_patch_module_version()


T = TypeVar("T")


class ControlFlowPatchAbleModule[T]:
    """A class that can be patched with control-flow modules."""

    def __init__(self) -> None:
        """Initialize the patchable module."""
        self.patcher_module_map: OrderedDict[
            str, ControlFlowPatchModuleMixin | list[ControlFlowPatchModuleMixin] | None
        ] = OrderedDict()
        type_hints = self.get_patch_module_type_hints()
        if type_hints:
            for patch_name, patch_type_hint in type_hints.items():
                patch_define = PatchDefine.from_type_hint(patch_type_hint)
                if patch_define.patch_type == PatchType.RANDOM_ORDER:
                    self.patcher_module_map[patch_name] = []
                else:
                    self.patcher_module_map[patch_name] = None

    def get_patch_module_type_hints(self) -> dict[str, Any] | None:
        """Get the type hints for the patchable modules."""
        type_hints = None

        patch_module_map_type = None
        if hasattr(type(self), "__orig_bases__"):
            for type_ in type(self).__orig_bases__:  # type: ignore
                if typing.get_origin(type_) is ControlFlowPatchAbleModule:  # type: ignore
                    args = typing.get_args(type_)
                    if args:
                        patch_module_map_type = args[0]
                    break

        if patch_module_map_type is not None:
            type_hints = typing.get_type_hints(patch_module_map_type)

        return type_hints

    def get_patch_module_define(self, patch_type: str) -> PatchDefine:
        """Get the patch definition for a given patch type."""
        patch_define = None

        type_hints = self.get_patch_module_type_hints()
        if type_hints is not None:
            type_hint = type_hints.get(patch_type, None)
            if type_hint is not None:
                patch_define = PatchDefine.from_type_hint(type_hint)

        if patch_define is None:
            raise Exception(f"patch_define {patch_type} not registered")
        return patch_define

    def add_patch(self, patch_type: str, patch_object: "ControlFlowPatchModuleMixin") -> None:
        """Add a patch to the module."""
        patch_define = self.get_patch_module_define(patch_type)

        if patch_define.patch_type == PatchType.EXCLUSIVE and self.patcher_module_map.get(patch_type) is not None:
            raise Exception(f"EXCLUSIVE patch {patch_type} has been applied.")

        if patch_define.patch_type == PatchType.EXCLUSIVE:
            self.patcher_module_map[patch_type] = patch_object
        elif patch_define.patch_type == PatchType.RANDOM_ORDER:
            patch_list = self.patcher_module_map.get(patch_type)
            if not isinstance(patch_list, list):
                patch_list = []
                self.patcher_module_map[patch_type] = patch_list
            patch_list.append(patch_object)

    def copy_patch_from_other(self, other_patch_module: Self) -> None:
        """Copy patches from another patchable module."""
        self_type_hints = self.get_patch_module_type_hints()
        other_type_hints = other_patch_module.get_patch_module_type_hints()
        if not (self_type_hints and other_type_hints):
            return
        for patch_type in other_type_hints:
            if patch_type in self_type_hints:
                if self_type_hints[patch_type] == other_type_hints[patch_type]:
                    self.patcher_module_map[patch_type] = other_patch_module.patcher_module_map[patch_type]
                else:
                    _logger.debug(
                        "Ignored %s has different types. self %s <-> other %s",
                        patch_type,
                        self_type_hints[patch_type],
                        other_type_hints[patch_type],
                    )
            else:
                _logger.debug("Ignored %s not found in self. other %s", patch_type, other_type_hints[patch_type])


class ControlFlowPatchAbleModuleMixin[T]:
    """A mixin for modules that can be patched with control-flow modules."""

    def __init__(self) -> None:
        """Initialize the mixin."""
        self.patcher_module_init()

    @property
    def patcher_module(
        self,
    ) -> OrderedDict[str, "ControlFlowPatchModuleMixin" | list["ControlFlowPatchModuleMixin"] | None]:
        """Get the patcher module map."""
        return self.patcher_module_inst.patcher_module_map

    def patcher_module_init(self) -> None:
        """Initialize the patcher module."""
        self.patcher_module_inst: ControlFlowPatchAbleModule[T] = ControlFlowPatchAbleModule[T]()
        self.patcher_module_dict = nn.ModuleDict()

    def patcher_module_add(self, path_type: str, patch_object: ControlFlowPatchModuleMixin) -> None:
        """Add a patch to the patcher module."""
        patch_name = patch_object.get_patch_module_name()

        if patch_name in self.patcher_module_dict:
            raise Exception(f"patch {path_type} have a conflicting name {patch_name}.")

        self.patcher_module_inst.add_patch(path_type, patch_object)

        self.patcher_module_dict[patch_name] = patch_object

    def patcher_module_rebase(self, other: Self) -> None:
        """Rebase the patcher module from another mixin instance."""
        self.patcher_module_inst.copy_patch_from_other(other.patcher_module_inst)
        self.patcher_module_update()

    def patcher_module_update(self) -> None:
        """Update the patcher module dictionary."""
        patcher_module_dict = nn.ModuleDict()

        def add_patch_object(patch_object: ControlFlowPatchModuleMixin) -> None:
            patch_name = patch_object.get_patch_module_name()
            if patch_name in self.patcher_module_dict:
                raise Exception(f"patch {patch_object} have a conflicting name {patch_name}.")
            patcher_module_dict[patch_name] = patch_object

        for patch_objects_val in self.patcher_module.values():
            patch_objects = patch_objects_val
            if not isinstance(patch_objects, list):
                patch_objects = [patch_objects]
            for patch_object in patch_objects:
                if patch_object:
                    add_patch_object(patch_object)

        self.patcher_module_dict = patcher_module_dict
