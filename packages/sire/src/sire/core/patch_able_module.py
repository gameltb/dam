# module witch controlflow can be patch
import logging
import typing
from collections import OrderedDict
from dataclasses import dataclass
from enum import IntEnum
from typing import Generic, Self, TypeVar

_logger = logging.getLogger(__name__)


class PatchType(IntEnum):
    EXCLUSIVE = 0
    RANDOM_ORDER = 1


@dataclass
class PatchDefine:
    patch_type: PatchType = PatchType.EXCLUSIVE

    @staticmethod
    def from_type_hint(patch_type_hint):
        if typing.get_origin(patch_type_hint) is list:
            patch_define = PatchDefine(PatchType.RANDOM_ORDER)
        else:
            patch_define = PatchDefine()

        return patch_define


class ControlFlowPatchModuleMixin:
    def get_patch_module_name(self) -> str:
        """

        Returns:
            str: control flow patch name.
        """
        raise NotImplementedError()

    def get_patch_module_version(self) -> str:
        """

        Returns:
            str: control flow patch version.
        """
        raise NotImplementedError()

    def get_patch_module_uid(self) -> str:
        """

        Returns:
            str: control flow patch uid.
        """
        return self.get_patch_module_name() + self.get_patch_module_version()


T = TypeVar("T")


class ControlFlowPatchAbleModule(Generic[T]):
    def __init__(self) -> None:
        self.patcher_module_map: T = OrderedDict()
        for patch_name, patch_type_hint in self.get_patch_module_type_hints().items():
            patch_define = PatchDefine.from_type_hint(patch_type_hint)
            if patch_define.patch_type == PatchType.RANDOM_ORDER:
                self.patcher_module_map[patch_name] = []
            else:
                self.patcher_module_map[patch_name] = None

    def get_patch_module_type_hints(self):
        type_hints = None

        patch_module_map_type = None
        for type_ in type(self).__orig_bases__:
            if typing.get_origin(type_) is ControlFlowPatchAbleModule:
                patch_module_map_type = typing.get_args(type_)[0]
                break

        if patch_module_map_type is not None:
            type_hints = typing.get_type_hints(patch_module_map_type)

        return type_hints

    def get_patch_module_define(self, patch_type: str):
        patch_define = None

        type_hints = self.get_patch_module_type_hints()
        if type_hints is not None:
            type_hint = type_hints.get(patch_type, None)
            if type_hint is not None:
                patch_define = PatchDefine.from_type_hint(type_hint)

        if patch_define is None:
            raise Exception(f"patch_define {patch_type} not registered")
        return patch_define

    def add_patch(self, patch_type: str, patch_object: ControlFlowPatchModuleMixin):
        patch_define = self.get_patch_module_define(patch_type)

        if patch_define.patch_type == PatchType.EXCLUSIVE:
            if patch_type in self.patcher_module_map:
                raise Exception(f"EXCLUSIVE patch {patch_type} has been applied.")

        if patch_define.patch_type == PatchType.EXCLUSIVE:
            self.patcher_module_map[patch_type] = patch_object
        elif patch_define.patch_type == PatchType.RANDOM_ORDER:
            if patch_type not in self.patcher_module_map:
                self.patcher_module_map[patch_type] = []
            self.patcher_module_map[patch_type].append(patch_object)

    def copy_patch_from_other(self, other_patch_module: Self):
        self_type_hints = self.get_patch_module_type_hints()
        other_type_hints = other_patch_module.get_patch_module_type_hints()
        for patch_type in other_type_hints:
            if patch_type in self_type_hints:
                if self_type_hints[patch_type] == other_type_hints[patch_type]:
                    self.patcher_module_map[patch_type] = other_patch_module.patcher_module_map[patch_type]
                else:
                    _logger.debug(
                        f"Ignored {patch_type} has different types. self {self_type_hints[patch_type]} <-> other {other_type_hints[patch_type]}"
                    )
            else:
                _logger.debug(f"Ignored {patch_type} not found in self. other {other_type_hints[patch_type]}")


class ControlFlowPatchAbleModuleMixin(Generic[T]):
    def __init__(self) -> None:
        self.patcher_module_init()

    @property
    def patcher_module(self):
        return self.patcher_module_inst.patcher_module_map

    def patcher_module_init(self) -> None:
        import torch

        self.patcher_module_inst = ControlFlowPatchAbleModule[T]()
        self.patcher_module_dict = torch.nn.ModuleDict()

    def patcher_module_add(self, path_type: str, patch_object: ControlFlowPatchModuleMixin):
        patch_name = patch_object.get_patch_module_name()

        if patch_name in self.patcher_module_dict:
            raise Exception(f"patch {path_type} have a conflicting name {patch_name}.")

        self.patcher_module_inst.add_patch(path_type, patch_object)

        self.patcher_module_dict[patch_name] = patch_object

    def patcher_module_rebase(self, other: Self):
        self.patcher_module_inst.copy_patch_from_other(other.patcher_module_inst)
        self.patcher_module_update()

    def patcher_module_update(self):
        import torch

        patcher_module_dict = torch.nn.ModuleDict()

        def add_patch_object(patch_object):
            patch_name = patch_object.get_patch_module_name()
            if patch_name in self.patcher_module_dict:
                raise Exception(f"patch {patch_object} have a conflicting name {patch_name}.")
            patcher_module_dict[patch_name] = patch_object

        for patch_type in self.patcher_module:
            patch_objects = self.patcher_module[patch_type]
            if not isinstance(patch_objects, list):
                patch_objects = [patch_objects]
            for patch_object in patch_objects:
                add_patch_object(patch_object)

        self.patcher_module_dict = patcher_module_dict
