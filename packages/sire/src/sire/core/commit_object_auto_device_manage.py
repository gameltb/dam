from typing import Any, NewType, Optional, TypeVar

import accelerate.hooks
import diffusers
import torch

from .commit_object import (
    AppliedCommitRefItem,
    BaseCommitObjectRef,
    CommitABC,
    CommitObjectProxy,
    CommitSquashItem,
    get_base_commit_object_ref,
)
from .runtime_resource_management import AutoManageWrapper
from .runtime_resource_user.pytorch_module import TorchModuleWrapper

# assume base_object is torch.nn.Module
T = TypeVar("T")


# 触发权重卸载后置于这种状态,当应用stack时首先让对象恢复正常(如果需要)
FUSE_WEIGHT_OFFLOADED_STATE_UUID = NewType("FUSE_WEIGHT_OFFLOADED_STATE_UUID", type(None))


class CommitWithAutoManage(CommitABC[T]):
    def __init__(self) -> None:
        super().__init__()
        self.am: Optional[AutoManageWrapper[T]] = None


class FuseWeightCommit(CommitWithAutoManage[T]):
    def apply(self, base_object: T, **kwargs: Any) -> None:
        # 是否可以推迟到真正执行时
        return super().apply(base_object, **kwargs)

    def revert(self, base_object: T) -> None:
        return super().revert(base_object)


class PipelineComponentsCommit(CommitWithAutoManage[diffusers.DiffusionPipeline]):  # type: ignore
    def __init__(self, component_name: str, component_commmit: CommitWithAutoManage[torch.nn.Module]):
        super().__init__()
        self.component_name = component_name
        self.component_commmit = component_commmit

    def apply(self, base_object: diffusers.DiffusionPipeline, **kwargs: Any) -> None:  # type: ignore
        self.component_commmit.am = self.am  # type: ignore
        component = base_object.components.get(self.component_name)
        if component:
            return self.component_commmit.apply(component, **kwargs)

    def revert(self, base_object: diffusers.DiffusionPipeline) -> None:  # type: ignore
        component = base_object.components.get(self.component_name)
        if component:
            return self.component_commmit.revert(component)


class CommitAutoManage(CommitABC[T]):
    def __init__(self, am: Optional[AutoManageWrapper[T]]):
        super().__init__()
        self.am = am

    def apply(self, base_object: T, **kwargs: Any) -> None:
        if self.am:
            self.am.load()

    def revert(self, base_object: T) -> None:
        if self.am and isinstance(self.am.user, TorchModuleWrapper) and self.am.user.use_accelerate:
            if isinstance(base_object, torch.nn.Module):
                accelerate.hooks.remove_hook_from_module(base_object, recurse=True)
                self.am.user.use_accelerate = False


class FuseWeightSquashItem(CommitSquashItem[T]):
    def __init__(self, am: AutoManageWrapper[T]):
        super().__init__()
        self.am = am

    def append(self, applied_commit_ref: AppliedCommitRefItem) -> None:
        return super().append(applied_commit_ref)

    def revert(self, base_object: T) -> None:
        return super().revert(base_object)


class AutoManageBaseCommitObjectRef(BaseCommitObjectRef[T]):
    def __init__(self, base_object: T):
        super().__init__(base_object)
        self.am: Optional[AutoManageWrapper[T]] = None
        self.am_commit: CommitAutoManage[T] = CommitAutoManage(self.am)

    def apply_commit(self, commit_list: list[CommitABC[T]]) -> None:
        if self.am is None:
            self.am = AutoManageWrapper(self.base_object)
            self.am_commit.am = self.am

        for commit in commit_list:
            if isinstance(commit, CommitWithAutoManage):
                commit.am = self.am
        if self.am:
            self.am.user.lock()

        is_fuse_weight_commit = [isinstance(commit, FuseWeightCommit) for commit in commit_list]
        if is_fuse_weight_commit.count(True) > 1:
            first_fuse_weight_commit = is_fuse_weight_commit.index(True)
            last_fuse_weight_commit = len(is_fuse_weight_commit) - is_fuse_weight_commit[::-1].index(True) - 1
            assert first_fuse_weight_commit != last_fuse_weight_commit
            super().apply_commit(commit_list[:first_fuse_weight_commit])
            if self.am:
                self.do_squash(FuseWeightSquashItem(self.am))
            super().apply_commit(commit_list[first_fuse_weight_commit : last_fuse_weight_commit + 1])
            self.stop_squash()
            super().apply_commit(commit_list[last_fuse_weight_commit + 1 :])
        else:
            super().apply_commit(commit_list)

    def rebase(self, commit_stack: list[CommitABC[T]]) -> None:
        if len(commit_stack) == 0 or commit_stack[-1] is not self.am_commit:
            commit_stack = commit_stack + [self.am_commit]

        # check if am unload becuse of on_resource_request ,
        # clean am_commit state of applied_commit_ref_stack by do an dyn revert_commit.
        if self.am and self.state_uuid == self.am_commit.commit_uuid:
            if not self.am.user.loaded:
                self.revert_commit()

        super().rebase(commit_stack)


class AutoManageCommitObjectProxy(CommitObjectProxy[T]):
    def __init__(self, base_object: T):
        super().__init__(
            get_base_commit_object_ref(base_object, AutoManageBaseCommitObjectRef)  # type: ignore
        )
