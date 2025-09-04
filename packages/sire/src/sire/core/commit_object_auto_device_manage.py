from typing import NewType, Optional, TypeVar

import diffusers
import torch

from .commit_object import (
    BaseCommitObjectRef,
    CommitABC,
    CommitObjectProxy,
    CommitSquashItem,
    get_base_commit_object_ref,
)
from .runtime_resource_management import AutoManageWrapper

# assume base_object is torch.nn.Module
T = TypeVar("T", bound=torch.nn.Module)

# 触发权重卸载后置于这种状态,当应用stack时首先让对象恢复正常(如果需要)
FUSE_WEIGHT_OFFLOADED_STATE_UUID = NewType("FUSE_WEIGHT_OFFLOADED", None)


class CommitWithAutoManage(CommitABC[T]):
    def __init__(self):
        super().__init__()
        self.am: Optional[AutoManageWrapper] = None


class FuseWeightCommit(CommitWithAutoManage):
    def apply(self, base_object, **kwargs):
        # 是否可以推迟到真正执行时
        return super().apply(base_object, **kwargs)

    def revert(self, base_object):
        return super().revert(base_object)


class PipelineComponentsCommit(CommitWithAutoManage[diffusers.DiffusionPipeline]):
    def __init__(self, component_name, component_commmit: CommitWithAutoManage):
        super().__init__()
        self.component_name = component_name
        self.component_commmit = component_commmit

    def apply(self, base_object, **kwargs):
        self.component_commmit.am = self.am
        base_object
        return self.component_commmit.apply(base_object.components.get(self.component_name), **kwargs)

    def revert(self, base_object):
        return self.component_commmit.revert(base_object.components.get(self.component_name))


class CommitAutoManage(CommitABC[T]):
    def __init__(self, am: AutoManageWrapper):
        super().__init__()
        self.am = am

    def apply(self, base_object, **kwargs):
        self.am.load()

    def revert(self, base_object):
        if hasattr(self.am.user, "use_accelerate") and self.am.user.use_accelerate:
            import accelerate.hooks

            accelerate.hooks.remove_hook_from_module(self.manage_object, recurse=True)
            self.am.user.use_accelerate = False


class FuseWeightSquashItem(CommitSquashItem):
    def __init__(self, am: AutoManageWrapper):
        super().__init__()
        self.am = am

    def append(self, applied_commit_ref):
        return super().append(applied_commit_ref)

    def revert(self, base_object):
        return super().revert(base_object)


class AutoManageBaseCommitObjectRef(BaseCommitObjectRef[T]):
    def __init__(self, base_object):
        super().__init__(base_object)
        self.am = None
        self.am_commit = CommitAutoManage(None)

    def apply_commit(self, commit_list):
        if self.am is None:
            self.am = AutoManageWrapper(self.base_object)

        self.am_commit.am = self.am

        for commit in commit_list:
            if isinstance(commit, CommitWithAutoManage):
                commit.am = self.am

        self.am.user.lock()

        is_fuse_weight_commit = [isinstance(commit, FuseWeightCommit) for commit in commit_list]
        if is_fuse_weight_commit.count(True) > 1:
            first_fuse_weight_commit = is_fuse_weight_commit.index(True)
            last_fuse_weight_commit = len(is_fuse_weight_commit) - is_fuse_weight_commit[::-1].index(True) - 1
            assert first_fuse_weight_commit != last_fuse_weight_commit
            super().apply_commit(commit_list[:first_fuse_weight_commit])
            self.do_squash(FuseWeightSquashItem(self.am))
            super().apply_commit(commit_list[first_fuse_weight_commit : last_fuse_weight_commit + 1])
            self.stop_squash()
            super().apply_commit(commit_list[last_fuse_weight_commit + 1 :])
        else:
            super().apply_commit(commit_list)

    def rebase(self, commit_stack):
        if len(commit_stack) == 0 or commit_stack[-1] is not self.am_commit:
            commit_stack = commit_stack + [self.am_commit]

        # check if am unload becuse of on_resource_request ,
        # clean am_commit state of applied_commit_ref_stack by do an dyn revert_commit.
        if self.state_uuid == self.am_commit.commit_uuid:
            if not self.am.user.loaded:
                self.revert_commit()

        super().rebase(commit_stack)


class AutoManageCommitObjectProxy(CommitObjectProxy[T]):
    def __init__(self, base_object=None):
        super().__init__(get_base_commit_object_ref(base_object, AutoManageBaseCommitObjectRef))
        self.base_object_ref: AutoManageBaseCommitObjectRef
