import dataclasses
import logging
import uuid
import weakref
from abc import ABC, abstractmethod
from typing import Callable, Generic, Optional, TypeVar, Union

_logger = logging.getLogger(__name__)

T = TypeVar("T")
UNKNOWN_STATE_UUID = None


class CommitABC(ABC, Generic[T]):
    def __init__(self):
        self.commit_uuid = uuid.uuid4()

    @abstractmethod
    def apply(self, base_object: T, **kwargs):
        pass

    @abstractmethod
    def revert(self, base_object: T):
        pass

    def get_revert_callable(self):
        return self.revert

    def release_revert_resource(self):
        pass


class CallableCommit(CommitABC, Generic[T]):
    def __init__(self, commit_callable: Callable[[T], Callable[[T], None]]):
        super().__init__()
        self.commit_callable = commit_callable

    def apply(self, base_object: T):
        self.revert_callable = self.commit_callable(base_object)

    def revert(self, base_object: T):
        self.revert_callable(base_object)

    def get_revert_callable(self):
        return self.revert_callable

    def release_revert_resource(self):
        self.revert_callable = None


@dataclasses.dataclass
class AppliedCommitRefItem:
    commit_ref: weakref.ref[CommitABC]
    commit_uuid: uuid.UUID
    revert_callable: Optional[Callable]
    squash: bool = False

    @classmethod
    def from_commit(cls, commit: CommitABC):
        return cls(
            commit_ref=weakref.ref(commit),
            commit_uuid=commit.commit_uuid,
            revert_callable=commit.get_revert_callable(),
        )


class CommitSquashItem(Generic[T]):
    def __init__(self):
        self.commit_stack_states: list[uuid.UUID] = []
        self.revert_callable_list: list[Callable] = []

    def append(self, applied_commit_ref: AppliedCommitRefItem):
        self.commit_stack_states.append(applied_commit_ref.commit_uuid)
        self.revert_callable_list.append(applied_commit_ref.revert_callable)
        applied_commit_ref.revert_callable = None

    def get_commit_stack_states(self):
        return self.commit_stack_states

    def revert(self, base_object: T):
        for revert_callable in reversed(self.revert_callable_list):
            revert_callable(base_object)


class BaseCommitObjectRef(Generic[T]):
    def __init__(self, base_object: T):
        self.object_uuid = uuid.uuid4()
        self.state_uuid = self.object_uuid
        self.base_object = base_object
        self.applied_commit_ref_stack: list[AppliedCommitRefItem] = []

        self._doing_squash = False
        self.active_commit_squash: Optional[CommitSquashItem] = None
        self.squash_commit_stack: list[CommitSquashItem] = []

    def apply_commit(self, commit_list: list[CommitABC]):
        for commit in commit_list:
            try:
                commit.apply(self.base_object)
                self.state_uuid = commit.commit_uuid
                applied_commit_ref = AppliedCommitRefItem.from_commit(commit)
                self.applied_commit_ref_stack.append(applied_commit_ref)
                if self.doing_squash:
                    self._do_commit_squash(applied_commit_ref)
            except Exception as e:
                self.state_uuid = UNKNOWN_STATE_UUID
                raise e

    def revert_commit(self, commit_count: int = 1):
        if commit_count == 0:
            return

        assert commit_count > 0
        assert len(self.applied_commit_ref_stack) >= commit_count

        self.stop_squash()

        if self._expand_revert_stack_with_squash(commit_count) > commit_count:
            raise Exception(f"revert {commit_count} commit but some commit is not full apart of an squash.")

        while commit_count > 0:
            applied_commit = self.applied_commit_ref_stack[-1]
            assert applied_commit.commit_uuid == self.state_uuid

            revert_callable_commit_count = self._expand_revert_stack_with_squash(1)
            commit_count -= revert_callable_commit_count

            revert_commit = self.applied_commit_ref_stack[-revert_callable_commit_count]

            try:
                if revert_commit.squash:
                    self.squash_commit_stack[-1].revert(self.base_object)
                    self.squash_commit_stack.pop()
                else:
                    revert_commit.revert_callable(self.base_object)
                self.applied_commit_ref_stack = self.applied_commit_ref_stack[:-revert_callable_commit_count]
                if len(self.applied_commit_ref_stack) > 0:
                    self.state_uuid = self.applied_commit_ref_stack[-1].commit_uuid
                else:
                    self.state_uuid = self.object_uuid
            except Exception as e:
                self.state_uuid = UNKNOWN_STATE_UUID
                raise e

    def rebase(self, commit_stack: list[CommitABC]):
        current_state_uuid = self.state_uuid
        target_state_uuid = self.object_uuid
        if len(commit_stack) != 0:
            target_state_uuid = commit_stack[-1].commit_uuid

        if current_state_uuid == target_state_uuid:
            return

        if len(commit_stack) == 0:
            self.reset()
            return

        pawn_state_uuid = current_state_uuid

        commit_stack_states = [self.object_uuid] + [p.commit_uuid for p in commit_stack]
        stack_need_apply = commit_stack
        stack_need_revert_count = 0

        _logger.debug(f"commit_stack_states : {commit_stack_states}")

        if pawn_state_uuid not in commit_stack_states:
            applied_commit_stack = self.applied_commit_ref_stack
            applied_commit_stack_states = [self.object_uuid] + [p.commit_uuid for p in applied_commit_stack]
            _logger.debug(f"applied_commit_stack_states : {applied_commit_stack_states}")
            common_path_len = 0
            for applied_state, need_apply_state in zip(applied_commit_stack_states, commit_stack_states):
                if applied_state != need_apply_state:
                    break
                else:
                    common_path_len += 1
            _logger.debug(f"common_path_len : {common_path_len}")
            assert common_path_len != 0
            stack_need_revert_count = self._expand_revert_stack_with_squash(
                len(applied_commit_stack_states[common_path_len:])
            )
            pawn_state_uuid = applied_commit_stack_states[-(stack_need_revert_count + 1)]

        assert pawn_state_uuid in commit_stack_states

        stack_need_apply = commit_stack[commit_stack_states.index(pawn_state_uuid) :]

        self.revert_commit(stack_need_revert_count)
        self.apply_commit(stack_need_apply)

    def do_squash(self, squash=None):
        if squash is None:
            squash = CommitSquashItem()
        self.active_commit_squash = squash
        self._doing_squash = True

    def stop_squash(self):
        self._doing_squash = False

    def get_active_squash(self):
        assert self.doing_squash
        return self.active_commit_squash

    def _do_commit_squash(self, commit_ref: AppliedCommitRefItem):
        active_squash = self.get_active_squash()
        active_squash.append(commit_ref)
        commit_ref.squash = True

        if len(self.squash_commit_stack) == 0 or self.squash_commit_stack[-1] is not active_squash:
            self.squash_commit_stack.append(active_squash)

    def _expand_revert_stack_with_squash(self, revert_commit_count: int):
        assert revert_commit_count > 0

        applied_commit_count = len(self.applied_commit_ref_stack)

        assert applied_commit_count >= revert_commit_count

        if len(self.squash_commit_stack) == 0:
            return revert_commit_count

        if not self.applied_commit_ref_stack[-revert_commit_count].squash:
            return revert_commit_count

        for commit in reversed(self.applied_commit_ref_stack[:-revert_commit_count]):
            if commit.squash:
                revert_commit_count += 1
            else:
                break

        return revert_commit_count

    @property
    def doing_squash(self):
        return self._doing_squash and len(self.squash_commit_stack) > 0

    def reset(self):
        self._reset()
        self.applied_commit_ref_stack = []
        self.state_uuid = self.object_uuid
        self.squash_commit_stack = []
        self.stop_squash()

    def _reset(self):
        self.revert_commit(len(self.applied_commit_ref_stack))


BASE_COMMIT_OBJECT_REF_OBJECT_MAP = weakref.WeakKeyDictionary()


def get_base_commit_object_ref(base_object, base_commit_object_ref_cls=BaseCommitObjectRef):
    """
    pipeline <- ref1
      ├ unet <- ref2
       ├ text_encoder
      ...
    Be careful with nested objects like this.

    Args:
        base_object (Object): Object.
        base_commit_object_ref_cls (type, optional): Ref class. Defaults to BaseCommitObjectRef.
    """
    if base_object not in BASE_COMMIT_OBJECT_REF_OBJECT_MAP:
        base_commit_object_ref = base_commit_object_ref_cls(base_object)
        BASE_COMMIT_OBJECT_REF_OBJECT_MAP[base_object] = weakref.ref(base_commit_object_ref)
    else:
        base_commit_object_ref = BASE_COMMIT_OBJECT_REF_OBJECT_MAP.get(base_object)()
    if type(base_commit_object_ref) is not base_commit_object_ref_cls:
        _logger.warning(
            f"want to get {base_commit_object_ref_cls.__name__} ref of {base_object} but aready have {base_commit_object_ref.__class__.__name__} ref for it."
        )
    return base_commit_object_ref


class CommitObjectProxy(Generic[T]):
    def __init__(self, base_object: Union[T, BaseCommitObjectRef[T]]):
        self.manager_uuid = uuid.uuid4()
        self.commit_stack: list[CommitABC[T]] = []

        self.base_object_ref: BaseCommitObjectRef[T] = None
        if isinstance(base_object, BaseCommitObjectRef):
            self.base_object_ref = base_object
        else:
            self.base_object_ref = get_base_commit_object_ref(base_object)

    def clone(self):
        new_manager = CommitObjectProxy[T](self.base_object_ref)
        new_manager.commit_stack = [*self.commit_stack]
        return new_manager

    def add_commit(self, commit: CommitABC[T]):
        self.commit_stack.append(commit)

    def clone_and_add_commit(self, commit: CommitABC[T]):
        new_self = self.clone()
        new_self.add_commit(commit)
        return new_self

    def clone_and_add_callable_commit(self, commit_callable: Callable[[T], Callable[[T], None]]):
        return self.clone_and_add_commit(CallableCommit(commit_callable))

    def apply_commit_stack(self):
        if self.base_object_ref.state_uuid is UNKNOWN_STATE_UUID:
            self.base_object_ref.reset()

        return self.base_object_ref.rebase(self.commit_stack)

    def get_current_object(self):
        self.apply_commit_stack()
        return self.base_object

    @property
    def base_object(self):
        return self.base_object_ref.base_object

    def __enter__(self):
        return self.get_current_object()

    def __exit__(self, exc_type, exc_value, traceback):
        pass


if __name__ in ("__main__", "<run_path>"):
    base = CommitObjectProxy(None)

    def a(o):
        def r(o):
            pass

        return r

    s1 = base.clone_and_add_callable_commit(a)
    s2 = s1.clone_and_add_callable_commit(a)
    s3 = s2.clone_and_add_callable_commit(a)

    base.base_object_ref.do_squash()
    with s3:
        print(s3.base_object_ref.applied_commit_ref_stack)
    base.base_object_ref.stop_squash()

    base.base_object_ref.do_squash()
    with s2:
        print(s2.base_object_ref.applied_commit_ref_stack)
    base.base_object_ref.stop_squash()

    s11 = s1.clone_and_add_callable_commit(a)

    base.base_object_ref.do_squash()
    with s11:
        print(s11.base_object_ref.applied_commit_ref_stack)
    base.base_object_ref.stop_squash()
