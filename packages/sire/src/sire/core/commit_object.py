"""Core components for Sire's commit-based object management."""

from __future__ import annotations

import dataclasses
import logging
import types
import uuid
import weakref
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from .runtime_resource_user.commit_object import CommitObjectProxyWrapper

_logger = logging.getLogger(__name__)

T = TypeVar("T")
UNKNOWN_STATE_UUID = uuid.UUID("00000000-0000-0000-0000-000000000000")


class CommitABC[T](ABC):
    """Abstract base class for a commit that can be applied to an object."""

    def __init__(self) -> None:
        """Initialize the commit with a unique UUID."""
        self.commit_uuid = uuid.uuid4()

    @abstractmethod
    def apply(self, base_object: T, **kwargs: Any) -> None:
        """Apply the commit to the base object."""

    @abstractmethod
    def revert(self, base_object: T) -> None:
        """Revert the commit from the base object."""

    def get_revert_callable(self) -> Callable[..., Any] | None:
        """Get a callable that can revert the commit."""
        return self.revert

    @abstractmethod
    def release_revert_resource(self) -> None:
        """Release any resources held by the revert callable."""


class CallableCommit(CommitABC[T]):
    """A commit that is defined by a callable."""

    def __init__(self, commit_callable: Callable[[T], Callable[[T], None]]):
        """
        Initialize the commit with a callable.

        Args:
            commit_callable: A callable that takes the base object and returns a revert callable.

        """
        super().__init__()
        self.commit_callable = commit_callable
        self.revert_callable: Callable[[T], None] | None = None

    def apply(self, base_object: T, **_kwargs: Any) -> None:
        """Apply the commit by calling the commit_callable."""
        self.revert_callable = self.commit_callable(base_object)

    def revert(self, base_object: T) -> None:
        """Revert the commit by calling the revert_callable."""
        if self.revert_callable:
            self.revert_callable(base_object)

    def get_revert_callable(self) -> Callable[[T], None] | None:
        """Get the revert callable."""
        return self.revert_callable

    def release_revert_resource(self) -> None:
        """Release the revert callable."""
        self.revert_callable = None


@dataclasses.dataclass
class AppliedCommitRefItem:
    """A reference to an applied commit."""

    commit_ref: weakref.ref[CommitABC[Any]]
    commit_uuid: uuid.UUID
    revert_callable: Callable[..., Any] | None
    squash: bool = False

    @classmethod
    def from_commit(cls, commit: CommitABC[Any]) -> AppliedCommitRefItem:
        """Create an AppliedCommitRefItem from a commit."""
        return cls(
            commit_ref=weakref.ref(commit),
            commit_uuid=commit.commit_uuid,
            revert_callable=commit.get_revert_callable(),
        )


class CommitSquashItem[T]:
    """An item that represents a squashed sequence of commits."""

    def __init__(self) -> None:
        """Initialize the squash item."""
        self.commit_stack_states: list[uuid.UUID] = []
        self.revert_callable_list: list[Callable[..., Any] | None] = []

    def append(self, applied_commit_ref: AppliedCommitRefItem) -> None:
        """Append a commit to the squash item."""
        self.commit_stack_states.append(applied_commit_ref.commit_uuid)
        self.revert_callable_list.append(applied_commit_ref.revert_callable)
        applied_commit_ref.revert_callable = None

    def get_commit_stack_states(self) -> list[uuid.UUID]:
        """Get the commit stack states."""
        return self.commit_stack_states

    def revert(self, base_object: T) -> None:
        """Revert all commits in the squash item."""
        for revert_callable in reversed(self.revert_callable_list):
            if revert_callable:
                revert_callable(base_object)


class BaseCommitObjectRef[T]:
    """A reference to a base object that can have commits applied to it."""

    def __init__(self, base_object: T):
        """
        Initialize the base commit object reference.

        Args:
            base_object: The object to be managed.

        """
        self.object_uuid = uuid.uuid4()
        self.state_uuid: uuid.UUID | None = self.object_uuid
        self.base_object = base_object
        self.applied_commit_ref_stack: list[AppliedCommitRefItem] = []

        self._doing_squash = False
        self.active_commit_squash: CommitSquashItem[T] | None = None
        self.squash_commit_stack: list[CommitSquashItem[T]] = []

    def apply_commit(self, commit_list: list[CommitABC[T]]) -> None:
        """Apply a list of commits to the base object."""
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

    def revert_commit(self, commit_count: int = 1) -> None:
        """Revert a number of commits from the base object."""
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
                elif revert_commit.revert_callable:
                    revert_commit.revert_callable(self.base_object)
                self.applied_commit_ref_stack = self.applied_commit_ref_stack[:-revert_callable_commit_count]
                if len(self.applied_commit_ref_stack) > 0:
                    self.state_uuid = self.applied_commit_ref_stack[-1].commit_uuid
                else:
                    self.state_uuid = self.object_uuid
            except Exception as e:
                self.state_uuid = UNKNOWN_STATE_UUID
                raise e

    def rebase(self, commit_stack: list[CommitABC[T]]) -> None:
        """Rebase the object to a new commit stack."""
        current_state_uuid = self.state_uuid
        target_state_uuid: uuid.UUID | None = self.object_uuid
        if len(commit_stack) != 0:
            target_state_uuid = commit_stack[-1].commit_uuid

        if current_state_uuid == target_state_uuid:
            return

        if len(commit_stack) == 0:
            self.reset()
            return

        pawn_state_uuid = current_state_uuid

        commit_stack_states = [self.object_uuid] + [p.commit_uuid for p in commit_stack]
        stack_need_apply: list[CommitABC[T]] = commit_stack
        stack_need_revert_count = 0

        _logger.debug("commit_stack_states : %s", commit_stack_states)

        if pawn_state_uuid not in commit_stack_states:
            applied_commit_stack = self.applied_commit_ref_stack
            applied_commit_stack_states: list[uuid.UUID] = [self.object_uuid] + [
                p.commit_uuid for p in applied_commit_stack
            ]
            _logger.debug("applied_commit_stack_states : %s", applied_commit_stack_states)
            common_path_len = 0
            for applied_state, need_apply_state in zip(applied_commit_stack_states, commit_stack_states, strict=False):
                if applied_state != need_apply_state:
                    break
                common_path_len += 1
            _logger.debug("common_path_len : %s", common_path_len)
            assert common_path_len != 0
            stack_need_revert_count = self._expand_revert_stack_with_squash(
                len(applied_commit_stack_states[common_path_len:])
            )
            pawn_state_uuid = applied_commit_stack_states[-(stack_need_revert_count + 1)]

        assert pawn_state_uuid in commit_stack_states

        stack_need_apply = commit_stack[commit_stack_states.index(pawn_state_uuid) :] if pawn_state_uuid else []

        self.revert_commit(stack_need_revert_count)
        self.apply_commit(stack_need_apply)

    def do_squash(self, squash: CommitSquashItem[T] | None = None) -> None:
        """Start squashing commits."""
        if squash is None:
            squash = CommitSquashItem()
        self.active_commit_squash = squash
        self._doing_squash = True

    def stop_squash(self) -> None:
        """Stop squashing commits."""
        self._doing_squash = False

    def get_active_squash(self) -> CommitSquashItem[T] | None:
        """Get the active squash item."""
        assert self.doing_squash
        return self.active_commit_squash

    def _do_commit_squash(self, commit_ref: AppliedCommitRefItem) -> None:
        active_squash = self.get_active_squash()
        if active_squash:
            active_squash.append(commit_ref)
        commit_ref.squash = True

        if (len(self.squash_commit_stack) == 0 or self.squash_commit_stack[-1] is not active_squash) and active_squash:
            self.squash_commit_stack.append(active_squash)

    def _expand_revert_stack_with_squash(self, revert_commit_count: int) -> int:
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
    def doing_squash(self) -> bool:
        """Whether commits are being squashed."""
        return self._doing_squash and len(self.squash_commit_stack) > 0

    def reset(self) -> None:
        """Reset the object to its initial state."""
        self._reset()
        self.applied_commit_ref_stack = []
        self.state_uuid = self.object_uuid
        self.squash_commit_stack = []
        self.stop_squash()

    def _reset(self) -> None:
        self.revert_commit(len(self.applied_commit_ref_stack))


BASE_COMMIT_OBJECT_REF_OBJECT_MAP: weakref.WeakKeyDictionary[Any, weakref.ref[BaseCommitObjectRef[Any]]] = (
    weakref.WeakKeyDictionary()
)

BCO = TypeVar("BCO", bound=BaseCommitObjectRef[Any])


def get_base_commit_object_ref[BCO: BaseCommitObjectRef[Any]](
    base_object: Any,
    base_commit_object_ref_cls: type[BCO] = BaseCommitObjectRef,  # type: ignore
) -> BCO:
    """
    Get or create a BaseCommitObjectRef for a given object.

    Pipeline <- ref1
      ├ unet <- ref2
       ├ text_encoder
      ...
    Be careful with nested objects like this.

    Args:
        base_object (Object): Object.
        base_commit_object_ref_cls (type, optional): Ref class. Defaults to BaseCommitObjectRef.

    """
    base_commit_object_ref_ref = BASE_COMMIT_OBJECT_REF_OBJECT_MAP.get(base_object)
    if base_commit_object_ref_ref:
        base_commit_object_ref = base_commit_object_ref_ref()
        if base_commit_object_ref and isinstance(base_commit_object_ref, base_commit_object_ref_cls):
            return base_commit_object_ref

    base_commit_object_ref = base_commit_object_ref_cls(base_object)
    BASE_COMMIT_OBJECT_REF_OBJECT_MAP[base_object] = weakref.ref(base_commit_object_ref)
    return base_commit_object_ref


class CommitObjectProxy[T]:
    """A proxy for an object that can have commits applied to it."""

    def __init__(self, base_object: T | BaseCommitObjectRef[T]):
        """
        Initialize the proxy.

        Args:
            base_object: The object to proxy, or a reference to it.

        """
        self.manager_uuid = uuid.uuid4()
        self.commit_stack: list[CommitABC[T]] = []
        self.am_ref: weakref.ref[CommitObjectProxyWrapper[T]] | None = None

        self.base_object_ref: BaseCommitObjectRef[T]
        if isinstance(base_object, BaseCommitObjectRef):
            self.base_object_ref = base_object
        else:
            self.base_object_ref = get_base_commit_object_ref(base_object)

    def clone(self) -> CommitObjectProxy[T]:
        """Clone the proxy."""
        new_manager = CommitObjectProxy[T](self.base_object_ref)
        new_manager.commit_stack = [*self.commit_stack]
        return new_manager

    def add_commit(self, commit: CommitABC[T]) -> None:
        """Add a commit to the stack."""
        self.commit_stack.append(commit)

    def clone_and_add_commit(self, commit: CommitABC[T]) -> CommitObjectProxy[T]:
        """Clone the proxy and add a commit."""
        new_self = self.clone()
        new_self.add_commit(commit)
        return new_self

    def clone_and_add_callable_commit(
        self, commit_callable: Callable[[T], Callable[[T], None]]
    ) -> CommitObjectProxy[T]:
        """Clone the proxy and add a callable commit."""
        return self.clone_and_add_commit(CallableCommit(commit_callable))

    def apply_commit_stack(self) -> None:
        """Apply the current commit stack to the base object."""
        if self.base_object_ref.state_uuid is UNKNOWN_STATE_UUID:
            self.base_object_ref.reset()

        self.base_object_ref.rebase(self.commit_stack)

    def get_current_object(self) -> T:
        """Get the object with the current commit stack applied."""
        self.apply_commit_stack()
        return self.base_object

    @property
    def base_object(self) -> T:
        """The underlying base object."""
        return self.base_object_ref.base_object

    def __enter__(self) -> T:
        """Enter the context manager, applying the commit stack."""
        return self.get_current_object()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> None:
        """Exit the context manager."""


if __name__ in ("__main__", "<run_path>"):
    base = CommitObjectProxy(object())

    def a(_o: Any) -> Callable[[Any], None]:
        """Define a dummy commit function for testing."""

        def r(_o: Any) -> None:
            pass

        return r

    s1 = base.clone_and_add_callable_commit(a)
    s2 = s1.clone_and_add_callable_commit(a)
    s3 = s2.clone_and_add_callable_commit(a)

    base.base_object_ref.do_squash()
    with s3:
        _logger.info(s3.base_object_ref.applied_commit_ref_stack)
    base.base_object_ref.stop_squash()

    base.base_object_ref.do_squash()
    with s2:
        _logger.info(s2.base_object_ref.applied_commit_ref_stack)
    base.base_object_ref.stop_squash()

    s11 = s1.clone_and_add_callable_commit(a)

    base.base_object_ref.do_squash()
    with s11:
        _logger.info(s11.base_object_ref.applied_commit_ref_stack)
    base.base_object_ref.stop_squash()
