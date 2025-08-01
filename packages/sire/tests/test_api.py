import pytest
import torch

import sire


# A simple model for testing
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)


@pytest.fixture(autouse=True)
def setup_sire():
    """Sets up Sire with default pools before each test."""
    sire.get_resource_management().__init__()  # Reset manager state for isolation
    sire.initialize()


def test_initialize_idempotent():
    """Tests that initialize can be called multiple times safely."""
    sire.initialize()
    sire.initialize()
    management = sire.get_resource_management()
    assert management.get_resource_pool(torch.device("cpu")) is not None


def test_manage_and_auto_manage_simple():
    model = SimpleModel()
    managed_model_wrapper = sire.manage(model)
    assert next(model.parameters()).device.type == "cpu"

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available, skipping GPU part of the test")

    managed_model_wrapper.user.runtime_resource_pool = sire.get_resource_management().get_resource_pool(
        torch.device("cuda", 0)
    )
    assert next(model.parameters()).device.type == "cpu"

    with sire.auto_manage(managed_model_wrapper) as am:
        execution_device = am.get_execution_device()
        assert execution_device.type == "cuda"
        assert next(model.parameters()).device.type == "cuda"
        dummy_input = torch.randn(1, 10).to(execution_device)
        model(dummy_input)

    assert next(model.parameters()).device.type == "cpu"


class ManagedCommit(sire.CommitWithAutoManage):
    def __init__(self):
        super().__init__()
        self.apply_called = False
        self.revert_called = False
        self.execution_device_at_apply = None

    def apply(self, base_object, **kwargs):
        self.apply_called = True
        assert self.am is not None
        self.execution_device_at_apply = self.am.get_execution_device()

    def revert(self, base_object):
        self.revert_called = True

    def get_runtime_device(self):
        return self.am.get_execution_device() if self.am else torch.device("cpu")


def test_commit_with_auto_manage():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available, skipping GPU part of the test")

    model = SimpleModel()
    proxy = sire.AutoManageCommitObjectProxy(model)

    proxy.base_object_ref.am.user.runtime_resource_pool = sire.get_resource_management().get_resource_pool(
        torch.device("cuda", 0)
    )

    commit = ManagedCommit()
    proxy_with_commit = proxy.clone_and_add_commit(commit)
    proxy_with_commit.apply_commit_stack()

    assert commit.apply_called
    assert commit.execution_device_at_apply.type == "cuda"
    assert next(model.parameters()).device.type == "cuda"

    proxy.apply_commit_stack()
    assert commit.revert_called
    assert next(model.parameters()).device.type == "cpu"


def test_automanage_hook():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available, skipping GPU part of the test")

    model = SimpleModel()
    from sire.core.runtime_resource_management import AutoManageHook

    hook = AutoManageHook.manage_module(model)
    hook.am.user.runtime_resource_pool = sire.get_resource_management().get_resource_pool(torch.device("cuda", 0))

    assert next(model.parameters()).device.type == "cpu"
    model(torch.randn(1, 10))
    assert next(model.parameters()).device.type == "cpu"
