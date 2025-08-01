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


def test_setup_default_pools():
    management = sire.get_resource_management()
    assert management.get_resource_pool(torch.device("cpu")) is not None
    if torch.cuda.is_available():
        assert management.get_resource_pool(torch.device("cuda", 0)) is not None


def test_manage_and_auto_manage_simple():
    model = SimpleModel()
    managed_model_wrapper = sire.manage(model)

    # Check that the model is initially on the CPU
    assert next(model.parameters()).device.type == "cpu"

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available, skipping GPU part of the test")

    # Force runtime device to be CUDA for the managed object
    managed_model_wrapper.user.runtime_resource_pool = sire.get_resource_management().get_resource_pool(
        torch.device("cuda", 0)
    )

    # Before entering the context, the model should be on CPU
    assert next(model.parameters()).device.type == "cpu"

    with sire.auto_manage(managed_model_wrapper) as am:
        # Inside the context, the model should be on the execution device (GPU)
        execution_device = am.get_execution_device()
        assert execution_device.type == "cuda"
        assert next(model.parameters()).device.type == "cuda"

        # You could even run a dummy forward pass
        dummy_input = torch.randn(1, 10).to(execution_device)
        model(dummy_input)

    # After exiting the context, the model should be offloaded back to CPU
    assert next(model.parameters()).device.type == "cpu"


class ContextAwareCommit(sire.CommitABC):
    def __init__(self, expected_device):
        super().__init__()
        self.apply_called = False
        self.expected_device = expected_device

    def apply(self, base_object, auto_manager=None, **kwargs):
        assert auto_manager is not None
        assert auto_manager.get_execution_device().type == self.expected_device
        self.apply_called = True

    def revert(self, base_object):
        pass


def test_commit_with_context():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available, skipping GPU part of the test")

    model = SimpleModel()
    proxy = sire.CommitObjectProxy(model)

    # The AutoManageWrapper for the proxy will be created here
    managed_proxy = sire.manage(proxy)

    # Force the underlying model's manager to use the GPU
    managed_proxy.user.base_object_am.user.runtime_resource_pool = sire.get_resource_management().get_resource_pool(
        torch.device("cuda", 0)
    )

    commit = ContextAwareCommit(expected_device="cuda")
    proxy.add_commit(commit)

    with sire.auto_manage(managed_proxy):
        # This will trigger the on_load of the CommitObjectProxyWrapper,
        # which in turn loads the base model to the correct device.
        # We also need to apply the commit stack to trigger the apply method.
        proxy.apply_commit_stack()

    assert commit.apply_called


def test_automanage_hook():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available, skipping GPU part of the test")

    model = SimpleModel()

    from sire.core.runtime_resource_management import AutoManageHook

    hook = AutoManageHook.manage_module(model)

    # Force the hook to use the GPU
    hook.am.user.runtime_resource_pool = sire.get_resource_management().get_resource_pool(torch.device("cuda", 0))

    assert next(model.parameters()).device.type == "cpu"

    dummy_input = torch.randn(1, 10)
    model(dummy_input)  # This should trigger the pre-forward and post-forward hooks

    # After the forward pass, the model should be back on the CPU
    assert next(model.parameters()).device.type == "cpu"
