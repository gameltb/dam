import pytest

pytest_plugins = ["dam_test_utils.fixtures"]

@pytest.fixture
def sire_resource():
    from dam_sire.resource import SireResource
    from sire.core.runtime_resource_user.pytorch_module import TorchModuleWrapper
    from dam_test_utils.fixtures import MockSentenceTransformer

    resource = SireResource()
    resource.register_model_type(MockSentenceTransformer, TorchModuleWrapper)
    return resource
