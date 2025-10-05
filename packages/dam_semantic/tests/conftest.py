import pytest
from dam_sire.resource import SireResource
from dam_test_utils.fixtures import MockSentenceTransformer
from sire.core.runtime_resource_user.pytorch_module import TorchModuleWrapper

pytest_plugins = ["dam_test_utils.fixtures"]


@pytest.fixture
def sire_resource():
    resource = SireResource()
    resource.register_model_type(MockSentenceTransformer, TorchModuleWrapper)
    return resource
