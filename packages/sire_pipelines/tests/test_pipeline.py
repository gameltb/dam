import pytest
import sire
import torch

from sire_pipelines.pipelines.simple_pipeline import SimplePipeline


@pytest.fixture(autouse=True)
def setup_sire_for_testing():
    """Sets up Sire with default pools and registers the torch wrapper."""
    sire.get_resource_management().__init__()
    sire.initialize()


def test_simple_pipeline_instantiation():
    """Tests that the pipeline can be created."""
    pipeline = SimplePipeline()
    assert pipeline.unet_managed is not None
    assert pipeline.vae_managed is not None
    assert pipeline._unet is not None
    assert pipeline._vae is not None
    print("Pipeline instantiated successfully.")


def test_simple_pipeline_call():
    """Tests that the pipeline can be called without errors."""
    pipeline = SimplePipeline()
    dummy_vae_input = torch.randn(1, 256)
    dummy_unet_input = torch.randn(1, 128)

    # This should run without raising exceptions
    output = pipeline(dummy_vae_input, dummy_unet_input)

    assert isinstance(output, torch.Tensor)
    print("Pipeline called successfully.")
