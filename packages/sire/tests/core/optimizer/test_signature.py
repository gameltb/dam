import torch
from torch import nn

from sire.core.optimizer.signature import ConfigSignatureGenerator


class SimpleModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()  # type: ignore
        self.linear1 = nn.Linear(10, 20)
        self.linear2 = nn.Linear(20, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.linear1(x))


def test_signature_generator_consistency() -> None:
    """Tests that the signature generator produces the same signature for the same inputs."""
    gen = ConfigSignatureGenerator()
    model = SimpleModel()
    args = (torch.randn(8, 10),)
    kwargs = {"an_option": True}
    dtype = torch.float32

    sig1, _ = gen.generate_config_signature(model, args, kwargs, dtype)  # type: ignore
    sig2, _ = gen.generate_config_signature(model, args, kwargs, dtype)  # type: ignore

    assert sig1 == sig2


def test_signature_generator_input_change_sensitivity() -> None:
    """Tests that the signature changes when model inputs change."""
    gen = ConfigSignatureGenerator()
    model = SimpleModel()
    dtype = torch.float32

    args1 = (torch.randn(8, 10),)
    kwargs1 = {"an_option": True}
    sig1, _ = gen.generate_config_signature(model, args1, kwargs1, dtype)  # type: ignore

    # Change args shape
    args2 = (torch.randn(16, 10),)
    kwargs2 = {"an_option": True}
    sig2, _ = gen.generate_config_signature(model, args2, kwargs2, dtype)  # type: ignore

    # Change kwargs value
    args3 = (torch.randn(8, 10),)
    kwargs3 = {"an_option": False}
    sig3, _ = gen.generate_config_signature(model, args3, kwargs3, dtype)  # type: ignore

    assert sig1 != sig2
    assert sig1 != sig3
    assert sig2 != sig3


def test_signature_generator_model_change_sensitivity() -> None:
    """Tests that the signature changes when the model changes."""
    gen = ConfigSignatureGenerator()
    args = (torch.randn(8, 10),)
    kwargs = {}
    dtype = torch.float32

    model1 = SimpleModel()
    sig1, _ = gen.generate_config_signature(model1, args, kwargs, dtype)  # type: ignore

    # Same architecture, but different weights
    model2 = SimpleModel()
    sig2, _ = gen.generate_config_signature(model2, args, kwargs, dtype)  # type: ignore

    # Different architecture
    class DifferentModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()  # type: ignore
            self.linear = nn.Linear(10, 5)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.linear(x)

    model3 = DifferentModel()
    sig3, _ = gen.generate_config_signature(model3, args, kwargs, dtype)  # type: ignore

    # Models 1 and 2 have different weights, so their weight hash should differ,
    # resulting in different final signatures.
    assert sig1 != sig2
    assert sig1 != sig3


def test_plan_identifier_generator() -> None:
    """Tests the generation of the plan identifier based on memory constraints."""
    gen = ConfigSignatureGenerator()
    mem1 = {"0": 8 * 1024**3, "cpu": 16 * 1024**3}
    mem2 = {"0": 16 * 1024**3, "cpu": 16 * 1024**3}
    mem3 = {"0": 8 * 1024**3, "cpu": 16 * 1024**3}

    plan_id1 = gen.generate_plan_identifier(mem1)
    plan_id2 = gen.generate_plan_identifier(mem2)
    plan_id3 = gen.generate_plan_identifier(mem3)

    assert plan_id1 != plan_id2
    assert plan_id1 == plan_id3
