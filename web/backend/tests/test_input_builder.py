from pathlib import Path

import pytest

from solar_web.api.schemas import RunCreateRequest, TensorInput
from solar_web.services.input_builder import InputContractError, ModelFileBuilder


def _request(code: str) -> RunCreateRequest:
    return RunCreateRequest(
        operator_code=code,
        inputs=[TensorInput(name="x", shape=[64, 128], dtype="float32")],
        hardware="H100_PCIe",
    )


def test_builds_deterministic_solar_model_file(tmp_path: Path) -> None:
    request = _request(
        """class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(4, 32, 64))

    def forward(self, x):
        return torch.matmul(self.weight, x)
"""
    )

    output = ModelFileBuilder().build(request, tmp_path / "model.py")
    source = output.read_text()

    assert source.startswith("import torch\nfrom torch import nn\n")
    assert "torch.manual_seed(0)" in source
    assert "torch.randn((64, 128), dtype=torch.float32)" in source
    assert "def get_init_inputs():" in source


def test_rejects_forward_arguments_that_do_not_match_inputs(tmp_path: Path) -> None:
    request = _request(
        """class Model(nn.Module):
    def forward(self, left, right):
        return left + right
"""
    )

    with pytest.raises(InputContractError, match="forward arguments"):
        ModelFileBuilder().build(request, tmp_path / "model.py")


def test_rejects_custom_get_inputs(tmp_path: Path) -> None:
    request = _request(
        """class Model(nn.Module):
    def forward(self, x):
        return x

def get_inputs():
    return []
"""
    )

    with pytest.raises(InputContractError, match="get_inputs"):
        ModelFileBuilder().build(request, tmp_path / "model.py")
