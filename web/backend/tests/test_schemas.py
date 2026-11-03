import pytest
from pydantic import ValidationError

from solar_web.api.schemas import RunCreateRequest, TensorInput


VALID_CODE = """class Model(nn.Module):
    def forward(self, x):
        return x
"""


def test_accepts_fp32_tensor_input() -> None:
    request = RunCreateRequest(
        operator_code=VALID_CODE,
        inputs=[TensorInput(name="x", shape=[64, 128], dtype="float32")],
        hardware="H100_PCIe",
    )

    assert request.inputs[0].shape == [64, 128]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("name", "not-valid"),
        ("shape", [64, 0]),
        ("dtype", "float16"),
    ],
)
def test_rejects_invalid_tensor_input(field: str, value: object) -> None:
    data = {"name": "x", "shape": [64, 128], "dtype": "float32"}
    data[field] = value

    with pytest.raises(ValidationError):
        TensorInput.model_validate(data)


def test_rejects_total_input_larger_than_limit() -> None:
    with pytest.raises(ValidationError, match="16,000,000"):
        RunCreateRequest(
            operator_code=VALID_CODE,
            inputs=[TensorInput(name="x", shape=[4_001, 4_000], dtype="float32")],
            hardware="H100_PCIe",
        )


@pytest.mark.parametrize("shape", [[True], ["16"]])
def test_rejects_non_integer_shape_values(shape: list[object]) -> None:
    with pytest.raises(ValidationError):
        TensorInput(name="x", shape=shape, dtype="float32")


def test_code_limit_is_measured_in_utf8_bytes() -> None:
    with pytest.raises(ValidationError, match="131,072 byte"):
        RunCreateRequest(
            operator_code="#" + "算" * 44_000,
            inputs=[TensorInput(name="x", shape=[1], dtype="float32")],
            hardware="H100_PCIe",
        )
