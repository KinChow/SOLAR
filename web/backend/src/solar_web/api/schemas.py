"""Validated public request schemas."""

from math import prod
from typing import List, Literal

from pydantic import BaseModel, Field, StrictInt, model_validator


MAX_TOTAL_ELEMENTS = 16_000_000
MAX_CODE_BYTES = 128 * 1024


class TensorInput(BaseModel):
    name: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    shape: List[StrictInt] = Field(min_length=1, max_length=8)
    dtype: Literal["float32"] = "float32"

    @model_validator(mode="after")
    def validate_shape(self) -> "TensorInput":
        if any(dimension <= 0 for dimension in self.shape):
            raise ValueError("tensor dimensions must be positive integers")
        return self


class RunCreateRequest(BaseModel):
    operator_code: str = Field(min_length=1, max_length=128 * 1024)
    inputs: List[TensorInput] = Field(min_length=1, max_length=8)
    hardware: str = Field(min_length=1, max_length=128)
    case_id: str = Field(default="default", pattern=r"^[A-Za-z0-9_.-]+$")

    @model_validator(mode="after")
    def validate_inputs(self) -> "RunCreateRequest":
        if len(self.operator_code.encode("utf-8")) > MAX_CODE_BYTES:
            raise ValueError("operator code exceeds the 131,072 byte limit")
        names = [tensor.name for tensor in self.inputs]
        if len(set(names)) != len(names):
            raise ValueError("tensor input names must be unique")
        total_elements = sum(prod(tensor.shape) for tensor in self.inputs)
        if total_elements > MAX_TOTAL_ELEMENTS:
            raise ValueError("tensor inputs exceed the 16,000,000 element limit")
        return self


class RunCreateResponse(BaseModel):
    run_id: str
