# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Versioned schemas shared by the four-stage pipeline."""

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


SCHEMA_VERSION = "1.0"
PHASE1_CONTRACT_VERSION = "phase1.v1"


class SchemaValidationError(ValueError):
    """Raised when an IR object violates its schema contract."""


class TensorRole(str, Enum):
    INPUT = "input"
    PARAMETER = "parameter"
    INTERMEDIATE = "intermediate"
    OUTPUT = "output"


class Phase1SupportLevel(str, Enum):
    """Highest Phase 1 capability proven for an API or graph node."""

    RECOGNIZED = "recognized"
    NORMALIZED = "normalized"
    COST_COMPLETE = "cost_complete"
    LB_SUPPORTED = "lb_supported"
    UNSUPPORTED = "unsupported"


def _plain(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value


@dataclass
class TensorSpec:
    tensor_id: str
    shape: List[int]
    dtype: str
    role: TensorRole
    layout: str = "unknown"
    stride: Optional[List[int]] = None
    alias_of: Optional[str] = None
    materialized: Optional[bool] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.tensor_id:
            raise SchemaValidationError("tensor_id must not be empty")
        if any(not isinstance(dim, int) or dim < 0 for dim in self.shape):
            raise SchemaValidationError(
                f"tensor '{self.tensor_id}' has a negative dimension or invalid shape"
            )
        if not self.dtype:
            raise SchemaValidationError(f"tensor '{self.tensor_id}' has no dtype")
        if self.stride is not None and len(self.stride) != len(self.shape):
            raise SchemaValidationError(
                f"tensor '{self.tensor_id}' stride rank does not match shape rank"
            )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TensorSpec":
        return cls(
            tensor_id=str(data["tensor_id"]),
            shape=[int(dim) for dim in data.get("shape", [])],
            dtype=str(data["dtype"]),
            role=TensorRole(data["role"]),
            layout=str(data.get("layout", "unknown")),
            stride=(
                [int(value) for value in data["stride"]]
                if data.get("stride") is not None
                else None
            ),
            alias_of=data.get("alias_of"),
            materialized=data.get("materialized"),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass
class WorkloadNode:
    node_id: str
    op_type: str
    input_tensors: List[str]
    output_tensors: List[str]
    attributes: Dict[str, Any] = field(default_factory=dict)
    source_location: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkloadNode":
        return cls(
            node_id=str(data["node_id"]),
            op_type=str(data["op_type"]),
            input_tensors=[str(value) for value in data.get("input_tensors", [])],
            output_tensors=[str(value) for value in data.get("output_tensors", [])],
            attributes=dict(data.get("attributes") or {}),
            source_location=data.get("source_location"),
        )


@dataclass
class WorkloadGraph:
    model_name: str
    source_file: str
    source_sha256: str
    tensors: Dict[str, TensorSpec]
    nodes: List[WorkloadNode]
    graph_inputs: List[str]
    graph_outputs: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise SchemaValidationError(
                f"unsupported schema version '{self.schema_version}'"
            )
        for tensor_id, tensor in self.tensors.items():
            if tensor_id != tensor.tensor_id:
                raise SchemaValidationError(
                    f"tensor key '{tensor_id}' does not match id '{tensor.tensor_id}'"
                )
            tensor.validate()
            if tensor.alias_of is not None and tensor.alias_of not in self.tensors:
                raise SchemaValidationError(
                    f"tensor '{tensor_id}' aliases unknown tensor '{tensor.alias_of}'"
                )
        for node in self.nodes:
            if not node.node_id or not node.op_type:
                raise SchemaValidationError("workload node id and op_type are required")
            for tensor_id in node.input_tensors + node.output_tensors:
                if tensor_id not in self.tensors:
                    raise SchemaValidationError(
                        f"node '{node.node_id}' references unknown tensor '{tensor_id}'"
                    )
        for tensor_id in self.graph_inputs + self.graph_outputs:
            if tensor_id not in self.tensors:
                raise SchemaValidationError(
                    f"graph references unknown tensor '{tensor_id}'"
                )

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return _plain(asdict(self))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkloadGraph":
        graph = cls(
            model_name=str(data["model_name"]),
            source_file=str(data["source_file"]),
            source_sha256=str(data["source_sha256"]),
            tensors={
                str(key): TensorSpec.from_dict(value)
                for key, value in (data.get("tensors") or {}).items()
            },
            nodes=[WorkloadNode.from_dict(value) for value in data.get("nodes", [])],
            graph_inputs=[str(value) for value in data.get("graph_inputs", [])],
            graph_outputs=[str(value) for value in data.get("graph_outputs", [])],
            metadata=dict(data.get("metadata") or {}),
            schema_version=str(data.get("schema_version", "")),
        )
        graph.validate()
        return graph
