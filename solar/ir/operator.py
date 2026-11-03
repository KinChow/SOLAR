# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage 2 semantic and execution graph schemas."""

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from solar.ir.schema import SCHEMA_VERSION, SchemaValidationError, TensorSpec, _plain


class OperatorKind(str, Enum):
    TENSOR_CONTRACTION = "tensor_contraction"
    ELEMENTWISE = "elementwise"
    REDUCTION = "reduction"
    NORMALIZATION = "normalization"
    DATA_MOVEMENT = "data_movement"
    INDEXED_ACCESS = "indexed_access"
    SCAN = "scan"
    COLLECTIVE = "collective"
    OPAQUE_KERNEL = "opaque_kernel"


@dataclass
class PhaseSpec:
    phase_id: str
    operation: str
    kind: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PhaseSpec":
        return cls(str(data["phase_id"]), str(data["operation"]), str(data["kind"]))


@dataclass
class ContractionSpec:
    equation: str
    operands: Dict[str, List[str]]
    dimensions: Dict[str, int]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContractionSpec":
        return cls(
            equation=str(data["equation"]),
            operands={
                str(name): [str(dim) for dim in dims]
                for name, dims in (data.get("operands") or {}).items()
            },
            dimensions={
                str(name): int(size)
                for name, size in (data.get("dimensions") or {}).items()
            },
        )


@dataclass
class OperatorNode:
    node_id: str
    kind: OperatorKind
    semantic_op: str
    input_tensors: List[str]
    output_tensors: List[str]
    phases: List[PhaseSpec]
    contraction: Optional[ContractionSpec] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    source_location: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OperatorNode":
        contraction = data.get("contraction")
        return cls(
            node_id=str(data["node_id"]),
            kind=OperatorKind(data["kind"]),
            semantic_op=str(data["semantic_op"]),
            input_tensors=[str(value) for value in data.get("input_tensors", [])],
            output_tensors=[str(value) for value in data.get("output_tensors", [])],
            phases=[PhaseSpec.from_dict(value) for value in data.get("phases", [])],
            contraction=(ContractionSpec.from_dict(contraction) if contraction else None),
            attributes=dict(data.get("attributes") or {}),
            source_location=data.get("source_location"),
        )


@dataclass
class OperatorGraph:
    model_name: str
    nodes: List[OperatorNode]
    tensors: Dict[str, TensorSpec]
    metadata: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise SchemaValidationError(f"unsupported schema version '{self.schema_version}'")
        seen = set()
        for node in self.nodes:
            if node.node_id in seen:
                raise SchemaValidationError(f"duplicate operator id '{node.node_id}'")
            seen.add(node.node_id)
            if node.kind == OperatorKind.TENSOR_CONTRACTION and node.contraction is None:
                raise SchemaValidationError(
                    f"tensor contraction '{node.node_id}' has no contraction spec"
                )
            for tensor_id in node.input_tensors + node.output_tensors:
                if tensor_id not in self.tensors:
                    raise SchemaValidationError(
                        f"operator '{node.node_id}' references unknown tensor '{tensor_id}'"
                    )

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return _plain(asdict(self))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OperatorGraph":
        graph = cls(
            model_name=str(data["model_name"]),
            nodes=[OperatorNode.from_dict(value) for value in data.get("nodes", [])],
            tensors={
                str(key): TensorSpec.from_dict(value)
                for key, value in (data.get("tensors") or {}).items()
            },
            metadata=dict(data.get("metadata") or {}),
            schema_version=str(data.get("schema_version", "")),
        )
        graph.validate()
        return graph


@dataclass
class ExecutionGroup:
    group_id: str
    operator_ids: List[str]
    implementation: str = "unknown"
    kernel_count: Optional[int] = None
    fusion: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionGroup":
        return cls(
            group_id=str(data["group_id"]),
            operator_ids=[str(value) for value in data.get("operator_ids", [])],
            implementation=str(data.get("implementation", "unknown")),
            kernel_count=(
                int(data["kernel_count"]) if data.get("kernel_count") is not None else None
            ),
            fusion=str(data.get("fusion", "unknown")),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass
class ExecutionGraph:
    model_name: str
    groups: List[ExecutionGroup]
    metadata: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise SchemaValidationError(f"unsupported schema version '{self.schema_version}'")
        group_ids = set()
        all_operator_ids = set()
        for group in self.groups:
            if not group.group_id or not group.operator_ids:
                raise SchemaValidationError("execution groups require an id and operators")
            if group.group_id in group_ids:
                raise SchemaValidationError(f"duplicate execution group '{group.group_id}'")
            group_ids.add(group.group_id)
            if group.kernel_count is not None and group.kernel_count < 0:
                raise SchemaValidationError(
                    f"execution group '{group.group_id}' has a negative kernel count"
                )
            overlap = all_operator_ids.intersection(group.operator_ids)
            if overlap:
                raise SchemaValidationError(
                    f"operator(s) mapped to multiple execution groups: {sorted(overlap)}"
                )
            all_operator_ids.update(group.operator_ids)

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return _plain(asdict(self))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionGraph":
        graph = cls(
            model_name=str(data["model_name"]),
            groups=[ExecutionGroup.from_dict(value) for value in data.get("groups", [])],
            metadata=dict(data.get("metadata") or {}),
            schema_version=str(data.get("schema_version", "")),
        )
        graph.validate()
        return graph
