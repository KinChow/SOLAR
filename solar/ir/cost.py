# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage 3 hardware-independent cost schemas."""

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional

from solar.ir.operator import OperatorKind
from solar.ir.schema import SCHEMA_VERSION, SchemaValidationError, _plain


@dataclass
class ResourceCost:
    macs: Optional[int] = None
    flops: Optional[int] = None
    alu_ops: Optional[int] = None
    sfu_ops: Optional[int] = None
    reduction_ops: Optional[int] = None
    compare_ops: Optional[int] = None
    index_ops: Optional[int] = None
    logical_read_bytes: Optional[int] = None
    logical_write_bytes: Optional[int] = None
    materialization_bytes: Optional[int] = None
    communication_bytes: Optional[int] = None

    def validate(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if value is not None and value < 0:
                raise SchemaValidationError(f"cost '{item.name}' must not be negative")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResourceCost":
        values = {}
        for item in fields(cls):
            value = data.get(item.name)
            values[item.name] = int(value) if value is not None else None
        return cls(**values)


@dataclass
class NodeCost:
    node_id: str
    operator_kind: OperatorKind
    dtype: str
    cost: ResourceCost
    status: str
    missing_costs: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NodeCost":
        return cls(
            node_id=str(data["node_id"]),
            operator_kind=OperatorKind(data["operator_kind"]),
            dtype=str(data["dtype"]),
            cost=ResourceCost.from_dict(data.get("cost") or {}),
            status=str(data["status"]),
            missing_costs=[str(value) for value in data.get("missing_costs", [])],
        )


@dataclass
class CostIR:
    model_name: str
    nodes: List[NodeCost]
    totals: ResourceCost
    dependencies: Dict[str, List[str]]
    completeness: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise SchemaValidationError(f"unsupported schema version '{self.schema_version}'")
        if self.completeness not in {"complete", "partial"}:
            raise SchemaValidationError(f"invalid cost completeness '{self.completeness}'")
        for node in self.nodes:
            node.cost.validate()
        self.totals.validate()

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return _plain(asdict(self))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CostIR":
        cost_ir = cls(
            model_name=str(data["model_name"]),
            nodes=[NodeCost.from_dict(value) for value in data.get("nodes", [])],
            totals=ResourceCost.from_dict(data.get("totals") or {}),
            dependencies={
                str(node_id): [str(value) for value in dependencies]
                for node_id, dependencies in (data.get("dependencies") or {}).items()
            },
            completeness=str(data["completeness"]),
            metadata=dict(data.get("metadata") or {}),
            schema_version=str(data.get("schema_version", "")),
        )
        cost_ir.validate()
        return cost_ir
