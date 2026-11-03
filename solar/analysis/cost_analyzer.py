# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage 3 hardware-independent cost analysis."""

from dataclasses import fields
from math import prod
from typing import Callable, Dict, List

from solar.common.types import TensorShapes
from solar.einsum import EinsumAnalyzer
from solar.ir.cost import CostIR, NodeCost, ResourceCost
from solar.ir.dependencies import derive_tensor_dependencies
from solar.ir.operator import ExecutionGraph, OperatorGraph, OperatorKind, OperatorNode


_DTYPE_BYTES = {
    "bool": 1,
    "uint8": 1,
    "int8": 1,
    "float16": 2,
    "bfloat16": 2,
    "int16": 2,
    "float32": 4,
    "int32": 4,
    "float64": 8,
    "int64": 8,
}

CostRule = Callable[[OperatorGraph, OperatorNode], NodeCost]


class CostAnalyzer:
    """Analyze normalized operators using explicit cost-rule registration."""

    def __init__(self) -> None:
        self._einsum = EinsumAnalyzer()
        self._registry: Dict[OperatorKind, CostRule] = {
            OperatorKind.TENSOR_CONTRACTION: self._tensor_contraction_cost,
        }

    def register(self, kind: OperatorKind, rule: CostRule) -> None:
        self._registry[kind] = rule

    def analyze(
        self,
        operator_graph: OperatorGraph,
        execution_graph: ExecutionGraph,
    ) -> CostIR:
        operator_graph.validate()
        execution_graph.validate()
        execution_ids = {
            operator_id for group in execution_graph.groups for operator_id in group.operator_ids
        }
        nodes: List[NodeCost] = []
        for operator in operator_graph.nodes:
            if operator.node_id not in execution_ids:
                raise ValueError(f"operator '{operator.node_id}' has no execution mapping")
            rule = self._registry.get(operator.kind)
            if rule is None:
                nodes.append(
                    NodeCost(
                        node_id=operator.node_id,
                        operator_kind=operator.kind,
                        dtype="unknown",
                        cost=ResourceCost(),
                        status="unsupported",
                        missing_costs=["all"],
                    )
                )
            else:
                nodes.append(rule(operator_graph, operator))

        totals = self._sum_costs([node.cost for node in nodes])
        tensor_dependencies = derive_tensor_dependencies(operator_graph)
        dependencies = {
            node.node_id: sorted(tensor_dependencies[node.node_id])
            for node in operator_graph.nodes
        }
        cost_ir = CostIR(
            model_name=operator_graph.model_name,
            nodes=nodes,
            totals=totals,
            dependencies=dependencies,
            completeness=("complete" if all(node.status == "supported" for node in nodes) else "partial"),
            metadata={"execution_graph_used": True},
        )
        cost_ir.validate()
        return cost_ir

    def _tensor_contraction_cost(
        self,
        graph: OperatorGraph,
        node: OperatorNode,
    ) -> NodeCost:
        if node.contraction is None:
            raise ValueError(f"operator '{node.node_id}' has no contraction spec")
        inputs = [graph.tensors[tensor_id] for tensor_id in node.input_tensors]
        outputs = [graph.tensors[tensor_id] for tensor_id in node.output_tensors]
        macs = int(
            self._einsum.get_compute_cost(
                node.semantic_op,
                TensorShapes(
                    inputs=[tensor.shape for tensor in inputs],
                    outputs=[tensor.shape for tensor in outputs],
                ),
                equation=node.contraction.equation,
            )
        )
        read_bytes = sum(prod(tensor.shape) * self._dtype_bytes(tensor.dtype) for tensor in inputs)
        write_bytes = sum(prod(tensor.shape) * self._dtype_bytes(tensor.dtype) for tensor in outputs)
        return NodeCost(
            node_id=node.node_id,
            operator_kind=node.kind,
            dtype=outputs[0].dtype if outputs else inputs[0].dtype,
            cost=ResourceCost(
                macs=macs,
                flops=2 * macs,
                alu_ops=0,
                sfu_ops=0,
                reduction_ops=0,
                compare_ops=0,
                index_ops=0,
                logical_read_bytes=int(read_bytes),
                logical_write_bytes=int(write_bytes),
                materialization_bytes=0,
                communication_bytes=0,
            ),
            status="supported",
        )

    @staticmethod
    def _dtype_bytes(dtype: str) -> int:
        normalized = dtype.removeprefix("torch.").lower()
        if normalized not in _DTYPE_BYTES:
            raise ValueError(f"unsupported dtype '{dtype}' for byte accounting")
        return _DTYPE_BYTES[normalized]

    @staticmethod
    def _sum_costs(costs: List[ResourceCost]) -> ResourceCost:
        values = {}
        for item in fields(ResourceCost):
            items = [getattr(cost, item.name) for cost in costs]
            values[item.name] = None if any(value is None for value in items) else sum(items)
        return ResourceCost(**values)


__all__ = ["CostAnalyzer"]
