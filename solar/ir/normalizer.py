# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage 2 workload semantic normalization."""

from typing import Callable, Dict, Tuple

from solar.common.types import TensorShapes
from solar.einsum import EinsumAnalyzer
from solar.ir.operator import (
    ContractionSpec,
    ExecutionGraph,
    ExecutionGroup,
    OperatorGraph,
    OperatorKind,
    OperatorNode,
    PhaseSpec,
)
from solar.ir.schema import WorkloadGraph, WorkloadNode


Normalizer = Callable[[WorkloadGraph, WorkloadNode], OperatorNode]


class OperatorNormalizer:
    """Normalize workload operations through an explicit handler registry."""

    def __init__(self) -> None:
        self._einsum = EinsumAnalyzer()
        self._registry: Dict[str, Normalizer] = {"matmul": self._normalize_matmul}

    def register(self, op_type: str, normalizer: Normalizer) -> None:
        self._registry[op_type.lower()] = normalizer

    def normalize(self, workload: WorkloadGraph) -> Tuple[OperatorGraph, ExecutionGraph]:
        workload.validate()
        nodes = []
        groups = []
        support_levels = {}
        issues = []
        for workload_node in workload.nodes:
            normalizer = self._registry.get(workload_node.op_type.lower())
            if normalizer is None:
                operator = OperatorNode(
                    node_id=workload_node.node_id,
                    kind=OperatorKind.OPAQUE_KERNEL,
                    semantic_op=workload_node.op_type,
                    input_tensors=list(workload_node.input_tensors),
                    output_tensors=list(workload_node.output_tensors),
                    phases=[
                        PhaseSpec(
                            phase_id=f"{workload_node.node_id}.unknown",
                            operation="unknown",
                            kind="unknown",
                        )
                    ],
                    attributes={"reason": "no_registered_normalizer"},
                    source_location=workload_node.source_location,
                )
                support_levels[workload_node.node_id] = "recognized"
                issues.append(
                    {
                        "code": "no_registered_normalizer",
                        "stage": "normalize",
                        "node_id": workload_node.node_id,
                        "message": f"no registered semantic normalizer for '{workload_node.op_type}'",
                        "source_location": workload_node.source_location,
                    }
                )
            else:
                operator = normalizer(workload, workload_node)
                support_levels[workload_node.node_id] = "normalized"
            nodes.append(operator)
            groups.append(
                ExecutionGroup(
                    group_id=f"{workload_node.node_id}.execution",
                    operator_ids=[workload_node.node_id],
                )
            )

        operator_graph = OperatorGraph(
            model_name=workload.model_name,
            nodes=nodes,
            tensors=dict(workload.tensors),
            metadata={
                "source_schema_version": workload.schema_version,
                "phase1_contract_version": "phase1.v1",
                "phase1_support_levels": support_levels,
                "phase1_issues": issues,
            },
        )
        execution_graph = ExecutionGraph(
            model_name=workload.model_name,
            groups=groups,
            metadata={
                "implementation_metadata": "not_provided",
                "phase1_contract_version": "phase1.v1",
            },
        )
        operator_graph.validate()
        execution_graph.validate()
        return operator_graph, execution_graph

    def _normalize_matmul(
        self,
        workload: WorkloadGraph,
        node: WorkloadNode,
    ) -> OperatorNode:
        input_shapes = [workload.tensors[tensor_id].shape for tensor_id in node.input_tensors]
        output_shapes = [workload.tensors[tensor_id].shape for tensor_id in node.output_tensors]
        einsum_op = self._einsum.get_einsum_op(
            "matmul",
            TensorShapes(inputs=input_shapes, outputs=output_shapes),
        )
        if not einsum_op.is_real_einsum or not einsum_op.equation:
            raise ValueError("registered Matmul handler did not produce a contraction")

        operands = {operand.name: list(operand.dims) for operand in einsum_op.operands}
        dimensions: Dict[str, int] = {}
        all_shapes = input_shapes + output_shapes
        for operand, shape in zip(einsum_op.operands, all_shapes):
            if len(operand.dims) != len(shape):
                raise ValueError(
                    f"operand '{operand.name}' rank does not match tensor shape"
                )
            for dim, size in zip(operand.dims, shape):
                previous = dimensions.get(dim)
                if previous is not None and previous != size:
                    raise ValueError(
                        f"dimension '{dim}' has inconsistent sizes {previous} and {size}"
                    )
                dimensions[dim] = int(size)

        return OperatorNode(
            node_id=node.node_id,
            kind=OperatorKind.TENSOR_CONTRACTION,
            semantic_op="matmul",
            input_tensors=list(node.input_tensors),
            output_tensors=list(node.output_tensors),
            phases=[
                PhaseSpec(
                    phase_id=f"{node.node_id}.contraction",
                    operation="multiply_accumulate",
                    kind="contraction",
                )
            ],
            contraction=ContractionSpec(
                equation=einsum_op.equation,
                operands=operands,
                dimensions=dimensions,
            ),
            attributes=dict(node.attributes),
            source_location=node.source_location,
        )


__all__ = ["OperatorNormalizer"]
