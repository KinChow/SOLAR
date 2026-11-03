# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from solar.ir import (
    ExecutionGraph,
    OperatorGraph,
    OperatorKind,
    OperatorNormalizer,
    TensorRole,
    TensorSpec,
    WorkloadGraph,
    WorkloadNode,
    dump_ir,
    load_ir,
)


def _workload() -> WorkloadGraph:
    return WorkloadGraph(
        model_name="Matmul",
        source_file="Matmul.py",
        source_sha256="b" * 64,
        tensors={
            "weight": TensorSpec("weight", [4, 32, 64], "float32", TensorRole.PARAMETER),
            "input": TensorSpec("input", [64, 128], "float32", TensorRole.INPUT),
            "output": TensorSpec("output", [4, 32, 128], "float32", TensorRole.OUTPUT),
        },
        nodes=[WorkloadNode("Model.matmul", "matmul", ["weight", "input"], ["output"])],
        graph_inputs=["input"],
        graph_outputs=["output"],
    )


def test_normalizes_matmul_as_tensor_contraction(tmp_path: Path) -> None:
    operator_graph, execution_graph = OperatorNormalizer().normalize(_workload())

    assert len(operator_graph.nodes) == 1
    node = operator_graph.nodes[0]
    assert node.kind == OperatorKind.TENSOR_CONTRACTION
    assert node.semantic_op == "matmul"
    assert node.input_tensors == ["weight", "input"]
    assert node.output_tensors == ["output"]
    assert node.contraction is not None
    assert node.contraction.equation == "B0MK,KN->B0MN"
    assert node.contraction.operands == {
        "Input": ["B0", "M", "K"],
        "Weight": ["K", "N"],
        "Output": ["B0", "M", "N"],
    }
    assert node.contraction.dimensions == {"B0": 4, "M": 32, "K": 64, "N": 128}
    assert [phase.operation for phase in node.phases] == ["multiply_accumulate"]

    group = execution_graph.groups[0]
    assert group.operator_ids == ["Model.matmul"]
    assert group.implementation == "unknown"
    assert group.kernel_count is None
    assert group.fusion == "unknown"

    operator_path = tmp_path / "operator_ir.yaml"
    execution_path = tmp_path / "execution_graph.yaml"
    dump_ir(operator_graph, operator_path)
    dump_ir(execution_graph, execution_path)
    assert load_ir(OperatorGraph, operator_path) == operator_graph
    assert load_ir(ExecutionGraph, execution_path) == execution_graph
