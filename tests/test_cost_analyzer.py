# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from solar.analysis import CostAnalyzer
from solar.ir import (
    CostIR,
    OperatorNormalizer,
    TensorRole,
    TensorSpec,
    WorkloadGraph,
    WorkloadNode,
    dump_ir,
    load_ir,
)


def _normalized_matmul():
    workload = WorkloadGraph(
        model_name="Matmul",
        source_file="Matmul.py",
        source_sha256="c" * 64,
        tensors={
            "weight": TensorSpec("weight", [4, 32, 64], "float32", TensorRole.PARAMETER),
            "input": TensorSpec("input", [64, 128], "float32", TensorRole.INPUT),
            "output": TensorSpec("output", [4, 32, 128], "float32", TensorRole.OUTPUT),
        },
        nodes=[WorkloadNode("Model.matmul", "matmul", ["weight", "input"], ["output"])],
        graph_inputs=["input"],
        graph_outputs=["output"],
    )
    return OperatorNormalizer().normalize(workload)


def test_analyzes_matmul_hardware_independent_cost(tmp_path: Path) -> None:
    operator_graph, execution_graph = _normalized_matmul()
    assert set(operator_graph.tensors) == {"weight", "input", "output"}

    cost_ir = CostAnalyzer().analyze(operator_graph, execution_graph)

    assert len(cost_ir.nodes) == 1
    cost = cost_ir.nodes[0].cost
    assert cost.macs == 1_048_576
    assert cost.flops == 2_097_152
    assert cost.alu_ops == 0
    assert cost.sfu_ops == 0
    assert cost.reduction_ops == 0
    assert cost.compare_ops == 0
    assert cost.index_ops == 0
    assert cost.logical_read_bytes == 65_536
    assert cost.logical_write_bytes == 65_536
    assert cost.materialization_bytes == 0
    assert cost.communication_bytes == 0
    assert cost_ir.totals.logical_read_bytes + cost_ir.totals.logical_write_bytes == 131_072
    assert cost_ir.dependencies == {"Model.matmul": []}

    output = tmp_path / "cost_ir.yaml"
    dump_ir(cost_ir, output)
    assert load_ir(CostIR, output) == cost_ir
