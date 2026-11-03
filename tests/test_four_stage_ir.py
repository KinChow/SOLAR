# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from solar.ir import (
    SCHEMA_VERSION,
    SchemaValidationError,
    TensorRole,
    TensorSpec,
    WorkloadGraph,
    WorkloadNode,
    dump_ir,
    load_ir,
)


def _matmul_workload() -> WorkloadGraph:
    tensors = {
        "weight": TensorSpec(
            tensor_id="weight",
            shape=[4, 32, 64],
            dtype="float32",
            role=TensorRole.PARAMETER,
        ),
        "input": TensorSpec(
            tensor_id="input",
            shape=[64, 128],
            dtype="float32",
            role=TensorRole.INPUT,
        ),
        "output": TensorSpec(
            tensor_id="output",
            shape=[4, 32, 128],
            dtype="float32",
            role=TensorRole.OUTPUT,
        ),
    }
    return WorkloadGraph(
        model_name="Matmul",
        source_file="examples/Matmul/Matmul.py",
        source_sha256="a" * 64,
        tensors=tensors,
        nodes=[
            WorkloadNode(
                node_id="Model.matmul",
                op_type="matmul",
                input_tensors=["weight", "input"],
                output_tensors=["output"],
            )
        ],
        graph_inputs=["input"],
        graph_outputs=["output"],
    )


def test_workload_graph_yaml_round_trip(tmp_path: Path) -> None:
    workload = _matmul_workload()
    output = tmp_path / "workload_graph.yaml"

    dump_ir(workload, output)
    restored = load_ir(WorkloadGraph, output)

    assert restored == workload
    assert restored.schema_version == SCHEMA_VERSION
    assert restored.to_dict()["tensors"]["weight"]["role"] == "parameter"


def test_workload_graph_rejects_negative_tensor_shape() -> None:
    workload = _matmul_workload()
    workload.tensors["input"].shape = [64, -1]

    with pytest.raises(SchemaValidationError, match="negative dimension"):
        workload.validate()


def test_workload_graph_rejects_missing_tensor_reference() -> None:
    workload = _matmul_workload()
    workload.nodes[0].input_tensors.append("missing")

    with pytest.raises(SchemaValidationError, match="unknown tensor 'missing'"):
        workload.validate()
