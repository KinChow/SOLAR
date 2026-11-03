# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from solar.graph import WorkloadExtractor
from solar.ir import TensorRole, load_ir, WorkloadGraph


REPO_ROOT = Path(__file__).resolve().parents[1]
MATMUL_MODEL = REPO_ROOT / "examples" / "Matmul" / "Matmul.py"


def test_extracts_matmul_workload_graph(tmp_path: Path) -> None:
    output_dir = tmp_path / "stage1"

    workload = WorkloadExtractor().extract(
        MATMUL_MODEL,
        output_dir,
        force_rerun=True,
    )

    assert (output_dir / "pytorch_graph.yaml").is_file()
    assert (output_dir / "workload_graph.yaml").is_file()
    assert load_ir(WorkloadGraph, output_dir / "workload_graph.yaml") == workload
    assert workload.model_name == "Matmul"
    assert len(workload.nodes) == 1

    node = workload.nodes[0]
    assert node.node_id == "Model.matmul"
    assert node.op_type == "matmul"
    assert [workload.tensors[tensor_id].shape for tensor_id in node.input_tensors] == [
        [4, 32, 64],
        [64, 128],
    ]
    assert workload.tensors[node.input_tensors[0]].role == TensorRole.PARAMETER
    assert workload.tensors[node.input_tensors[1]].role == TensorRole.INPUT
    assert workload.tensors[node.output_tensors[0]].role == TensorRole.OUTPUT
    assert all(tensor.dtype == "float32" for tensor in workload.tensors.values())
    assert workload.graph_inputs == [node.input_tensors[1]]
    assert workload.graph_outputs == node.output_tensors


def test_matmul_workload_ids_are_stable(tmp_path: Path) -> None:
    extractor = WorkloadExtractor()
    first = extractor.extract(MATMUL_MODEL, tmp_path / "first", force_rerun=True)
    second = extractor.extract(MATMUL_MODEL, tmp_path / "second", force_rerun=True)

    assert list(first.tensors) == list(second.tensors)
    assert [node.node_id for node in first.nodes] == [
        node.node_id for node in second.nodes
    ]
