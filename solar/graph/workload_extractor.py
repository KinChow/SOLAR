# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage 1 adapter from Solar's PyTorch graph to WorkloadGraph."""

import hashlib
from pathlib import Path
from typing import Any, Dict, List

import yaml

from solar.common.types import ProcessingConfig
from solar.graph.pytorch_processor import PyTorchProcessor
from solar.ir import TensorRole, TensorSpec, WorkloadGraph, WorkloadNode, dump_ir


_TENSOR_ROLE = {
    "auxiliary-tensor": TensorRole.INPUT,
    "input-tensor": TensorRole.INPUT,
    "parameter-tensor": TensorRole.PARAMETER,
    "hidden-tensor": TensorRole.INTERMEDIATE,
    "output-tensor": TensorRole.OUTPUT,
}


def _canonical_dtype(value: str) -> str:
    dtype = str(value or "unknown")
    return dtype.removeprefix("torch.")


class WorkloadExtractor:
    """Extract a typed workload while preserving the legacy Stage 1 artifact."""

    def __init__(self, *, debug: bool = False) -> None:
        self.debug = debug

    def extract(
        self,
        model_file: Path,
        output_dir: Path,
        *,
        case_id: str = "default",
        force_rerun: bool = False,
        save_graph: bool = False,
    ) -> WorkloadGraph:
        if not case_id or any(char.isspace() for char in case_id):
            raise ValueError("case_id must be a non-empty token without whitespace")
        model_path = Path(model_file).resolve()
        out_dir = Path(output_dir)
        processor = PyTorchProcessor(
            ProcessingConfig(
                debug=self.debug,
                force_rerun=force_rerun,
                save_graph=save_graph,
                output_dir=str(out_dir),
            )
        )
        if not processor.process_model_file(str(model_path), str(out_dir)):
            raise RuntimeError(f"failed to extract PyTorch graph from {model_path}")

        legacy_path = out_dir / "pytorch_graph.yaml"
        with legacy_path.open() as stream:
            legacy = yaml.safe_load(stream) or {}
        workload = self._from_legacy_graph(legacy, model_path, case_id=case_id)
        dump_ir(workload, out_dir / "workload_graph.yaml")
        return workload

    def _from_legacy_graph(
        self,
        legacy: Dict[str, Any],
        model_path: Path,
        *,
        case_id: str = "default",
    ) -> WorkloadGraph:
        layers = legacy.get("layers") or {}
        tensors: Dict[str, TensorSpec] = {}
        nodes: List[WorkloadNode] = []

        for node_id, node in layers.items():
            node_type = str(node.get("type", "unknown"))
            role = _TENSOR_ROLE.get(node_type)
            if role is None:
                continue
            shapes = list(node.get("output_shapes") or node.get("input_shapes") or [])
            dtypes = list(node.get("output_dtypes") or node.get("input_dtypes") or [])
            if len(shapes) != 1:
                raise ValueError(
                    f"tensor node '{node_id}' must describe exactly one tensor shape"
                )
            tensors[str(node_id)] = TensorSpec(
                tensor_id=str(node_id),
                shape=[int(dim) for dim in shapes[0]],
                dtype=_canonical_dtype(dtypes[0] if dtypes else "unknown"),
                role=role,
                materialized=True,
                metadata={"source": "pytorch_graph.yaml"},
            )

        for node_id, node in layers.items():
            if str(node.get("node_class", "")) == "TensorNode":
                continue
            connections = node.get("connections") or {}
            nodes.append(
                WorkloadNode(
                    node_id=str(node_id),
                    op_type=str(node.get("type", "unknown")),
                    input_tensors=[str(value) for value in connections.get("inputs", [])],
                    output_tensors=[str(value) for value in connections.get("outputs", [])],
                    attributes=dict(node.get("module_args") or {}),
                    source_location=str(model_path),
                )
            )

        workload = WorkloadGraph(
            model_name=str(legacy.get("model_name") or model_path.stem),
            source_file=str(model_path),
            source_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
            tensors=tensors,
            nodes=nodes,
            graph_inputs=[
                tensor_id
                for tensor_id, tensor in tensors.items()
                if tensor.role == TensorRole.INPUT
            ],
            graph_outputs=[
                tensor_id
                for tensor_id, tensor in tensors.items()
                if tensor.role == TensorRole.OUTPUT
            ],
            metadata={
                "execution_path": "fixed_forward",
                "legacy_artifact": "pytorch_graph.yaml",
                "phase1_contract_version": "phase1.v1",
                "phase1_case_id": case_id,
                "phase1_input_constraints": {
                    "dynamic_shape": False,
                    "target_hardware": "P800",
                    "multi_device_collective": False,
                },
            },
        )
        workload.validate()
        return workload


__all__ = ["WorkloadExtractor"]
