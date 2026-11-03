# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data dependencies derived from Operator IR tensor flow."""

from typing import Dict, List

from solar.ir.operator import OperatorGraph


TensorDependencies = Dict[str, Dict[str, List[str]]]


def derive_tensor_dependencies(graph: OperatorGraph) -> TensorDependencies:
    """Map each consumer to its producers and the tensors connecting them."""
    graph.validate()
    producers = {
        tensor_id: node.node_id
        for node in graph.nodes
        for tensor_id in node.output_tensors
    }
    dependencies: TensorDependencies = {node.node_id: {} for node in graph.nodes}
    for node in graph.nodes:
        for tensor_id in node.input_tensors:
            source = producers.get(tensor_id)
            if source is None or source == node.node_id:
                continue
            tensor_ids = dependencies[node.node_id].setdefault(source, [])
            if tensor_id not in tensor_ids:
                tensor_ids.append(tensor_id)
    return dependencies
