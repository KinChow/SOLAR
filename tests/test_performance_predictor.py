# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from solar.analysis import CostAnalyzer
from solar.ir import (
    DiagnosisSection,
    LowerBoundComponents,
    OperatorKind,
    OperatorNormalizer,
    PerformanceSection,
    PerformanceReport,
    SchemaValidationError,
    SupportSection,
    SupportStatus,
    TensorRole,
    TensorSpec,
    WorkloadGraph,
    WorkloadNode,
    dump_ir,
    load_ir,
)
from solar.perf import PerformancePredictor


def _matmul_cost():
    workload = WorkloadGraph(
        model_name="Matmul",
        source_file="Matmul.py",
        source_sha256="d" * 64,
        tensors={
            "weight": TensorSpec("weight", [4, 32, 64], "float32", TensorRole.PARAMETER),
            "input": TensorSpec("input", [64, 128], "float32", TensorRole.INPUT),
            "output": TensorSpec("output", [4, 32, 128], "float32", TensorRole.OUTPUT),
        },
        nodes=[WorkloadNode("Model.matmul", "matmul", ["weight", "input"], ["output"])],
        graph_inputs=["input"],
        graph_outputs=["output"],
    )
    operators, execution = OperatorNormalizer().normalize(workload)
    return CostAnalyzer().analyze(operators, execution), execution


def test_predicts_matmul_lower_bound_from_legacy_h100_config(tmp_path: Path) -> None:
    cost_ir, execution_graph = _matmul_cost()

    report = PerformancePredictor().predict(
        cost_ir,
        execution_graph,
        hardware="H100_PCIe",
    )

    assert report.hardware == "H100_PCIe"
    assert report.performance.components.compute_lower_bound_us == pytest.approx(
        0.020560313725490196, rel=1e-12
    )
    assert report.performance.components.memory_lower_bound_us == pytest.approx(
        0.06428879733176378, rel=1e-12
    )
    assert report.performance.theoretical_lower_bound_us == pytest.approx(
        0.06428879733176378, rel=1e-12
    )
    assert report.performance.completeness == "complete"
    assert report.diagnosis.bottleneck == "memory"
    assert report.diagnosis.execution_path == "cuda_core+dram"
    assert report.support.status == SupportStatus.SUPPORTED
    assert report.support.confidence == "high"
    assert report.support.missing_capabilities == []

    output = tmp_path / "performance_report_H100_PCIe.yaml"
    dump_ir(report, output)
    assert load_ir(PerformanceReport, output) == report


def test_unknown_operator_is_explicitly_unsupported() -> None:
    workload = WorkloadGraph(
        model_name="Unknown",
        source_file="Unknown.py",
        source_sha256="e" * 64,
        tensors={
            "input": TensorSpec("input", [16], "float32", TensorRole.INPUT),
            "output": TensorSpec("output", [16], "float32", TensorRole.OUTPUT),
        },
        nodes=[WorkloadNode("custom", "custom_op", ["input"], ["output"])],
        graph_inputs=["input"],
        graph_outputs=["output"],
    )
    operators, execution = OperatorNormalizer().normalize(workload)
    assert operators.nodes[0].kind == OperatorKind.OPAQUE_KERNEL

    cost_ir = CostAnalyzer().analyze(operators, execution)
    assert cost_ir.completeness == "partial"
    assert cost_ir.nodes[0].status == "unsupported"

    report = PerformancePredictor().predict(
        cost_ir,
        execution,
        hardware="H100_PCIe",
    )

    assert report.performance.theoretical_lower_bound_us is None
    assert report.performance.completeness == "partial"
    assert report.support.status == SupportStatus.UNSUPPORTED
    assert report.support.missing_capabilities


def test_partial_capability_cannot_publish_a_graph_lower_bound() -> None:
    cost_ir, execution_graph = _matmul_cost()
    cost_ir.totals.sfu_ops = 1

    report = PerformancePredictor().predict(
        cost_ir,
        execution_graph,
        hardware="H100_PCIe",
    )

    assert report.performance.theoretical_lower_bound_us is None
    assert report.performance.completeness == "partial"
    assert report.support.status == SupportStatus.UNSUPPORTED
    assert "sfu_ops_per_second" in report.support.missing_capabilities


def test_supported_status_requires_a_complete_lower_bound() -> None:
    report = PerformanceReport(
        model_name="Demo",
        hardware="P800",
        performance=PerformanceSection(
            theoretical_lower_bound_us=None,
            components=LowerBoundComponents(None, None, None, None, None),
            completeness="partial",
        ),
        diagnosis=DiagnosisSection(None, None),
        support=SupportSection(
            status=SupportStatus.SUPPORTED,
            confidence="low",
            missing_capabilities=["physical_dram_bandwidth"],
            assumptions=[],
            evidence_ids=[],
        ),
    )

    with pytest.raises(SchemaValidationError, match="complete theoretical lower bound"):
        report.validate()
