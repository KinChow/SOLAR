# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Orchestration for Solar's versioned four-stage offline pipeline."""

from dataclasses import dataclass
from pathlib import Path

import yaml

from solar.analysis import CostAnalyzer
from solar.graph import WorkloadExtractor
from solar.ir import (
    SCHEMA_VERSION,
    CostIR,
    ExecutionGraph,
    OperatorGraph,
    OperatorNormalizer,
    PerformanceReport,
    WorkloadGraph,
    dump_ir,
)
from solar.perf import PerformancePredictor
from solar.phase1 import Phase1Output, build_output


@dataclass(frozen=True)
class PipelineConfig:
    model_file: Path
    output_dir: Path
    hardware: str = "H100_PCIe"
    case_id: str = "default"
    force_rerun: bool = False
    save_graph: bool = False
    debug: bool = False


@dataclass(frozen=True)
class PipelineRunResult:
    workload_graph: WorkloadGraph
    operator_graph: OperatorGraph
    execution_graph: ExecutionGraph
    cost_ir: CostIR
    performance_report: PerformanceReport
    manifest_path: Path
    phase1_output: Phase1Output | None = None


class SolarPipeline:
    """Run all four deterministic stages and write their artifacts."""

    def run(self, config: PipelineConfig) -> PipelineRunResult:
        output_dir = Path(config.output_dir)
        stage1_dir = output_dir / "stage1"
        stage2_dir = output_dir / "stage2"
        stage3_dir = output_dir / "stage3"
        stage4_dir = output_dir / "stage4"

        workload = WorkloadExtractor(debug=config.debug).extract(
            Path(config.model_file),
            stage1_dir,
            case_id=config.case_id,
            force_rerun=config.force_rerun,
            save_graph=config.save_graph,
        )
        operators, execution = OperatorNormalizer().normalize(workload)
        operator_path = stage2_dir / "operator_ir.yaml"
        execution_path = stage2_dir / "execution_graph.yaml"
        dump_ir(operators, operator_path)
        dump_ir(execution, execution_path)

        cost_ir = CostAnalyzer().analyze(operators, execution)
        cost_path = stage3_dir / "cost_ir.yaml"
        dump_ir(cost_ir, cost_path)

        report = PerformancePredictor().predict(
            cost_ir,
            execution,
            hardware=config.hardware,
        )
        report_path = stage4_dir / f"performance_report_{report.hardware}.yaml"
        dump_ir(report, report_path)

        phase1_output = None
        phase1_path = None
        if config.hardware == "P800":
            phase1_output = build_output(
                operators,
                cost_ir,
                report,
                case_id=config.case_id,
            )
            phase1_path = output_dir / "phase1" / "phase1_output.yaml"
            dump_ir(phase1_output, phase1_path)

        artifacts = {
            "pytorch_graph": stage1_dir / "pytorch_graph.yaml",
            "workload_graph": stage1_dir / "workload_graph.yaml",
            "operator_ir": operator_path,
            "execution_graph": execution_path,
            "cost_ir": cost_path,
            "performance_report": report_path,
        }
        if phase1_path is not None:
            artifacts["phase1_output"] = phase1_path
        manifest_path = output_dir / "manifest.yaml"
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "phase1_contract_version": "phase1.v1",
            "phase1": {
                "case_id": config.case_id,
                "target_hardware": config.hardware,
                "status": (
                    phase1_output.level.value
                    if phase1_output is not None
                    else "not_applicable"
                ),
                "support_levels": operators.metadata.get("phase1_support_levels", {}),
                "issues": operators.metadata.get("phase1_issues", []),
                "artifact": (
                    str(phase1_path.relative_to(output_dir))
                    if phase1_path is not None
                    else None
                ),
            },
            "source": {
                "file": workload.source_file,
                "sha256": workload.source_sha256,
            },
            "hardware": report.hardware,
            "artifacts": {
                name: str(path.relative_to(output_dir)) for name, path in artifacts.items()
            },
        }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w") as stream:
            yaml.safe_dump(manifest, stream, sort_keys=False)

        return PipelineRunResult(
            workload_graph=workload,
            operator_graph=operators,
            execution_graph=execution,
            cost_ir=cost_ir,
            performance_report=report,
            manifest_path=manifest_path,
            phase1_output=phase1_output,
        )


__all__ = ["PipelineConfig", "PipelineRunResult", "SolarPipeline"]
