# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import yaml

from solar.perf.perf_model import EinsumGraphPerfModel
from solar.pipeline import PipelineConfig, SolarPipeline


ROOT = Path(__file__).resolve().parents[1]
MATMUL_MODEL = ROOT / "examples" / "Matmul" / "Matmul.py"


def test_pipeline_writes_versioned_artifacts_and_manifest(tmp_path: Path) -> None:
    output_dir = tmp_path / "output_four_stage"

    result = SolarPipeline().run(
        PipelineConfig(
            model_file=MATMUL_MODEL,
            output_dir=output_dir,
            hardware="H100_PCIe",
            force_rerun=True,
        )
    )

    expected = {
        "stage1/pytorch_graph.yaml",
        "stage1/workload_graph.yaml",
        "stage2/operator_ir.yaml",
        "stage2/execution_graph.yaml",
        "stage3/cost_ir.yaml",
        "stage4/performance_report_H100_PCIe.yaml",
        "manifest.yaml",
    }
    assert {
        str(path.relative_to(output_dir))
        for path in output_dir.rglob("*.yaml")
    } == expected
    assert result.performance_report.performance.theoretical_lower_bound_us == pytest.approx(
        0.06428879733176378,
        rel=1e-12,
    )
    assert "estimated_runtime_us" not in result.performance_report.to_dict()["performance"]

    manifest = yaml.safe_load((output_dir / "manifest.yaml").read_text())
    assert manifest["schema_version"] == "1.0"
    assert manifest["source"]["file"] == str(MATMUL_MODEL.resolve())
    assert len(manifest["source"]["sha256"]) == 64
    assert manifest["hardware"] == "H100_PCIe"
    assert set(manifest["artifacts"]) == {
        "pytorch_graph",
        "workload_graph",
        "operator_ir",
        "execution_graph",
        "cost_ir",
        "performance_report",
    }
    assert all(not Path(path).is_absolute() for path in manifest["artifacts"].values())


def test_matmul_lower_bound_matches_legacy_fused_roofline(tmp_path: Path) -> None:
    result = SolarPipeline().run(
        PipelineConfig(
            model_file=MATMUL_MODEL,
            output_dir=tmp_path / "new",
            hardware="H100_PCIe",
            force_rerun=True,
        )
    )
    legacy_analysis = tmp_path / "analysis.yaml"
    legacy_analysis.write_text(
        yaml.safe_dump(
            {
                "total": {
                    "macs": 1_048_576,
                    "flops": 2_097_152,
                    "other_ops": 0,
                    "unfused_elements": 32_768,
                    "fused_elements": 32_768,
                    "fused_prefetched_elements": 32_768,
                    "model_io_elements": 32_768,
                    "intermediate_elements": 0,
                }
            }
        )
    )
    legacy = EinsumGraphPerfModel().predict(
        legacy_analysis,
        tmp_path / "legacy",
        arch_config="H100_PCIe",
        precision="fp32",
    )

    assert legacy is not None
    assert result.performance_report.performance.theoretical_lower_bound_us == pytest.approx(
        legacy["fused"]["runtime_ms"] * 1_000,
        rel=1e-12,
    )
