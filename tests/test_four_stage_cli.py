# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from solar.cli.analyze_cost import main as analyze_main
from solar.cli.estimate_model import main as estimate_main
from solar.cli.extract_workload import main as extract_main
from solar.cli.normalize_operators import main as normalize_main
from solar.cli.predict_performance import main as predict_main
from solar.ir import PerformanceReport, load_ir


ROOT = Path(__file__).resolve().parents[1]
MATMUL_MODEL = ROOT / "examples" / "Matmul" / "Matmul.py"


def test_independent_clis_match_unified_cli(tmp_path: Path) -> None:
    independent = tmp_path / "independent"
    unified = tmp_path / "unified"

    assert extract_main(
        [
            "--model-file",
            str(MATMUL_MODEL),
            "--output-dir",
            str(independent / "stage1"),
            "--force-rerun",
        ]
    ) == 0
    assert normalize_main(
        [
            "--workload-graph",
            str(independent / "stage1" / "workload_graph.yaml"),
            "--output-dir",
            str(independent / "stage2"),
        ]
    ) == 0
    assert analyze_main(
        [
            "--operator-ir",
            str(independent / "stage2" / "operator_ir.yaml"),
            "--execution-graph",
            str(independent / "stage2" / "execution_graph.yaml"),
            "--output-dir",
            str(independent / "stage3"),
        ]
    ) == 0
    assert predict_main(
        [
            "--cost-ir",
            str(independent / "stage3" / "cost_ir.yaml"),
            "--execution-graph",
            str(independent / "stage2" / "execution_graph.yaml"),
            "--output-dir",
            str(independent / "stage4"),
            "--hardware",
            "H100_PCIe",
        ]
    ) == 0
    assert estimate_main(
        [
            "--model-file",
            str(MATMUL_MODEL),
            "--output-dir",
            str(unified),
            "--hardware",
            "H100_PCIe",
            "--force-rerun",
        ]
    ) == 0

    independent_report = load_ir(
        PerformanceReport,
        independent / "stage4" / "performance_report_H100_PCIe.yaml",
    )
    unified_report = load_ir(
        PerformanceReport,
        unified / "stage4" / "performance_report_H100_PCIe.yaml",
    )
    assert independent_report == unified_report
