from pathlib import Path

import yaml

from solar_web.domain.models import StageId
from solar_web.services.artifact_reader import ArtifactReader


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False))


def test_reads_stage_one_graph_and_tensor_table(tmp_path: Path) -> None:
    _write(
        tmp_path / "stage1" / "workload_graph.yaml",
        {
            "model_name": "Matmul",
            "tensors": {
                "x": {"tensor_id": "x", "shape": [64, 128], "dtype": "float32", "role": "input"},
                "y": {"tensor_id": "y", "shape": [4, 32, 128], "dtype": "float32", "role": "output"},
            },
            "nodes": [{"node_id": "matmul", "op_type": "matmul", "input_tensors": ["x"], "output_tensors": ["y"]}],
        },
    )

    result = ArtifactReader(tmp_path).read_stage(StageId.WORKLOAD)

    assert result["summary"] == {"model_name": "Matmul", "operator_count": 1, "tensor_count": 2}
    assert len(result["graph"]["nodes"]) == 3
    assert result["tensors"][0]["dtype"] == "float32"
    assert "workload_graph" in result["raw_yaml"]


def test_reads_final_result_with_supported_lower_bound(tmp_path: Path) -> None:
    _write(
        tmp_path / "stage4" / "performance_report_H100_PCIe.yaml",
        {
            "hardware": "H100_PCIe",
            "performance": {
                "theoretical_lower_bound_us": 0.06428879733176378,
                "completeness": "complete",
                "components": {"compute_lower_bound_us": 0.02, "memory_lower_bound_us": 0.064},
            },
            "diagnosis": {"bottleneck": "memory", "execution_path": "cuda_core+dram"},
            "support": {"status": "supported", "confidence": "high", "assumptions": [], "missing_capabilities": []},
        },
    )

    final = ArtifactReader(tmp_path).read_final_result()

    assert final["theoretical_lower_bound_us"] == 0.06428879733176378
    assert final["support_status"] == "supported"
    assert final["bottleneck"] == "memory"


def test_reads_stage_two_semantics_and_opaque_warning(tmp_path: Path) -> None:
    _write(
        tmp_path / "stage2" / "operator_ir.yaml",
        {
            "model_name": "Mixed",
            "schema_version": "1.0",
            "tensors": {
                "input": {"tensor_id": "input", "shape": [32, 128], "dtype": "float32", "role": "input"},
                "Model.hidden-tensor": {"tensor_id": "Model.hidden-tensor", "shape": [32, 256], "dtype": "float32", "role": "intermediate"},
                "output": {"tensor_id": "output", "shape": [32, 256], "dtype": "float32", "role": "output"},
            },
            "nodes": [
                {
                    "node_id": "matmul",
                    "kind": "tensor_contraction",
                    "semantic_op": "matmul",
                    "input_tensors": ["input"],
                    "output_tensors": ["Model.hidden-tensor"],
                    "phases": [{"phase_id": "matmul.contraction", "operation": "multiply_accumulate", "kind": "contraction"}],
                    "contraction": {"equation": "MK,KN->MN", "operands": {}, "dimensions": {}},
                },
                {
                    "node_id": "custom",
                    "kind": "opaque_kernel",
                    "semantic_op": "custom_op",
                    "input_tensors": ["Model.hidden-tensor"],
                    "output_tensors": ["output"],
                    "phases": [{"phase_id": "custom.unknown", "operation": "unknown", "kind": "unknown"}],
                },
            ],
        },
    )
    _write(
        tmp_path / "stage2" / "execution_graph.yaml",
        {
            "groups": [
                {"group_id": "g0", "operator_ids": ["matmul"], "implementation": "unknown"},
                {"group_id": "g1", "operator_ids": ["custom"], "implementation": "unknown"},
            ]
        },
    )

    result = ArtifactReader(tmp_path).read_stage(StageId.OPERATOR_IR)

    assert result["summary"]["operator_count"] == 2
    assert result["operators"][0]["equation"] == "MK,KN->MN"
    assert result["warnings"] == ["custom: no registered semantic normalizer"]
    assert result["execution_groups"][0]["operator_ids"] == ["matmul"]
    assert [node["id"] for node in result["graph"]["nodes"]] == ["matmul", "custom"]
    assert result["graph"]["edges"] == [
        {
            "source": "matmul",
            "target": "custom",
            "kind": "data_dependency",
            "tensor_ids": ["Model.hidden-tensor"],
            "label": "hidden-tensor [32x256]",
        }
    ]


def test_reads_stage_three_cost_totals(tmp_path: Path) -> None:
    _write(
        tmp_path / "stage3" / "cost_ir.yaml",
        {
            "completeness": "complete",
            "totals": {"macs": 1_048_576, "flops": 2_097_152, "logical_read_bytes": 65_536, "logical_write_bytes": 65_536},
            "nodes": [
                {"node_id": "input", "operator_kind": "data_movement", "status": "supported", "cost": {"logical_read_bytes": 65_536}},
                {"node_id": "matmul", "operator_kind": "tensor_contraction", "status": "supported", "cost": {"macs": 1_048_576}},
            ],
            "dependencies": {"input": [], "matmul": ["input"]},
        },
    )

    result = ArtifactReader(tmp_path).read_stage(StageId.COST)

    assert result["summary"]["completeness"] == "complete"
    assert result["totals"]["flops"] == 2_097_152
    assert result["totals"]["logical_bytes"] == 131_072
    assert result["graph"]["edges"] == [{"source": "input", "target": "matmul"}]


def test_reads_stage_four_performance_components(tmp_path: Path) -> None:
    _write(
        tmp_path / "stage4" / "performance_report_H100_PCIe.yaml",
        {
            "hardware": "H100_PCIe",
            "performance": {"theoretical_lower_bound_us": 0.064, "completeness": "complete", "components": {"compute_lower_bound_us": 0.02, "memory_lower_bound_us": 0.064}},
            "diagnosis": {"bottleneck": "memory", "execution_path": "cuda_core+dram"},
            "support": {"status": "supported", "confidence": "high", "assumptions": ["dram_only_memory_path"], "missing_capabilities": []},
        },
    )

    result = ArtifactReader(tmp_path).read_stage(StageId.PERFORMANCE)

    assert result["summary"]["hardware"] == "H100_PCIe"
    assert result["components"]["memory_lower_bound_us"] == 0.064
    assert result["support"]["status"] == "supported"
