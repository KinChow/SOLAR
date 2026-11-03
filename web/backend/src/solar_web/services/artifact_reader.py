"""Translate four-stage YAML artifacts into UI-oriented view models."""

from pathlib import Path
from typing import Any, Dict

import yaml

from solar.ir import OperatorGraph, derive_tensor_dependencies
from solar_web.domain.models import StageId


class ArtifactReader:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)

    def read_stage(self, stage_id: StageId) -> Dict[str, Any]:
        readers = {
            StageId.WORKLOAD: self._read_workload,
            StageId.OPERATOR_IR: self._read_operator_ir,
            StageId.COST: self._read_cost,
            StageId.PERFORMANCE: self._read_performance,
        }
        return readers[stage_id]()

    def read_final_result(self) -> Dict[str, Any]:
        report, _ = self._load_performance_report()
        performance = report.get("performance") or {}
        diagnosis = report.get("diagnosis") or {}
        support = report.get("support") or {}
        return {
            "theoretical_lower_bound_us": performance.get("theoretical_lower_bound_us"),
            "bottleneck": diagnosis.get("bottleneck"),
            "support_status": support.get("status"),
            "confidence": support.get("confidence"),
        }

    def result_completeness(self, stage_id: StageId) -> str:
        if stage_id == StageId.OPERATOR_IR:
            operators, _ = self._load("stage2/operator_ir.yaml")
            return "partial" if any(node.get("kind") == "opaque_kernel" for node in operators.get("nodes") or []) else "complete"
        if stage_id == StageId.COST:
            cost, _ = self._load("stage3/cost_ir.yaml")
            return str(cost.get("completeness", "partial"))
        if stage_id == StageId.PERFORMANCE:
            report, _ = self._load_performance_report()
            return str((report.get("performance") or {}).get("completeness", "partial"))
        return "complete"

    def _read_workload(self) -> Dict[str, Any]:
        workload, raw = self._load("stage1/workload_graph.yaml")
        tensors = list((workload.get("tensors") or {}).values())
        operators = workload.get("nodes") or []
        graph_nodes = [
            {"id": tensor["tensor_id"], "label": tensor["tensor_id"], "kind": "tensor", "role": tensor.get("role")}
            for tensor in tensors
        ] + [
            {"id": node["node_id"], "label": node.get("op_type", node["node_id"]), "kind": "operator"}
            for node in operators
        ]
        edges = []
        for node in operators:
            edges.extend({"source": tensor_id, "target": node["node_id"]} for tensor_id in node.get("input_tensors") or [])
            edges.extend({"source": node["node_id"], "target": tensor_id} for tensor_id in node.get("output_tensors") or [])
        return {
            "summary": {"model_name": workload.get("model_name"), "operator_count": len(operators), "tensor_count": len(tensors)},
            "graph": {"nodes": graph_nodes, "edges": edges},
            "tensors": tensors,
            "operators": operators,
            "raw_yaml": {"workload_graph": raw},
        }

    def _read_operator_ir(self) -> Dict[str, Any]:
        operators, operator_raw = self._load("stage2/operator_ir.yaml")
        execution, execution_raw = self._load("stage2/execution_graph.yaml")
        operator_graph = OperatorGraph.from_dict(operators)
        nodes = operators.get("nodes") or []
        operator_rows = [
            {
                "node_id": node.get("node_id"),
                "kind": node.get("kind"),
                "semantic_op": node.get("semantic_op"),
                "equation": (node.get("contraction") or {}).get("equation"),
                "dimensions": (node.get("contraction") or {}).get("dimensions"),
                "phases": node.get("phases") or [],
            }
            for node in nodes
        ]
        warnings = [
            f"{node.get('node_id')}: no registered semantic normalizer"
            for node in nodes
            if node.get("kind") == "opaque_kernel"
        ]
        graph_nodes = [
            {
                "id": str(node.get("node_id")),
                "label": str(node.get("semantic_op") or node.get("node_id")),
                "kind": node.get("kind"),
            }
            for node in nodes
        ]
        dependencies = derive_tensor_dependencies(operator_graph)
        graph_edges = [
            {
                "source": source,
                "target": target,
                "kind": "data_dependency",
                "tensor_ids": tensor_ids,
                "label": self._tensor_edge_label(operator_graph, tensor_ids),
            }
            for target, sources in dependencies.items()
            for source, tensor_ids in sources.items()
        ]
        return {
            "summary": {"model_name": operators.get("model_name"), "operator_count": len(nodes), "execution_group_count": len(execution.get("groups") or [])},
            "operators": operator_rows,
            "execution_groups": execution.get("groups") or [],
            "graph": {"nodes": graph_nodes, "edges": graph_edges},
            "warnings": warnings,
            "raw_yaml": {"operator_ir": operator_raw, "execution_graph": execution_raw},
        }

    @staticmethod
    def _tensor_edge_label(graph: OperatorGraph, tensor_ids: list[str]) -> str:
        labels = []
        for tensor_id in tensor_ids:
            shape = "x".join(str(dim) for dim in graph.tensors[tensor_id].shape)
            display_id = tensor_id.rsplit(".", 1)[-1]
            labels.append(f"{display_id} [{shape}]" if shape else display_id)
        return ", ".join(labels)

    def _read_cost(self) -> Dict[str, Any]:
        cost, raw = self._load("stage3/cost_ir.yaml")
        totals = dict(cost.get("totals") or {})
        read_bytes = totals.get("logical_read_bytes")
        write_bytes = totals.get("logical_write_bytes")
        totals["logical_bytes"] = (read_bytes + write_bytes) if read_bytes is not None and write_bytes is not None else None
        nodes = cost.get("nodes") or []
        dependencies = cost.get("dependencies") or {}
        return {
            "summary": {"completeness": cost.get("completeness"), "node_count": len(nodes)},
            "totals": totals,
            "nodes": nodes,
            "dependencies": dependencies,
            "graph": {
                "nodes": [
                    {
                        "id": str(node.get("node_id")),
                        "label": str(node.get("node_id")),
                        "kind": node.get("operator_kind"),
                        "status": node.get("status"),
                    }
                    for node in nodes
                ],
                "edges": [
                    {"source": str(source), "target": str(target)}
                    for target, sources in dependencies.items()
                    for source in sources
                ],
            },
            "raw_yaml": {"cost_ir": raw},
        }

    def _read_performance(self) -> Dict[str, Any]:
        report, raw = self._load_performance_report()
        performance = report.get("performance") or {}
        return {
            "summary": {"hardware": report.get("hardware"), "completeness": performance.get("completeness"), "theoretical_lower_bound_us": performance.get("theoretical_lower_bound_us")},
            "components": performance.get("components") or {},
            "diagnosis": report.get("diagnosis") or {},
            "support": report.get("support") or {},
            "raw_yaml": {"performance_report": raw},
        }

    def _load_performance_report(self) -> tuple[Dict[str, Any], str]:
        matches = sorted((self.run_dir / "stage4").glob("performance_report_*.yaml"))
        if len(matches) != 1:
            raise FileNotFoundError("expected exactly one Stage 4 performance report")
        return self._load(str(matches[0].relative_to(self.run_dir)))

    def _load(self, relative_path: str) -> tuple[Dict[str, Any], str]:
        path = self.run_dir / relative_path
        raw = path.read_text()
        return yaml.safe_load(raw) or {}, raw
