# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage 4 multi-resource lower-bound prediction."""

from typing import Dict, List, Optional, Tuple

from solar.ir.cost import CostIR
from solar.ir.operator import ExecutionGraph
from solar.ir.report import (
    DiagnosisSection,
    LowerBoundComponents,
    PerformanceReport,
    PerformanceSection,
    SupportSection,
    SupportStatus,
)
from solar.perf.capability import HardwareCapability, HardwareCapabilityLoader


class PerformancePredictor:
    """Map Cost IR to target hardware capability without running the workload."""

    def __init__(self, loader: Optional[HardwareCapabilityLoader] = None) -> None:
        self._loader = loader or HardwareCapabilityLoader()

    def predict(
        self,
        cost_ir: CostIR,
        execution_graph: ExecutionGraph,
        *,
        hardware: str,
    ) -> PerformanceReport:
        cost_ir.validate()
        execution_graph.validate()
        if cost_ir.model_name != execution_graph.model_name:
            raise ValueError("Cost IR and Execution Graph model names do not match")
        capability = self._loader.load(hardware)
        missing: List[str] = []

        compute_us, compute_resource = self._compute_time(cost_ir, capability, missing)
        memory_us = self._memory_time(cost_ir, capability, missing)
        sfu_us = self._unsupported_resource_time(
            cost_ir.totals.sfu_ops, "sfu_ops_per_second", missing
        )
        reduction_us = self._unsupported_resource_time(
            cost_ir.totals.reduction_ops, "reduction_ops_per_second", missing
        )
        communication_us = self._unsupported_resource_time(
            cost_ir.totals.communication_bytes, "communication_bytes_per_second", missing
        )
        components = {
            "compute": compute_us,
            "memory": memory_us,
            "sfu": sfu_us,
            "reduction": reduction_us,
            "communication": communication_us,
        }
        known = {name: value for name, value in components.items() if value is not None}
        candidate_lower_bound = max(known.values()) if known else None
        bottleneck = max(known, key=known.get) if known else None  # type: ignore[arg-type]
        complete = cost_ir.completeness == "complete" and not missing
        lower_bound = candidate_lower_bound if complete else None
        dram = capability.memory_resources.get("dram")
        execution_path = (
            f"{compute_resource}+{dram.resource_id}"
            if compute_resource is not None and dram is not None
            else None
        )
        report = PerformanceReport(
            model_name=cost_ir.model_name,
            hardware=capability.name,
            performance=PerformanceSection(
                theoretical_lower_bound_us=lower_bound,
                components=LowerBoundComponents(
                    compute_lower_bound_us=compute_us,
                    memory_lower_bound_us=memory_us,
                    sfu_lower_bound_us=sfu_us,
                    reduction_lower_bound_us=reduction_us,
                    communication_lower_bound_us=communication_us,
                ),
                completeness="complete" if complete else "partial",
            ),
            diagnosis=DiagnosisSection(
                bottleneck=bottleneck,
                execution_path=execution_path,
            ),
            support=SupportSection(
                status=(
                    SupportStatus.SUPPORTED
                    if complete
                    else SupportStatus.UNSUPPORTED
                ),
                confidence="high" if complete else "low",
                missing_capabilities=missing,
                assumptions=["perfect_resource_overlap", "dram_only_memory_path"],
                evidence_ids=[capability.source],
            ),
            metadata={"backend": "nvidia_roofline"},
        )
        report.validate()
        return report

    @staticmethod
    def _compute_time(
        cost_ir: CostIR,
        capability: HardwareCapability,
        missing: List[str],
    ) -> Tuple[Optional[float], Optional[str]]:
        macs = cost_ir.totals.macs
        if macs is None:
            missing.append("macs")
            return None, None
        if macs == 0:
            return 0.0, None
        dtypes = {node.dtype for node in cost_ir.nodes if node.cost.macs}
        if len(dtypes) != 1:
            missing.append("single_compute_dtype")
            return None, None
        dtype = next(iter(dtypes))
        for resource_id in ("tensor_core", "cuda_core"):
            resource = capability.compute_resources.get(resource_id)
            if resource is not None and dtype in resource.macs_per_cycle:
                rate = resource.macs_per_cycle[dtype] * capability.frequency_hz
                return macs / rate * 1e6, resource_id
        missing.append(f"macs_per_cycle.{dtype}")
        return None, None

    @staticmethod
    def _memory_time(
        cost_ir: CostIR,
        capability: HardwareCapability,
        missing: List[str],
    ) -> Optional[float]:
        read_bytes = cost_ir.totals.logical_read_bytes
        write_bytes = cost_ir.totals.logical_write_bytes
        if read_bytes is None or write_bytes is None:
            missing.append("logical_memory_bytes")
            return None
        dram = capability.memory_resources.get("dram")
        if dram is None or dram.bytes_per_cycle <= 0:
            missing.append("dram_bytes_per_cycle")
            return None
        return (read_bytes + write_bytes) / (
            dram.bytes_per_cycle * capability.frequency_hz
        ) * 1e6

    @staticmethod
    def _unsupported_resource_time(
        amount: Optional[int],
        capability_name: str,
        missing: List[str],
    ) -> Optional[float]:
        if amount is None:
            missing.append(capability_name.replace("_per_second", ""))
            return None
        if amount == 0:
            return 0.0
        missing.append(capability_name)
        return None


__all__ = ["PerformancePredictor"]
