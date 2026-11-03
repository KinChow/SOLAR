"""Build the Phase 1 contract result from the deterministic IR stages."""

from typing import List

from solar.ir.cost import CostIR
from solar.ir.operator import OperatorGraph
from solar.ir.report import PerformanceReport, SupportStatus
from solar.phase1.contract import (
    Phase1CaseOutput,
    Phase1NodeResult,
    Phase1Output,
    Phase1SupportLevel,
)


def build_output(
    operators: OperatorGraph,
    costs: CostIR,
    report: PerformanceReport,
    *,
    case_id: str,
) -> Phase1Output:
    """Translate stage evidence into an explicit Phase 1 support decision."""

    if report.hardware != "P800":
        raise ValueError(
            f"phase1 output requires a P800 performance report, got '{report.hardware}'"
        )

    cost_by_id = {node.node_id: node for node in costs.nodes}
    nodes: List[Phase1NodeResult] = []
    for operator in operators.nodes:
        cost = cost_by_id.get(operator.node_id)
        if cost is None:
            level = Phase1SupportLevel.RECOGNIZED
            reason = "missing_cost_result"
        elif cost.status != "supported":
            level = Phase1SupportLevel.UNSUPPORTED
            reason = ",".join(cost.missing_costs) or "cost_rule_unsupported"
        elif report.performance.completeness == "complete":
            level = Phase1SupportLevel.LB_SUPPORTED
            reason = None
        else:
            level = Phase1SupportLevel.COST_COMPLETE
            reason = "graph_contains_unsupported_or_missing_capability"

        nodes.append(
            Phase1NodeResult(
                node_id=operator.node_id,
                api=operator.semantic_op,
                semantic_family=operator.kind.value,
                level=level,
                reason=reason,
            )
        )

    if report.support.status == SupportStatus.UNSUPPORTED:
        case_level = Phase1SupportLevel.UNSUPPORTED
    elif report.performance.theoretical_lower_bound_us is not None:
        case_level = Phase1SupportLevel.LB_SUPPORTED
    elif nodes:
        case_level = min(
            (node.level for node in nodes),
            key=lambda value: {
                Phase1SupportLevel.UNSUPPORTED: -1,
                Phase1SupportLevel.RECOGNIZED: 0,
                Phase1SupportLevel.NORMALIZED: 1,
                Phase1SupportLevel.COST_COMPLETE: 2,
                Phase1SupportLevel.LB_SUPPORTED: 3,
            }[value],
        )
    else:
        case_level = Phase1SupportLevel.UNSUPPORTED

    case = Phase1CaseOutput(
        case_id=case_id,
        level=case_level,
        nodes=nodes,
        theoretical_lower_bound_us=report.performance.theoretical_lower_bound_us,
        failure_reason=(
            "; ".join(report.support.missing_capabilities)
            if case_level == Phase1SupportLevel.UNSUPPORTED
            else None
        ),
        assumptions=list(report.support.assumptions),
        missing_capabilities=list(report.support.missing_capabilities),
    )
    result = Phase1Output(
        model_name=operators.model_name,
        hardware="P800",
        cases=[case],
        level=case_level,
        metadata={
            "source": "operator_ir+cost_ir+performance_report",
            "performance_support_status": report.support.status.value,
        },
    )
    result.validate()
    return result


__all__ = ["build_output"]
