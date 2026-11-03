from pathlib import Path

import pytest

from solar.ir import (
    SchemaValidationError,
    TensorRole,
    TensorSpec,
    WorkloadGraph,
    WorkloadNode,
    dump_ir,
    load_ir,
)
from solar.phase1 import (
    Phase1CaseInput,
    Phase1CaseOutput,
    Phase1Input,
    Phase1NodeResult,
    Phase1Output,
    Phase1SupportLevel,
    build_output,
)
from solar.analysis import CostAnalyzer
from solar.ir import OperatorNormalizer, SupportStatus
from solar.perf import PerformancePredictor


def _workload() -> WorkloadGraph:
    return WorkloadGraph(
        model_name="Demo",
        source_file="demo.py",
        source_sha256="a" * 64,
        tensors={
            "x": TensorSpec("x", [4], "float32", TensorRole.INPUT),
            "y": TensorSpec("y", [4], "float32", TensorRole.OUTPUT),
        },
        nodes=[WorkloadNode("add", "add", ["x"], ["y"])],
        graph_inputs=["x"],
        graph_outputs=["y"],
    )


def test_phase1_input_round_trip_and_case_isolation(tmp_path: Path) -> None:
    value = Phase1Input(
        model_name="Demo",
        cases=[Phase1CaseInput("short", _workload(), guards={"n": 4})],
        api_manifest_version="V3.3.1.0",
    )
    path = tmp_path / "phase1_input.yaml"
    dump_ir(value, path)
    restored = load_ir(Phase1Input, path)
    assert restored == value
    assert restored.cases[0].guards == {"n": 4}


def test_phase1_input_rejects_duplicate_cases_and_non_p800() -> None:
    workload = _workload()
    value = Phase1Input("Demo", [Phase1CaseInput("x", workload), Phase1CaseInput("x", workload)])
    with pytest.raises(SchemaValidationError, match="duplicate case_id"):
        value.validate()
    value = Phase1Input("Demo", [Phase1CaseInput("x", workload)], hardware="H20")
    with pytest.raises(SchemaValidationError, match="hardware"):
        value.validate()


def test_phase1_output_requires_reason_and_consistent_weakest_level() -> None:
    unsupported = Phase1CaseOutput(
        "x",
        Phase1SupportLevel.UNSUPPORTED,
        [Phase1NodeResult("n", "torch.custom", None, Phase1SupportLevel.UNSUPPORTED, "opaque")],
    )
    supported = Phase1CaseOutput(
        "y",
        Phase1SupportLevel.LB_SUPPORTED,
        [Phase1NodeResult("n", "torch.add", "elementwise", Phase1SupportLevel.LB_SUPPORTED, theoretical_lower_bound_us=1.0)],
        theoretical_lower_bound_us=1.0,
        assumptions=["resident inputs"],
    )
    output = Phase1Output("Demo", "P800", [unsupported, supported], Phase1SupportLevel.UNSUPPORTED)
    output.validate()
    output.level = Phase1SupportLevel.LB_SUPPORTED
    with pytest.raises(SchemaValidationError, match="weakest case"):
        output.validate()


def test_phase1_lower_bound_must_be_finite_and_supported() -> None:
    result = Phase1NodeResult("n", "torch.add", "elementwise", Phase1SupportLevel.NORMALIZED, theoretical_lower_bound_us=1.0)
    with pytest.raises(SchemaValidationError, match="requires lb_supported"):
        result.validate()
    result = Phase1NodeResult("n", "torch.add", "elementwise", Phase1SupportLevel.LB_SUPPORTED, theoretical_lower_bound_us=float("inf"))
    with pytest.raises(SchemaValidationError, match="finite"):
        result.validate()


def test_phase1_output_contract_does_not_emit_runtime_estimate() -> None:
    case = Phase1CaseOutput(
        "x",
        Phase1SupportLevel.LB_SUPPORTED,
        [
            Phase1NodeResult(
                "n",
                "torch.add",
                "elementwise",
                Phase1SupportLevel.LB_SUPPORTED,
            )
        ],
        theoretical_lower_bound_us=1.0,
    )
    output = Phase1Output("Demo", "P800", [case], Phase1SupportLevel.LB_SUPPORTED)

    assert "estimated_runtime_us" not in output.to_dict()["cases"][0]


def test_unknown_operator_stays_explicitly_unsupported_in_phase1() -> None:
    workload = _workload()
    operators, execution = OperatorNormalizer().normalize(workload)
    costs = CostAnalyzer().analyze(operators, execution)
    report = PerformancePredictor().predict(costs, execution, hardware="H100_PCIe")

    assert report.support.status == SupportStatus.UNSUPPORTED
    with pytest.raises(ValueError, match="requires a P800 performance report"):
        build_output(operators, costs, report, case_id="default")
    report.hardware = "P800"
    output = build_output(operators, costs, report, case_id="default")

    assert output.hardware == "P800"
    assert output.level == Phase1SupportLevel.UNSUPPORTED
    assert output.cases[0].theoretical_lower_bound_us is None
    assert output.cases[0].nodes[0].reason == "all"
