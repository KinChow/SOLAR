# SPDX-License-Identifier: Apache-2.0

"""Serializable Phase 1 input and output contracts."""

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Dict, List, Optional

from solar.ir.schema import (
    PHASE1_CONTRACT_VERSION,
    Phase1SupportLevel,
    SchemaValidationError,
    WorkloadGraph,
    _plain,
)


_LEVEL_RANK = {
    Phase1SupportLevel.UNSUPPORTED: -1,
    Phase1SupportLevel.RECOGNIZED: 0,
    Phase1SupportLevel.NORMALIZED: 1,
    Phase1SupportLevel.COST_COMPLETE: 2,
    Phase1SupportLevel.LB_SUPPORTED: 3,
}


def _level(value: Any) -> Phase1SupportLevel:
    try:
        return value if isinstance(value, Phase1SupportLevel) else Phase1SupportLevel(value)
    except (TypeError, ValueError) as error:
        raise SchemaValidationError(f"invalid phase1 support level '{value}'") from error


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchemaValidationError(f"{name} must be a non-empty string")
    return value


def _lower_bound(value: Optional[float], name: str) -> None:
    if value is not None and (
        not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value < 0
    ):
        raise SchemaValidationError(f"{name} must be finite and non-negative")


@dataclass
class Phase1CaseInput:
    """One concrete input path represented by an extracted workload graph."""

    case_id: str
    workload: WorkloadGraph
    guards: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _required_text(self.case_id, "case_id")
        if any(char.isspace() for char in self.case_id):
            raise SchemaValidationError("case_id must not contain whitespace")
        if not isinstance(self.workload, WorkloadGraph):
            raise SchemaValidationError("workload must be a WorkloadGraph")
        self.workload.validate()
        if not isinstance(self.guards, dict):
            raise SchemaValidationError("guards must be a mapping")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Phase1CaseInput":
        value = cls(
            case_id=str(data.get("case_id", "")),
            workload=WorkloadGraph.from_dict(data.get("workload") or {}),
            guards=dict(data.get("guards") or {}),
        )
        value.validate()
        return value


@dataclass
class Phase1Input:
    model_name: str
    cases: List[Phase1CaseInput]
    hardware: str = "P800"
    api_manifest_version: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    contract_version: str = PHASE1_CONTRACT_VERSION

    def validate(self) -> None:
        if self.contract_version != PHASE1_CONTRACT_VERSION:
            raise SchemaValidationError(
                f"unsupported phase1 contract version '{self.contract_version}'"
            )
        _required_text(self.model_name, "model_name")
        if self.hardware != "P800":
            raise SchemaValidationError("phase1 hardware must be 'P800'")
        if not self.cases:
            raise SchemaValidationError("phase1 input requires at least one case")
        seen = set()
        for case in self.cases:
            case.validate()
            if case.case_id in seen:
                raise SchemaValidationError(f"duplicate case_id '{case.case_id}'")
            seen.add(case.case_id)
            if case.workload.model_name != self.model_name:
                raise SchemaValidationError(
                    f"case '{case.case_id}' model_name does not match phase1 input"
                )

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return _plain(asdict(self))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Phase1Input":
        value = cls(
            model_name=str(data.get("model_name", "")),
            cases=[Phase1CaseInput.from_dict(item) for item in data.get("cases", [])],
            hardware=str(data.get("hardware", "")),
            api_manifest_version=str(data.get("api_manifest_version", "")),
            metadata=dict(data.get("metadata") or {}),
            contract_version=str(data.get("contract_version", "")),
        )
        value.validate()
        return value


@dataclass
class Phase1NodeResult:
    node_id: str
    api: str
    semantic_family: Optional[str]
    level: Phase1SupportLevel
    reason: Optional[str] = None
    source_location: Optional[str] = None
    theoretical_lower_bound_us: Optional[float] = None

    def validate(self) -> None:
        _required_text(self.node_id, "node_id")
        _required_text(self.api, "api")
        self.level = _level(self.level)
        if self.semantic_family is not None:
            _required_text(self.semantic_family, "semantic_family")
        if self.level == Phase1SupportLevel.UNSUPPORTED and not self.reason:
            raise SchemaValidationError("unsupported node result requires a reason")
        _lower_bound(self.theoretical_lower_bound_us, "theoretical_lower_bound_us")
        if self.theoretical_lower_bound_us is not None and (
            _LEVEL_RANK[self.level] < _LEVEL_RANK[Phase1SupportLevel.LB_SUPPORTED]
        ):
            raise SchemaValidationError(
                "theoretical_lower_bound_us requires lb_supported"
            )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Phase1NodeResult":
        value = cls(
            node_id=str(data.get("node_id", "")),
            api=str(data.get("api", "")),
            semantic_family=data.get("semantic_family"),
            level=_level(data.get("level")),
            reason=data.get("reason"),
            source_location=data.get("source_location"),
            theoretical_lower_bound_us=(
                float(data["theoretical_lower_bound_us"])
                if data.get("theoretical_lower_bound_us") is not None
                else None
            ),
        )
        value.validate()
        return value


@dataclass
class Phase1CaseOutput:
    case_id: str
    level: Phase1SupportLevel
    nodes: List[Phase1NodeResult]
    theoretical_lower_bound_us: Optional[float] = None
    failure_reason: Optional[str] = None
    assumptions: List[str] = field(default_factory=list)
    missing_capabilities: List[str] = field(default_factory=list)

    def validate(self) -> None:
        _required_text(self.case_id, "case_id")
        self.level = _level(self.level)
        _lower_bound(self.theoretical_lower_bound_us, "theoretical_lower_bound_us")
        if self.theoretical_lower_bound_us is not None and (
            _LEVEL_RANK[self.level] < _LEVEL_RANK[Phase1SupportLevel.LB_SUPPORTED]
        ):
            raise SchemaValidationError("case lower bound requires lb_supported")
        if self.level == Phase1SupportLevel.UNSUPPORTED and not (
            self.failure_reason
            or self.missing_capabilities
            or any(node.reason for node in self.nodes)
        ):
            raise SchemaValidationError(
                "unsupported case output requires a reason or missing capability"
            )
        seen = set()
        for node in self.nodes:
            node.validate()
            if node.node_id in seen:
                raise SchemaValidationError(f"duplicate node_id '{node.node_id}'")
            seen.add(node.node_id)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Phase1CaseOutput":
        value = cls(
            case_id=str(data.get("case_id", "")),
            level=_level(data.get("level")),
            nodes=[Phase1NodeResult.from_dict(item) for item in data.get("nodes", [])],
            theoretical_lower_bound_us=(
                float(data["theoretical_lower_bound_us"])
                if data.get("theoretical_lower_bound_us") is not None
                else None
            ),
            failure_reason=data.get("failure_reason"),
            assumptions=[str(value) for value in data.get("assumptions", [])],
            missing_capabilities=[
                str(value) for value in data.get("missing_capabilities", [])
            ],
        )
        value.validate()
        return value


@dataclass
class Phase1Output:
    model_name: str
    hardware: str
    cases: List[Phase1CaseOutput]
    level: Phase1SupportLevel
    metadata: Dict[str, Any] = field(default_factory=dict)
    contract_version: str = PHASE1_CONTRACT_VERSION

    def validate(self) -> None:
        if self.contract_version != PHASE1_CONTRACT_VERSION:
            raise SchemaValidationError(
                f"unsupported phase1 contract version '{self.contract_version}'"
            )
        _required_text(self.model_name, "model_name")
        if self.hardware != "P800":
            raise SchemaValidationError("phase1 hardware must be 'P800'")
        if not self.cases:
            raise SchemaValidationError("phase1 output requires at least one case")
        seen = set()
        for case in self.cases:
            case.validate()
            if case.case_id in seen:
                raise SchemaValidationError(f"duplicate case_id '{case.case_id}'")
            seen.add(case.case_id)
        self.level = _level(self.level)
        weakest = min(self.cases, key=lambda case: _LEVEL_RANK[case.level]).level
        if self.level != weakest:
            raise SchemaValidationError(
                f"overall level '{self.level.value}' does not match weakest case "
                f"'{weakest.value}'"
            )

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return _plain(asdict(self))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Phase1Output":
        value = cls(
            model_name=str(data.get("model_name", "")),
            hardware=str(data.get("hardware", "")),
            cases=[Phase1CaseOutput.from_dict(item) for item in data.get("cases", [])],
            level=_level(data.get("level")),
            metadata=dict(data.get("metadata") or {}),
            contract_version=str(data.get("contract_version", "")),
        )
        value.validate()
        return value


__all__ = [
    "Phase1CaseInput",
    "Phase1CaseOutput",
    "Phase1Input",
    "Phase1NodeResult",
    "Phase1Output",
    "Phase1SupportLevel",
]
