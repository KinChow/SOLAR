# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage 4 performance report schema."""

from dataclasses import asdict, dataclass, field
from enum import Enum
import math
from typing import Any, Dict, List, Optional

from solar.ir.schema import SCHEMA_VERSION, SchemaValidationError, _plain


class SupportStatus(str, Enum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    AI_SUGGESTED = "ai_suggested"


@dataclass
class LowerBoundComponents:
    compute_lower_bound_us: Optional[float]
    memory_lower_bound_us: Optional[float]
    sfu_lower_bound_us: Optional[float]
    reduction_lower_bound_us: Optional[float]
    communication_lower_bound_us: Optional[float]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LowerBoundComponents":
        def optional_float(name: str) -> Optional[float]:
            value = data.get(name)
            return float(value) if value is not None else None

        return cls(**{name: optional_float(name) for name in cls.__annotations__})


@dataclass
class PerformanceSection:
    theoretical_lower_bound_us: Optional[float]
    components: LowerBoundComponents
    completeness: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PerformanceSection":
        lower_bound = data.get("theoretical_lower_bound_us")
        return cls(
            theoretical_lower_bound_us=(float(lower_bound) if lower_bound is not None else None),
            components=LowerBoundComponents.from_dict(data.get("components") or {}),
            completeness=str(data["completeness"]),
        )


@dataclass
class DiagnosisSection:
    bottleneck: Optional[str]
    execution_path: Optional[str]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DiagnosisSection":
        return cls(
            bottleneck=data.get("bottleneck"),
            execution_path=data.get("execution_path"),
        )


@dataclass
class SupportSection:
    status: SupportStatus
    confidence: str
    missing_capabilities: List[str]
    assumptions: List[str]
    evidence_ids: List[str]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SupportSection":
        return cls(
            status=SupportStatus(data["status"]),
            confidence=str(data["confidence"]),
            missing_capabilities=[str(value) for value in data.get("missing_capabilities", [])],
            assumptions=[str(value) for value in data.get("assumptions", [])],
            evidence_ids=[str(value) for value in data.get("evidence_ids", [])],
        )


@dataclass
class PerformanceReport:
    model_name: str
    hardware: str
    performance: PerformanceSection
    diagnosis: DiagnosisSection
    support: SupportSection
    metadata: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise SchemaValidationError(f"unsupported schema version '{self.schema_version}'")
        if self.performance.completeness not in {"complete", "partial"}:
            raise SchemaValidationError(
                f"invalid performance completeness '{self.performance.completeness}'"
            )
        for name, value in asdict(self.performance.components).items():
            if value is not None and (not math.isfinite(value) or value < 0):
                raise SchemaValidationError(
                    f"component '{name}' must be finite and non-negative"
                )
        lower_bound = self.performance.theoretical_lower_bound_us
        if lower_bound is not None and (
            not math.isfinite(lower_bound) or lower_bound < 0
        ):
            raise SchemaValidationError(
                "theoretical_lower_bound_us must be finite and non-negative"
            )
        if self.support.status == SupportStatus.SUPPORTED:
            if lower_bound is None or self.performance.completeness != "complete":
                raise SchemaValidationError(
                    "supported performance requires a complete theoretical lower bound"
                )
        elif lower_bound is not None:
            raise SchemaValidationError(
                "theoretical_lower_bound_us requires supported status"
            )

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return _plain(asdict(self))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PerformanceReport":
        report = cls(
            model_name=str(data["model_name"]),
            hardware=str(data["hardware"]),
            performance=PerformanceSection.from_dict(data.get("performance") or {}),
            diagnosis=DiagnosisSection.from_dict(data.get("diagnosis") or {}),
            support=SupportSection.from_dict(data.get("support") or {}),
            metadata=dict(data.get("metadata") or {}),
            schema_version=str(data.get("schema_version", "")),
        )
        report.validate()
        return report
