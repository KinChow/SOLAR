# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Intermediate representations for Solar's four-stage pipeline."""

from solar.ir.io import dump_ir, load_ir
from solar.ir.cost import CostIR, NodeCost, ResourceCost
from solar.ir.dependencies import TensorDependencies, derive_tensor_dependencies
from solar.ir.normalizer import OperatorNormalizer
from solar.ir.operator import (
    ContractionSpec,
    ExecutionGraph,
    ExecutionGroup,
    OperatorGraph,
    OperatorKind,
    OperatorNode,
    PhaseSpec,
)
from solar.ir.report import (
    DiagnosisSection,
    LowerBoundComponents,
    PerformanceReport,
    PerformanceSection,
    SupportSection,
    SupportStatus,
)
from solar.ir.schema import (
    PHASE1_CONTRACT_VERSION,
    Phase1SupportLevel,
    SCHEMA_VERSION,
    SchemaValidationError,
    TensorRole,
    TensorSpec,
    WorkloadGraph,
    WorkloadNode,
)
__all__ = [
    "SCHEMA_VERSION",
    "PHASE1_CONTRACT_VERSION",
    "Phase1SupportLevel",
    "ContractionSpec",
    "CostIR",
    "ExecutionGraph",
    "ExecutionGroup",
    "OperatorGraph",
    "OperatorKind",
    "OperatorNode",
    "OperatorNormalizer",
    "NodeCost",
    "PhaseSpec",
    "DiagnosisSection",
    "LowerBoundComponents",
    "PerformanceReport",
    "PerformanceSection",
    "SupportSection",
    "SupportStatus",
    "ResourceCost",
    "SchemaValidationError",
    "TensorRole",
    "TensorSpec",
    "TensorDependencies",
    "WorkloadGraph",
    "WorkloadNode",
    "dump_ir",
    "derive_tensor_dependencies",
    "load_ir",
]
