# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage 4 CLI: Cost IR and hardware capability to Performance Report."""

import argparse
from pathlib import Path
from typing import Optional, Sequence

from solar.ir import CostIR, ExecutionGraph, dump_ir, load_ir
from solar.perf import PerformancePredictor


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Predict a theoretical performance lower bound.")
    parser.add_argument("--cost-ir", type=Path, required=True)
    parser.add_argument("--execution-graph", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hardware", default="H100_PCIe")
    args = parser.parse_args(argv)

    cost_ir = load_ir(CostIR, args.cost_ir)
    execution = load_ir(ExecutionGraph, args.execution_graph)
    report = PerformancePredictor().predict(cost_ir, execution, hardware=args.hardware)
    output = args.output_dir / f"performance_report_{report.hardware}.yaml"
    dump_ir(report, output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
