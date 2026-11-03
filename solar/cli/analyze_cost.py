# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage 3 CLI: Operator IR to hardware-independent Cost IR."""

import argparse
from pathlib import Path
from typing import Optional, Sequence

from solar.analysis import CostAnalyzer
from solar.ir import ExecutionGraph, OperatorGraph, dump_ir, load_ir


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze hardware-independent costs.")
    parser.add_argument("--operator-ir", type=Path, required=True)
    parser.add_argument("--execution-graph", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    operators = load_ir(OperatorGraph, args.operator_ir)
    execution = load_ir(ExecutionGraph, args.execution_graph)
    cost_ir = CostAnalyzer().analyze(operators, execution)
    dump_ir(cost_ir, args.output_dir / "cost_ir.yaml")
    print(args.output_dir / "cost_ir.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
