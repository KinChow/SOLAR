# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage 2 CLI: WorkloadGraph to Operator IR and Execution Graph."""

import argparse
from pathlib import Path
from typing import Optional, Sequence

from solar.ir import OperatorNormalizer, WorkloadGraph, dump_ir, load_ir


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Normalize workload operator semantics.")
    parser.add_argument("--workload-graph", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    workload = load_ir(WorkloadGraph, args.workload_graph)
    operators, execution = OperatorNormalizer().normalize(workload)
    dump_ir(operators, args.output_dir / "operator_ir.yaml")
    dump_ir(execution, args.output_dir / "execution_graph.yaml")
    print(args.output_dir / "operator_ir.yaml")
    print(args.output_dir / "execution_graph.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
