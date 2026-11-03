# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage 1 CLI: PyTorch model to WorkloadGraph."""

import argparse
from pathlib import Path
from typing import Optional, Sequence

from solar.graph import WorkloadExtractor


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Extract a versioned workload graph.")
    parser.add_argument("--model-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case-id", default="default")
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--save-graph", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)

    WorkloadExtractor(debug=args.debug).extract(
        args.model_file,
        args.output_dir,
        case_id=args.case_id,
        force_rerun=args.force_rerun,
        save_graph=args.save_graph,
    )
    print(args.output_dir / "workload_graph.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
