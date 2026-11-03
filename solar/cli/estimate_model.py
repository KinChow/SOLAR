# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unified CLI for Solar's four-stage offline pipeline."""

import argparse
from pathlib import Path
from typing import Optional, Sequence

from solar.pipeline import PipelineConfig, SolarPipeline


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the four-stage offline estimator.")
    parser.add_argument("--model-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hardware", default="H100_PCIe")
    parser.add_argument("--case-id", default="default")
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--save-graph", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)

    result = SolarPipeline().run(
        PipelineConfig(
            model_file=args.model_file,
            output_dir=args.output_dir,
            hardware=args.hardware,
            case_id=args.case_id,
            force_rerun=args.force_rerun,
            save_graph=args.save_graph,
            debug=args.debug,
        )
    )
    print(result.manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
