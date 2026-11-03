#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOLAR_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MODEL_FILE="${SCRIPT_DIR}/Matmul.py"
OUTPUT_DIR="${SOLAR_MATMUL_FOUR_STAGE_OUTPUT_DIR:-${SCRIPT_DIR}/output_four_stage}"

cd "${SOLAR_ROOT}"

python3 -m solar.cli.estimate_model \
  --model-file "${MODEL_FILE}" \
  --output-dir "${OUTPUT_DIR}" \
  --hardware H100_PCIe \
  --force-rerun

echo "Four-stage manifest: ${OUTPUT_DIR}/manifest.yaml"
echo "Performance report: ${OUTPUT_DIR}/stage4/performance_report_H100_PCIe.yaml"
