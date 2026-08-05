#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Configure the locked local environment while preserving the generic
# install.sh and install_uv.sh interfaces.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${PROJECT_ROOT}/../../.." && pwd)"
SOLAR_PYVENV_ROOT="${SOLAR_PYVENV_ROOT:-${WORKSPACE_ROOT}/pyvenv}"
UV_BIN="${UV_BIN:-${SOLAR_PYVENV_ROOT}/uv/bin/uv}"
UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-${SOLAR_PYVENV_ROOT}/solar}"
UV_CACHE_DIR="${UV_CACHE_DIR:-${SOLAR_PYVENV_ROOT}/uv-cache}"

if [[ ! -x "${UV_BIN}" ]]; then
    echo "uv not found at ${UV_BIN}; set UV_BIN to the uv executable." >&2
    exit 1
fi

export UV_PROJECT_ENVIRONMENT UV_CACHE_DIR
export PATH="$(dirname "${UV_BIN}"):${PATH}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

bash "${PROJECT_ROOT}/install_uv.sh" "$@"

PYTHON_BIN="${UV_PROJECT_ENVIRONMENT}/bin/python"
"${UV_BIN}" pip check --python "${PYTHON_BIN}"

TORCHVIEW_DIR="${PROJECT_ROOT}/third_party/torchview"
TORCHVIEW_PATCH="${PROJECT_ROOT}/patches/torchview-parameter-tensors.patch"
EXPECTED_PATCH_ID="$(git patch-id --stable < "${TORCHVIEW_PATCH}" | awk '{print $1}')"
ACTUAL_PATCH_ID="$(git -C "${TORCHVIEW_DIR}" diff --no-ext-diff | git patch-id --stable | awk '{print $1}')"
UNTRACKED_FILES="$(git -C "${TORCHVIEW_DIR}" ls-files --others --exclude-standard)"
if [[ "${ACTUAL_PATCH_ID}" != "${EXPECTED_PATCH_ID}" ]] || [[ -n "${UNTRACKED_FILES}" ]]; then
    echo "Patched torchview contains unexpected changes." >&2
    exit 1
fi

if ! command -v dot >/dev/null 2>&1; then
    echo "Graphviz 'dot' is required for PDF output." >&2
    exit 1
fi

"${PYTHON_BIN}" - <<'PY'
import importlib.metadata as metadata
import subprocess
import sys

import torch
import torchview
from solar.graph.pytorch_processor import _check_torchview_parameter_support

_check_torchview_parameter_support()
print(f"python: {sys.version.split()[0]}")
print(f"torch: {torch.__version__}")
print(f"torchview: {metadata.version('torchview')} ({torchview.__file__})")
print(f"dot: {subprocess.check_output(['dot', '-V'], stderr=subprocess.STDOUT, text=True).strip()}")
print("torchview parameter-tensor probe: PASS")
PY

echo "environment: ${UV_PROJECT_ENVIRONMENT}"
echo "torchview commit: $(git -C "${PROJECT_ROOT}/third_party/torchview" rev-parse HEAD)"
