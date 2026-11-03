#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEB_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${WEB_DIR}/../../../../" && pwd)"
PYVENV_ROOT="${SOLAR_PYVENV_ROOT:-${WORKSPACE_ROOT}/pyvenv}"
UV_BIN="${UV_BIN:-${PYVENV_ROOT}/uv/bin/uv}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-${PYVENV_ROOT}/solar-web}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${PYVENV_ROOT}/uv-cache}"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

if [[ ! -x "${UV_BIN}" ]]; then
    echo "uv was not found at ${UV_BIN}. Set UV_BIN or install the repository uv environment first." >&2
    exit 1
fi
if ! command -v npm >/dev/null 2>&1; then
    echo "npm is required to install the Web frontend." >&2
    exit 1
fi

echo "Syncing backend into ${UV_PROJECT_ENVIRONMENT}"
"${UV_BIN}" sync --frozen --project "${WEB_DIR}/backend" --python 3.10.20

echo "Installing locked frontend dependencies"
npm ci --prefix "${WEB_DIR}/frontend"

echo "Installing Playwright Chromium"
(cd "${WEB_DIR}/frontend" && npx playwright install chromium)

echo "Backend Python: $("${UV_PROJECT_ENVIRONMENT}/bin/python" --version)"
echo "Backend environment: ${UV_PROJECT_ENVIRONMENT}"
echo "Frontend Node: $(node --version)"
echo "Setup complete"
