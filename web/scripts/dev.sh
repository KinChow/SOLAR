#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEB_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${WEB_DIR}/../../../../" && pwd)"
PYVENV_ROOT="${SOLAR_PYVENV_ROOT:-${WORKSPACE_ROOT}/pyvenv}"
BACKEND_ENV="${UV_PROJECT_ENVIRONMENT:-${PYVENV_ROOT}/solar-web}"

if [[ ! -x "${BACKEND_ENV}/bin/python" ]]; then
    echo "Backend environment not found. Run: bash ${SCRIPT_DIR}/setup.sh" >&2
    exit 1
fi

cleanup() {
    kill "${BACKEND_PID:-}" "${FRONTEND_PID:-}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

cd "${WEB_DIR}/backend"
"${BACKEND_ENV}/bin/python" -m uvicorn solar_web.main:app --host 127.0.0.1 --port "${SOLAR_WEB_API_PORT:-8000}" &
BACKEND_PID=$!

cd "${WEB_DIR}/frontend"
npm run dev -- --port "${SOLAR_WEB_PORT:-5173}" &
FRONTEND_PID=$!

echo "SOLAR Web: http://127.0.0.1:${SOLAR_WEB_PORT:-5173}"
wait "${BACKEND_PID}" "${FRONTEND_PID}"
