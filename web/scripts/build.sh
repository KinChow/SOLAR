#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEB_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

npm run build --prefix "${WEB_DIR}/frontend"
echo "Frontend build: ${WEB_DIR}/frontend/dist"
