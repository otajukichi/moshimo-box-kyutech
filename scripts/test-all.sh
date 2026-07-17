#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"
ENV_DIR="${APP_ENV_DIR}"
PYTHON_BIN="${APP_PYTHON}"
NPM_BIN="${APP_NPM}"

if [[ ! -x "${PYTHON_BIN}" || ! -x "${NPM_BIN}" ]]; then
  echo "[moshimo-box] App environment is missing. Run scripts/setup-app-env.sh first." >&2
  exit 1
fi

cd "${ROOT_DIR}"
"${PYTHON_BIN}" -m pytest -q

PATH="${ENV_DIR}/bin:${PATH}" "${NPM_BIN}" run build --prefix frontend

if [[ "${1:-}" == "--e2e" ]]; then
  PATH="${ENV_DIR}/bin:${PATH}" "${NPM_BIN}" run test:e2e --prefix frontend
fi
