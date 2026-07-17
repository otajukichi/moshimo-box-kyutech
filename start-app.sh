#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${ROOT_DIR}/scripts/lib/common.sh"
ENV_DIR="${APP_ENV_DIR}"
PYTHON_BIN="${PYTHON_BIN:-${APP_PYTHON}}"
FRONTEND_DIST="${ROOT_DIR}/frontend/dist"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[moshimo-box] App environment is missing: ${PYTHON_BIN}" >&2
  echo "[moshimo-box] Run: ${ROOT_DIR}/scripts/bootstrap.sh" >&2
  exit 1
fi

if [[ ! -d "${FRONTEND_DIST}" ]]; then
  echo "[moshimo-box] Frontend build is missing: ${FRONTEND_DIST}" >&2
  echo "[moshimo-box] Run: ${ROOT_DIR}/scripts/setup-app-env.sh" >&2
  exit 1
fi

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
DEFAULT_PORT="$(
  "${PYTHON_BIN}" -c \
    'from backend.app.config import ConfigManager; print(ConfigManager().developer.server.default_port)'
)"
PORT="${1:-${DEFAULT_PORT}}"

if [[ ! "${PORT}" =~ ^[0-9]+$ ]] || (( PORT < 1024 || PORT > 65535 )); then
  echo "[moshimo-box] Port must be a number between 1024 and 65535." >&2
  exit 2
fi

export PORT

PREFIX="${JUPYTERHUB_SERVICE_PREFIX:-/}"
if [[ "${PREFIX}" != */ ]]; then
  PREFIX="${PREFIX}/"
fi
CONFIG_PUBLIC_BASE_URL="$(
  "${PYTHON_BIN}" -c \
    'from backend.app.config import ConfigManager; print(ConfigManager().developer.app.public_base_url)'
)"
PUBLIC_BASE_URL="${PUBLIC_BASE_URL:-${CONFIG_PUBLIC_BASE_URL}}"
ACCESS_URL="${PUBLIC_BASE_URL}${PREFIX}proxy/${PORT}/"

echo "[moshimo-box] URL: ${ACCESS_URL}"
echo "[moshimo-box] Binding: 127.0.0.1:${PORT}"
echo "[moshimo-box] Environment: ${ENV_DIR}"

cd "${ROOT_DIR}"
exec "${PYTHON_BIN}" -m uvicorn \
  backend.app.main:app \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --proxy-headers
