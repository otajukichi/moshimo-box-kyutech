#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"
ENV_DIR="${APP_ENV_DIR}"

mkdir -p "${ENV_ROOT}"

if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
  echo "[moshimo-box] Creating conda environment: ${ENV_DIR}"
  "${CONDA_BIN}" create \
    --prefix "${ENV_DIR}" \
    python=3.12 \
    pip \
    nodejs=22 \
    -y
else
  echo "[moshimo-box] Reusing conda environment: ${ENV_DIR}"
fi

echo "[moshimo-box] Installing Python dependencies"
"${ENV_DIR}/bin/python" -m pip install -r "${ROOT_DIR}/backend/requirements.txt"

echo "[moshimo-box] Installing frontend dependencies"
PATH="${ENV_DIR}/bin:${PATH}" "${ENV_DIR}/bin/npm" install \
  --prefix "${ROOT_DIR}/frontend"

echo "[moshimo-box] Building frontend"
PATH="${ENV_DIR}/bin:${PATH}" "${ENV_DIR}/bin/npm" run build \
  --prefix "${ROOT_DIR}/frontend"

echo "[moshimo-box] Setup complete."
echo "[moshimo-box] Start with: ${ROOT_DIR}/start-app.sh"
