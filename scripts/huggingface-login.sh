#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"
moshimo_require_app_env

HF_BIN="${APP_ENV_DIR}/bin/hf"
LEGACY_HF_BIN="${APP_ENV_DIR}/bin/huggingface-cli"

if [[ ! -x "${HF_BIN}" && ! -x "${LEGACY_HF_BIN}" ]]; then
  echo "[moshimo-box] Installing the local Hugging Face login client"
  "${APP_PYTHON}" -m pip install "huggingface-hub==0.36.2"
fi

echo "[moshimo-box] The token is stored in this JupyterHub account, not in Git."
if [[ -x "${HF_BIN}" ]]; then
  exec "${HF_BIN}" auth login
fi
exec "${LEGACY_HF_BIN}" login
