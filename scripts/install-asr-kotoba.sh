#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"
ENV_DIR="${APP_ENV_DIR}"
PYTHON_BIN="${APP_PYTHON}"
MODEL_DIR="${MODEL_ROOT}/asr/kotoba-whisper-v2.0-faster"
CACHE_DIR="${MODEL_ROOT}/shared-cache/huggingface"
REVISION="f44edd35eaeb2274e85ac7b31fb2c6f59ff1c4bc"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[moshimo-box] App environment is missing. Run scripts/setup-app-env.sh first." >&2
  exit 1
fi

mkdir -p "${MODEL_DIR}" "${CACHE_DIR}"

echo "[moshimo-box] Installing faster-whisper dependencies"
"${PYTHON_BIN}" -m pip install -r "${ROOT_DIR}/workers/requirements/asr-faster-whisper.lock.txt"

echo "[moshimo-box] Downloading Kotoba-Whisper revision ${REVISION}"
HF_HOME="${CACHE_DIR}" "${PYTHON_BIN}" - "${MODEL_DIR}" "${REVISION}" <<'PYMODEL'
from pathlib import Path
import sys

from huggingface_hub import snapshot_download

model_dir = Path(sys.argv[1])
revision = sys.argv[2]
snapshot_download(
    repo_id="kotoba-tech/kotoba-whisper-v2.0-faster",
    revision=revision,
    local_dir=model_dir,
)
required = {
    "config.json",
    "model.bin",
    "preprocessor_config.json",
    "tokenizer.json",
    "vocabulary.json",
}
missing = sorted(name for name in required if not (model_dir / name).is_file())
if missing:
    raise SystemExit(f"Downloaded model is incomplete: {', '.join(missing)}")
print(f"[moshimo-box] Model ready: {model_dir}")
PYMODEL
