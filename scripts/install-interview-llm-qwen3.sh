#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"
ENV_DIR="${APP_ENV_DIR}"
PYTHON_BIN="${APP_PYTHON}"
MODEL_DIR="${MODEL_ROOT}/llm/qwen3-4b-instruct-2507"
CACHE_DIR="${MODEL_ROOT}/shared-cache/huggingface"
REVISION="cdbee75f17c01a7cc42f958dc650907174af0554"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[moshimo-box] App environment is missing. Run scripts/setup-app-env.sh first." >&2
  exit 1
fi

mkdir -p "${MODEL_DIR}" "${CACHE_DIR}"

echo "[moshimo-box] Installing PyTorch with CUDA 12.8 support"
"${PYTHON_BIN}" -m pip install \
  torch==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu128

echo "[moshimo-box] Installing Transformers dependencies"
"${PYTHON_BIN}" -m pip install \
  -r "${ROOT_DIR}/workers/requirements/interview-llm-transformers.txt"

echo "[moshimo-box] Downloading Qwen3-4B-Instruct-2507 revision ${REVISION}"
HF_HOME="${CACHE_DIR}" "${PYTHON_BIN}" - "${MODEL_DIR}" "${REVISION}" <<'PYMODEL'
from pathlib import Path
import sys

from huggingface_hub import snapshot_download

model_dir = Path(sys.argv[1])
revision = sys.argv[2]
snapshot_download(
    repo_id="Qwen/Qwen3-4B-Instruct-2507",
    revision=revision,
    local_dir=model_dir,
)
required = {
    "config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
}
missing = sorted(name for name in required if not (model_dir / name).is_file())
if missing:
    raise SystemExit(f"Downloaded model is incomplete: {', '.join(missing)}")
if len(list(model_dir.glob("model-*.safetensors"))) != 3:
    raise SystemExit("Downloaded model shards are incomplete")
print(f"[moshimo-box] Model ready: {model_dir}")
PYMODEL
