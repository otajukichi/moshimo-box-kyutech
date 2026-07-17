#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"

ENV_DIR="${ENV_ROOT}/gpt-oss"
PYTHON_BIN="${ENV_DIR}/bin/python"
MODEL_DIR="${MODEL_ROOT}/llm/gpt-oss-20b"
CACHE_DIR="${MODEL_ROOT}/shared-cache/huggingface"
REVISION="6cee5e81ee83917806bbde320786a8fb61efebee"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[moshimo-box] Creating the GPT-OSS Conda environment"
  "${CONDA_BIN}" create -y -p "${ENV_DIR}" python=3.12 pip
else
  echo "[moshimo-box] Reusing the GPT-OSS environment: ${ENV_DIR}"
fi

mkdir -p "${MODEL_DIR}" "${CACHE_DIR}"

echo "[moshimo-box] Installing the pinned GPT-OSS runtime"
"${PYTHON_BIN}" -m pip install --upgrade pip setuptools wheel
"${PYTHON_BIN}" -m pip install \
  -r "${ROOT_DIR}/backend/requirements.txt" \
  -r "${ROOT_DIR}/workers/requirements/gpt-oss-vllm.txt"

"${PYTHON_BIN}" - <<'PYRUNTIME'
import torch
import vllm

cuda_runtime = torch.version.cuda or ""
if not cuda_runtime.startswith("12."):
    raise SystemExit(
        f"GPT-OSS requires the CUDA 12.x vLLM wheel on this server; got {cuda_runtime or 'unknown'}"
    )
print(f"[moshimo-box] vLLM {vllm.__version__} / PyTorch CUDA {cuda_runtime}")
PYRUNTIME

echo "[moshimo-box] Downloading openai/gpt-oss-20b revision ${REVISION}"
HF_HOME="${CACHE_DIR}" "${PYTHON_BIN}" - "${MODEL_DIR}" "${REVISION}" <<'PYMODEL'
from pathlib import Path
import sys

from huggingface_hub import snapshot_download

model_dir = Path(sys.argv[1])
revision = sys.argv[2]
snapshot_download(
    repo_id="openai/gpt-oss-20b",
    revision=revision,
    local_dir=model_dir,
    allow_patterns=[
        "LICENSE",
        "README.md",
        "USAGE_POLICY",
        "chat_template.jinja",
        "config.json",
        "generation_config.json",
        "model-*.safetensors",
        "model.safetensors.index.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ],
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
    raise SystemExit("Downloaded GPT-OSS model shards are incomplete")
print(f"[moshimo-box] GPT-OSS model ready: {model_dir}")
PYMODEL

echo "[moshimo-box] GPT-OSS environment: ${ENV_DIR}"
