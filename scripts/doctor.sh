#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"

ALLOW_DEBUG=false
case "${1:-}" in
  "")
    ;;
  --allow-debug)
    ALLOW_DEBUG=true
    ;;
  -h|--help)
    echo "Usage: ./scripts/doctor.sh [--allow-debug]"
    exit 0
    ;;
  *)
    echo "Usage: ./scripts/doctor.sh [--allow-debug]" >&2
    exit 2
    ;;
esac

errors=0
warnings=0

ok() {
  echo "[ OK ] $*"
}

warn() {
  echo "[WARN] $*"
  warnings=$((warnings + 1))
}

fail() {
  echo "[FAIL] $*"
  errors=$((errors + 1))
}

check_command() {
  local command_name="$1"
  if command -v "${command_name}" >/dev/null 2>&1; then
    ok "${command_name}: $(command -v "${command_name}")"
  else
    fail "${command_name} is not available"
  fi
}

echo "Moshimo Box preflight"
echo "====================="
moshimo_print_paths
echo

check_command git
check_command ffmpeg
check_command ffprobe
check_command nvidia-smi

if [[ -x "${CONDA_BIN}" ]]; then
  ok "conda: ${CONDA_BIN}"
else
  fail "conda was not found: ${CONDA_BIN}"
fi

if [[ -x "${APP_PYTHON}" ]]; then
  ok "app Python: ${APP_PYTHON}"
else
  fail "app environment is missing; run ./scripts/bootstrap.sh"
fi

if [[ -x "${APP_NPM}" ]]; then
  ok "npm: ${APP_NPM}"
else
  fail "npm is missing from the app environment"
fi

GPT_OSS_MODEL_DIR="${MODEL_ROOT}/llm/gpt-oss-20b"
if [[ -d "${GPT_OSS_MODEL_DIR}" ]]; then
  if [[ -x "${GPT_OSS_PYTHON}" ]]; then
    if version_line="$("${GPT_OSS_PYTHON}" -c 'import openai_harmony, vllm; print(f"vLLM {vllm.__version__} / Harmony available")' 2>/dev/null)"; then
      ok "GPT-OSS runtime: ${version_line}"
    else
      fail "GPT-OSS runtime imports failed; rerun ./scripts/install-models.sh gpt-oss"
    fi
  else
    fail "GPT-OSS environment is missing: ${GPT_OSS_PYTHON}"
  fi
  shard_count="$(find "${GPT_OSS_MODEL_DIR}" -maxdepth 1 -type f -name 'model-*.safetensors' | wc -l)"
  if [[ -f "${GPT_OSS_MODEL_DIR}/config.json"     && -f "${GPT_OSS_MODEL_DIR}/model.safetensors.index.json"     && -f "${GPT_OSS_MODEL_DIR}/tokenizer.json"     && "${shard_count}" -eq 3 ]]; then
    ok "GPT-OSS 20B checkpoint is complete"
  else
    fail "GPT-OSS 20B checkpoint is incomplete; rerun ./scripts/install-models.sh gpt-oss"
  fi
fi

if [[ -f "${ROOT_DIR}/frontend/dist/index.html" ]]; then
  ok "frontend build exists"
else
  fail "frontend build is missing; run ./scripts/bootstrap.sh"
fi

if [[ -f "${ROOT_DIR}/config/local.yaml" ]]; then
  ok "local developer configuration exists"
else
  fail "config/local.yaml is missing; run ./scripts/bootstrap.sh"
fi

if [[ -f "${ROOT_DIR}/config/model-catalog.local.yaml" ]]; then
  ok "local model catalog exists"
else
  fail "local model catalog is missing; install and activate models"
fi

for directory in "${ROOT_DIR}/data/sessions" "${ROOT_DIR}/logs"; do
  if [[ -d "${directory}" && -w "${directory}" ]]; then
    ok "writable: ${directory}"
  else
    fail "not writable: ${directory}"
  fi
done

if command -v nvidia-smi >/dev/null 2>&1; then
  gpu_line="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"
  if [[ -n "${gpu_line}" ]]; then
    ok "GPU visible: ${gpu_line}"
  else
    fail "nvidia-smi cannot see an assigned GPU"
  fi
fi

if [[ -x "${APP_PYTHON}" ]]; then
  if "${APP_PYTHON}" - "${ROOT_DIR}" "${ALLOW_DEBUG}" <<'PYCHECK'
from __future__ import annotations

from pathlib import Path
import sys

root = Path(sys.argv[1])
allow_debug = sys.argv[2].lower() == "true"
sys.path.insert(0, str(root))

from backend.app.config import ConfigManager
from backend.app.schemas import WorkerRole

config = ConfigManager(root)
print(f"[INFO] profile: {config.staff.quality_profile.value}")
print(f"[INFO] debug mode: {config.developer.app.debug_mode}")
print(f"[INFO] environment root: {config.environment_root}")
print(f"[INFO] model root: {config.model_root}")

if config.developer.app.debug_mode and not allow_debug:
    raise SystemExit(
        "[FAIL] debug mode is enabled; use config/local.yaml for production "
        "or pass --allow-debug"
    )

allowed_stubs = {
    WorkerRole.AUDIO_PREPROCESS,
    WorkerRole.INTERVIEW_TTS,
}
problems = []
for role, model_id in config.staff.stage_models.items():
    entry = config.catalog.entry(model_id)
    state = "stub" if entry.is_stub else "ready"
    print(f"[MODEL] {role.value}: {model_id} ({state})")
    if entry.is_stub and role not in allowed_stubs:
        problems.append(f"{role.value} still uses a stub")
    elif not config.catalog.is_available(entry):
        problems.append(f"{role.value} is unavailable: {model_id}")

if problems:
    raise SystemExit("[FAIL] " + "; ".join(problems))
PYCHECK
  then
    ok "application configuration and selected workers"
  else
    fail "application configuration or selected workers"
  fi
fi

if [[ -z "${JUPYTERHUB_SERVICE_PREFIX:-}" ]]; then
  warn "JUPYTERHUB_SERVICE_PREFIX is not visible in this terminal"
else
  ok "JupyterHub proxy prefix: ${JUPYTERHUB_SERVICE_PREFIX}"
fi

if [[ -x "${APP_PYTHON}" ]]; then
  if "${APP_PYTHON}" -c     'from huggingface_hub import get_token; raise SystemExit(0 if get_token() else 1)'     >/dev/null 2>&1; then
    ok "Hugging Face authentication is available"
  else
    warn "Hugging Face authentication is absent; runtime works if gated weights are already present"
  fi
fi

echo
df -h "${MODEL_ROOT}" 2>/dev/null | tail -1 || true
echo
echo "Result: ${errors} error(s), ${warnings} warning(s)"
if (( errors > 0 )); then
  exit 1
fi

echo "Preflight passed. Grant camera and microphone access in Microsoft Edge on first use."
