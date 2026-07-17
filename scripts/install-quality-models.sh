#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"
FLUX_PYTHON="${ENV_ROOT}/flux2-klein/bin/python"
CACHE_DIR="${MODEL_ROOT}/shared-cache/huggingface"
QWEN_DIR="${MODEL_ROOT}/llm/qwen3-vl-8b-instruct"
QWEN_REVISION="0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
FLUX_DIR="${MODEL_ROOT}/image/flux2-klein-9b"
FLUX_REVISION="92196c8e11f7b6cf2b7493e037d8c5345c559216"
TARGET="${1:-all}"

case "${TARGET}" in
  all|qwen|flux) ;;
  *)
    echo "Usage: $0 [all|qwen|flux]" >&2
    exit 2
    ;;
esac

if [[ ! -x "${APP_PYTHON}" || ! -x "${FLUX_PYTHON}" ]]; then
  echo "[moshimo-box] Run the base and generation installers first." >&2
  exit 1
fi

mkdir -p "${CACHE_DIR}" "${MODEL_ROOT}/llm" "${MODEL_ROOT}/image"

install_qwen() {
  if ! "${APP_PYTHON}" -c "import torchvision" >/dev/null 2>&1; then
    "${APP_PYTHON}" -m pip install --no-deps torchvision==0.26.0 \
      --index-url https://download.pytorch.org/whl/cu128
  fi

  HF_HOME="${CACHE_DIR}" "${APP_PYTHON}" - \
    "${QWEN_DIR}" "${QWEN_REVISION}" <<'PYQWEN'
from pathlib import Path
import sys

from huggingface_hub import snapshot_download

target = Path(sys.argv[1])
revision = sys.argv[2]
snapshot_download(
    repo_id="Qwen/Qwen3-VL-8B-Instruct",
    revision=revision,
    local_dir=target,
)
required = [
    target / "config.json",
    target / "preprocessor_config.json",
    target / "model.safetensors.index.json",
]
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise SystemExit("Qwen3-VL download is incomplete: " + ", ".join(missing))
print("[moshimo-box] Qwen3-VL 8B weights are ready")
PYQWEN

  "${APP_PYTHON}" -c \
    "from transformers import AutoProcessor, Qwen3VLForConditionalGeneration; print('Qwen3-VL adapter ready')"
}

install_flux() {
  local hf_token
  hf_token="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"
  if [[ -z "${hf_token}" ]]; then
    hf_token="$("${FLUX_PYTHON}" -c \
      'from huggingface_hub import get_token; print(get_token() or "")')"
  fi
  if [[ -z "${hf_token}" ]]; then
    echo "[moshimo-box] Hugging Face authentication is not configured." >&2
    exit 1
  fi

  HF_HOME="${CACHE_DIR}" HF_TOKEN="${hf_token}" "${FLUX_PYTHON}" - \
    "${FLUX_DIR}" "${FLUX_REVISION}" <<'PYFLUX'
from pathlib import Path
import os
import sys

from huggingface_hub import snapshot_download

target = Path(sys.argv[1])
revision = sys.argv[2]
try:
    snapshot_download(
        repo_id="black-forest-labs/FLUX.2-klein-9B",
        revision=revision,
        local_dir=target,
        token=os.environ["HF_TOKEN"],
    )
except Exception as exc:
    raise SystemExit(
        "FLUX.2 Klein 9B download failed. Accept its gated non-commercial "
        "license on Hugging Face and configure HF_TOKEN, then rerun: " + str(exc)
    ) from exc

required = [
    target / "model_index.json",
    target / "transformer",
    target / "text_encoder",
    target / "vae",
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit("FLUX.2 Klein 9B download is incomplete: " + ", ".join(missing))
print("[moshimo-box] FLUX.2 Klein 9B weights are ready")
PYFLUX

  "${FLUX_PYTHON}" -c \
    "from diffusers import Flux2KleinPipeline; print('FLUX.2 Klein adapter ready')"
}

if [[ "${TARGET}" == "all" || "${TARGET}" == "qwen" ]]; then
  install_qwen
fi
if [[ "${TARGET}" == "all" || "${TARGET}" == "flux" ]]; then
  install_flux
fi

echo "[moshimo-box] Requested quality model installation completed."
