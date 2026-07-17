#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"
FISH_ENV_DIR="${ENV_ROOT}/generation"
FISH_PYTHON_BIN="${FISH_ENV_DIR}/bin/python"
FLUX_ENV_DIR="${ENV_ROOT}/flux2-klein"
FLUX_PYTHON_BIN="${FLUX_ENV_DIR}/bin/python"
FISH_SOURCE="${MODEL_ROOT}/fish-speech-source"
FISH_COMMIT="e5e292632cb11e7a27b2b7487f58f612bc101e13"
FISH_MODEL_DIR="${MODEL_ROOT}/tts/fish-s2-pro"
FISH_REVISION="1de9996b6be38b745688de084d87a5633f714e4e"
FLUX_MODEL_DIR="${MODEL_ROOT}/image/flux2-klein-4b"
FLUX_REVISION="e7b7dc27f91deacad38e78976d1f2b499d76a294"
DIFFUSERS_COMMIT="8f02e2c07f48ff2b53ced8392940e28aa9bd0019"
TRANSFORMERS_COMMIT="150eb7c9ed4091294c829fa0e9466b090cb0f87f"
CACHE_DIR="${MODEL_ROOT}/shared-cache/huggingface"

mkdir -p "${ENV_ROOT}" "${MODEL_ROOT}/tts" "${MODEL_ROOT}/image" "${CACHE_DIR}"

if [[ ! -x "${FISH_PYTHON_BIN}" ]]; then
  echo "[moshimo-box] Creating Fish environment: ${FISH_ENV_DIR}"
  "${CONDA_BIN}" create --prefix "${FISH_ENV_DIR}" python=3.12 pip -y
else
  echo "[moshimo-box] Reusing Fish environment: ${FISH_ENV_DIR}"
fi

if [[ ! -x "${FLUX_PYTHON_BIN}" ]]; then
  echo "[moshimo-box] Creating FLUX environment: ${FLUX_ENV_DIR}"
  "${CONDA_BIN}" create --prefix "${FLUX_ENV_DIR}" python=3.12 pip -y
else
  echo "[moshimo-box] Reusing FLUX environment: ${FLUX_ENV_DIR}"
fi

if [[ ! -d "${FISH_SOURCE}/.git" ]]; then
  echo "[moshimo-box] Cloning Fish Speech"
  git clone https://github.com/fishaudio/fish-speech.git "${FISH_SOURCE}"
fi

git -C "${FISH_SOURCE}" fetch origin "${FISH_COMMIT}" --depth 1
git -C "${FISH_SOURCE}" checkout --detach "${FISH_COMMIT}"

# PyAudio is an official Fish dependency. Installing PortAudio through Conda
# keeps the project environment self-contained and avoids system changes.
"${CONDA_BIN}" install --prefix "${FISH_ENV_DIR}" -c conda-forge portaudio libsndfile -y

"${FISH_PYTHON_BIN}" -m pip install --upgrade pip setuptools wheel
"${FISH_PYTHON_BIN}" -m pip install \
  torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128
"${FISH_PYTHON_BIN}" -m pip install -r "${ROOT_DIR}/backend/requirements.txt"
"${FISH_PYTHON_BIN}" -m pip install \
  -r "${ROOT_DIR}/workers/requirements/fish-s2-pro.txt"
# AudioTools has an obsolete protobuf upper bound. Fish's own lock file
# overrides it, so install these two official runtime packages without asking
# pip to resolve their unrelated training extras.
"${FISH_PYTHON_BIN}" -m pip install --no-deps \
  descript-audio-codec==1.0.0 \
  descript-audiotools==0.7.2
"${FISH_PYTHON_BIN}" -m pip install --no-deps -e "${FISH_SOURCE}"

"${FLUX_PYTHON_BIN}" -m pip install --upgrade pip setuptools wheel
"${FLUX_PYTHON_BIN}" -m pip install \
  torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128
"${FLUX_PYTHON_BIN}" -m pip install -r "${ROOT_DIR}/backend/requirements.txt"
"${FLUX_PYTHON_BIN}" -m pip install \
  "git+https://github.com/huggingface/diffusers.git@${DIFFUSERS_COMMIT}" \
  "git+https://github.com/huggingface/transformers.git@${TRANSFORMERS_COMMIT}" \
  accelerate==1.12.0 sentencepiece==0.2.1 soundfile==0.13.1

DOWNLOAD_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"
if [[ -z "${DOWNLOAD_TOKEN}" ]]; then
  DOWNLOAD_TOKEN="$("${FISH_PYTHON_BIN}" -c \
    'from huggingface_hub import get_token; print(get_token() or "")')"
fi
if [[ -z "${DOWNLOAD_TOKEN}" ]]; then
  echo "[moshimo-box] Hugging Face authentication is required." >&2
  echo "[moshimo-box] Run: ./scripts/huggingface-login.sh" >&2
  exit 1
fi

HF_HOME="${CACHE_DIR}" HF_TOKEN="${DOWNLOAD_TOKEN}" "${FISH_PYTHON_BIN}" - \
  "${FISH_MODEL_DIR}" "${FISH_REVISION}" <<'PYFISH'
from pathlib import Path
import sys

from huggingface_hub import snapshot_download

fish_dir = Path(sys.argv[1])
fish_revision = sys.argv[2]

snapshot_download(
    repo_id="fishaudio/s2-pro",
    revision=fish_revision,
    local_dir=fish_dir,
)

required = [
    fish_dir / "config.json",
    fish_dir / "codec.pth",
    fish_dir / "model.safetensors.index.json",
]
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise SystemExit("Downloaded Fish model is incomplete: " + ", ".join(missing))
print("[moshimo-box] Fish S2 Pro weights are ready")
PYFISH

HF_HOME="${CACHE_DIR}" HF_TOKEN="${DOWNLOAD_TOKEN}" "${FLUX_PYTHON_BIN}" - \
  "${FLUX_MODEL_DIR}" "${FLUX_REVISION}" <<'PYFLUX'
from pathlib import Path
import sys

from huggingface_hub import snapshot_download

flux_dir = Path(sys.argv[1])
flux_revision = sys.argv[2]

snapshot_download(
    repo_id="black-forest-labs/FLUX.2-klein-4B",
    revision=flux_revision,
    local_dir=flux_dir,
)

required = [
    flux_dir / "model_index.json",
    flux_dir / "transformer" / "diffusion_pytorch_model.safetensors",
    flux_dir / "text_encoder" / "model.safetensors.index.json",
    flux_dir / "vae" / "diffusion_pytorch_model.safetensors",
]
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise SystemExit("Downloaded FLUX model is incomplete: " + ", ".join(missing))
print("[moshimo-box] FLUX.2 Klein weights are ready")
PYFLUX

"${FISH_PYTHON_BIN}" - <<'PYCHECK'
import torch
from fish_speech.inference_engine import TTSInferenceEngine

print(f"[moshimo-box] Fish torch={torch.__version__} cuda={torch.cuda.is_available()}")
print(f"[moshimo-box] Fish adapter={TTSInferenceEngine.__name__}")
PYCHECK

"${FLUX_PYTHON_BIN}" - <<'PYCHECK'
import torch
from diffusers import Flux2KleinPipeline

print(f"[moshimo-box] FLUX torch={torch.__version__} cuda={torch.cuda.is_available()}")
print(f"[moshimo-box] FLUX adapter={Flux2KleinPipeline.__name__}")
PYCHECK

echo "[moshimo-box] Generation models installed."
echo "[moshimo-box] Install EchoMimicV3 next, then run scripts/activate-generation-models.sh"
