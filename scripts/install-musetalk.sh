#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"
ENV_DIR="${ENV_ROOT}/musetalk"
PYTHON_BIN="${ENV_DIR}/bin/python"
SOURCE_DIR="${MODEL_ROOT}/musetalk-source"
SOURCE_PATCH="${ROOT_DIR}/patches/musetalk-1.5-static-image-cleanup.patch"
MUSE_COMMIT="0a89dec45a0192b824e3cf4daf96c239440c5ed8"
MUSE_REVISION="3ef28bc5cff08c90ad8178a25f1b570cd800170f"
VAE_REVISION="31f26fdeee1355a5c34592e401dd41e45d25a493"
WHISPER_REVISION="169d4a4341b33bc18d8881c4b69c2e104e1cc0af"
DWPOSE_REVISION="1a7144101628d69ee7a3768d1ee3a094070dc388"
CACHE_DIR="${MODEL_ROOT}/shared-cache/huggingface"
INSTALL_MARKER="${ENV_DIR}/.moshimo-musetalk-${MUSE_COMMIT}"

mkdir -p "${ENV_ROOT}" "${MODEL_ROOT}" "${CACHE_DIR}"

if [[ ! -d "${SOURCE_DIR}/.git" ]]; then
  git clone https://github.com/TMElyralab/MuseTalk.git "${SOURCE_DIR}"
fi
git -C "${SOURCE_DIR}" fetch origin "${MUSE_COMMIT}" --depth 1
git -C "${SOURCE_DIR}" checkout --detach "${MUSE_COMMIT}"
if git -C "${SOURCE_DIR}" apply --check "${SOURCE_PATCH}"; then
  git -C "${SOURCE_DIR}" apply "${SOURCE_PATCH}"
elif ! git -C "${SOURCE_DIR}" apply --reverse --check "${SOURCE_PATCH}"; then
  echo "[moshimo-box] MuseTalk compatibility patch does not apply cleanly" >&2
  exit 1
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[moshimo-box] Creating MuseTalk environment: ${ENV_DIR}"
  "${CONDA_BIN}" create --prefix "${ENV_DIR}" python=3.10 pip -y
else
  echo "[moshimo-box] Reusing MuseTalk environment: ${ENV_DIR}"
fi

if [[ ! -f "${INSTALL_MARKER}" ]]; then
  "${PYTHON_BIN}" -m pip install --upgrade pip setuptools wheel
  "${PYTHON_BIN}" -m pip install \
    torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118
  "${PYTHON_BIN}" -m pip install -r "${ROOT_DIR}/backend/requirements.txt"
  "${PYTHON_BIN}" -m pip install \
    diffusers==0.30.2 accelerate==0.28.0 numpy==1.23.5 \
    opencv-python==4.9.0.80 soundfile==0.12.1 transformers==4.39.2 \
    huggingface_hub==0.30.2 librosa==0.11.0 einops==0.8.1 \
    imageio[ffmpeg] omegaconf ffmpeg-python moviepy tqdm scipy gdown requests
  "${PYTHON_BIN}" -m pip install --no-cache-dir -U openmim
  "${ENV_DIR}/bin/mim" install mmengine
  "${ENV_DIR}/bin/mim" install "mmcv==2.0.1"
  "${ENV_DIR}/bin/mim" install "mmdet==3.1.0"
  # chumpy's legacy setup imports pip and cannot build in pip's isolated env.
  "${PYTHON_BIN}" -m pip install --no-build-isolation "chumpy==0.70"
  "${ENV_DIR}/bin/mim" install "mmpose==1.1.0"
  touch "${INSTALL_MARKER}"
fi

HF_HOME="${CACHE_DIR}" "${PYTHON_BIN}" - \
  "${SOURCE_DIR}" "${MUSE_REVISION}" "${VAE_REVISION}" \
  "${WHISPER_REVISION}" "${DWPOSE_REVISION}" <<'PYWEIGHTS'
from pathlib import Path
import sys

import gdown
from huggingface_hub import snapshot_download

source = Path(sys.argv[1])
muse_revision, vae_revision, whisper_revision, dwpose_revision = sys.argv[2:6]
models = source / "models"
models.mkdir(parents=True, exist_ok=True)

snapshot_download(
    repo_id="TMElyralab/MuseTalk",
    revision=muse_revision,
    local_dir=models,
    allow_patterns=["musetalkV15/musetalk.json", "musetalkV15/unet.pth"],
)
snapshot_download(
    repo_id="stabilityai/sd-vae-ft-mse",
    revision=vae_revision,
    local_dir=models / "sd-vae",
    allow_patterns=["config.json", "diffusion_pytorch_model.bin"],
)
snapshot_download(
    repo_id="openai/whisper-tiny",
    revision=whisper_revision,
    local_dir=models / "whisper",
    allow_patterns=["config.json", "pytorch_model.bin", "preprocessor_config.json"],
)
snapshot_download(
    repo_id="yzd-v/DWPose",
    revision=dwpose_revision,
    local_dir=models / "dwpose",
    allow_patterns=["dw-ll_ucoco_384.pth"],
)

face_dir = models / "face-parse-bisent"
face_dir.mkdir(parents=True, exist_ok=True)
face_model = face_dir / "79999_iter.pth"
if not face_model.is_file():
    result = gdown.download(
        id="154JgKpzCPW82qINcVieuPH3fZ2e0P812",
        output=str(face_model),
        quiet=False,
    )
    if not result:
        raise SystemExit("Failed to download MuseTalk face parser")
PYWEIGHTS

FACE_DIR="${SOURCE_DIR}/models/face-parse-bisent"
S3FD_DIR="${SOURCE_DIR}/musetalk/utils/face_detection/detection/sfd"
mkdir -p "${FACE_DIR}" "${S3FD_DIR}"
if [[ ! -f "${FACE_DIR}/resnet18-5c106cde.pth" ]]; then
  curl -L https://download.pytorch.org/models/resnet18-5c106cde.pth \
    -o "${FACE_DIR}/resnet18-5c106cde.pth"
fi
if [[ ! -f "${S3FD_DIR}/s3fd.pth" ]]; then
  curl -L https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth \
    -o "${S3FD_DIR}/s3fd.pth"
fi

required=(
  "${SOURCE_DIR}/models/musetalkV15/musetalk.json"
  "${SOURCE_DIR}/models/musetalkV15/unet.pth"
  "${SOURCE_DIR}/models/sd-vae/config.json"
  "${SOURCE_DIR}/models/sd-vae/diffusion_pytorch_model.bin"
  "${SOURCE_DIR}/models/whisper/config.json"
  "${SOURCE_DIR}/models/whisper/pytorch_model.bin"
  "${SOURCE_DIR}/models/whisper/preprocessor_config.json"
  "${SOURCE_DIR}/models/dwpose/dw-ll_ucoco_384.pth"
  "${SOURCE_DIR}/models/face-parse-bisent/79999_iter.pth"
  "${SOURCE_DIR}/models/face-parse-bisent/resnet18-5c106cde.pth"
)
for path in "${required[@]}"; do
  if [[ ! -f "${path}" ]]; then
    echo "[moshimo-box] Missing MuseTalk file: ${path}" >&2
    exit 1
  fi
done

(
  cd "${SOURCE_DIR}"
  PYTHONPATH="${SOURCE_DIR}" "${PYTHON_BIN}" -m scripts.inference --help >/dev/null
)
PYTHONPATH="${ROOT_DIR}" "${PYTHON_BIN}" -m backend.app.workers.runtime --help >/dev/null

echo "[moshimo-box] MuseTalk 1.5 is installed and import-validated."
