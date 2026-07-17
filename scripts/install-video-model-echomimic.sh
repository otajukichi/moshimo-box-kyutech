#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"
ENV_DIR="${ENV_ROOT}/echomimic-v3"
PYTHON_BIN="${ENV_DIR}/bin/python"
SOURCE_DIR="${MODEL_ROOT}/echomimic-v3-source"
SOURCE_COMMIT="7e89489ca51c0d008fc1963ec6c03fc5bd0b9397"
VIDEO_MODEL_ROOT="${MODEL_ROOT}/video/echomimic-v3-flash"
BASE_DIR="${VIDEO_MODEL_ROOT}/Wan2.1-Fun-V1.1-1.3B-InP"
BASE_REVISION="fc913c34361f4ec879e2f9c78b4f11ae50a937d1"
FLASH_REVISION="311e176905a8c4c24b240b530488fe636ce4d249"
WAV2VEC_DIR="${VIDEO_MODEL_ROOT}/chinese-wav2vec2-base"
WAV2VEC_SHA256="b86f5be7b752fc655c27387a75712f733315a30f976e5875491599615399e773"
HF_CACHE_DIR="${MODEL_ROOT}/shared-cache/huggingface"
MS_CACHE_DIR="${MODEL_ROOT}/shared-cache/modelscope"

mkdir -p "${ENV_ROOT}" "${VIDEO_MODEL_ROOT}" "${HF_CACHE_DIR}" "${MS_CACHE_DIR}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[moshimo-box] Creating EchoMimicV3 environment: ${ENV_DIR}"
  "${CONDA_BIN}" create --prefix "${ENV_DIR}" python=3.10 pip -y
else
  echo "[moshimo-box] Reusing EchoMimicV3 environment: ${ENV_DIR}"
fi

if [[ ! -d "${SOURCE_DIR}/.git" ]]; then
  echo "[moshimo-box] Cloning EchoMimicV3"
  git clone https://github.com/antgroup/echomimic_v3.git "${SOURCE_DIR}"
fi

git -C "${SOURCE_DIR}" fetch origin "${SOURCE_COMMIT}" --depth 1
git -C "${SOURCE_DIR}" checkout --detach "${SOURCE_COMMIT}"

"${PYTHON_BIN}" -m pip install --upgrade pip setuptools wheel
"${PYTHON_BIN}" -m pip install \
  torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128
"${PYTHON_BIN}" -m pip install -r "${ROOT_DIR}/backend/requirements.txt"
"${PYTHON_BIN}" -m pip install \
  -r "${ROOT_DIR}/workers/requirements/echomimic-v3-flash.txt"

HF_HOME="${HF_CACHE_DIR}" "${PYTHON_BIN}" - \
  "${BASE_DIR}" "${BASE_REVISION}" "${VIDEO_MODEL_ROOT}" "${FLASH_REVISION}" <<'PYHF'
from pathlib import Path
import sys

from huggingface_hub import snapshot_download

base_dir = Path(sys.argv[1])
base_revision = sys.argv[2]
model_root = Path(sys.argv[3])
flash_revision = sys.argv[4]

snapshot_download(
    repo_id="alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP",
    revision=base_revision,
    local_dir=base_dir,
)
snapshot_download(
    repo_id="BadToBest/EchoMimicV3",
    revision=flash_revision,
    local_dir=model_root,
    allow_patterns=["echomimicv3-flash-pro/*"],
)
PYHF

MODELSCOPE_CACHE="${MS_CACHE_DIR}" "${PYTHON_BIN}" - \
  "${WAV2VEC_DIR}" "${WAV2VEC_SHA256}" "${MS_CACHE_DIR}" <<'PYMS'
from hashlib import sha256
from pathlib import Path
import shutil
import sys

from modelscope.hub.snapshot_download import snapshot_download

target = Path(sys.argv[1])
expected_hash = sys.argv[2]
cache_dir = Path(sys.argv[3])
model_file = target / "model.safetensors"
if not model_file.is_file() or sha256(model_file.read_bytes()).hexdigest() != expected_hash:
    cached = Path(
        snapshot_download(
            "TencentGameMate/chinese-wav2vec2-base",
            revision="master",
            cache_dir=str(cache_dir),
        )
    )
    shutil.copytree(cached, target, dirs_exist_ok=True)
actual_hash = sha256(model_file.read_bytes()).hexdigest()
if actual_hash != expected_hash:
    raise SystemExit(f"Unexpected chinese-wav2vec2 model hash: {actual_hash}")
print("[moshimo-box] chinese-wav2vec2-base verified")
PYMS

"${PYTHON_BIN}" - \
  "${ROOT_DIR}" "${SOURCE_DIR}" "${BASE_DIR}" "${VIDEO_MODEL_ROOT}" "${WAV2VEC_DIR}" <<'PYCHECK'
from pathlib import Path
import subprocess
import sys

root = Path(sys.argv[1])
source = Path(sys.argv[2])
base = Path(sys.argv[3])
model_root = Path(sys.argv[4])
wav2vec = Path(sys.argv[5])
required = [
    source / "config" / "config.yaml",
    base / "config.json",
    base / "diffusion_pytorch_model.safetensors",
    base / "Wan2.1_VAE.pth",
    base / "models_t5_umt5-xxl-enc-bf16.pth",
    base / "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
    model_root / "echomimicv3-flash-pro" / "diffusion_pytorch_model.safetensors",
    wav2vec / "config.json",
    wav2vec / "model.safetensors",
    root / "workers" / "runners" / "echomimic_v3_flash.py",
]
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise SystemExit("EchoMimicV3 installation is incomplete: " + ", ".join(missing))
subprocess.run(
    [sys.executable, str(root / "workers" / "runners" / "echomimic_v3_flash.py"), "--help"],
    cwd=source,
    env={**__import__("os").environ, "PYTHONPATH": str(source)},
    check=True,
    stdout=subprocess.DEVNULL,
)
subprocess.run(
    [sys.executable, "-m", "backend.app.workers.runtime", "--help"],
    cwd=root,
    env={**__import__("os").environ, "PYTHONPATH": str(root)},
    check=True,
    stdout=subprocess.DEVNULL,
)
print("[moshimo-box] EchoMimicV3 runner and worker runtime import successfully")
PYCHECK

echo "[moshimo-box] EchoMimicV3 Flash installed."
echo "[moshimo-box] Run scripts/activate-generation-models.sh after all generation models are installed."
