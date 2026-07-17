#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"
moshimo_require_app_env

GEN_PYTHON="${ENV_ROOT}/generation/bin/python"
FLUX_PYTHON="${ENV_ROOT}/flux2-klein/bin/python"
MUSETALK_PYTHON="${ENV_ROOT}/musetalk/bin/python"
VIDEO_PYTHON="${ENV_ROOT}/echomimic-v3/bin/python"

"${APP_PYTHON}" - \
  "${ROOT_DIR}" \
  "${MODEL_ROOT}" \
  "${APP_PYTHON}" \
  "${GEN_PYTHON}" \
  "${FLUX_PYTHON}" \
  "${MUSETALK_PYTHON}" \
  "${VIDEO_PYTHON}" \
  "${GPT_OSS_PYTHON}" <<'PYACTIVATE'
from __future__ import annotations

from datetime import date
import os
from pathlib import Path
import shutil
import subprocess
import sys

import yaml

root = Path(sys.argv[1])
model_root = Path(sys.argv[2])
app_python = Path(sys.argv[3])
generation_python = Path(sys.argv[4])
flux_python = Path(sys.argv[5])
musetalk_python = Path(sys.argv[6])
video_python = Path(sys.argv[7])
gpt_oss_python = Path(sys.argv[8])


def paths_exist(*paths: Path) -> bool:
    return all(path.exists() for path in paths)


def import_ready(python: Path, statement: str, *, cwd: Path = root) -> bool:
    if not python.is_file():
        return False
    try:
        completed = subprocess.run(
            [str(python), "-c", statement],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"[moshimo-box] Import check failed: {python}: {exc}", file=sys.stderr)
        return False
    if completed.returncode != 0:
        detail = completed.stderr.strip().splitlines()
        tail = detail[-1] if detail else "unknown import error"
        print(f"[moshimo-box] Import check failed: {python}: {tail}", file=sys.stderr)
        return False
    return True


asr_dir = model_root / "asr" / "kotoba-whisper-v2.0-faster"
qwen4_dir = model_root / "llm" / "qwen3-4b-instruct-2507"
qwen_vl_dir = model_root / "llm" / "qwen3-vl-8b-instruct"
gpt_oss_dir = model_root / "llm" / "gpt-oss-20b"
fish_dir = model_root / "tts" / "fish-s2-pro"
flux4_dir = model_root / "image" / "flux2-klein-4b"
flux9_dir = model_root / "image" / "flux2-klein-9b"
muse_source = model_root / "musetalk-source"
echo_source = model_root / "echomimic-v3-source"
echo_dir = model_root / "video" / "echomimic-v3-flash"

availability = {
    "kotoba-whisper-v2-faster-fp16": (
        paths_exist(
            asr_dir / "config.json",
            asr_dir / "model.bin",
            asr_dir / "tokenizer.json",
        )
        and import_ready(app_python, "from faster_whisper import WhisperModel")
    ),
    "qwen3-4b-instruct-2507-bf16": (
        paths_exist(
            qwen4_dir / "config.json",
            qwen4_dir / "model.safetensors.index.json",
            qwen4_dir / "tokenizer.json",
        )
        and import_ready(
            app_python,
            "from transformers import AutoModelForCausalLM, AutoTokenizer",
        )
    ),
    "qwen3-vl-8b-instruct-bf16": (
        paths_exist(
            qwen_vl_dir / "config.json",
            qwen_vl_dir / "preprocessor_config.json",
            qwen_vl_dir / "model.safetensors.index.json",
        )
        and import_ready(
            app_python,
            "from transformers import AutoProcessor, Qwen3VLForConditionalGeneration",
        )
    ),
    "gpt-oss-20b-mxfp4-vllm": (
        paths_exist(
            gpt_oss_dir / "config.json",
            gpt_oss_dir / "model.safetensors.index.json",
            gpt_oss_dir / "tokenizer.json",
        )
        and len(list(gpt_oss_dir.glob("model-*.safetensors"))) == 3
        and import_ready(
            gpt_oss_python,
            "from vllm import LLM; from openai_harmony import load_harmony_encoding",
        )
    ),
    "fish-s2-pro-bf16": (
        paths_exist(
            fish_dir / "config.json",
            fish_dir / "codec.pth",
            fish_dir / "model.safetensors.index.json",
        )
        and import_ready(
            generation_python,
            "from fish_speech.inference_engine import TTSInferenceEngine",
        )
    ),
    "flux2-klein-4b-bf16": (
        paths_exist(flux4_dir / "model_index.json")
        and import_ready(flux_python, "from diffusers import Flux2KleinPipeline")
    ),
    "flux2-klein-9b-bf16": (
        paths_exist(
            flux9_dir / "model_index.json",
            flux9_dir / "transformer",
            flux9_dir / "text_encoder",
            flux9_dir / "vae",
        )
        and import_ready(flux_python, "from diffusers import Flux2KleinPipeline")
    ),
    "musetalk-1.5-fp16": (
        paths_exist(
            muse_source / "models" / "musetalkV15" / "musetalk.json",
            muse_source / "models" / "musetalkV15" / "unet.pth",
            muse_source / "models" / "sd-vae" / "config.json",
            muse_source / "models" / "whisper" / "pytorch_model.bin",
            muse_source / "models" / "dwpose" / "dw-ll_ucoco_384.pth",
            muse_source / "models" / "face-parse-bisent" / "79999_iter.pth",
        )
        and import_ready(
            musetalk_python,
            "import torch, cv2, diffusers, transformers, mmcv, mmpose",
            cwd=muse_source,
        )
    ),
    "echomimic-v3-flash-bf16": (
        paths_exist(
            echo_source / "config" / "config.yaml",
            echo_dir / "Wan2.1-Fun-V1.1-1.3B-InP" / "config.json",
            echo_dir / "echomimicv3-flash-pro" / "diffusion_pytorch_model.safetensors",
            echo_dir / "chinese-wav2vec2-base" / "model.safetensors",
        )
        and video_python.is_file()
        and (root / "workers" / "runners" / "echomimic_v3_flash.py").is_file()
    ),
}

base_catalog = yaml.safe_load(
    (root / "config" / "model-catalog.yaml").read_text(encoding="utf-8")
)
all_roles = list(base_catalog["profiles"]["balanced"])

local_path = root / "config" / "model-catalog.local.yaml"
local = yaml.safe_load(local_path.read_text(encoding="utf-8")) if local_path.exists() else {}
local = local or {}
local["schema_version"] = "1.0"

utility_roles = {
    "final_asr_worker",
    "interview_summary_worker",
    "episode_selector",
    "reference_frame_selector",
    "voice_reference_selector",
    "lip_sync_worker",
    "video_postprocess_worker",
}
profiles: dict[str, dict[str, str]] = {}
for profile_name in ("fast", "balanced", "quality"):
    selected = {role: "foundation-stub" for role in all_roles}
    selected.update({role: "pipeline-utilities-v1" for role in utility_roles})

    if availability["kotoba-whisper-v2-faster-fp16"]:
        selected["streaming_asr_worker"] = "kotoba-whisper-v2-faster-fp16"

    default_interview_model = (
        "qwen3-4b-instruct-2507-bf16"
        if availability["qwen3-4b-instruct-2507-bf16"]
        else (
            "qwen3-vl-8b-instruct-bf16"
            if availability["qwen3-vl-8b-instruct-bf16"]
            else "foundation-stub"
        )
    )
    interview_model = (
        "gpt-oss-20b-mxfp4-vllm"
        if profile_name == "quality"
        and availability["gpt-oss-20b-mxfp4-vllm"]
        else default_interview_model
    )
    selected["interview_llm_worker"] = interview_model

    if profile_name == "fast":
        design_model = interview_model
        image_model = (
            "flux2-klein-4b-bf16"
            if availability["flux2-klein-4b-bf16"]
            else "foundation-stub"
        )
    elif (
        profile_name == "quality"
        and availability["gpt-oss-20b-mxfp4-vllm"]
    ):
        design_model = "gpt-oss-20b-mxfp4-vllm"
        image_model = (
            "flux2-klein-9b-bf16"
            if availability["flux2-klein-9b-bf16"]
            else (
                "flux2-klein-4b-bf16"
                if availability["flux2-klein-4b-bf16"]
                else "foundation-stub"
            )
        )
    else:
        design_model = (
            "qwen3-vl-8b-instruct-bf16"
            if availability["qwen3-vl-8b-instruct-bf16"]
            else interview_model
        )
        image_model = (
            "flux2-klein-9b-bf16"
            if availability["flux2-klein-9b-bf16"]
            else (
                "flux2-klein-4b-bf16"
                if availability["flux2-klein-4b-bf16"]
                else "foundation-stub"
            )
        )
    selected["script_design_llm_worker"] = design_model
    selected["script_safety_review_worker"] = design_model
    selected["image_generation_worker"] = image_model

    if availability["fish-s2-pro-bf16"]:
        selected["voice_clone_tts_worker"] = "fish-s2-pro-bf16"

    if profile_name == "quality" and availability["echomimic-v3-flash-bf16"]:
        selected["video_generation_worker"] = "echomimic-v3-flash-bf16"
    elif availability["musetalk-1.5-fp16"]:
        selected["video_generation_worker"] = "musetalk-1.5-fp16"
    elif availability["echomimic-v3-flash-bf16"]:
        selected["video_generation_worker"] = "echomimic-v3-flash-bf16"

    profiles[profile_name] = selected

local["profiles"] = profiles
models = local.setdefault("models", [])
positions = {
    item.get("id"): item
    for item in models
    if isinstance(item, dict) and item.get("id")
}
for model_id, ready in availability.items():
    item = positions.get(model_id)
    if item is None:
        item = {"id": model_id}
        models.append(item)
    item.update(
        installed=ready,
        validated=ready,
        last_healthcheck="passed" if ready else "not_run",
        checked_at=date.today().isoformat(),
    )

local_path.write_text(
    yaml.safe_dump(local, allow_unicode=True, sort_keys=False),
    encoding="utf-8",
)

print(f"[moshimo-box] Updated {local_path}")
for model_id, ready in availability.items():
    print(f"[moshimo-box] {'ready' if ready else 'missing'}: {model_id}")
PYACTIVATE
