#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"

TARGET="${1:-balanced}"
case "${TARGET}" in
  core|fast|balanced|full) ;;
  *)
    echo "Usage: ./scripts/install-models.sh [core|fast|balanced|full]" >&2
    echo "  core: interview ASR and LLM only" >&2
    echo "  fast: core, Fish TTS, FLUX 4B, MuseTalk" >&2
    echo "  balanced: fast, Qwen3-VL 8B, FLUX 9B (recommended)" >&2
    echo "  full: balanced plus EchoMimicV3" >&2
    exit 2
    ;;
esac

moshimo_require_app_env
moshimo_print_paths

if [[ "${TARGET}" != "core" ]]; then
  if ! "${APP_PYTHON}" -c     'from huggingface_hub import get_token; raise SystemExit(0 if get_token() else 1)'     >/dev/null 2>&1; then
    echo "[moshimo-box] Hugging Face login is required for the generation stack." >&2
    echo "[moshimo-box] Accept the gated model terms, then run:" >&2
    echo "  ./scripts/huggingface-login.sh" >&2
    exit 3
  fi
fi

"${ROOT_DIR}/scripts/install-asr-kotoba.sh"
"${ROOT_DIR}/scripts/install-interview-llm-qwen3.sh"

if [[ "${TARGET}" != "core" ]]; then
  "${ROOT_DIR}/scripts/install-generation-models.sh"
  "${ROOT_DIR}/scripts/install-musetalk.sh"
fi

if [[ "${TARGET}" == "balanced" || "${TARGET}" == "full" ]]; then
  "${ROOT_DIR}/scripts/install-quality-models.sh"
fi

if [[ "${TARGET}" == "full" ]]; then
  "${ROOT_DIR}/scripts/install-video-model-echomimic.sh"
fi

"${ROOT_DIR}/scripts/activate-installed-models.sh"

echo
echo "[moshimo-box] Model installation completed: ${TARGET}"
echo "[moshimo-box] Verify with: ./scripts/doctor.sh"
