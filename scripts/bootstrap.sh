#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"

MODE=""
PUBLIC_BASE_URL=""

usage() {
  cat <<'EOF'
Usage: ./scripts/bootstrap.sh [--production|--debug] [--public-base-url URL]

Environment overrides:
  MOSHIMO_ENV_ROOT    Conda environment parent
  MOSHIMO_MODEL_ROOT  Model and download-cache parent
  CONDA_BIN           Conda executable
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --production)
      MODE="production"
      shift
      ;;
    --debug)
      MODE="debug"
      shift
      ;;
    --public-base-url)
      if [[ $# -lt 2 ]]; then
        echo "[moshimo-box] --public-base-url requires a value" >&2
        exit 2
      fi
      PUBLIC_BASE_URL="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[moshimo-box] Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

moshimo_print_paths

if [[ ! -x "${CONDA_BIN}" ]]; then
  echo "[moshimo-box] Conda was not found: ${CONDA_BIN}" >&2
  exit 1
fi

"${ROOT_DIR}/scripts/setup-app-env.sh"

"${APP_PYTHON}" - "${ROOT_DIR}" "${ENV_ROOT}" "${MODEL_ROOT}"   "${MODE}" "${PUBLIC_BASE_URL}" <<'PYCONFIG'
from __future__ import annotations

from pathlib import Path
import sys

import yaml

root = Path(sys.argv[1])
environment_root = Path(sys.argv[2])
model_root = Path(sys.argv[3])
mode = sys.argv[4]
public_base_url = sys.argv[5]

local_path = root / "config" / "local.yaml"
is_new = not local_path.exists()
local = yaml.safe_load(local_path.read_text(encoding="utf-8")) if not is_new else {}
local = local or {}

app = local.setdefault("app", {})
if mode:
    app["debug_mode"] = mode == "debug"
elif is_new:
    app["debug_mode"] = False
if public_base_url:
    app["public_base_url"] = public_base_url.rstrip("/")

storage = local.setdefault("storage", {})
storage.setdefault("environment_root", str(environment_root))
if model_root.resolve() != (root / "models").resolve():
    storage.setdefault("model_root", str(model_root))

local_path.write_text(
    yaml.safe_dump(local, allow_unicode=True, sort_keys=False),
    encoding="utf-8",
)

catalog_path = root / "config" / "model-catalog.local.yaml"
if not catalog_path.exists():
    catalog_path.write_text(
        'schema_version: "1.0"
models: []
',
        encoding="utf-8",
    )

for relative in (
    "data/runtime",
    "data/sessions",
    "data/metrics",
    "logs",
):
    (root / relative).mkdir(parents=True, exist_ok=True)

print(f"[moshimo-box] Local configuration: {local_path}")
print(f"[moshimo-box] Local model catalog: {catalog_path}")
PYCONFIG

echo
echo "[moshimo-box] Bootstrap complete."
echo "[moshimo-box] Next:"
echo "  1. Accept gated model terms in Hugging Face."
echo "  2. Run ./scripts/huggingface-login.sh"
echo "  3. Run ./scripts/install-models.sh balanced"
echo "  4. Run ./scripts/doctor.sh"
