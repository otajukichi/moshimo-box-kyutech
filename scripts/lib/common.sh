#!/usr/bin/env bash

# Shared paths for setup, model installers, tests, and app startup.
# A clone under <workspace>/repositories uses <workspace>/env by default.

if [[ -n "${MOSHIMO_COMMON_LOADED:-}" ]]; then
  return 0
fi
MOSHIMO_COMMON_LOADED=1

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

moshimo_default_env_root() {
  local repository_parent workspace_root
  repository_parent="$(dirname "${ROOT_DIR}")"
  if [[ "$(basename "${repository_parent}")" == "repositories" ]]; then
    workspace_root="$(dirname "${repository_parent}")"
  else
    workspace_root="$(dirname "${ROOT_DIR}")"
  fi
  printf '%s/env/moshimo-box-kyutech\n' "${workspace_root}"
}

ENV_ROOT="${MOSHIMO_ENV_ROOT:-$(moshimo_default_env_root)}"
MODEL_ROOT="${MOSHIMO_MODEL_ROOT:-${ROOT_DIR}/models}"
if [[ "${ENV_ROOT}" != /* ]]; then
  ENV_ROOT="${ROOT_DIR}/${ENV_ROOT}"
fi
if [[ "${MODEL_ROOT}" != /* ]]; then
  MODEL_ROOT="${ROOT_DIR}/${MODEL_ROOT}"
fi

APP_ENV_DIR="${ENV_ROOT}/app"
APP_PYTHON="${APP_ENV_DIR}/bin/python"
APP_NPM="${APP_ENV_DIR}/bin/npm"

if [[ -z "${CONDA_BIN:-}" ]]; then
  CONDA_BIN="$(command -v conda 2>/dev/null || true)"
  CONDA_BIN="${CONDA_BIN:-/opt/conda/bin/conda}"
fi

# The Python configuration loader uses these values for relative model-catalog
# executables and for an optional shared model directory.
export MOSHIMO_ENV_ROOT="${ENV_ROOT}"
export MOSHIMO_MODEL_ROOT="${MODEL_ROOT}"
export MOSHIMO__STORAGE__ENVIRONMENT_ROOT="${MOSHIMO__STORAGE__ENVIRONMENT_ROOT:-${ENV_ROOT}}"
export MOSHIMO__STORAGE__MODEL_ROOT="${MOSHIMO__STORAGE__MODEL_ROOT:-${MODEL_ROOT}}"

moshimo_require_app_env() {
  if [[ ! -x "${APP_PYTHON}" ]]; then
    echo "[moshimo-box] App environment is missing: ${APP_PYTHON}" >&2
    echo "[moshimo-box] Run: ${ROOT_DIR}/scripts/bootstrap.sh" >&2
    return 1
  fi
}

moshimo_print_paths() {
  echo "[moshimo-box] Project: ${ROOT_DIR}"
  echo "[moshimo-box] Environments: ${ENV_ROOT}"
  echo "[moshimo-box] Models: ${MODEL_ROOT}"
}
