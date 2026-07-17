from __future__ import annotations

import shutil
import time
from pathlib import Path

import pytest
import yaml

from backend.app.config import ConfigManager, ROOT_DIR


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    shutil.copytree(ROOT_DIR / "config", tmp_path / "config")
    (tmp_path / "config" / "model-catalog.local.yaml").unlink(missing_ok=True)
    (tmp_path / "data" / "runtime").mkdir(parents=True)
    (tmp_path / "data" / "sessions").mkdir(parents=True)
    (tmp_path / "data" / "metrics").mkdir(parents=True)
    (tmp_path / "logs").mkdir()
    (tmp_path / "models").mkdir()
    return tmp_path


@pytest.fixture
def fast_config(project_root: Path) -> ConfigManager:
    local_path = project_root / "config" / "local.yaml"
    local_path.write_text(
        yaml.safe_dump(
            {
                "app": {"debug_mode": True},
                "pipeline": {"stub_step_delay_seconds": 0.01},
                "worker_runtime": {"process_isolation_enabled": False},
                "capture": {
                    "stale_check_interval_seconds": 1,
                    "stale_session_seconds": 30,
                },
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    return ConfigManager(project_root, environ={"MOSHIMO_TEST": "1"})


def wait_until_ready(client, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        preparation = client.get("/api/runtime/status").json()["preparation"]
        if preparation["state"] == "ready":
            return
        time.sleep(0.02)
    raise AssertionError("worker preparation did not become ready")
