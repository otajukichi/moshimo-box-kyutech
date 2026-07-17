from __future__ import annotations

import json
from pathlib import Path

from backend.app.config import ConfigManager
from backend.app.model_catalog import _merge_catalog
from backend.app.schemas import QualityProfile


def test_config_precedence(project_root: Path) -> None:
    (project_root / "config" / "local.yaml").write_text(
        "staff_defaults:\n  generation_time_limit_seconds: 2100\n",
        encoding="utf-8",
    )
    (project_root / ".env").write_text(
        "MOSHIMO__STAFF_DEFAULTS__GENERATION_TIME_LIMIT_SECONDS=2400\n",
        encoding="utf-8",
    )
    runtime_path = project_root / "data" / "runtime" / "staff-settings.json"
    runtime_path.write_text(
        json.dumps({"generation_time_limit_seconds": 2700}),
        encoding="utf-8",
    )

    manager = ConfigManager(project_root, environ={"MOSHIMO_TEST": "1"})

    assert manager.developer.staff_defaults.generation_time_limit_seconds == 2400
    assert manager.staff.generation_time_limit_seconds == 2700
    assert len(manager.staff.stage_models) == 16


def test_jupyter_proxy_base_path(project_root: Path) -> None:
    manager = ConfigManager(
        project_root,
        environ={
            "JUPYTERHUB_SERVICE_PREFIX": "/user/demo/",
            "PORT": "8789",
        },
    )

    assert manager.developer.app.base_path == "/user/demo/proxy/8789/"


def test_custom_model_selection_is_validated(project_root: Path) -> None:
    manager = ConfigManager(project_root, environ={"MOSHIMO_TEST": "1"})
    custom = manager.staff.model_copy(deep=True)
    custom.quality_profile = "custom"
    custom.stage_models["video_generation_worker"] = "missing-model"

    try:
        manager.validate_staff(custom)
    except ValueError as exc:
        assert "missing-model" in str(exc)
    else:
        raise AssertionError("missing model must be rejected")


def test_custom_profile_is_preserved_after_save_and_reload(project_root: Path) -> None:
    manager = ConfigManager(project_root, environ={"MOSHIMO_TEST": "1"})
    custom = manager.staff.model_copy(deep=True)
    custom.quality_profile = QualityProfile.CUSTOM

    saved = manager.save_staff(custom)
    reloaded = ConfigManager(project_root, environ={"MOSHIMO_TEST": "1"})

    assert saved.quality_profile.value == "custom"
    assert reloaded.staff.quality_profile.value == "custom"
    assert reloaded.staff.stage_models == custom.stage_models


def test_local_model_catalog_overrides_machine_fields(project_root: Path) -> None:
    (project_root / "config" / "model-catalog.local.yaml").write_text(
        """schema_version: \"1.0\"
models:
  - id: \"foundation-stub\"
    checked_at: \"2026-07-16\"
    description: \"local override\"
""",
        encoding="utf-8",
    )

    manager = ConfigManager(project_root, environ={"MOSHIMO_TEST": "1"})
    entry = manager.catalog.entry("foundation-stub")

    assert entry.checked_at == "2026-07-16"
    assert entry.description == "local override"
    assert entry.adapter_entrypoint == "backend.app.workers.base:create_stub_worker"


def test_relative_worker_python_uses_environment_root(project_root: Path) -> None:
    environment_root = project_root / "worker-envs"
    expected = environment_root / "app" / "bin" / "python"
    manager = ConfigManager(
        project_root,
        environ={
            "MOSHIMO__STORAGE__ENVIRONMENT_ROOT": str(environment_root),
        },
    )
    entry = manager.catalog.entry("foundation-stub")

    assert manager.catalog.python_bin(entry) == expected


def test_profile_overrides_do_not_leak_through_yaml_aliases() -> None:
    shared_profile = {"video_generation_worker": "foundation-stub"}
    base = {
        "schema_version": "1.0",
        "profiles": {
            "fast": shared_profile,
            "balanced": shared_profile,
            "quality": shared_profile,
        },
        "models": [],
    }
    local = {
        "schema_version": "1.0",
        "profiles": {
            "fast": {"video_generation_worker": "fast-video"},
            "balanced": {"video_generation_worker": "balanced-video"},
            "quality": {"video_generation_worker": "quality-video"},
        },
    }

    merged = _merge_catalog(base, local)
