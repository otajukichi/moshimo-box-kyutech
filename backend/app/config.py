from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import yaml
from dotenv import dotenv_values
from pydantic import BaseModel, Field

from .model_catalog import ModelCatalogRepository
from .schemas import (
    QualityProfile,
    Rarity,
    StaffLimits,
    StaffSettings,
    WorkerRole,
)


ROOT_DIR = Path(__file__).resolve().parents[2]


class AppSection(BaseModel):
    name: str
    public_base_url: str
    base_path: str = "/"
    debug_mode: bool = False


class ServerSection(BaseModel):
    host: str = "127.0.0.1"
    default_port: int = 8789


class StorageSection(BaseModel):
    session_root: str
    staff_settings_path: str
    metrics_db_path: str
    model_root: str
    log_root: str
    environment_root: str | None = None


class EpisodesSection(BaseModel):
    source_dir: str
    effects_path: str
    rarity_weights: dict[Rarity, float]


class ModelCatalogSection(BaseModel):
    source_path: str
    local_source_path: str | None = None


class PipelineSection(BaseModel):
    stub_step_delay_seconds: float = Field(ge=0)
    worker_restart_count: int = Field(default=1, ge=0)
    json_correction_retry_count: int = Field(default=1, ge=0)
    gpu_release_wait_seconds: float = Field(default=1.0, ge=0)
    cpu_memory_warning_mb: int = Field(default=65536, gt=0)
    cpu_memory_hard_limit_mb: int = Field(default=98304, gt=0)


class CaptureSection(BaseModel):
    video_chunk_seconds: int = Field(default=5, gt=0)
    silence_seconds: float = Field(default=1.8, gt=0)
    speech_start_threshold: float = Field(default=0.025, gt=0)
    response_max_seconds: int = Field(default=30, gt=0)
    upload_retry_count: int = Field(default=3, ge=0)
    browser_queue_limit_mb: int = Field(default=200, gt=0)
    max_chunk_size_mb: int = Field(default=64, gt=0)
    finalize_timeout_seconds: int = Field(default=15, gt=0)
    stale_session_seconds: int = Field(default=3600, gt=0)
    stale_check_interval_seconds: int = Field(default=60, gt=0)
    camera_stable_seconds: float = Field(default=1.5, gt=0)
    brightness_min: int = Field(default=45, ge=0, le=255)
    brightness_max: int = Field(default=220, ge=0, le=255)
    debug_short_answer_count: int = Field(default=2, gt=0)
    debug_short_time_limit_seconds: int = Field(default=45, gt=0)


class WorkerRuntimeSection(BaseModel):
    process_isolation_enabled: bool = True
    host: str = "127.0.0.1"
    startup_timeout_seconds: int = Field(default=20, gt=0)
    request_timeout_seconds: int = Field(default=120, gt=0)
    shutdown_timeout_seconds: int = Field(default=8, gt=0)
    auth_header: str = "X-Moshimo-Worker-Key"


class LoggingSection(BaseModel):
    level: str = "INFO"
    log_transcripts: bool = False
    log_prompts: bool = False
    log_media_paths: bool = False


class DeveloperConfig(BaseModel):
    app: AppSection
    server: ServerSection
    storage: StorageSection
    episodes: EpisodesSection
    model_catalog: ModelCatalogSection
    pipeline: PipelineSection
    capture: CaptureSection
    worker_runtime: WorkerRuntimeSection
    logging: LoggingSection
    staff_defaults: StaffSettings
    staff_limits: StaffLimits


def normalize_base_path(value: str | None) -> str:
    if not value:
        return "/"
    result = value if value.startswith("/") else f"/{value}"
    return result if result.endswith("/") else f"{result}/"


def _deep_update(base: dict[str, Any], patch: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in patch.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            base[key] = _deep_update(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError(f"設定ファイルのルートはオブジェクトである必要があります: {path}")
    return value


def _set_nested(target: dict[str, Any], path: list[str], value: Any) -> None:
    cursor = target
    for key in path[:-1]:
        next_value = cursor.get(key)
        if not isinstance(next_value, dict):
            next_value = {}
            cursor[key] = next_value
        cursor = next_value
    cursor[path[-1]] = value


def _environment_patch(root_dir: Path, environ: Mapping[str, str] | None) -> dict[str, Any]:
    combined: dict[str, str] = {
        key: value
        for key, value in dotenv_values(root_dir / ".env").items()
        if value is not None
    }
    combined.update(dict(environ if environ is not None else os.environ))

    patch: dict[str, Any] = {}
    prefix = "MOSHIMO__"
    for key, raw_value in combined.items():
        if not key.startswith(prefix):
            continue
        path = [part.lower() for part in key[len(prefix) :].split("__") if part]
        if path:
            _set_nested(patch, path, yaml.safe_load(raw_value))
    return patch


class ConfigManager:
    precedence = ["config/default.yaml", "config/local.yaml", ".env", "staff-settings.json"]

    def __init__(
        self,
        root_dir: Path = ROOT_DIR,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self.root_dir = root_dir.resolve()
        self.environ = environ
        self.developer = self._load_developer()
        self.catalog = ModelCatalogRepository(
            self.model_catalog_path,
            self.model_root,
            self.model_catalog_local_path,
            self.environment_root,
        )
        self.staff = self._load_staff()

    def resolve_path(self, value: str) -> Path:
        path = Path(value).expanduser()
        return path if path.is_absolute() else (self.root_dir / path).resolve()

    @property
    def session_root(self) -> Path:
        return self.resolve_path(self.developer.storage.session_root)

    @property
    def staff_settings_path(self) -> Path:
        return self.resolve_path(self.developer.storage.staff_settings_path)

    @property
    def metrics_db_path(self) -> Path:
        return self.resolve_path(self.developer.storage.metrics_db_path)

    @property
    def episode_dir(self) -> Path:
        return self.resolve_path(self.developer.episodes.source_dir)

    @property
    def effects_path(self) -> Path:
        return self.resolve_path(self.developer.episodes.effects_path)

    @property
    def model_catalog_path(self) -> Path:
        return self.resolve_path(self.developer.model_catalog.source_path)



    @property
    def model_catalog_local_path(self) -> Path | None:
        value = self.developer.model_catalog.local_source_path
        return self.resolve_path(value) if value else None

    @property
    def model_root(self) -> Path:
        return self.resolve_path(self.developer.storage.model_root)

    @property
    def environment_root(self) -> Path:
        configured = self.developer.storage.environment_root
        if configured:
            return self.resolve_path(configured)

        runtime_environment = self.environ if self.environ is not None else os.environ
        explicit = runtime_environment.get("MOSHIMO_ENV_ROOT")
        if explicit:
            return self.resolve_path(explicit)

        current_environment = Path(sys.executable).resolve().parent.parent
        if current_environment.name == "app":
            return current_environment.parent

        repository_parent = self.root_dir.parent
        workspace_root = (
            repository_parent.parent
            if repository_parent.name == "repositories"
            else repository_parent
        )
        return (workspace_root / "env" / "moshimo-box-kyutech").resolve()

    @property
    def log_root(self) -> Path:
        return self.resolve_path(self.developer.storage.log_root)

    def _load_developer(self) -> DeveloperConfig:
        raw = _load_yaml(self.root_dir / "config" / "default.yaml")
        raw = _deep_update(raw, _load_yaml(self.root_dir / "config" / "local.yaml"))
        raw = _deep_update(raw, _environment_patch(self.root_dir, self.environ))

        runtime_environment = self.environ if self.environ is not None else os.environ
        service_prefix = runtime_environment.get("JUPYTERHUB_SERVICE_PREFIX", "")
        port = runtime_environment.get("PORT", "")
        if service_prefix and port:
            raw.setdefault("app", {})["base_path"] = normalize_base_path(
                f"{service_prefix.rstrip('/')}/proxy/{port}/"
            )
        else:
            raw.setdefault("app", {})["base_path"] = normalize_base_path(
                raw.get("app", {}).get("base_path")
            )
        return DeveloperConfig.model_validate(raw)

    def _migrate_staff(self, raw: dict[str, Any]) -> dict[str, Any]:
        migrated = dict(raw)
        if "demo_time_limit_seconds" in migrated and "generation_time_limit_seconds" not in migrated:
            migrated["generation_time_limit_seconds"] = migrated["demo_time_limit_seconds"]
        if "generation_quality" in migrated and "quality_profile" not in migrated:
            migrated["quality_profile"] = (
                "quality" if migrated["generation_quality"] == "high" else "balanced"
            )
        return migrated

    def _load_staff(self) -> StaffSettings:
        raw = self.developer.staff_defaults.model_dump(mode="json")
        path = self.staff_settings_path
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                saved = self._migrate_staff(json.load(handle))
            raw = _deep_update(raw, saved)
        settings = StaffSettings.model_validate(raw)
        return self.validate_staff(self.normalize_stage_models(settings))

    def normalize_stage_models(self, settings: StaffSettings) -> StaffSettings:
        normalized = settings.model_copy(deep=True)
        if normalized.quality_profile != QualityProfile.CUSTOM:
            normalized.stage_models = self.catalog.profile_models(normalized.quality_profile)
        else:
            defaults = self.catalog.profile_models(QualityProfile.BALANCED)
            normalized.stage_models = {
                role: normalized.stage_models.get(role, defaults[role])
                for role in WorkerRole
            }
        return normalized

    def validate_staff(self, settings: StaffSettings) -> StaffSettings:
        settings = self.normalize_stage_models(settings)
        limits = self.developer.staff_limits
        checks = {
            "generation_time_limit_seconds": limits.generation_time_limit_seconds,
            "target_transcript_chars": limits.target_transcript_chars,
            "minimum_transcript_chars": limits.minimum_transcript_chars,
            "conversation_time_limit_seconds": limits.conversation_time_limit_seconds,
        }
        for field_name, limit in checks.items():
            value = getattr(settings, field_name)
            if not limit.min <= value <= limit.max:
                raise ValueError(f"{field_name} は {limit.min} から {limit.max} の範囲で指定してください")
        for role in WorkerRole:
            catalog_id = settings.stage_models.get(role)
            if not catalog_id:
                raise ValueError(f"{role} のモデルが選択されていません")
            self.catalog.validate_selection(role, catalog_id)
        return settings

    def save_staff(self, settings: StaffSettings) -> StaffSettings:
        validated = self.validate_staff(settings)
        path = self.staff_settings_path
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(validated.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
        self.staff = validated
        return validated

    def reset_staff(self) -> StaffSettings:
        path = self.staff_settings_path
        if path.exists():
            path.unlink()
        self.staff = self.validate_staff(
            self.developer.staff_defaults.model_copy(deep=True)
        )
        return self.staff
