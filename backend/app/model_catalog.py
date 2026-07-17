from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import yaml

from .schemas import (
    ModelCatalog,
    ModelCatalogEntry,
    ModelOption,
    QualityProfile,
    SCHEMA_VERSION,
    WorkerModelSpec,
    WorkerRole,
)


def _read_yaml(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError(f"モデルカタログのルートはオブジェクトである必要があります: {path}")
    return value


def _deep_update(base: dict[str, Any], patch: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in patch.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            base[key] = _deep_update(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def _merge_catalog(base: dict[str, Any], local: dict[str, Any]) -> dict[str, Any]:
    if not local:
        return base
    merged = copy.deepcopy(base)
    if local.get("schema_version") not in {None, merged.get("schema_version")}:
        raise ValueError("共通カタログとローカルカタログのschema_versionが一致しません")

    if isinstance(local.get("profiles"), Mapping):
        merged["profiles"] = {
            key: copy.deepcopy(value)
            for key, value in merged.get("profiles", {}).items()
        }
        _deep_update(merged["profiles"], local["profiles"])

    entries = list(merged.get("models", []))
    positions = {entry.get("id"): index for index, entry in enumerate(entries)}
    for patch in local.get("models", []):
        if not isinstance(patch, dict) or not patch.get("id"):
            raise ValueError("ローカルモデル定義にはidが必要です")
        model_id = patch["id"]
        if model_id in positions:
            index = positions[model_id]
            entries[index] = _deep_update(copy.deepcopy(entries[index]), patch)
        else:
            positions[model_id] = len(entries)
            entries.append(copy.deepcopy(patch))
    merged["models"] = entries
    return merged


class ModelCatalogRepository:
    def __init__(
        self,
        source_path: Path,
        model_root: Path,
        local_source_path: Path | None = None,
        environment_root: Path | None = None,
    ) -> None:
        self.source_path = source_path
        self.local_source_path = local_source_path
        self.model_root = model_root
        self.environment_root = environment_root or source_path.parent.parent / "env"
        self.catalog = self._load()
        self._entries = {entry.id: entry for entry in self.catalog.models}

    def _load(self) -> ModelCatalog:
        raw = _merge_catalog(
            _read_yaml(self.source_path),
            _read_yaml(self.local_source_path),
        )
        catalog = ModelCatalog.model_validate(raw)
        if catalog.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"未対応のモデルカタログschema_versionです: {catalog.schema_version}"
            )
        return catalog

    def reload(self) -> ModelCatalog:
        self.catalog = self._load()
        self._entries = {entry.id: entry for entry in self.catalog.models}
        return self.catalog

    def entry(self, catalog_id: str) -> ModelCatalogEntry:
        try:
            return self._entries[catalog_id]
        except KeyError as exc:
            raise ValueError(f"モデルカタログに存在しません: {catalog_id}") from exc

    def is_available(self, entry: ModelCatalogEntry) -> bool:
        if not entry.installed or not entry.validated:
            return False
        if entry.last_healthcheck != "passed":
            return False
        if entry.is_stub:
            return True
        if entry.model_path:
            path = Path(entry.model_path).expanduser()
            path = path if path.is_absolute() else self.model_root / path
            if not path.exists():
                return False
        return self.python_bin(entry).exists()

    def python_bin(self, entry: ModelCatalogEntry) -> Path:
        path = Path(entry.python_bin).expanduser()
        return path if path.is_absolute() else (self.environment_root / path).resolve()

    def options(self) -> list[ModelOption]:
        return [
            ModelOption(
                id=entry.id,
                label=entry.label,
                description=entry.description,
                roles=entry.roles,
                backend=entry.backend,
                model_id=entry.model_id,
                revision=entry.revision,
                dtype=entry.dtype,
                quantization=entry.quantization,
                device=entry.device,
                is_stub=entry.is_stub,
            )
            for entry in self.catalog.models
            if self.is_available(entry)
        ]

    def profile_models(
        self,
        profile: QualityProfile,
    ) -> dict[WorkerRole, str]:
        if profile == QualityProfile.CUSTOM:
            profile = QualityProfile.BALANCED
        try:
            return dict(self.catalog.profiles[profile])
        except KeyError as exc:
            raise ValueError(f"モデルプリセットが存在しません: {profile}") from exc

    def validate_selection(self, role: WorkerRole, catalog_id: str) -> None:
        entry = self.entry(catalog_id)
        if role not in entry.roles:
            raise ValueError(f"{catalog_id} は {role} では使用できません")
        if not self.is_available(entry):
            raise ValueError(f"{catalog_id} はインストール・検証済みではありません")

    def spec(self, role: WorkerRole, catalog_id: str) -> WorkerModelSpec:
        self.validate_selection(role, catalog_id)
        entry = self.entry(catalog_id)
        model_path: str | None = None
        if entry.model_path:
            path = Path(entry.model_path).expanduser()
            path = path if path.is_absolute() else self.model_root / path
            model_path = str(path.resolve())
        parameters = {
            key: self._resolve_parameter_path(key, value)
            for key, value in entry.parameters.items()
        }
        return WorkerModelSpec(
            worker=role,
            backend=entry.backend,
            catalog_id=entry.id,
            model_id=entry.model_id,
            model_revision=entry.revision,
            dtype=entry.dtype,
            quantization=entry.quantization,
            device=entry.device,
            adapter_entrypoint=entry.adapter_entrypoint,
            model_path=model_path,
            parameters=parameters,
            timeout_seconds=entry.timeout_seconds,
            fallback_model_id=entry.fallback_model_id,
        )

    def _resolve_parameter_path(self, key: str, value: Any) -> Any:
        if isinstance(value, dict):
            return {
                child_key: self._resolve_parameter_path(child_key, child_value)
                for child_key, child_value in value.items()
            }
        if isinstance(value, list):
            return [self._resolve_parameter_path(key, item) for item in value]
        if not isinstance(value, str) or not key.endswith(("_path", "_dir")):
            return value
        path = Path(value).expanduser()
        path = path if path.is_absolute() else self.model_root / path
        return str(path.resolve())
