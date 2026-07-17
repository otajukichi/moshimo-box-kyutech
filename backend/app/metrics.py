from __future__ import annotations

import asyncio
import sqlite3
from datetime import datetime
from pathlib import Path

from .schemas import WorkerRequest, WorkerResult


class MetricsStore:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(self._initialize_sync)

    def _initialize_sync(self) -> None:
        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS worker_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    recorded_at TEXT NOT NULL,
                    anonymous_session_id TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    worker_role TEXT NOT NULL,
                    backend TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    model_revision TEXT NOT NULL,
                    dtype TEXT NOT NULL,
                    quantization TEXT NOT NULL,
                    device TEXT NOT NULL,
                    success INTEGER NOT NULL,
                    error_code TEXT,
                    load_time_ms INTEGER NOT NULL,
                    processing_time_ms INTEGER NOT NULL,
                    unload_time_ms INTEGER NOT NULL,
                    peak_vram_mb INTEGER NOT NULL,
                    peak_cpu_memory_mb INTEGER NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_worker_runs_role ON worker_runs(worker_role)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_worker_runs_model ON worker_runs(model_id, model_revision)"
            )

    async def record(
        self,
        request: WorkerRequest,
        result: WorkerResult | None,
        *,
        error_code: str | None = None,
    ) -> None:
        async with self._lock:
            await asyncio.to_thread(
                self._record_sync,
                request,
                result,
                error_code,
            )

    def _record_sync(
        self,
        request: WorkerRequest,
        result: WorkerResult | None,
        error_code: str | None,
    ) -> None:
        metrics = result.metrics if result else None
        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                """
                INSERT INTO worker_runs (
                    recorded_at, anonymous_session_id, request_id, worker_role,
                    backend, model_id, model_revision, dtype, quantization, device,
                    success, error_code, load_time_ms, processing_time_ms,
                    unload_time_ms, peak_vram_mb, peak_cpu_memory_mb
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    request.session_id,
                    request.request_id,
                    request.worker.value,
                    request.model.backend,
                    request.model.model_id,
                    request.model.model_revision,
                    request.model.dtype,
                    request.model.quantization,
                    request.model.device,
                    1 if result and result.error is None else 0,
                    error_code or (result.error.code if result and result.error else None),
                    metrics.load_time_ms if metrics else 0,
                    metrics.processing_time_ms if metrics else 0,
                    metrics.unload_time_ms if metrics else 0,
                    metrics.peak_vram_mb if metrics else 0,
                    metrics.peak_cpu_memory_mb if metrics else 0,
                ),
            )
