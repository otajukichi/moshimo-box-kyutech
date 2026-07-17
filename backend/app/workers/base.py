from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable

from ..schemas import (
    SCHEMA_VERSION,
    WorkerHealth,
    WorkerModelSpec,
    WorkerProgressEvent,
    WorkerRequest,
    WorkerResult,
    WorkerRole,
    WorkerMetrics,
)


ProgressCallback = Callable[[WorkerProgressEvent], Awaitable[None]]


class WorkerAdapter(ABC):
    role: WorkerRole

    @abstractmethod
    async def load(self, model: WorkerModelSpec) -> WorkerHealth:
        raise NotImplementedError

    @abstractmethod
    async def healthcheck(self) -> WorkerHealth:
        raise NotImplementedError

    @abstractmethod
    async def run(
        self,
        request: WorkerRequest,
        progress: ProgressCallback | None = None,
    ) -> WorkerResult:
        raise NotImplementedError

    @abstractmethod
    async def cancel(self, request_id: str) -> None:
        raise NotImplementedError

    @abstractmethod
    async def unload(self) -> None:
        raise NotImplementedError


def create_stub_worker(role: WorkerRole) -> WorkerAdapter:
    return StubWorker(role)


class StubWorker(WorkerAdapter):
    def __init__(self, role: WorkerRole) -> None:
        self.role = role
        self._model: WorkerModelSpec | None = None
        self._cancelled: set[str] = set()

    async def load(self, model: WorkerModelSpec) -> WorkerHealth:
        if model.schema_version != SCHEMA_VERSION or model.worker != self.role:
            raise ValueError("worker_model_schema_mismatch")
        await asyncio.sleep(0.04)
        self._model = model
        return await self.healthcheck()

    async def healthcheck(self) -> WorkerHealth:
        return WorkerHealth(
            worker=self.role,
            loaded=self._model is not None,
            ready=self._model is not None,
            backend=self._model.backend if self._model else None,
            model_id=self._model.model_id if self._model else None,
        )

    async def run(
        self,
        request: WorkerRequest,
        progress: ProgressCallback | None = None,
    ) -> WorkerResult:
        if request.schema_version != SCHEMA_VERSION:
            raise ValueError("worker_request_schema_mismatch")
        if self._model is None:
            raise RuntimeError("worker_not_loaded")
        if request.worker != self.role:
            raise ValueError("worker_role_mismatch")

        started = time.perf_counter()
        delay = float(request.metadata.get("stub_delay_seconds", 0.2))
        for fraction, message in (
            (0.15, "入力を確認しています"),
            (0.55, "未接続スタブを実行しています"),
            (1.0, "スタブ処理が完了しました"),
        ):
            if request.request_id in self._cancelled:
                self._cancelled.discard(request.request_id)
                raise asyncio.CancelledError
            await asyncio.sleep(max(0.01, delay / 3))
            if progress:
                await progress(
                    WorkerProgressEvent(
                        request_id=request.request_id,
                        worker=self.role,
                        progress=fraction,
                        message=message,
                    )
                )

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return WorkerResult(
            request_id=request.request_id,
            worker=self.role,
            backend=self._model.backend,
            model_id=self._model.model_id,
            model_revision=self._model.model_revision,
            implemented=False,
            metadata={
                "session_id": request.session_id,
                "message": "このワーカーは未接続です",
            },
            metrics=WorkerMetrics(processing_time_ms=elapsed_ms),
        )

    async def cancel(self, request_id: str) -> None:
        self._cancelled.add(request_id)

    async def unload(self) -> None:
        await asyncio.sleep(0.02)
        self._model = None
        self._cancelled.clear()
