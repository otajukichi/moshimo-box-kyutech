from __future__ import annotations

import asyncio
import logging
import os
import secrets
import socket
import sys
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import httpx

from ..config import ConfigManager
from ..metrics import MetricsStore
from ..schemas import (
    ROLE_GROUP,
    WORKER_GROUP_ROLES,
    PreparationGroupStatus,
    PreparationPublic,
    PreparationState,
    SessionRecord,
    StaffSettings,
    WorkerGroup,
    WorkerHealth,
    WorkerLifecycleState,
    WorkerProgressEvent,
    WorkerRequest,
    WorkerResult,
    WorkerRole,
    WorkerStatusPublic,
    utc_now,
)
from .base import WorkerAdapter
from .factory import create_worker_adapter


LOGGER = logging.getLogger(__name__)


class WorkerExecutionError(RuntimeError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        worker: WorkerRole | None = None,
        phase: str | None = None,
        detail: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.worker = worker
        self.phase = phase
        self.detail = detail or message


StatusCallback = Callable[
    [WorkerStatusPublic, WorkerProgressEvent | None],
    Awaitable[None],
]


@dataclass
class WorkerHandle:
    role: WorkerRole
    catalog_id: str
    adapter: WorkerAdapter | None = None
    process: asyncio.subprocess.Process | None = None
    client: httpx.AsyncClient | None = None
    port: int | None = None
    log_handle: BinaryIO | None = None
    log_path: Path | None = None
    log_start_offset: int = 0


class WorkerProcessSupervisor:
    def __init__(self, config: ConfigManager, auth_key: str) -> None:
        self.config = config
        self.auth_key = auth_key
        self.handles: dict[WorkerRole, WorkerHandle] = {}

    @staticmethod
    def _free_port(host: str) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.bind((host, 0))
            return int(server.getsockname()[1])

    @staticmethod
    def _startup_failure_detail(
        handle: WorkerHandle,
        return_code: int | None,
    ) -> str:
        exit_detail = (
            f"exit_code={return_code}"
            if return_code is not None
            else "exit_code=not_available"
        )
        if handle.log_handle:
            try:
                handle.log_handle.flush()
            except (OSError, ValueError):
                pass
        if not handle.log_path:
            return exit_detail
        try:
            size = handle.log_path.stat().st_size
            start = min(handle.log_start_offset, size)
            start = max(start, size - 4096)
            with handle.log_path.open("rb") as log_file:
                log_file.seek(start)
                tail = log_file.read().decode("utf-8", errors="replace").strip()
        except OSError:
            return exit_detail
        return f"{exit_detail}\n{tail[-1800:]}" if tail else exit_detail

    async def start(self, role: WorkerRole, catalog_id: str) -> WorkerHandle:
        existing = self.handles.get(role)
        if existing and existing.catalog_id == catalog_id:
            if existing.adapter is not None:
                return existing
            if existing.process and existing.process.returncode is None:
                return existing
        if existing:
            await self.stop(role)

        if not self.config.developer.worker_runtime.process_isolation_enabled:
            entry = self.config.catalog.entry(catalog_id)
            handle = WorkerHandle(
                role=role,
                catalog_id=catalog_id,
                adapter=create_worker_adapter(role, entry.adapter_entrypoint),
            )
            self.handles[role] = handle
            return handle

        entry = self.config.catalog.entry(catalog_id)
        host = self.config.developer.worker_runtime.host
        port = self._free_port(host)
        python_bin = self.config.catalog.python_bin(entry)
        executable = str(python_bin) if python_bin.exists() else sys.executable
        command = [
            executable,
            *entry.command,
            "--host",
            host,
            "--port",
            str(port),
            "--role",
            role.value,
        ]
        environment = os.environ.copy()
        environment["MOSHIMO_WORKER_KEY"] = self.auth_key
        environment["MOSHIMO_WORKER_AUTH_HEADER"] = (
            self.config.developer.worker_runtime.auth_header
        )
        existing_python_path = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            f"{self.config.root_dir}:{existing_python_path}"
            if existing_python_path
            else str(self.config.root_dir)
        )

        log_path = self.config.log_root / f"worker-{role.value}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_start_offset = log_path.stat().st_size if log_path.exists() else 0
        log_handle = log_path.open("ab")
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=self.config.root_dir,
            env=environment,
            stdout=log_handle,
            stderr=log_handle,
        )
        client = httpx.AsyncClient(
            base_url=f"http://{host}:{port}",
            headers={
                self.config.developer.worker_runtime.auth_header: self.auth_key,
            },
            timeout=self.config.developer.worker_runtime.request_timeout_seconds,
        )
        handle = WorkerHandle(
            role=role,
            catalog_id=catalog_id,
            process=process,
            client=client,
            port=port,
            log_handle=log_handle,
            log_path=log_path,
            log_start_offset=log_start_offset,
        )
        self.handles[role] = handle

        deadline = time.monotonic() + self.config.developer.worker_runtime.startup_timeout_seconds
        while time.monotonic() < deadline:
            if process.returncode is not None:
                detail = self._startup_failure_detail(handle, process.returncode)
                await self.stop(role)
                raise WorkerExecutionError(
                    "worker_process_exited",
                    f"{role.value} のプロセスが起動中に終了しました",
                    worker=role,
                    phase="worker_start",
                    detail=detail,
                )
            try:
                response = await client.get("/health")
                if response.status_code == 200:
                    return handle
            except httpx.HTTPError:
                pass
            await asyncio.sleep(0.1)

        detail = self._startup_failure_detail(handle, process.returncode)
        await self.stop(role)
        raise WorkerExecutionError(
            "worker_start_timeout",
            f"{role.value} の起動確認が時間内に完了しませんでした",
            worker=role,
            phase="worker_start",
            detail=detail,
        )


    @staticmethod
    def _response_error(
        response: httpx.Response,
        role: WorkerRole,
        phase: str,
    ) -> WorkerExecutionError:
        code = "worker_runtime_error"
        message = f"{role.value} returned HTTP {response.status_code}"
        detail_text = message
        error_phase = phase
        try:
            payload = response.json()
            raw_detail = payload.get("detail")
            if isinstance(raw_detail, dict):
                code = str(raw_detail.get("code") or code)
                message = str(raw_detail.get("message") or message)
                detail_text = (
                    f"{raw_detail.get('exception_type', 'worker error')}: "
                    f"{message}"
                )
                error_phase = str(raw_detail.get("phase") or phase)
            elif raw_detail:
                message = str(raw_detail)
                detail_text = message
        except (TypeError, ValueError):
            detail_text = response.text[:1200] or message
        return WorkerExecutionError(
            code,
            message,
            worker=role,
            phase=error_phase,
            detail=detail_text[:1200],
        )

    async def load(self, role: WorkerRole, catalog_id: str) -> tuple[WorkerHealth, int]:
        handle = await self.start(role, catalog_id)
        spec = self.config.catalog.spec(role, catalog_id)
        started = time.perf_counter()
        try:
            if handle.adapter:
                health = await handle.adapter.load(spec)
            else:
                assert handle.client is not None
                response = await handle.client.post(
                    "/load",
                    json=spec.model_dump(mode="json"),
                )
                if not response.is_success:
                    raise self._response_error(response, role, "model_load")
                health = WorkerHealth.model_validate(response.json())
        except WorkerExecutionError:
            raise
        except Exception as exc:
            raise WorkerExecutionError(
                str(getattr(exc, "code", "worker_load_error")),
                f"{role.value} のモデル読み込みに失敗しました",
                worker=role,
                phase=str(getattr(exc, "phase", "model_load")),
                detail=str(exc)[:1200],
            ) from exc
        return health, int((time.perf_counter() - started) * 1000)

    async def healthcheck(self, role: WorkerRole) -> WorkerHealth:
        handle = self.handles[role]
        if handle.adapter:
            return await handle.adapter.healthcheck()
        assert handle.client is not None
        response = await handle.client.get("/health")
        if not response.is_success:
            raise self._response_error(response, role, "healthcheck")
        return WorkerHealth.model_validate(response.json())

    async def _poll_progress(
        self,
        handle: WorkerHandle,
        request: WorkerRequest,
        seen: int,
        progress: Callable[[WorkerProgressEvent], Awaitable[None]] | None,
    ) -> int:
        if progress is None or handle.client is None:
            return seen
        try:
            response = await handle.client.get(f"/progress/{request.request_id}")
            if not response.is_success:
                return seen
            events = response.json().get("events", [])
            for value in events[seen:]:
                await progress(WorkerProgressEvent.model_validate(value))
            return len(events)
        except (httpx.HTTPError, TypeError, ValueError):
            return seen

    async def run(
        self,
        request: WorkerRequest,
        progress: Callable[[WorkerProgressEvent], Awaitable[None]] | None = None,
    ) -> WorkerResult:
        handle = self.handles.get(request.worker)
        if not handle:
            raise WorkerExecutionError(
                "worker_not_started",
                request.worker.value,
                worker=request.worker,
                phase="worker_start",
            )
        timeout_seconds = min(request.deadline_seconds, request.model.timeout_seconds)
        try:
            async with asyncio.timeout(timeout_seconds):
                if handle.adapter:
                    result = await handle.adapter.run(request, progress)
                else:
                    assert handle.client is not None
                    seen = 0
                    run_task = asyncio.create_task(
                        handle.client.post(
                            "/run",
                            json=request.model_dump(mode="json"),
                        )
                    )
                    while not run_task.done():
                        await asyncio.sleep(0.15)
                        seen = await self._poll_progress(
                            handle, request, seen, progress
                        )
                    response = await run_task
                    await self._poll_progress(handle, request, seen, progress)
                    if not response.is_success:
                        raise self._response_error(
                            response, request.worker, "inference"
                        )
                    result = WorkerResult.model_validate(response.json())
        except WorkerExecutionError:
            raise
        except TimeoutError as exc:
            await self.cancel(request.worker, request.request_id)
            raise WorkerExecutionError(
                "worker_timeout",
                request.worker.value,
                worker=request.worker,
                phase="inference",
                detail=f"{timeout_seconds}秒のワーカー制限時間を超過しました",
            ) from exc
        except httpx.HTTPError as exc:
            raise WorkerExecutionError(
                "worker_transport_error",
                request.worker.value,
                worker=request.worker,
                phase="worker_transport",
                detail=str(exc)[:1200],
            ) from exc
        except Exception as exc:
            raise WorkerExecutionError(
                str(getattr(exc, "code", "worker_runtime_error")),
                request.worker.value,
                worker=request.worker,
                phase=str(getattr(exc, "phase", "inference")),
                detail=str(exc)[:1200],
            ) from exc

        handle = self.handles.get(request.worker)
        if handle and handle.process:
            result.metrics.peak_cpu_memory_mb = max(
                result.metrics.peak_cpu_memory_mb,
                self._process_rss_mb(handle.process.pid),
            )
        return result



    @staticmethod
    def _process_rss_mb(pid: int) -> int:
        try:
            for line in Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines():
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) // 1024
        except (OSError, ValueError, IndexError):
            return 0
        return 0

    async def cancel(self, role: WorkerRole, request_id: str) -> None:
        handle = self.handles.get(role)
        if not handle:
            return
        if handle.adapter:
            await handle.adapter.cancel(request_id)
            return
        if handle.client:
            try:
                await handle.client.post(f"/cancel/{request_id}")
            except httpx.HTTPError:
                pass

    async def stop(self, role: WorkerRole) -> int:
        handle = self.handles.pop(role, None)
        if not handle:
            return 0
        started = time.perf_counter()
        if handle.adapter:
            await handle.adapter.unload()
            return int((time.perf_counter() - started) * 1000)

        if handle.client:
            try:
                await handle.client.post("/unload")
            except httpx.HTTPError:
                pass
            await handle.client.aclose()
        if handle.process and handle.process.returncode is None:
            handle.process.terminate()
            try:
                async with asyncio.timeout(
                    self.config.developer.worker_runtime.shutdown_timeout_seconds
                ):
                    await handle.process.wait()
            except TimeoutError:
                handle.process.kill()
                await handle.process.wait()
        if handle.log_handle:
            handle.log_handle.close()
        return int((time.perf_counter() - started) * 1000)

    async def stop_group(self, group: WorkerGroup) -> None:
        for role in WORKER_GROUP_ROLES[group]:
            await self.stop(role)

    async def stop_all(self) -> None:
        for role in list(self.handles):
            await self.stop(role)


class WorkerOrchestrator:
    def __init__(self, config: ConfigManager, metrics: MetricsStore) -> None:
        self.config = config
        self.metrics = metrics
        self.auth_key = secrets.token_urlsafe(32)
        self.supervisor = WorkerProcessSupervisor(config, self.auth_key)
        self.gpu_lock = asyncio.Lock()
        self.preparation_state = PreparationState.STOPPED
        self.preparation_message = "AIワーカーは停止しています"
        self.statuses: dict[WorkerRole, WorkerStatusPublic] = {
            role: WorkerStatusPublic(role=role, group=ROLE_GROUP[role])
            for role in WorkerRole
        }
        self.active_requests: dict[WorkerRole, str] = {}
        self._prepare_lock = asyncio.Lock()

    def status(self) -> PreparationPublic:
        groups = [
            PreparationGroupStatus(
                group=group,
                state=self._group_state(group),
                roles=[self.statuses[role].model_copy(deep=True) for role in roles],
            )
            for group, roles in WORKER_GROUP_ROLES.items()
        ]
        return PreparationPublic(
            state=self.preparation_state,
            message=self.preparation_message,
            groups=groups,
        )

    def _group_state(self, group: WorkerGroup) -> WorkerLifecycleState:
        states = [self.statuses[role].state for role in WORKER_GROUP_ROLES[group]]
        if any(state == WorkerLifecycleState.FAILED for state in states):
            return WorkerLifecycleState.FAILED
        if any(state == WorkerLifecycleState.RUNNING for state in states):
            return WorkerLifecycleState.RUNNING
        if any(
            state in {WorkerLifecycleState.STARTING, WorkerLifecycleState.LOADING}
            for state in states
        ):
            return WorkerLifecycleState.LOADING
        if all(state == WorkerLifecycleState.READY for state in states):
            return WorkerLifecycleState.READY
        return WorkerLifecycleState.STOPPED

    def _set_status(
        self,
        role: WorkerRole,
        state: WorkerLifecycleState,
        *,
        phase: str | None = None,
        catalog_id: str | None = None,
        model_id: str | None = None,
        model_revision: str | None = None,
        backend: str | None = None,
        dtype: str | None = None,
        quantization: str | None = None,
        device: str | None = None,
        request_id: str | None = None,
        attempt: int | None = None,
        progress: float | None = None,
        message: str | None = None,
        detail: str | None = None,
        error_code: str | None = None,
    ) -> None:
        status = self.statuses[role]
        phase_changed = status.state != state or (phase is not None and status.phase != phase)
        status.state = state
        status.phase = phase if phase is not None else status.phase
        status.catalog_id = catalog_id if catalog_id is not None else status.catalog_id
        status.model_id = model_id if model_id is not None else status.model_id
        status.model_revision = (
            model_revision if model_revision is not None else status.model_revision
        )
        status.backend = backend if backend is not None else status.backend
        status.dtype = dtype if dtype is not None else status.dtype
        status.quantization = (
            quantization if quantization is not None else status.quantization
        )
        status.device = device if device is not None else status.device
        status.request_id = request_id if request_id is not None else status.request_id
        status.attempt = attempt if attempt is not None else status.attempt
        status.progress = progress if progress is not None else status.progress
        status.message = message
        status.detail = detail
        status.error_code = error_code
        now = utc_now()
        if phase_changed or status.phase_started_at is None:
            status.phase_started_at = now
        status.updated_at = now

    async def _publish(
        self,
        role: WorkerRole,
        state: WorkerLifecycleState,
        callback: StatusCallback | None = None,
        event: WorkerProgressEvent | None = None,
        **values: object,
    ) -> None:
        self._set_status(role, state, **values)
        if callback is not None:
            await callback(self.statuses[role].model_copy(deep=True), event)

    def reset_generation_statuses(self) -> None:
        for role in WorkerRole:
            if ROLE_GROUP[role] != WorkerGroup.INTERVIEW:
                self.statuses[role] = WorkerStatusPublic(
                    role=role,
                    group=ROLE_GROUP[role],
                )

    async def prepare_interview(self, settings: StaffSettings | None = None) -> None:
        async with self._prepare_lock:
            selected = settings or self.config.staff
            self.preparation_state = PreparationState.LOADING
            self.preparation_message = "インタビューに必要なAIを準備しています"
            try:
                await self.supervisor.stop_group(WorkerGroup.INTERVIEW)
                for role in WORKER_GROUP_ROLES[WorkerGroup.INTERVIEW]:
                    catalog_id = selected.stage_models[role]
                    spec = self.config.catalog.spec(role, catalog_id)
                    self._set_status(
                        role,
                        WorkerLifecycleState.LOADING,
                        model_id=spec.model_id,
                        backend=spec.backend,
                        progress=0.2,
                        message="モデルを読み込んでいます",
                    )
                    health, load_ms = await self.supervisor.load(role, catalog_id)
                    if not health.ready:
                        raise WorkerExecutionError("worker_healthcheck_failed", role.value)
                    self.statuses[role].load_time_ms = load_ms
                    self._set_status(
                        role,
                        WorkerLifecycleState.READY,
                        progress=1.0,
                        message="準備完了",
                    )
                self.preparation_state = PreparationState.READY
                self.preparation_message = "インタビューを開始できます"
            except Exception as exc:
                LOGGER.exception("interview worker preparation failed")
                self.preparation_state = PreparationState.FAILED
                self.preparation_message = "AIワーカーの準備に失敗しました"
                for role in WORKER_GROUP_ROLES[WorkerGroup.INTERVIEW]:
                    if self.statuses[role].state != WorkerLifecycleState.READY:
                        self._set_status(
                            role,
                            WorkerLifecycleState.FAILED,
                            error_code=getattr(exc, "code", "worker_preparation_failed"),
                            message="準備に失敗しました",
                        )
                await self.supervisor.stop_group(WorkerGroup.INTERVIEW)

    async def reconfigure(
        self,
        previous: StaffSettings,
        current: StaffSettings,
    ) -> None:
        changed = any(
            previous.stage_models.get(role) != current.stage_models.get(role)
            for role in WORKER_GROUP_ROLES[WorkerGroup.INTERVIEW]
        )
        if changed or self.preparation_state != PreparationState.READY:
            await self.prepare_interview(current)

    async def run_prepared_role(
        self,
        session: SessionRecord,
        role: WorkerRole,
        *,
        input_paths: dict[str, str],
        output_dir: str,
        deadline_seconds: int,
        metadata: dict[str, object] | None = None,
    ) -> WorkerResult:
        if ROLE_GROUP[role] != WorkerGroup.INTERVIEW:
            raise ValueError(f"prepared_worker_must_be_interview_role: {role.value}")

        settings = session.settings_snapshot
        selected_id = settings.stage_models[role]
        selected_entry = self.config.catalog.entry(selected_id)
        candidates = [selected_id]
        if settings.auto_model_fallback and selected_entry.fallback_model_id:
            candidates.append(selected_entry.fallback_model_id)

        last_error: WorkerExecutionError | None = None
        for candidate_index, catalog_id in enumerate(candidates):
            spec = self.config.catalog.spec(role, catalog_id)
            attempts = self.config.developer.pipeline.worker_restart_count + 1
            for attempt in range(attempts):
                request = WorkerRequest(
                    worker=role,
                    session_id=session.session_id,
                    model=spec,
                    deadline_seconds=max(1, deadline_seconds),
                    input_paths=input_paths,
                    output_dir=output_dir,
                    metadata={
                        "stub_delay_seconds": self.config.developer.pipeline.stub_step_delay_seconds,
                        **(metadata or {}),
                    },
                )
                self.active_requests[role] = request.request_id
                try:
                    lock = (
                        self.gpu_lock
                        if spec.device.lower().startswith("cuda")
                        else _NullAsyncLock()
                    )
                    async with lock:
                        load_ms = 0
                        handle = self.supervisor.handles.get(role)
                        if handle is None or handle.catalog_id != catalog_id:
                            health, load_ms = await self.supervisor.load(role, catalog_id)
                        else:
                            health = await self.supervisor.healthcheck(role)
                            if not health.ready:
                                health, load_ms = await self.supervisor.load(
                                    role, catalog_id
                                )
                        if not health.ready:
                            raise WorkerExecutionError(
                                "worker_healthcheck_failed", role.value
                            )
                        if load_ms:
                            self.statuses[role].load_time_ms = load_ms
                        self._set_status(
                            role,
                            WorkerLifecycleState.RUNNING,
                            model_id=spec.model_id,
                            backend=spec.backend,
                            progress=0.5,
                            message=(
                                "会話の続きを考えています"
                                if role == WorkerRole.INTERVIEW_LLM
                                else "回答を文字起こししています"
                            ),
                        )
                        result = await self.supervisor.run(request)
                    result.metrics.load_time_ms = load_ms
                    self.statuses[role].processing_time_ms = (
                        result.metrics.processing_time_ms
                    )
                    self.statuses[role].peak_vram_mb = result.metrics.peak_vram_mb
                    self.statuses[role].peak_cpu_memory_mb = (
                        result.metrics.peak_cpu_memory_mb
                    )
                    self._set_status(
                        role,
                        WorkerLifecycleState.READY,
                        progress=1.0,
                        message="準備完了",
                    )
                    await self.metrics.record(request, result)
                    return result
                except WorkerExecutionError as exc:
                    last_error = exc
                    await self.metrics.record(request, None, error_code=exc.code)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    LOGGER.exception("prepared worker execution failed: role=%s", role.value)
                    last_error = WorkerExecutionError(
                        "worker_runtime_error",
                        f"{role.value} の実行中に予期しないエラーが発生しました",
                    )
                    await self.metrics.record(
                        request,
                        None,
                        error_code=last_error.code,
                    )
                finally:
                    self.active_requests.pop(role, None)

                await self.supervisor.stop(role)
                if attempt + 1 < attempts:
                    self._set_status(
                        role,
                        WorkerLifecycleState.STARTING,
                        progress=0.1,
                        message="同じモデルを再起動しています",
                    )

            if candidate_index + 1 < len(candidates):
                next_entry = self.config.catalog.entry(candidates[candidate_index + 1])
                session.model_switch_notice = f"{role.value}を軽量モデルへ切り替えました"
                self._set_status(
                    role,
                    WorkerLifecycleState.STARTING,
                    model_id=next_entry.model_id,
                    backend=next_entry.backend,
                    progress=0.1,
                    message="軽量モデルへ切り替えています",
                )

        error = last_error or WorkerExecutionError("worker_failed", role.value)
        self._set_status(
            role,
            WorkerLifecycleState.FAILED,
            progress=1.0,
            error_code=error.code,
            message="処理に失敗しました",
        )
        raise error

    async def run_role(
        self,
        session: SessionRecord,
        role: WorkerRole,
        *,
        input_paths: dict[str, str],
        output_dir: str,
        deadline_seconds: int,
        metadata: dict[str, object] | None = None,
        on_status: StatusCallback | None = None,
    ) -> WorkerResult:
        settings = session.settings_snapshot
        selected_id = settings.stage_models[role]
        selected_entry = self.config.catalog.entry(selected_id)
        candidates = [selected_id]
        if settings.auto_model_fallback and selected_entry.fallback_model_id:
            candidates.append(selected_entry.fallback_model_id)

        last_error: WorkerExecutionError | None = None
        for candidate_index, catalog_id in enumerate(candidates):
            spec = self.config.catalog.spec(role, catalog_id)
            status_model: dict[str, object] = {
                "catalog_id": spec.catalog_id,
                "model_id": spec.model_id,
                "model_revision": spec.model_revision,
                "backend": spec.backend,
                "dtype": spec.dtype,
                "quantization": spec.quantization,
                "device": spec.device,
            }
            attempts = self.config.developer.pipeline.worker_restart_count + 1
            for attempt_index in range(attempts):
                attempt = attempt_index + 1
                request = WorkerRequest(
                    worker=role,
                    session_id=session.session_id,
                    model=spec,
                    deadline_seconds=max(1, deadline_seconds),
                    input_paths=input_paths,
                    output_dir=output_dir,
                    metadata={
                        "stub_delay_seconds": self.config.developer.pipeline.stub_step_delay_seconds,
                        **(metadata or {}),
                    },
                )
                self.active_requests[role] = request.request_id
                started = time.perf_counter()
                await self._publish(
                    role,
                    WorkerLifecycleState.STARTING,
                    on_status,
                    phase="worker_start",
                    request_id=request.request_id,
                    attempt=attempt,
                    progress=0.03,
                    message="ワーカープロセスを起動しています",
                    detail=f"試行 {attempt}/{attempts}",
                    error_code=None,
                    **status_model,
                )
                try:
                    lock = (
                        self.gpu_lock
                        if spec.device.lower().startswith("cuda")
                        else _NullAsyncLock()
                    )
                    async with lock:
                        await self.supervisor.start(role, catalog_id)
                        await self._publish(
                            role,
                            WorkerLifecycleState.LOADING,
                            on_status,
                            phase="model_load",
                            request_id=request.request_id,
                            attempt=attempt,
                            progress=0.12,
                            message="モデルを読み込んでいます",
                            detail=f"{spec.model_id} / {spec.device}",
                            **status_model,
                        )
                        health, load_ms = await self.supervisor.load(role, catalog_id)
                        if not health.ready:
                            raise WorkerExecutionError(
                                "worker_healthcheck_failed",
                                f"{role.value} のロード後確認に失敗しました",
                                worker=role,
                                phase="healthcheck",
                            )
                        self.statuses[role].load_time_ms = load_ms
                        await self._publish(
                            role,
                            WorkerLifecycleState.RUNNING,
                            on_status,
                            phase="inference",
                            request_id=request.request_id,
                            attempt=attempt,
                            progress=0.28,
                            message="モデル推論を開始しました",
                            detail=f"ロード {load_ms}ms",
                            **status_model,
                        )

                        async def relay_progress(event: WorkerProgressEvent) -> None:
                            await self._publish(
                                role,
                                WorkerLifecycleState.RUNNING,
                                on_status,
                                event,
                                phase=event.phase,
                                request_id=request.request_id,
                                attempt=attempt,
                                progress=min(0.88, 0.28 + event.progress * 0.58),
                                message=event.message,
                                detail=event.detail,
                                **status_model,
                            )

                        result = await self.supervisor.run(
                            request,
                            relay_progress,
                        )
                        self.statuses[role].processing_time_ms = (
                            result.metrics.processing_time_ms
                        )
                        self.statuses[role].peak_vram_mb = result.metrics.peak_vram_mb
                        self.statuses[role].peak_cpu_memory_mb = (
                            result.metrics.peak_cpu_memory_mb
                        )
                        await self._publish(
                            role,
                            WorkerLifecycleState.UNLOADING,
                            on_status,
                            phase="model_unload",
                            request_id=request.request_id,
                            attempt=attempt,
                            progress=0.93,
                            message="モデルを解放しています",
                            detail="次の工程のためGPUメモリを解放します",
                            **status_model,
                        )
                        unload_ms = await self.supervisor.stop(role)
                        result.metrics.load_time_ms = load_ms
                        result.metrics.unload_time_ms = unload_ms
                        if spec.device.lower().startswith("cuda"):
                            await self._publish(
                                role,
                                WorkerLifecycleState.UNLOADING,
                                on_status,
                                phase="gpu_release",
                                request_id=request.request_id,
                                attempt=attempt,
                                progress=0.97,
                                message="GPUメモリの解放を確認しています",
                                detail=f"アンロード {unload_ms}ms",
                                **status_model,
                            )
                            await asyncio.sleep(
                                self.config.developer.pipeline.gpu_release_wait_seconds
                            )
                    await self._publish(
                        role,
                        WorkerLifecycleState.READY,
                        on_status,
                        phase="completed",
                        request_id=request.request_id,
                        attempt=attempt,
                        progress=1.0,
                        message="処理が完了しました",
                        detail=(
                            f"ロード {result.metrics.load_time_ms}ms / "
                            f"推論 {result.metrics.processing_time_ms}ms / "
                            f"解放 {result.metrics.unload_time_ms}ms"
                        ),
                        **status_model,
                    )
                    await self.metrics.record(request, result)
                    return result
                except WorkerExecutionError as exc:
                    last_error = WorkerExecutionError(
                        exc.code,
                        str(exc),
                        worker=exc.worker or role,
                        phase=exc.phase or self.statuses[role].phase or "runtime",
                        detail=exc.detail,
                    )
                    self.statuses[role].processing_time_ms = int(
                        (time.perf_counter() - started) * 1000
                    )
                    await self._publish(
                        role,
                        WorkerLifecycleState.FAILED,
                        on_status,
                        phase=last_error.phase,
                        request_id=request.request_id,
                        attempt=attempt,
                        progress=1.0,
                        message="この工程の処理に失敗しました",
                        detail=last_error.detail,
                        error_code=last_error.code,
                        **status_model,
                    )
                    await self.metrics.record(request, None, error_code=last_error.code)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    LOGGER.exception("worker execution failed: role=%s", role.value)
                    last_error = WorkerExecutionError(
                        "worker_runtime_error",
                        f"{role.value} の実行中に予期しないエラーが発生しました",
                        worker=role,
                        phase=self.statuses[role].phase or "runtime",
                        detail=str(exc)[:1200],
                    )
                    await self._publish(
                        role,
                        WorkerLifecycleState.FAILED,
                        on_status,
                        phase=last_error.phase,
                        request_id=request.request_id,
                        attempt=attempt,
                        progress=1.0,
                        message="この工程の処理に失敗しました",
                        detail=last_error.detail,
                        error_code=last_error.code,
                        **status_model,
                    )
                    await self.metrics.record(request, None, error_code=last_error.code)
                finally:
                    self.active_requests.pop(role, None)
                    if role in self.supervisor.handles:
                        try:
                            await self._publish(
                                role,
                                WorkerLifecycleState.UNLOADING,
                                on_status,
                                phase="failure_cleanup",
                                request_id=request.request_id,
                                attempt=attempt,
                                progress=0.98,
                                message="失敗したモデルを解放しています",
                                detail=None,
                                **status_model,
                            )
                            await asyncio.shield(self.supervisor.stop(role))
                        except Exception:
                            LOGGER.exception(
                                "worker cleanup failed: role=%s",
                                role.value,
                            )

                if attempt < attempts:
                    await self._publish(
                        role,
                        WorkerLifecycleState.STARTING,
                        on_status,
                        phase="retry",
                        request_id=request.request_id,
                        attempt=attempt + 1,
                        progress=0.02,
                        message="同じモデルで再試行します",
                        detail=f"次の試行 {attempt + 1}/{attempts}",
                        **status_model,
                    )

            if candidate_index + 1 < len(candidates):
                next_entry = self.config.catalog.entry(candidates[candidate_index + 1])
                session.model_switch_notice = f"{role.value}を軽量モデルへ切り替えました"
                await self._publish(
                    role,
                    WorkerLifecycleState.STARTING,
                    on_status,
                    phase="model_fallback",
                    model_id=next_entry.model_id,
                    backend=next_entry.backend,
                    progress=0.02,
                    message="軽量モデルへ切り替えています",
                    detail=f"次のモデル: {next_entry.model_id}",
                )

        error = last_error or WorkerExecutionError(
            "worker_failed",
            role.value,
            worker=role,
            phase="runtime",
        )
        await self._publish(
            role,
            WorkerLifecycleState.FAILED,
            on_status,
            phase=error.phase or "runtime",
            progress=1.0,
            message="処理を継続できません",
            detail=error.detail,
            error_code=error.code,
        )
        raise error

    async def cancel_all(self) -> None:
        for role, request_id in list(self.active_requests.items()):
            await self.supervisor.cancel(role, request_id)
        self.active_requests.clear()

    async def release_interview(self) -> None:
        await self.supervisor.stop_group(WorkerGroup.INTERVIEW)
        self.preparation_state = PreparationState.STOPPED
        self.preparation_message = "動画生成中はインタビュー用モデルを解放しています"
        for role in WORKER_GROUP_ROLES[WorkerGroup.INTERVIEW]:
            self._set_status(
                role,
                WorkerLifecycleState.STOPPED,
                progress=0,
                message="生成処理のため解放しました",
            )

    async def emergency_stop(self) -> None:
        await self.cancel_all()
        await self.supervisor.stop_all()
        self.preparation_state = PreparationState.STOPPED
        self.preparation_message = "緊急停止しました。再準備を開始します"
        for role in WorkerRole:
            self._set_status(role, WorkerLifecycleState.STOPPED, progress=0, message=None)

    async def shutdown(self) -> None:
        await self.cancel_all()
        await self.supervisor.stop_all()


class _NullAsyncLock:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False
