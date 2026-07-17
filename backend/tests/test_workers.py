from __future__ import annotations

import asyncio
import json
import shutil
import subprocess

import pytest

from backend.app.schemas import (
    WorkerRequest,
    WorkerRole,
)
from backend.app.workers.base import StubWorker


def test_common_worker_lifecycle(fast_config) -> None:
    async def scenario() -> None:
        role = WorkerRole.IMAGE_GENERATION
        worker = StubWorker(role)
        model = fast_config.catalog.spec(role, "foundation-stub")

        loaded = await worker.load(model)
        assert loaded.ready is True

        request = WorkerRequest(
            worker=role,
            session_id="anonymous-session",
            model=model,
            deadline_seconds=10,
            output_dir=str(fast_config.session_root / "output"),
            metadata={"stub_delay_seconds": 0.01},
        )
        events = []

        async def progress(event) -> None:
            events.append(event)

        result = await worker.run(request, progress)
        assert result.worker == role
        assert result.implemented is False
        assert events[-1].progress == 1

        await worker.unload()
        health = await worker.healthcheck()
        assert health.loaded is False

    asyncio.run(scenario())


def test_voice_reference_accepts_webm_without_container_duration(
    fast_config,
    tmp_path,
) -> None:
    from backend.app.workers.adapters.pipeline_utilities import (
        PipelineUtilitiesAdapter,
    )

    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        pytest.skip("ffmpeg and ffprobe are required")

    answers_dir = tmp_path / "answers"
    answers_dir.mkdir()
    source_path = answers_dir / "audio-000000.webm"
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:sample_rate=48000:duration=3",
        "-c:a",
        "libopus",
        "-f",
        "webm",
        "pipe:1",
    ]
    with source_path.open("wb") as output:
        subprocess.run(command, stdout=output, check=True)

    format_duration = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(source_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert format_duration.stdout.strip() == "N/A"

    transcript_path = tmp_path / "transcript.json"
    transcript_path.write_text(
        json.dumps(
            [{"speaker": "visitor", "text": "筋力トレーニングを続けたいです"}],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    async def scenario() -> None:
        role = WorkerRole.VOICE_REFERENCE_SELECTOR
        worker = PipelineUtilitiesAdapter(role)
        model = fast_config.catalog.spec(role, "pipeline-utilities-v1")
        await worker.load(model)
        request = WorkerRequest(
            worker=role,
            session_id="media-recorder-duration-test",
            model=model,
            deadline_seconds=10,
            output_dir=str(tmp_path / "output"),
            input_paths={
                "audio_answers_dir": str(answers_dir),
                "transcript": str(transcript_path),
            },
        )

        result = await worker.run(request)

        assert result.metadata["candidate_count"] == 1
        assert result.metadata["selected_count"] == 1
        assert result.metadata["duration_seconds"] >= 2.8
        assert (tmp_path / "output" / "voice-reference.wav").is_file()

    asyncio.run(scenario())


def test_isolated_worker_process_is_removed_after_cancellation(tmp_path) -> None:
    from backend.app.config import ConfigManager, ROOT_DIR
    from backend.app.metrics import MetricsStore
    from backend.app.schemas import SessionRecord
    from backend.app.workers.manager import WorkerOrchestrator

    async def scenario() -> None:
        config = ConfigManager(
            ROOT_DIR,
            environ={
                "MOSHIMO__STORAGE__STAFF_SETTINGS_PATH": str(
                    tmp_path / "staff-settings.json"
                ),
                "MOSHIMO__STORAGE__METRICS_DB_PATH": str(tmp_path / "metrics.sqlite3"),
                "MOSHIMO__WORKER_RUNTIME__PROCESS_ISOLATION_ENABLED": "true",
                "MOSHIMO__WORKER_RUNTIME__STARTUP_TIMEOUT_SECONDS": "10",
            },
        )
        metrics = MetricsStore(config.metrics_db_path)
        await metrics.initialize()
        orchestrator = WorkerOrchestrator(config, metrics)
        role = WorkerRole.IMAGE_GENERATION
        session = SessionRecord(session_id="process-test", settings_snapshot=config.staff)

        task = asyncio.create_task(
            orchestrator.run_role(
                session,
                role,
                input_paths={},
                output_dir=str(tmp_path / "output"),
                deadline_seconds=10,
                metadata={"stub_delay_seconds": 2.0},
            )
        )
        deadline = asyncio.get_running_loop().time() + 5
        while role not in orchestrator.supervisor.handles:
            if asyncio.get_running_loop().time() >= deadline:
                raise AssertionError("isolated worker did not start")
            await asyncio.sleep(0.05)

        handle = orchestrator.supervisor.handles[role]
        assert handle.process is not None
        assert handle.process.returncode is None

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert role not in orchestrator.supervisor.handles
        assert handle.process.returncode is not None
        await orchestrator.shutdown()

    asyncio.run(scenario())



def test_prepared_interview_worker_stays_loaded(fast_config, tmp_path) -> None:
    from backend.app.metrics import MetricsStore
    from backend.app.schemas import SessionRecord
    from backend.app.workers.manager import WorkerOrchestrator

    async def scenario() -> None:
        metrics = MetricsStore(fast_config.metrics_db_path)
        await metrics.initialize()
        orchestrator = WorkerOrchestrator(fast_config, metrics)
        await orchestrator.prepare_interview(fast_config.staff)
        role = WorkerRole.STREAMING_ASR
        original_handle = orchestrator.supervisor.handles[role]
        session = SessionRecord(
            session_id="prepared-worker-test",
            settings_snapshot=fast_config.staff,
        )

        result = await orchestrator.run_prepared_role(
            session,
            role,
            input_paths={},
            output_dir=str(tmp_path / "output"),
            deadline_seconds=10,
            metadata={"stub_delay_seconds": 0.01},
        )

        assert result.worker == role
        assert orchestrator.supervisor.handles[role] is original_handle
        assert (await orchestrator.supervisor.healthcheck(role)).ready is True
        await orchestrator.shutdown()

    asyncio.run(scenario())


def test_generation_worker_reports_lifecycle_phases(fast_config, tmp_path) -> None:
    from backend.app.metrics import MetricsStore
    from backend.app.schemas import SessionRecord
    from backend.app.workers.manager import WorkerOrchestrator

    async def scenario() -> None:
        metrics = MetricsStore(fast_config.metrics_db_path)
        await metrics.initialize()
        orchestrator = WorkerOrchestrator(fast_config, metrics)
        role = WorkerRole.IMAGE_GENERATION
        session = SessionRecord(
            session_id="progress-test",
            settings_snapshot=fast_config.staff,
        )
        phases: list[str] = []

        async def on_status(status, event) -> None:
            if status.phase:
                phases.append(status.phase)

        await orchestrator.run_role(
            session,
            role,
            input_paths={},
            output_dir=str(tmp_path / "output"),
            deadline_seconds=10,
            metadata={"stub_delay_seconds": 0.01},
            on_status=on_status,
        )

        assert "worker_start" in phases
        assert "model_load" in phases
        assert "inference" in phases
        assert "model_unload" in phases
        assert phases[-1] == "completed"
        await orchestrator.shutdown()

    asyncio.run(scenario())


def test_worker_http_error_preserves_remote_diagnostics(fast_config) -> None:
    import httpx

    from backend.app.workers.manager import WorkerProcessSupervisor

    response = httpx.Response(
        500,
        request=httpx.Request("POST", "http://worker/run"),
        json={
            "detail": {
                "code": "generation_llm_stage_failed",
                "phase": "script.narration_script",
                "message": "ナレーション生成に失敗しました",
                "exception_type": "GenerationLlmStageError",
            }
        },
    )
    error = WorkerProcessSupervisor._response_error(
        response,
        WorkerRole.SCRIPT_DESIGN_LLM,
        "inference",
    )

    assert error.code == "generation_llm_stage_failed"
    assert error.phase == "script.narration_script"
    assert "GenerationLlmStageError" in error.detail


def test_worker_startup_failure_detail_uses_current_process_log(tmp_path) -> None:
    from backend.app.workers.manager import (
        WorkerHandle,
        WorkerProcessSupervisor,
    )

    log_path = tmp_path / "worker-video_generation_worker.log"
    log_path.write_text("stale process error\n", encoding="utf-8")
    start_offset = log_path.stat().st_size
    with log_path.open("ab") as log_handle:
        log_handle.write(b"Traceback\nImportError: cannot import name 'StrEnum'\n")
        handle = WorkerHandle(
            role=WorkerRole.VIDEO_GENERATION,
            catalog_id="echomimic-v3-flash-bf16",
            log_handle=log_handle,
            log_path=log_path,
            log_start_offset=start_offset,
        )

        detail = WorkerProcessSupervisor._startup_failure_detail(handle, 1)

    assert "exit_code=1" in detail
    assert "ImportError: cannot import name 'StrEnum'" in detail
    assert "stale process error" not in detail


def test_isolated_worker_relays_adapter_progress(tmp_path) -> None:
    from backend.app.config import ConfigManager, ROOT_DIR
    from backend.app.metrics import MetricsStore
    from backend.app.schemas import SessionRecord
    from backend.app.workers.manager import WorkerOrchestrator

    async def scenario() -> None:
        config = ConfigManager(
            ROOT_DIR,
            environ={
                "MOSHIMO__STORAGE__STAFF_SETTINGS_PATH": str(
                    tmp_path / "staff-settings.json"
                ),
                "MOSHIMO__STORAGE__METRICS_DB_PATH": str(tmp_path / "metrics.sqlite3"),
                "MOSHIMO__WORKER_RUNTIME__PROCESS_ISOLATION_ENABLED": "true",
                "MOSHIMO__WORKER_RUNTIME__STARTUP_TIMEOUT_SECONDS": "10",
            },
        )
        metrics = MetricsStore(config.metrics_db_path)
        await metrics.initialize()
        orchestrator = WorkerOrchestrator(config, metrics)
        role = WorkerRole.REFERENCE_FRAME_SELECTOR
        session = SessionRecord(
            session_id="isolated-progress-test",
            settings_snapshot=config.staff,
        )
        session.settings_snapshot.stage_models[role] = "foundation-stub"
        adapter_messages: list[str] = []

        async def on_status(status, event) -> None:
            if event is not None:
                adapter_messages.append(event.message)

        await orchestrator.run_role(
            session,
            role,
            input_paths={},
            output_dir=str(tmp_path / "output"),
            deadline_seconds=10,
            metadata={"stub_delay_seconds": 0.06},
            on_status=on_status,
        )

        assert "入力を確認しています" in adapter_messages
        assert "スタブ処理が完了しました" in adapter_messages
        await orchestrator.shutdown()

    asyncio.run(scenario())
