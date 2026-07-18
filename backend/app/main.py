from __future__ import annotations

import asyncio
import logging
import mimetypes
from contextlib import asynccontextmanager
from typing import Any
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import ConfigManager
from .contracts import InterviewTurnInput, InterviewTurnOutput
from .episodes import EpisodeRepository
from .logging_config import configure_logging, session_event
from .metrics import MetricsStore
from .schemas import (
    AddUtteranceRequest,
    AnswerCompleteRequest,
    ConsentRequest,
    ConversationPhase,
    DebugTestMode,
    DeviceCheckReport,
    PreparationState,
    RuntimeOptions,
    SessionState,
    StaffSettings,
    WorkerRole,
    WORKER_GROUP_ROLES,
    utc_now,
)
from .services.generation import GenerationService
from .session_store import (
    SessionNotFoundError,
    SessionStore,
    public_session,
    seconds_since,
)
from .workers.manager import WorkerExecutionError, WorkerOrchestrator


LOGGER = logging.getLogger(__name__)


def create_app(config_manager: ConfigManager | None = None) -> FastAPI:
    config = config_manager or ConfigManager()
    configure_logging(config.developer.logging.level, config.log_root)
    episodes = EpisodeRepository(
        config.episode_dir,
        config.effects_path,
        config.developer.episodes.rarity_weights,
    )
    store = SessionStore(config.session_root)
    metrics = MetricsStore(config.metrics_db_path)
    workers = WorkerOrchestrator(config, metrics)
    generation = GenerationService(config, store, episodes, workers)
    background_tasks: set[asyncio.Task[Any]] = set()

    def spawn(coroutine: Any) -> None:
        task = asyncio.create_task(coroutine)
        background_tasks.add(task)
        task.add_done_callback(background_tasks.discard)

    async def stale_session_sweeper() -> None:
        capture = config.developer.capture
        while True:
            await asyncio.sleep(capture.stale_check_interval_seconds)
            deleted_session_id = await store.clear_if_stale(capture.stale_session_seconds)
            if deleted_session_id:
                await generation.cancel(deleted_session_id)
                session_event(
                    LOGGER,
                    "session_deleted",
                    session_id=deleted_session_id,
                    reason="browser_stale_timeout",
                )
                await workers.prepare_interview(config.staff)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        await store.startup_cleanup()
        await metrics.initialize()
        workers.preparation_state = PreparationState.LOADING
        workers.preparation_message = "インタビューに必要なAIを準備しています"
        spawn(workers.prepare_interview(config.staff))
        spawn(stale_session_sweeper())
        try:
            yield
        finally:
            for task in list(background_tasks):
                task.cancel()
            if background_tasks:
                await asyncio.gather(*background_tasks, return_exceptions=True)
            await generation.shutdown()
            await workers.shutdown()
            await store.clear()

    application = FastAPI(title=config.developer.app.name, lifespan=lifespan)
    application.state.config = config
    application.state.episodes = episodes
    application.state.store = store
    application.state.metrics = metrics
    application.state.workers = workers
    application.state.generation = generation

    @application.exception_handler(SessionNotFoundError)
    async def session_not_found_handler(
        _: Request,
        exc: SessionNotFoundError,
    ) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    async def auto_finish(session_id: str, reason: str) -> None:
        await asyncio.sleep(config.developer.capture.finalize_timeout_seconds)
        try:
            session = await store.require(session_id)
        except SessionNotFoundError:
            return
        if session.state == SessionState.CONVERSATION:
            await generation.start(session, reason)

    async def generate_next_interview_turn(session) -> InterviewTurnOutput:
        settings = session.settings_snapshot
        elapsed_seconds = seconds_since(session.conversation_started_at)
        turn = InterviewTurnInput(
            transcript=session.transcript,
            state=session.interview_state.model_copy(deep=True),
            target_transcript_chars=settings.target_transcript_chars,
            minimum_transcript_chars=settings.minimum_transcript_chars,
            conversation_time_limit_seconds=settings.conversation_time_limit_seconds,
            remaining_time_seconds=max(
                0,
                settings.conversation_time_limit_seconds - elapsed_seconds,
            ),
            thinking_enabled=False,
        )
        turn.state.visitor_char_count = session.visitor_char_count
        turn.state.elapsed_seconds = elapsed_seconds
        output_dir = (
            store.session_dir(session.session_id)
            / "intermediate"
            / "interview-llm"
            / f"{session.interview_state.answer_count:06d}"
        )
        result = await workers.run_prepared_role(
            session,
            WorkerRole.INTERVIEW_LLM,
            input_paths={},
            output_dir=str(output_dir),
            deadline_seconds=30,
            metadata={
                "interview_turn": turn.model_dump(mode="json"),
                "max_new_tokens": 256,
            },
        )
        if not result.implemented:
            raise WorkerExecutionError(
                "interview_llm_unimplemented",
                "対話LLMが未接続です",
            )
        return InterviewTurnOutput.model_validate(
            result.metadata.get("interview_turn")
        )

    async def current_with_automatic_transitions():
        session = await store.current(touch=True)
        if session is None:
            return None
        settings = session.settings_snapshot
        if (
            session.state == SessionState.GENERATING
            and seconds_since(session.generation_started_at)
            >= settings.generation_time_limit_seconds
        ):
            await generation.fail_timeout(session)
            return session
        if session.state == SessionState.CONVERSATION and not session.auto_finish_scheduled:
            elapsed = seconds_since(session.conversation_started_at)
            short_limit_reached = (
                config.developer.app.debug_mode
                and settings.debug_test_mode == DebugTestMode.SHORT
                and elapsed >= config.developer.capture.debug_short_time_limit_seconds
            )
            if elapsed >= settings.conversation_time_limit_seconds or short_limit_reached:
                session.auto_finish_scheduled = True
                session.conversation_phase = ConversationPhase.CLOSING
                session.interview_state.should_end = True
                session.interview_state.end_reason = (
                    "debug_short_time_limit"
                    if short_limit_reached
                    else "conversation_time_limit"
                )
                await store.persist(session)
                spawn(auto_finish(session.session_id, session.interview_state.end_reason))
        return session

    def settings_payload() -> dict[str, Any]:
        return {
            "settings": config.staff.model_dump(mode="json"),
            "preparation": workers.status().model_dump(mode="json"),
            "options": RuntimeOptions(
                models=config.catalog.options(),
                profile_models={
                    profile: config.catalog.profile_models(profile)
                    for profile in (
                        "fast",
                        "balanced",
                        "quality",
                    )
                },
                episodes=episodes.summaries(),
                worker_groups={
                    group: list(roles) for group, roles in WORKER_GROUP_ROLES.items()
                },
                limits=config.developer.staff_limits,
                debug_mode=config.developer.app.debug_mode,
                config_precedence=config.precedence,
            ).model_dump(mode="json"),
        }

    @application.get("/api/health")
    async def health() -> dict[str, Any]:
        return {
            "ok": True,
            "app_name": config.developer.app.name,
            "base_path": config.developer.app.base_path,
            "debug_mode": config.developer.app.debug_mode,
            "pipeline": "local-generation-stack",
            "preparation": workers.preparation_state,
        }

    @application.get("/api/config")
    async def frontend_config() -> dict[str, Any]:
        capture = config.developer.capture
        return {
            "app_name": config.developer.app.name,
            "base_path": config.developer.app.base_path,
            "debug_mode": config.developer.app.debug_mode,
            "capture": {
                "video_chunk_seconds": capture.video_chunk_seconds,
                "silence_seconds": capture.silence_seconds,
                "speech_start_threshold": capture.speech_start_threshold,
                "response_max_seconds": capture.response_max_seconds,
                "upload_retry_count": capture.upload_retry_count,
                "browser_queue_limit_mb": capture.browser_queue_limit_mb,
                "camera_stable_seconds": capture.camera_stable_seconds,
                "brightness_min": capture.brightness_min,
                "brightness_max": capture.brightness_max,
            },
            "notices": {
                "privacy": "入力された情報は今回のデモにのみ使用し、終了後に削除します。",
                "fiction": "生成される内容はフィクションであり、実際の将来を示すものではありません。",
                "implementation": "収録、文字起こし、台本、本人声、未来画像、動画生成を研究室サーバー内で処理します。",
            },
        }

    @application.get("/api/runtime/status")
    async def runtime_status() -> dict[str, Any]:
        return {"preparation": workers.status().model_dump(mode="json")}

    @application.post("/api/runtime/retry")
    async def retry_runtime() -> dict[str, Any]:
        if workers.preparation_state == PreparationState.LOADING:
            return {"preparation": workers.status().model_dump(mode="json")}
        workers.preparation_state = PreparationState.LOADING
        workers.preparation_message = "AIワーカーを再準備しています"
        spawn(workers.prepare_interview(config.staff))
        return {"preparation": workers.status().model_dump(mode="json")}

    @application.get("/api/settings")
    async def get_settings() -> dict[str, Any]:
        return settings_payload()

    @application.put("/api/settings")
    async def update_settings(settings: StaffSettings) -> dict[str, Any]:
        previous = config.staff.model_copy(deep=True)
        try:
            validated = config.validate_staff(settings)
            if validated.episode_selection == "fixed":
                episodes.select_episode(validated)
            config.save_staff(validated)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if await store.current() is None:
            workers.mark_interview_reconfigure_pending(previous, config.staff)
            spawn(workers.reconfigure(previous, config.staff))
        return settings_payload()

    @application.post("/api/settings/reset")
    async def reset_settings() -> dict[str, Any]:
        previous = config.staff.model_copy(deep=True)
        config.reset_staff()
        if await store.current() is None:
            workers.mark_interview_reconfigure_pending(previous, config.staff)
            spawn(workers.reconfigure(previous, config.staff))
        return settings_payload()

    @application.get("/api/session/current")
    async def get_current_session() -> dict[str, Any]:
        session = await current_with_automatic_transitions()
        return {
            "session": public_session(session).model_dump(mode="json")
            if session
            else None
        }

    @application.post("/api/session/start")
    async def start_session() -> dict[str, Any]:
        active = await store.current(touch=True)
        if active is not None:
            return {"session": public_session(active).model_dump(mode="json")}
        if workers.preparation_state != PreparationState.READY:
            raise HTTPException(status_code=409, detail="interview_workers_not_ready")
        session = await store.create(config.staff)
        session_event(LOGGER, "session_started", session_id=session.session_id)
        return {"session": public_session(session).model_dump(mode="json")}

    @application.post("/api/session/{session_id}/consent")
    async def consent(session_id: str, payload: ConsentRequest) -> dict[str, Any]:
        session = await store.require(session_id)
        if session.state != SessionState.CONSENT:
            raise HTTPException(status_code=409, detail="consent_state_required")
        if not payload.voice_clone_consent:
            raise HTTPException(status_code=422, detail="voice_clone_consent_required")
        await store.begin_device_check(session)
        session_event(LOGGER, "consent_accepted", session_id=session.session_id)
        return {"session": public_session(session).model_dump(mode="json")}

    @application.post("/api/session/{session_id}/device-check/denied")
    async def device_check_denied(session_id: str) -> dict[str, Any]:
        session = await store.require(session_id)
        if session.state != SessionState.DEVICE_CHECK:
            raise HTTPException(status_code=409, detail="device_check_state_required")
        await store.mark_permissions_denied(session)
        return {"session": public_session(session).model_dump(mode="json")}

    @application.post("/api/session/{session_id}/device-check/complete")
    async def device_check_complete(
        session_id: str,
        report: DeviceCheckReport,
    ) -> dict[str, Any]:
        session = await store.require(session_id)
        if session.state != SessionState.DEVICE_CHECK:
            raise HTTPException(status_code=409, detail="device_check_state_required")
        await store.begin_conversation(session, report)
        session_event(
            LOGGER,
            "device_check_completed",
            session_id=session.session_id,
            width=report.camera_width,
            height=report.camera_height,
            face_check_supported=report.face_check_supported,
        )
        return {"session": public_session(session).model_dump(mode="json")}

    @application.post("/api/session/{session_id}/media/chunk")
    async def upload_media_chunk(
        session_id: str,
        request: Request,
        kind: str = Query(pattern="^(video|audio)$"),
        sequence: int = Query(ge=0),
        mime_type: str = Query(default="application/octet-stream", max_length=100),
    ) -> dict[str, Any]:
        session = await store.require(session_id)
        if session.state not in {SessionState.DEVICE_CHECK, SessionState.CONVERSATION}:
            raise HTTPException(status_code=409, detail="capture_state_required")
        payload = await request.body()
        max_bytes = config.developer.capture.max_chunk_size_mb * 1024 * 1024
        if not payload or len(payload) > max_bytes:
            raise HTTPException(status_code=413, detail="invalid_media_chunk_size")
        await store.save_media_chunk(
            session,
            kind=kind,
            sequence=sequence,
            payload=payload,
            mime_type=mime_type,
        )
        return {"ok": True, "bytes": len(payload)}

    @application.post("/api/session/{session_id}/media/upload-failure")
    async def media_upload_failure(session_id: str) -> dict[str, Any]:
        session = await store.require(session_id)
        await store.mark_upload_failure(session)
        return {"ok": True}

    @application.post("/api/session/{session_id}/conversation/ai-finished")
    async def ai_finished(session_id: str) -> dict[str, Any]:
        session = await store.require(session_id)
        if session.state != SessionState.CONVERSATION:
            raise HTTPException(status_code=409, detail="conversation_state_required")
        await store.mark_ai_finished(session)
        return {"session": public_session(session).model_dump(mode="json")}

    @application.post("/api/session/{session_id}/conversation/answer-complete")
    async def answer_complete(
        session_id: str,
        payload: AnswerCompleteRequest,
    ) -> dict[str, Any]:
        session = await store.require(session_id)
        if session.state != SessionState.CONVERSATION:
            raise HTTPException(status_code=409, detail="conversation_state_required")
        await store.complete_answer(session, silence_reason=payload.silence_reason)
        settings = session.settings_snapshot
        warning: str | None = None
        transcript_text = ""
        try:
            audio_path = store.answer_audio_path(session.session_id, payload.sequence)
            asr_output_dir = (
                store.session_dir(session.session_id)
                / "intermediate"
                / "asr"
                / "answers"
                / f"{payload.sequence:06d}"
            )
            asr_result = await workers.run_prepared_role(
                session,
                WorkerRole.STREAMING_ASR,
                input_paths={"audio": str(audio_path)},
                output_dir=str(asr_output_dir),
                deadline_seconds=60,
                metadata={
                    "language": "ja",
                    "beam_size": 5,
                    "chunk_length": 15,
                    "condition_on_previous_text": False,
                    "vad_filter": False,
                    "sequence": payload.sequence,
                    "duration_ms": payload.duration_ms,
                },
            )
            transcript_text = str(asr_result.metadata.get("text", "")).strip()
            if asr_result.implemented and transcript_text:
                await store.add_transcript(session, "visitor", transcript_text)
        except (FileNotFoundError, WorkerExecutionError) as exc:
            warning = "この回答の文字起こしに失敗しました。会話は続行します。"
            session_event(
                LOGGER,
                "answer_asr_failed",
                session_id=session.session_id,
                error_code=getattr(exc, "code", "answer_audio_not_found"),
                sequence=payload.sequence,
            )

        short_complete = (
            config.developer.app.debug_mode
            and settings.debug_test_mode == DebugTestMode.SHORT
            and session.interview_state.answer_count
            >= config.developer.capture.debug_short_answer_count
        )
        target_reached = session.visitor_char_count >= settings.target_transcript_chars
        time_reached = (
            seconds_since(session.conversation_started_at)
            >= settings.conversation_time_limit_seconds
        )
        if session.auto_finish_scheduled:
            session.conversation_phase = ConversationPhase.CLOSING
            await store.persist(session)
        elif short_complete or target_reached or time_reached:
            session.auto_finish_scheduled = True
            session.conversation_phase = ConversationPhase.CLOSING
            session.interview_state.should_end = True
            session.interview_state.end_reason = (
                "debug_short_complete"
                if short_complete
                else "target_transcript_reached"
                if target_reached
                else "conversation_time_limit"
            )
            await store.persist(session)
            spawn(auto_finish(session.session_id, session.interview_state.end_reason))
        else:
            try:
                turn = await generate_next_interview_turn(session)
                await store.apply_interview_turn(session, turn)
            except (WorkerExecutionError, ValueError) as exc:
                warning = "会話の続きの生成に失敗したため、標準の返答で続けます。"
                session_event(
                    LOGGER,
                    "interview_llm_failed",
                    session_id=session.session_id,
                    error_code=getattr(
                        exc,
                        "code",
                        "interview_llm_invalid_output",
                    ),
                )
                await store.advance_question(session)
        return {
            "session": public_session(session).model_dump(mode="json"),
            "warning": warning,
        }

    @application.post("/api/session/{session_id}/conversation/utterance")
    async def add_utterance(
        session_id: str,
        payload: AddUtteranceRequest,
    ) -> dict[str, Any]:
        if not config.developer.app.debug_mode:
            raise HTTPException(status_code=404, detail="debug_mode_disabled")
        session = await store.require(session_id)
        if session.state != SessionState.CONVERSATION:
            raise HTTPException(status_code=409, detail="conversation_state_required")
        await store.add_transcript(session, "visitor", payload.text)
        if session.auto_finish_scheduled:
            session.conversation_phase = ConversationPhase.CLOSING
            await store.persist(session)
        elif (
            session.visitor_char_count
            >= session.settings_snapshot.target_transcript_chars
        ):
            session.auto_finish_scheduled = True
            session.conversation_phase = ConversationPhase.CLOSING
            session.interview_state.should_end = True
            session.interview_state.end_reason = "target_transcript_reached"
            await store.persist(session)
            spawn(auto_finish(session.session_id, "target_transcript_reached"))
        else:
            session.conversation_phase = ConversationPhase.LISTENING
            await store.persist(session)
        session_event(
            LOGGER,
            "debug_utterance_added",
            session_id=session.session_id,
            visitor_char_count=session.visitor_char_count,
        )
        return {"session": public_session(session).model_dump(mode="json")}

    @application.post("/api/session/{session_id}/conversation/finish")
    async def finish_conversation(session_id: str) -> dict[str, Any]:
        session = await store.require(session_id)
        if session.state in {SessionState.GENERATING, SessionState.REVIEW}:
            return {"session": public_session(session).model_dump(mode="json")}
        if session.state != SessionState.CONVERSATION:
            raise HTTPException(status_code=409, detail="conversation_state_required")
        reason = (
            session.interview_state.end_reason
            if session.auto_finish_scheduled and session.interview_state.end_reason
            else "operator_finished"
        )
        await generation.start(session, reason)
        return {"session": public_session(session).model_dump(mode="json")}

    @application.post("/api/session/{session_id}/generation/complete")
    async def force_generation_complete(session_id: str) -> dict[str, Any]:
        if not config.developer.app.debug_mode:
            raise HTTPException(status_code=404, detail="debug_mode_disabled")
        session = await store.require(session_id)
        if session.state != SessionState.GENERATING:
            raise HTTPException(status_code=409, detail="generation_state_required")
        await generation.force_complete(session)
        return {"session": public_session(session).model_dump(mode="json")}

    @application.post("/api/session/{session_id}/review/regenerate")
    async def regenerate_review(session_id: str) -> dict[str, Any]:
        session = await store.require(session_id)
        if session.state != SessionState.REVIEW:
            raise HTTPException(status_code=409, detail="review_state_required")
        await generation.regenerate(session)
        return {"session": public_session(session).model_dump(mode="json")}

    @application.get("/api/session/{session_id}/output/{filename:path}")
    async def session_output(session_id: str, filename: str) -> FileResponse:
        session = await store.require(session_id)
        artifact = session.video_artifact
        if (
            session.state != SessionState.REVIEW
            or not artifact
            or not artifact.implemented
        ):
            raise HTTPException(status_code=404, detail="video_artifact_not_available")
        expected_name = (
            artifact.media_url.rsplit("/", 1)[-1] if artifact.media_url else ""
        )
        if not expected_name or filename != expected_name:
            raise HTTPException(status_code=404, detail="video_artifact_not_found")
        output_path = store.session_dir(session_id) / "output" / expected_name
        if not output_path.is_file():
            raise HTTPException(status_code=404, detail="video_artifact_not_found")
        return FileResponse(
            output_path,
            media_type="video/mp4",
            headers={"Cache-Control": "no-store"},
        )

    @application.get("/api/session/{session_id}/debug/artifacts")
    async def debug_artifacts(session_id: str) -> dict[str, Any]:
        if not config.developer.app.debug_mode:
            raise HTTPException(status_code=404, detail="debug_mode_disabled")
        session = await store.require(session_id)
        if session.state not in {SessionState.ERROR, SessionState.REVIEW}:
            raise HTTPException(status_code=409, detail="debug_artifact_state_required")
        artifacts = store.list_debug_artifacts(session_id)
        for artifact in artifacts:
            relative_path = quote(str(artifact["path"]), safe="/")
            artifact["media_url"] = (
                f"api/session/{session_id}/debug/artifact/{relative_path}"
            )
        return {"retained": store.session_dir(session_id).is_dir(), "artifacts": artifacts}

    @application.get("/api/session/{session_id}/debug/artifact/{artifact_path:path}")
    async def debug_artifact_file(
        session_id: str,
        artifact_path: str,
    ) -> FileResponse:
        if not config.developer.app.debug_mode:
            raise HTTPException(status_code=404, detail="debug_mode_disabled")
        session = await store.require(session_id)
        if session.state not in {SessionState.ERROR, SessionState.REVIEW}:
            raise HTTPException(status_code=409, detail="debug_artifact_state_required")
        try:
            path = store.resolve_debug_artifact(session_id, artifact_path)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        return FileResponse(
            path,
            media_type=media_type,
            headers={"Cache-Control": "no-store"},
        )

    @application.post("/api/session/{session_id}/debug/error")
    async def debug_error(session_id: str) -> dict[str, Any]:
        if not config.developer.app.debug_mode:
            raise HTTPException(status_code=404, detail="debug_mode_disabled")
        session = await store.require(session_id)
        await generation.cancel(session.session_id)
        await generation.fail(
            session,
            "debug_error",
            "デバッグ操作によるエラーです。生成データを一時保持しています。",
        )
        return {"session": public_session(session).model_dump(mode="json")}

    @application.post("/api/session/{session_id}/abandon")
    async def abandon_interview(session_id: str) -> dict[str, Any]:
        session = await store.require(session_id)
        if session.state not in {SessionState.DEVICE_CHECK, SessionState.CONVERSATION}:
            return {"deleted": False}
        await generation.cancel(session.session_id)
        deleted_id = await store.clear()
        if deleted_id:
            session_event(
                LOGGER,
                "session_deleted",
                session_id=deleted_id,
                reason="interview_page_unloaded",
            )
        return {"deleted": bool(deleted_id)}

    @application.post("/api/control/emergency-stop")
    async def emergency_stop() -> dict[str, Any]:
        session = await store.current()
        if session is not None:
            await generation.cancel(session.session_id)
            session_id = session.session_id
        else:
            session_id = "none"
        await workers.emergency_stop()
        await store.clear()
        session_event(
            LOGGER,
            "emergency_stop",
            session_id=session_id,
            reason="operator_confirmed",
        )
        workers.preparation_state = PreparationState.LOADING
        workers.preparation_message = "緊急停止後の再準備をしています"
        spawn(workers.prepare_interview(config.staff))
        return {
            "session": None,
            "preparation": workers.status().model_dump(mode="json"),
        }

    @application.post("/api/control/reset")
    async def reset_demo() -> dict[str, Any]:
        session = await store.current()
        if session is not None:
            await generation.cancel(session.session_id)
        deleted_session_id = await store.clear()
        if deleted_session_id:
            session_event(
                LOGGER,
                "session_deleted",
                session_id=deleted_session_id,
                reason="operator_reset",
            )
        workers.preparation_state = PreparationState.LOADING
        workers.preparation_message = "次のインタビューを準備しています"
        spawn(workers.prepare_interview(config.staff))
        return {
            "session": None,
            "preparation": workers.status().model_dump(mode="json"),
        }

    frontend_dist = config.root_dir / "frontend" / "dist"
    if frontend_dist.exists():
        application.mount(
            "/",
            StaticFiles(directory=frontend_dist, html=True),
            name="frontend",
        )

    return application


app = create_app()
