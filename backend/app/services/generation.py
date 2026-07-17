from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from ..config import ConfigManager
from ..episodes import EpisodeRepository
from ..logging_config import session_event
from ..schemas import (
    GenerationEvent,
    GenerationStep,
    SessionRecord,
    SessionState,
    StepStatus,
    VideoArtifact,
    WorkerLifecycleState,
    WorkerProgressEvent,
    WorkerResult,
    WorkerRole,
    WorkerStatusPublic,
    utc_now,
)
from ..session_store import SessionStore, seconds_since
from ..workers.manager import WorkerExecutionError, WorkerOrchestrator


LOGGER = logging.getLogger(__name__)

STEP_DEFINITIONS: list[tuple[str, str, tuple[WorkerRole, ...]]] = [
    (
        "reflection",
        "あなたとの会話を振り返っています",
        (
            WorkerRole.FINAL_ASR,
            WorkerRole.INTERVIEW_SUMMARY,
            WorkerRole.REFERENCE_FRAME_SELECTOR,
        ),
    ),
    (
        "future-event",
        "未来の出来事を探しています",
        (
            WorkerRole.EPISODE_SELECTOR,
            WorkerRole.SCRIPT_DESIGN_LLM,
            WorkerRole.SCRIPT_SAFETY_REVIEW,
        ),
    ),
    (
        "future-portrait",
        "未来のあなたを描いています",
        (
            WorkerRole.VOICE_REFERENCE_SELECTOR,
            WorkerRole.IMAGE_GENERATION,
        ),
    ),
    (
        "message-video",
        "メッセージ動画を作っています",
        (
            WorkerRole.VOICE_CLONE_TTS,
            WorkerRole.VIDEO_GENERATION,
            WorkerRole.LIP_SYNC,
            WorkerRole.VIDEO_POSTPROCESS,
        ),
    ),
]


class GenerationService:
    def __init__(
        self,
        config: ConfigManager,
        store: SessionStore,
        episodes: EpisodeRepository,
        workers: WorkerOrchestrator,
    ) -> None:
        self.config = config
        self.store = store
        self.episodes = episodes
        self.workers = workers
        self.tasks: dict[str, asyncio.Task[None]] = {}

    async def start(
        self,
        session: SessionRecord,
        reason: str,
        *,
        clear_generated_data: bool = False,
    ) -> SessionRecord:
        await self.cancel(session.session_id)
        settings = session.settings_snapshot
        episode, effect, final_rarity = self.episodes.select(settings)
        if clear_generated_data:
            await self.store.clear_generated_data(session.session_id)

        session.state = SessionState.GENERATING
        session.state_changed_at = utc_now()
        session.generation_started_at = utc_now()
        session.generation_steps = [
            GenerationStep(id=step_id, label=label)
            for step_id, label, _ in STEP_DEFINITIONS
        ]
        session.generation_events = []
        session.selected_episode_id = episode.id
        session.selected_episode_name = episode.name
        session.selected_effect_id = effect.id
        session.selected_effect_name = effect.name
        session.base_rarity = episode.base_rarity
        session.final_rarity = final_rarity
        session.quality_profile = settings.quality_profile
        session.video_artifact = None
        session.error_code = None
        session.error_message = None
        session.error_detail = None
        session.failed_worker_role = None
        session.failed_worker_phase = None
        session.completion_reason = reason
        session.auto_finish_scheduled = False
        session.model_switch_notice = None
        self.workers.reset_generation_statuses()
        self._sync_worker_statuses(session)
        await self.store.persist(session)

        session_event(
            LOGGER,
            "generation_started",
            session_id=session.session_id,
            episode_id=episode.id,
            effect_id=effect.id,
            base_rarity=episode.base_rarity,
            final_rarity=final_rarity,
            quality_profile=settings.quality_profile,
            reason=reason,
        )
        task = asyncio.create_task(self._run(session))
        self.tasks[session.session_id] = task
        task.add_done_callback(lambda _: self.tasks.pop(session.session_id, None))
        return session

    async def regenerate(self, session: SessionRecord) -> SessionRecord:
        return await self.start(
            session,
            "operator_regenerated",
            clear_generated_data=True,
        )

    async def _run(self, session: SessionRecord) -> None:
        timeout_seconds = session.settings_snapshot.generation_time_limit_seconds
        try:
            async with asyncio.timeout(timeout_seconds):
                self._append_event(
                    session,
                    GenerationEvent(
                        phase="interview_release",
                        progress=0,
                        message="インタビュー用モデルを解放しています",
                    ),
                )
                await self.workers.release_interview()
                self._sync_worker_statuses(session)
                await self.store.persist(session)
                artifacts = self._initial_artifacts(session)

                for step, (_, _, roles) in zip(session.generation_steps, STEP_DEFINITIONS):
                    for candidate in session.generation_steps:
                        if candidate.status == StepStatus.CURRENT:
                            candidate.status = StepStatus.COMPLETED
                    step.status = StepStatus.CURRENT
                    self._append_event(
                        session,
                        GenerationEvent(
                            phase=f"step.{step.id}",
                            progress=0,
                            message=step.label,
                        ),
                    )
                    await self.store.persist(session)
                    session_event(
                        LOGGER,
                        "generation_step_started",
                        session_id=session.session_id,
                        step_id=step.id,
                    )

                    for role in roles:
                        if (
                            role == WorkerRole.SCRIPT_SAFETY_REVIEW
                            and session.settings_snapshot.episode_mode == "underground"
                        ):
                            status = self.workers.statuses[role]
                            status.state = WorkerLifecycleState.SKIPPED
                            status.progress = 1
                            status.message = "アングラモードのため省略"
                            self._sync_worker_statuses(session)
                            await self.store.persist(session)
                            continue
                        if (
                            role == WorkerRole.SCRIPT_SAFETY_REVIEW
                            and "inline_safety_review" in artifacts
                        ):
                            status = self.workers.statuses[role]
                            status.state = WorkerLifecycleState.SKIPPED
                            status.progress = 1
                            status.phase = "safety.model_reuse"
                            status.message = "設計モデルを解放せず、そのままジャッジ済み"
                            status.model_id = session.settings_snapshot.stage_models.get(role)
                            self._append_event(
                                session,
                                GenerationEvent(
                                    role=role,
                                    state=WorkerLifecycleState.SKIPPED,
                                    phase="safety.model_reuse",
                                    progress=1,
                                    message="同じモデルプロセスで公開用ジャッジを完了しました",
                                    model_id=status.model_id,
                                ),
                            )
                            self._sync_worker_statuses(session)
                            await self.store.persist(session)
                            continue
                        result = await self._run_role(session, role, artifacts)
                        artifacts.update(result.output_paths)

                    step.status = StepStatus.COMPLETED
                    await self.store.persist(session)

                await self._complete(session, artifacts)
        except TimeoutError:
            await self.fail_timeout(session)
        except asyncio.CancelledError:
            raise
        except WorkerExecutionError as exc:
            LOGGER.exception(
                "event=generation_failed session_id=%s error_code=%s",
                session.session_id,
                exc.code,
            )
            await self.fail(
                session,
                exc.code,
                "AIワーカーの処理に失敗したため、安全に終了しました。",
                worker=exc.worker,
                phase=exc.phase,
                detail=exc.detail,
            )
        except Exception as exc:
            LOGGER.exception(
                "event=generation_failed session_id=%s error_type=%s",
                session.session_id,
                type(exc).__name__,
            )
            await self.fail(
                session,
                "generation_failed",
                "生成処理を安全に終了しました。運営スタッフにお知らせください。",
                phase="generation",
                detail=f"{type(exc).__name__}: {exc}",
            )

    def _initial_artifacts(self, session: SessionRecord) -> dict[str, str]:
        session_dir = self.store.session_dir(session.session_id)
        return {
            "session_dir": str(session_dir),
            "transcript": str(session_dir / "input" / "transcript.json"),
            "interview_state": str(session_dir / "input" / "interview-state.json"),
            "audio_answers_dir": str(session_dir / "input" / "audio" / "answers"),
            "video_chunks_dir": str(session_dir / "input" / "video" / "chunks"),
        }

    async def _run_role(
        self,
        session: SessionRecord,
        role: WorkerRole,
        artifacts: dict[str, str],
    ) -> WorkerResult:
        remaining = max(
            1,
            session.settings_snapshot.generation_time_limit_seconds
            - seconds_since(session.generation_started_at),
        )
        session_dir = self.store.session_dir(session.session_id)
        output_dir = session_dir / "intermediate" / role.value
        output_dir.mkdir(parents=True, exist_ok=True)
        if not session.selected_episode_id or not session.selected_effect_id:
            raise WorkerExecutionError(
                "generation_selection_missing",
                "エピソードまたは追加演出が選択されていません",
            )
        episode = self.episodes.get_episode(session.selected_episode_id)
        effect = self.episodes.get_effect(session.selected_effect_id)
        async def on_status(
            status: WorkerStatusPublic,
            event: WorkerProgressEvent | None,
        ) -> None:
            await self._record_worker_status(session, status, event)

        result = await self.workers.run_role(
            session,
            role,
            input_paths=dict(artifacts),
            output_dir=str(output_dir),
            deadline_seconds=remaining,
            metadata={
                "episode_id": session.selected_episode_id,
                "effect_id": session.selected_effect_id,
                "episode": episode.model_dump(mode="json"),
                "effect": effect.model_dump(mode="json"),
                "final_rarity": (
                    session.final_rarity.value
                    if session.final_rarity
                    else episode.base_rarity.value
                ),
                "episode_mode": session.settings_snapshot.episode_mode,
                "target_video_seconds": 20,
                "inline_safety_review": (
                    role == WorkerRole.SCRIPT_DESIGN_LLM
                    and session.settings_snapshot.episode_mode != "underground"
                    and session.settings_snapshot.stage_models.get(
                        WorkerRole.SCRIPT_DESIGN_LLM
                    )
                    == session.settings_snapshot.stage_models.get(
                        WorkerRole.SCRIPT_SAFETY_REVIEW
                    )
                ),
                "remaining_time_seconds": remaining,
                "person_information": session.interview_state.acquired_information,
                "prohibited_expressions": [
                    "本人の容姿、性格、進路、家族、健康、経済状況への侮辱",
                    "実在人物や既存作品の人物になったと断定する表現",
                    "将来を予言または保証する表現",
                ],
                "capabilities": {
                    "image_model": {
                        "catalog_id": session.settings_snapshot.stage_models.get(
                            WorkerRole.IMAGE_GENERATION
                        ),
                        "single_reference_editing": True,
                        "max_resolution": 768,
                    },
                    "video_model": {
                        "catalog_id": session.settings_snapshot.stage_models.get(
                            WorkerRole.VIDEO_GENERATION
                        ),
                        "inputs": ["image", "audio", "text"],
                    },
                    "voice_model": {
                        "catalog_id": session.settings_snapshot.stage_models.get(
                            WorkerRole.VOICE_CLONE_TTS
                        ),
                        "reference_audio": True,
                    },
                },
            },
            on_status=on_status,
        )
        self._sync_worker_statuses(session)
        await self.store.persist(session)
        return result

    async def _record_worker_status(
        self,
        session: SessionRecord,
        status: WorkerStatusPublic,
        event: WorkerProgressEvent | None,
    ) -> None:
        self._sync_worker_statuses(session)
        self._append_event(
            session,
            GenerationEvent(
                created_at=event.created_at if event else utc_now(),
                role=status.role,
                state=status.state,
                phase=(event.phase if event else status.phase) or status.state.value,
                progress=status.progress,
                message=(event.message if event else status.message) or status.state.value,
                detail=event.detail if event else status.detail,
                model_id=status.model_id,
                backend=status.backend,
                device=status.device,
                request_id=status.request_id,
                attempt=status.attempt,
                error_code=status.error_code,
            ),
        )
        await self.store.persist(session)


    @staticmethod
    def _append_event(session: SessionRecord, event: GenerationEvent) -> None:
        if session.generation_events:
            previous = session.generation_events[-1]
            if (
                previous.role == event.role
                and previous.phase == event.phase
                and previous.message == event.message
                and previous.request_id == event.request_id
                and previous.error_code == event.error_code
            ):
                session.generation_events[-1] = event
                return
        session.generation_events.append(event)
        if len(session.generation_events) > 160:
            session.generation_events = session.generation_events[-160:]

    def _sync_worker_statuses(self, session: SessionRecord) -> None:
        session.worker_statuses = [
            self.workers.statuses[role].model_copy(deep=True)
            for role in WorkerRole
        ]

    async def _complete(
        self,
        session: SessionRecord,
        artifacts: dict[str, str],
    ) -> None:
        session_dir = self.store.session_dir(session.session_id)
        output_dir = session_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        final_video_value = artifacts.get("final_video")
        final_video = Path(final_video_value) if final_video_value else None
        implemented = bool(final_video and final_video.is_file())
        metadata_path = output_dir / (
            "video-metadata.json" if implemented else "video-placeholder.json"
        )
        metadata = {
            "schema_version": "1.0",
            "implemented": implemented,
            "session_id": session.session_id,
            "episode_id": session.selected_episode_id,
            "effect_id": session.selected_effect_id,
            "base_rarity": session.base_rarity,
            "final_rarity": session.final_rarity,
            "quality_profile": session.quality_profile,
            "target_video_seconds": 20,
            "media_file": final_video.name if implemented and final_video else None,
            "message": (
                "未来の自分からのメッセージ動画が完成しました"
                if implemented
                else "動画生成ワーカーは未接続です"
            ),
        }
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        session.video_artifact = VideoArtifact(
            implemented=implemented,
            media_url=(
                f"api/session/{session.session_id}/output/{final_video.name}"
                if implemented and final_video
                else None
            ),
            metadata_path=str(metadata_path.relative_to(session_dir)),
            message=str(metadata["message"]),
        )
        session.state = SessionState.REVIEW
        session.state_changed_at = utc_now()
        self._sync_worker_statuses(session)
        await self.store.persist(session)
        session_event(
            LOGGER,
            "generation_completed",
            session_id=session.session_id,
            episode_id=session.selected_episode_id or "none",
            final_rarity=session.final_rarity or "none",
            implemented=implemented,
        )

    async def force_complete(self, session: SessionRecord) -> SessionRecord:
        await self.cancel(session.session_id)
        await self.workers.cancel_all()
        for step in session.generation_steps:
            step.status = StepStatus.COMPLETED
        await self._complete(session, {})
        return session

    async def fail_timeout(self, session: SessionRecord) -> None:
        task = self.tasks.get(session.session_id)
        if task is not None and task is not asyncio.current_task():
            await self.cancel(session.session_id)
        await self.fail(
            session,
            "generation_time_limit_exceeded",
            "動画生成の制限時間に達したため、処理を安全に終了しました。",
        )

    async def fail(
        self,
        session: SessionRecord,
        code: str,
        message: str,
        *,
        worker: WorkerRole | None = None,
        phase: str | None = None,
        detail: str | None = None,
    ) -> None:
        await self.workers.cancel_all()
        self._sync_worker_statuses(session)
        for step in session.generation_steps:
            if step.status == StepStatus.CURRENT:
                step.status = StepStatus.FAILED

        failed_status = next(
            (
                status
                for status in session.worker_statuses
                if status.role == worker
            ),
            None,
        )
        if failed_status is None:
            failed_status = next(
                (
                    status
                    for status in session.worker_statuses
                    if status.state == WorkerLifecycleState.FAILED
                ),
                None,
            )
        resolved_worker = worker or (failed_status.role if failed_status else None)
        resolved_phase = phase or (failed_status.phase if failed_status else None)
        resolved_detail = detail or (failed_status.detail if failed_status else None)

        session.state = SessionState.ERROR
        session.state_changed_at = utc_now()
        session.error_code = code
        session.error_message = message
        session.error_detail = resolved_detail
        session.failed_worker_role = resolved_worker
        session.failed_worker_phase = resolved_phase
        self._append_event(
            session,
            GenerationEvent(
                role=resolved_worker,
                state=WorkerLifecycleState.FAILED,
                phase=resolved_phase or "generation_failed",
                progress=1,
                message="生成処理を継続できないため安全に終了しました",
                detail=resolved_detail,
                model_id=failed_status.model_id if failed_status else None,
                backend=failed_status.backend if failed_status else None,
                device=failed_status.device if failed_status else None,
                request_id=failed_status.request_id if failed_status else None,
                attempt=failed_status.attempt if failed_status else 0,
                error_code=code,
            ),
        )
        if self.config.developer.app.debug_mode:
            await self.store.persist(session)
        else:
            await self.store.purge_files(session.session_id)
        session_event(
            LOGGER,
            "session_error",
            session_id=session.session_id,
            error_code=code,
            worker=resolved_worker or "none",
            phase=resolved_phase or "unknown",
        )

    async def cancel(self, session_id: str) -> None:
        task = self.tasks.pop(session_id, None)
        if task is None or task.done():
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def shutdown(self) -> None:
        for session_id in list(self.tasks):
            await self.cancel(session_id)
