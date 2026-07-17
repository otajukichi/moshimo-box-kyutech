from __future__ import annotations

import asyncio
import json
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from .schemas import (
    CaptureStats,
    ConversationPhase,
    DeviceCheckReport,
    SessionPublic,
    SessionRecord,
    SessionState,
    StaffSettings,
    TranscriptEntry,
    WorkerRole,
    utc_now,
)

if TYPE_CHECKING:
    from .contracts import InterviewTurnOutput


INTERVIEW_QUESTIONS = [
    ("future-question", "未来の自分に何か聞きたいことはありますか？"),
    (
        "future-scene",
        "いいね〜。未来の自分からその答えが届く場所は、どんな景色になっていそう？",
    ),
    (
        "future-surprise",
        "それ、面白い！未来の自分が予想外の一言を返すなら、どんな感じだと思う？",
    ),
    (
        "future-message",
        "だんだん未来が見えてきたね。未来の自分から一言だけ届くなら、何と聞こえてきそう？",
    ),
]

FALLBACK_INTERVIEW_QUESTIONS = (
    ("future-invention", "その未来に一つだけ発明を持ち込めるなら、どんなものを試してみたい？"),
    ("future-action", "その未来へ着いた瞬間、まず何をしてみたい？"),
    ("future-rule", "今とは違う不思議なルールがあるなら、どんな世界が面白そう？"),
    ("future-souvenir", "未来から一つ持ち帰れるなら、何を選んでみたい？"),
    ("future-crisis", "未来のあなたが大活躍するなら、どんなピンチを解決してほしい？"),
    ("future-title", "その未来の一日を映画にするなら、どんなタイトルを付けたい？"),
)

DEBUG_IMAGE_EXTENSIONS = frozenset({".jpeg", ".jpg", ".png", ".webp"})
DEBUG_AUDIO_EXTENSIONS = frozenset({".flac", ".m4a", ".mp3", ".ogg", ".wav"})
DEBUG_VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".webm"})
DEBUG_TEXT_EXTENSIONS = frozenset({".json", ".log", ".md", ".txt", ".yaml", ".yml"})
DEBUG_ARTIFACT_EXTENSIONS = (
    DEBUG_IMAGE_EXTENSIONS
    | DEBUG_AUDIO_EXTENSIONS
    | DEBUG_VIDEO_EXTENSIONS
    | DEBUG_TEXT_EXTENSIONS
)


class SessionNotFoundError(LookupError):
    pass


class SessionStore:
    def __init__(self, session_root: Path) -> None:
        self.session_root = session_root
        self._active: SessionRecord | None = None
        self._lock = asyncio.Lock()

    async def startup_cleanup(self) -> None:
        self.session_root.mkdir(parents=True, exist_ok=True)
        for child in self.session_root.iterdir():
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            elif child.name != ".gitkeep":
                child.unlink(missing_ok=True)

    async def create(self, settings: StaffSettings) -> SessionRecord:
        async with self._lock:
            if self._active is not None:
                await self._purge_files_unlocked(self._active.session_id)
            session = SessionRecord(
                session_id=uuid.uuid4().hex,
                settings_snapshot=settings.model_copy(deep=True),
                quality_profile=settings.quality_profile,
            )
            self._active = session
            self._create_session_directories(session.session_id)
            await self._persist_unlocked(session)
            return session

    async def current(self, *, touch: bool = False) -> SessionRecord | None:
        async with self._lock:
            if self._active and touch:
                self._active.last_seen_at = utc_now()
            return self._active

    async def require(self, session_id: str | None = None) -> SessionRecord:
        async with self._lock:
            if self._active is None:
                raise SessionNotFoundError("active_session_not_found")
            if session_id is not None and self._active.session_id != session_id:
                raise SessionNotFoundError("session_not_found")
            self._active.last_seen_at = utc_now()
            return self._active

    async def begin_device_check(self, session: SessionRecord) -> None:
        session.state = SessionState.DEVICE_CHECK
        session.state_changed_at = utc_now()
        session.device_check_started_at = utc_now()
        await self.persist(session)

    async def begin_conversation(
        self,
        session: SessionRecord,
        report: DeviceCheckReport,
    ) -> None:
        capture = session.capture_stats
        capture.camera_permission = "granted"
        capture.microphone_permission = "granted"
        capture.camera_width = report.camera_width
        capture.camera_height = report.camera_height
        capture.camera_fps = report.camera_fps
        capture.face_check_supported = report.face_check_supported
        capture.face_detected = report.face_detected
        capture.brightness = report.brightness
        capture.recording_started_at = capture.recording_started_at or utc_now()
        session.state = SessionState.CONVERSATION
        session.state_changed_at = utc_now()
        session.conversation_started_at = utc_now()
        session.conversation_phase = ConversationPhase.SPEAKING
        self._set_question(session, 0)
        await self.persist(session)

    async def mark_permissions_denied(self, session: SessionRecord) -> None:
        session.capture_stats.camera_permission = "denied"
        session.capture_stats.microphone_permission = "denied"
        await self.persist(session)

    async def mark_ai_finished(self, session: SessionRecord) -> None:
        session.conversation_phase = ConversationPhase.LISTENING
        await self.persist(session)

    async def complete_answer(
        self,
        session: SessionRecord,
        *,
        silence_reason: str,
    ) -> None:
        session.interview_state.answer_count += 1
        session.capture_stats.last_silence_reason = silence_reason
        session.conversation_phase = ConversationPhase.THINKING
        session.interview_state.visitor_char_count = session.visitor_char_count
        await self.persist(session)

    async def advance_question(self, session: SessionRecord) -> None:
        next_index = session.interview_state.answer_count
        self._set_question(session, next_index)
        session.conversation_phase = ConversationPhase.SPEAKING
        await self.persist(session)

    async def apply_interview_turn(
        self,
        session: SessionRecord,
        turn: "InterviewTurnOutput",
    ) -> None:
        state = session.interview_state
        state.acquired_information = {
            **state.acquired_information,
            **turn.acquired_information,
        }
        state.asked_topics = list(
            dict.fromkeys([*state.asked_topics, *turn.asked_topics])
        )
        state.next_topics = turn.next_topics
        state.visitor_char_count = session.visitor_char_count
        state.elapsed_seconds = seconds_since(session.conversation_started_at)
        state.should_end = False
        state.end_reason = "continue"
        state.current_question_id = f"llm-{state.answer_count}"
        state.current_question_text = turn.next_utterance
        state.next_utterance = turn.next_utterance
        session.transcript.append(
            TranscriptEntry(speaker="ai", text=turn.next_utterance)
        )
        session.conversation_phase = ConversationPhase.SPEAKING
        await self.persist(session)

    def _set_question(self, session: SessionRecord, index: int) -> None:
        if index < len(INTERVIEW_QUESTIONS):
            question_id, question_text = INTERVIEW_QUESTIONS[index]
        else:
            fallback_index = (index - len(INTERVIEW_QUESTIONS)) % len(
                FALLBACK_INTERVIEW_QUESTIONS
            )
            base_id, question_text = FALLBACK_INTERVIEW_QUESTIONS[fallback_index]
            question_id = f"{base_id}-{index}"
        session.interview_state.current_question_id = question_id
        session.interview_state.current_question_text = question_text
        session.interview_state.next_utterance = question_text
        if question_id not in session.interview_state.asked_topics:
            session.interview_state.asked_topics.append(question_id)
            session.transcript.append(TranscriptEntry(speaker="ai", text=question_text))

    async def add_transcript(
        self,
        session: SessionRecord,
        speaker: str,
        text: str,
    ) -> None:
        session.transcript.append(TranscriptEntry(speaker=speaker, text=text))
        session.interview_state.visitor_char_count = session.visitor_char_count
        session.conversation_phase = ConversationPhase.THINKING
        await self.persist(session)

    async def persist(self, session: SessionRecord) -> None:
        async with self._lock:
            await self._persist_unlocked(session)

    async def _persist_unlocked(self, session: SessionRecord) -> None:
        session_dir = self._session_dir(session.session_id)
        if not session_dir.exists():
            return
        session.interview_state.elapsed_seconds = seconds_since(
            session.conversation_started_at
        )
        if session.capture_stats.recording_started_at:
            session.capture_stats.recording_duration_seconds = seconds_since(
                session.capture_stats.recording_started_at
            )
        metadata = session.model_dump(
            mode="json",
            exclude={"transcript", "interview_state"},
        )
        self._atomic_json(session_dir / "session.json", metadata)
        self._atomic_json(
            session_dir / "input" / "transcript.json",
            [entry.model_dump(mode="json") for entry in session.transcript],
        )
        self._atomic_json(
            session_dir / "input" / "interview-state.json",
            session.interview_state.model_dump(mode="json"),
        )

    @staticmethod
    def _atomic_json(path: Path, value: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    async def save_media_chunk(
        self,
        session: SessionRecord,
        *,
        kind: str,
        sequence: int,
        payload: bytes,
        mime_type: str,
    ) -> Path:
        if kind not in {"video", "audio"}:
            raise ValueError("unsupported_media_kind")
        extension = ".webm"
        if "ogg" in mime_type:
            extension = ".ogg"
        elif "mp4" in mime_type:
            extension = ".mp4"
        elif "wav" in mime_type:
            extension = ".wav"
        folder = (
            self._session_dir(session.session_id) / "input" / kind / "chunks"
            if kind == "video"
            else self._session_dir(session.session_id) / "input" / "audio" / "answers"
        )
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{kind}-{sequence:06d}{extension}"
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(payload)
        temporary.replace(path)
        if kind == "video":
            session.capture_stats.video_chunk_count += 1
        else:
            session.capture_stats.audio_segment_count += 1
        session.capture_stats.uploaded_bytes += len(payload)
        await self.persist(session)
        return path

    def answer_audio_path(self, session_id: str, sequence: int) -> Path:
        folder = self._session_dir(session_id) / "input" / "audio" / "answers"
        candidates = sorted(
            path
            for path in folder.glob(f"audio-{sequence:06d}.*")
            if not path.name.endswith(".tmp")
        )
        if not candidates:
            raise FileNotFoundError(f"answer_audio_not_found: {sequence}")
        return max(candidates, key=lambda candidate: candidate.stat().st_mtime_ns)

    async def mark_upload_failure(self, session: SessionRecord) -> None:
        session.capture_stats.upload_failure_count += 1
        await self.persist(session)

    async def clear_generated_data(self, session_id: str) -> None:
        async with self._lock:
            root = self._session_dir(session_id)
            for relative in ("intermediate", "output"):
                target = root / relative
                shutil.rmtree(target, ignore_errors=True)
                target.mkdir(parents=True, exist_ok=True)

    async def purge_files(self, session_id: str) -> None:
        async with self._lock:
            await self._purge_files_unlocked(session_id)

    async def _purge_files_unlocked(self, session_id: str) -> None:
        shutil.rmtree(self._session_dir(session_id), ignore_errors=True)

    async def clear(self) -> str | None:
        async with self._lock:
            if self._active is None:
                return None
            session_id = self._active.session_id
            await self._purge_files_unlocked(session_id)
            self._active = None
            return session_id

    async def clear_if_stale(self, stale_seconds: int) -> str | None:
        async with self._lock:
            if not self._active:
                return None
            if seconds_since(self._active.last_seen_at) < stale_seconds:
                return None
            session_id = self._active.session_id
            await self._purge_files_unlocked(session_id)
            self._active = None
            return session_id

    def list_debug_artifacts(self, session_id: str) -> list[dict[str, object]]:
        root = self._session_dir(session_id)
        if not root.is_dir():
            return []

        candidates = [
            root / "session.json",
            root / "input" / "transcript.json",
            root / "input" / "interview-state.json",
        ]
        for directory_name in ("intermediate", "output"):
            directory = root / directory_name
            if directory.is_dir():
                candidates.extend(path for path in directory.rglob("*") if path.is_file())

        artifacts: list[dict[str, object]] = []
        for path in sorted(set(candidates)):
            suffix = path.suffix.lower()
            if not path.is_file() or suffix not in DEBUG_ARTIFACT_EXTENSIONS:
                continue
            relative_path = path.relative_to(root).as_posix()
            kind = self._debug_artifact_kind(suffix)
            artifacts.append(
                {
                    "path": relative_path,
                    "name": path.name,
                    "kind": kind,
                    "size_bytes": path.stat().st_size,
                    "text_preview": (
                        self._debug_text_preview(path) if kind == "text" else None
                    ),
                }
            )
        return artifacts

    def resolve_debug_artifact(self, session_id: str, relative_path: str) -> Path:
        root = self._session_dir(session_id).resolve()
        relative = Path(relative_path)
        parts = relative.parts
        allowed_input_files = {
            "input/transcript.json",
            "input/interview-state.json",
        }
        normalized = relative.as_posix()
        allowed = (
            normalized == "session.json"
            or normalized in allowed_input_files
            or (parts and parts[0] in {"intermediate", "output"})
        )
        if relative.is_absolute() or ".." in parts or not allowed:
            raise FileNotFoundError("debug_artifact_not_allowed")

        candidate = (root / relative).resolve()
        if root not in candidate.parents or not candidate.is_file():
            raise FileNotFoundError("debug_artifact_not_found")
        if candidate.suffix.lower() not in DEBUG_ARTIFACT_EXTENSIONS:
            raise FileNotFoundError("debug_artifact_type_not_allowed")
        return candidate

    @staticmethod
    def _debug_artifact_kind(suffix: str) -> str:
        if suffix in DEBUG_IMAGE_EXTENSIONS:
            return "image"
        if suffix in DEBUG_AUDIO_EXTENSIONS:
            return "audio"
        if suffix in DEBUG_VIDEO_EXTENSIONS:
            return "video"
        return "text"

    @staticmethod
    def _debug_text_preview(path: Path, limit: int = 16000) -> str:
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return f"プレビューを読み込めませんでした: {exc}"
        if len(content) > limit:
            return f"{content[:limit]}\n…（以降省略）"
        return content

    def session_dir(self, session_id: str) -> Path:
        return self._session_dir(session_id)

    def _create_session_directories(self, session_id: str) -> None:
        root = self._session_dir(session_id)
        for relative in (
            "input/audio/answers",
            "input/video/chunks",
            "intermediate",
            "output",
        ):
            (root / relative).mkdir(parents=True, exist_ok=True)

    def _session_dir(self, session_id: str) -> Path:
        return self.session_root / session_id


def seconds_since(value: datetime | None) -> int:
    if value is None:
        return 0
    return max(0, int((utc_now() - value).total_seconds()))


def public_session(session: SessionRecord) -> SessionPublic:
    settings = session.settings_snapshot
    capture = session.capture_stats.model_copy(deep=True)
    if capture.recording_started_at:
        capture.recording_duration_seconds = seconds_since(capture.recording_started_at)
    statuses = {item.role: item for item in session.worker_statuses}
    ordered_statuses = [
        statuses.get(role) or session_status_default(role)
        for role in WorkerRole
    ]
    latest_visitor_transcript = next(
        (entry.text for entry in reversed(session.transcript) if entry.speaker == "visitor"),
        None,
    )
    return SessionPublic(
        session_id=session.session_id,
        state=session.state,
        created_at=session.created_at,
        state_changed_at=session.state_changed_at,
        elapsed_seconds=seconds_since(session.created_at),
        conversation_elapsed_seconds=seconds_since(session.conversation_started_at),
        generation_elapsed_seconds=seconds_since(session.generation_started_at),
        conversation_phase=session.conversation_phase,
        visitor_char_count=session.visitor_char_count,
        latest_visitor_transcript=latest_visitor_transcript,
        target_transcript_chars=settings.target_transcript_chars,
        minimum_transcript_chars=settings.minimum_transcript_chars,
        conversation_time_limit_seconds=settings.conversation_time_limit_seconds,
        generation_time_limit_seconds=settings.generation_time_limit_seconds,
        generation_steps=session.generation_steps,
        worker_statuses=ordered_statuses,
        generation_events=session.generation_events,
        selected_episode_id=session.selected_episode_id,
        selected_episode_name=session.selected_episode_name,
        selected_effect_id=session.selected_effect_id,
        selected_effect_name=session.selected_effect_name,
        base_rarity=session.base_rarity,
        final_rarity=session.final_rarity,
        quality_profile=session.quality_profile,
        stage_models=settings.stage_models,
        allow_video_download=settings.allow_video_download,
        episode_mode=settings.episode_mode,
        current_question_id=session.interview_state.current_question_id,
        current_question_text=session.interview_state.current_question_text,
        answer_count=session.interview_state.answer_count,
        capture_stats=capture,
        video_artifact=session.video_artifact,
        error_code=session.error_code,
        error_message=session.error_message,
        error_detail=session.error_detail,
        failed_worker_role=session.failed_worker_role,
        failed_worker_phase=session.failed_worker_phase,
        completion_reason=session.completion_reason,
        model_switch_notice=session.model_switch_notice,
    )


def session_status_default(role: WorkerRole):
    from .schemas import ROLE_GROUP, WorkerStatusPublic

    return WorkerStatusPublic(role=role, group=ROLE_GROUP[role])
