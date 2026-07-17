from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

try:
    from enum import StrEnum
except ImportError:  # Python 3.10 model-worker environments
    from enum import Enum

    class StrEnum(str, Enum):
        def __str__(self) -> str:
            return str(self.value)


SCHEMA_VERSION = "1.0"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class SessionState(StrEnum):
    CONSENT = "consent"
    DEVICE_CHECK = "device_check"
    CONVERSATION = "conversation"
    GENERATING = "generating"
    REVIEW = "review"
    ERROR = "error"
    STOPPED = "stopped"


class ConversationPhase(StrEnum):
    WAITING = "waiting"
    SPEAKING = "speaking"
    LISTENING = "listening"
    THINKING = "thinking"
    CLOSING = "closing"


class StepStatus(StrEnum):
    PENDING = "pending"
    CURRENT = "current"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"


class Rarity(StrEnum):
    R = "R"
    SR = "SR"
    SSR = "SSR"
    UR = "UR"


class QualityProfile(StrEnum):
    QUALITY = "quality"
    BALANCED = "balanced"
    FAST = "fast"
    CUSTOM = "custom"


class DebugTestMode(StrEnum):
    NORMAL = "normal"
    SHORT = "short"


class PreparationState(StrEnum):
    LOADING = "loading"
    READY = "ready"
    FAILED = "failed"
    STOPPED = "stopped"


class WorkerRole(StrEnum):
    AUDIO_PREPROCESS = "audio_preprocess_worker"
    STREAMING_ASR = "streaming_asr_worker"
    FINAL_ASR = "final_asr_worker"
    INTERVIEW_LLM = "interview_llm_worker"
    INTERVIEW_TTS = "interview_tts_worker"
    INTERVIEW_SUMMARY = "interview_summary_worker"
    EPISODE_SELECTOR = "episode_selector"
    SCRIPT_DESIGN_LLM = "script_design_llm_worker"
    SCRIPT_SAFETY_REVIEW = "script_safety_review_worker"
    REFERENCE_FRAME_SELECTOR = "reference_frame_selector"
    VOICE_REFERENCE_SELECTOR = "voice_reference_selector"
    IMAGE_GENERATION = "image_generation_worker"
    VOICE_CLONE_TTS = "voice_clone_tts_worker"
    VIDEO_GENERATION = "video_generation_worker"
    LIP_SYNC = "lip_sync_worker"
    VIDEO_POSTPROCESS = "video_postprocess_worker"


class WorkerGroup(StrEnum):
    INTERVIEW = "interview"
    MATERIAL_PREPARATION = "material_preparation"
    GENERATION = "generation"
    FINISHING = "finishing"


WORKER_GROUP_ROLES: dict[WorkerGroup, tuple[WorkerRole, ...]] = {
    WorkerGroup.INTERVIEW: (
        WorkerRole.AUDIO_PREPROCESS,
        WorkerRole.STREAMING_ASR,
        WorkerRole.INTERVIEW_LLM,
        WorkerRole.INTERVIEW_TTS,
    ),
    WorkerGroup.MATERIAL_PREPARATION: (
        WorkerRole.FINAL_ASR,
        WorkerRole.INTERVIEW_SUMMARY,
        WorkerRole.EPISODE_SELECTOR,
        WorkerRole.SCRIPT_DESIGN_LLM,
        WorkerRole.SCRIPT_SAFETY_REVIEW,
    ),
    WorkerGroup.GENERATION: (
        WorkerRole.REFERENCE_FRAME_SELECTOR,
        WorkerRole.VOICE_REFERENCE_SELECTOR,
        WorkerRole.IMAGE_GENERATION,
        WorkerRole.VOICE_CLONE_TTS,
    ),
    WorkerGroup.FINISHING: (
        WorkerRole.VIDEO_GENERATION,
        WorkerRole.LIP_SYNC,
        WorkerRole.VIDEO_POSTPROCESS,
    ),
}

ROLE_GROUP: dict[WorkerRole, WorkerGroup] = {
    role: group for group, roles in WORKER_GROUP_ROLES.items() for role in roles
}


class WorkerLifecycleState(StrEnum):
    STOPPED = "stopped"
    STARTING = "starting"
    LOADING = "loading"
    READY = "ready"
    RUNNING = "running"
    CANCELLING = "cancelling"
    UNLOADING = "unloading"
    FAILED = "failed"
    SKIPPED = "skipped"


class RangeLimit(BaseModel):
    min: int
    max: int
    step: int = 1


class StaffLimits(BaseModel):
    generation_time_limit_seconds: RangeLimit
    target_transcript_chars: RangeLimit
    minimum_transcript_chars: RangeLimit
    conversation_time_limit_seconds: RangeLimit


class StaffSettings(BaseModel):
    generation_time_limit_seconds: int = 1800
    quality_profile: QualityProfile = QualityProfile.BALANCED
    stage_models: dict[WorkerRole, str] = Field(default_factory=dict)
    episode_mode: Literal["formal", "underground"] = "formal"
    episode_selection: Literal["random", "fixed"] = "random"
    fixed_episode_id: str | None = None
    auto_model_fallback: bool = False
    simple_video_fallback: bool = False
    allow_video_download: bool = False
    target_transcript_chars: int = 700
    minimum_transcript_chars: int = 400
    conversation_time_limit_seconds: int = 180
    debug_test_mode: DebugTestMode = DebugTestMode.NORMAL

    @model_validator(mode="after")
    def validate_relationships(self) -> "StaffSettings":
        if self.minimum_transcript_chars > self.target_transcript_chars:
            raise ValueError("最低文字数は目標文字数以下にしてください")
        return self


class ModelLicenseRecord(BaseModel):
    code_license: str | None = None
    weight_license: str | None = None
    research_use: str | None = None
    noncommercial_demo: str | None = None
    redistribution: str | None = None
    output_terms: str | None = None
    attribution: str | None = None
    restrictions: list[str] = Field(default_factory=list)


class ModelCatalogEntry(BaseModel):
    id: str
    label: str
    description: str
    roles: list[WorkerRole]
    backend: str
    model_id: str
    revision: str
    dtype: str
    quantization: str
    device: str
    environment: str
    python_bin: str
    command: list[str] = Field(default_factory=list)
    adapter_entrypoint: str = "backend.app.workers.base:create_stub_worker"
    model_path: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: int = Field(gt=0)
    fallback_model_id: str | None = None
    installed: bool = False
    validated: bool = False
    last_healthcheck: Literal["passed", "failed", "not_run"] = "not_run"
    is_stub: bool = False
    official_sources: list[str] = Field(default_factory=list)
    checked_at: str | None = None
    license: ModelLicenseRecord = Field(default_factory=ModelLicenseRecord)


class ModelCatalog(BaseModel):
    schema_version: str = SCHEMA_VERSION
    profiles: dict[QualityProfile, dict[WorkerRole, str]]
    models: list[ModelCatalogEntry]


class ModelOption(BaseModel):
    id: str
    label: str
    description: str
    roles: list[WorkerRole]
    backend: str
    model_id: str
    revision: str
    dtype: str
    quantization: str
    device: str
    enabled: bool = True
    is_stub: bool = False


class Episode(BaseModel):
    schema_version: str = SCHEMA_VERSION
    id: str
    name: str
    content: str
    base_rarity: Rarity
    weight: float = Field(gt=0)
    enabled: bool = True
    formal_mode_allowed: bool = True
    public_demo_allowed: bool = True
    limited_only: bool = False
    min_age: int | None = None
    max_age: int | None = None
    generation_instruction: str = ""
    tags: list[str] = Field(default_factory=list)


class EpisodeEffect(BaseModel):
    schema_version: str = SCHEMA_VERSION
    id: str
    name: str
    content: str
    rarity_upgrade_steps: int = Field(default=0, ge=0, le=3)
    weight: float = Field(gt=0)
    enabled: bool = True
    formal_mode_allowed: bool = True
    generation_instruction: str = ""


class EpisodeSummary(BaseModel):
    id: str
    name: str
    base_rarity: Rarity
    formal_mode_allowed: bool
    public_demo_allowed: bool
    limited_only: bool


class TranscriptEntry(BaseModel):
    speaker: Literal["visitor", "ai"]
    text: str
    created_at: datetime = Field(default_factory=utc_now)


class InterviewState(BaseModel):
    schema_version: str = SCHEMA_VERSION
    acquired_information: dict[str, Any] = Field(default_factory=dict)
    asked_topics: list[str] = Field(default_factory=list)
    next_topics: list[str] = Field(default_factory=list)
    visitor_char_count: int = 0
    elapsed_seconds: int = 0
    answer_count: int = 0
    current_question_id: str | None = None
    current_question_text: str | None = None
    should_end: bool = False
    end_reason: str | None = None
    next_utterance: str | None = None


class CaptureStats(BaseModel):
    camera_permission: Literal["unknown", "granted", "denied"] = "unknown"
    microphone_permission: Literal["unknown", "granted", "denied"] = "unknown"
    camera_width: int | None = None
    camera_height: int | None = None
    camera_fps: float | None = None
    face_check_supported: bool = False
    face_detected: bool | None = None
    brightness: float | None = None
    video_chunk_count: int = 0
    audio_segment_count: int = 0
    uploaded_bytes: int = 0
    upload_failure_count: int = 0
    last_silence_reason: str | None = None
    recording_started_at: datetime | None = None
    recording_duration_seconds: int = 0


class GenerationStep(BaseModel):
    id: str
    label: str
    status: StepStatus = StepStatus.PENDING


class WorkerStatusPublic(BaseModel):
    role: WorkerRole
    group: WorkerGroup
    state: WorkerLifecycleState = WorkerLifecycleState.STOPPED
    phase: str | None = None
    catalog_id: str | None = None
    model_id: str | None = None
    model_revision: str | None = None
    backend: str | None = None
    dtype: str | None = None
    quantization: str | None = None
    device: str | None = None
    request_id: str | None = None
    attempt: int = 0
    progress: float = Field(default=0, ge=0, le=1)
    message: str | None = None
    detail: str | None = None
    phase_started_at: datetime | None = None
    updated_at: datetime | None = None
    load_time_ms: int | None = None
    processing_time_ms: int | None = None
    peak_vram_mb: int | None = None
    peak_cpu_memory_mb: int | None = None
    error_code: str | None = None


class GenerationEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    created_at: datetime = Field(default_factory=utc_now)
    role: WorkerRole | None = None
    state: WorkerLifecycleState | None = None
    phase: str
    progress: float = Field(default=0, ge=0, le=1)
    message: str
    detail: str | None = None
    model_id: str | None = None
    backend: str | None = None
    device: str | None = None
    request_id: str | None = None
    attempt: int = 0
    error_code: str | None = None


class VideoArtifact(BaseModel):
    implemented: bool = False
    media_url: str | None = None
    metadata_path: str | None = None
    message: str = "動画生成ワーカーは未接続です"
    ai_generated_label: str = "AI生成映像"


class SessionRecord(BaseModel):
    session_id: str
    settings_snapshot: StaffSettings
    state: SessionState = SessionState.CONSENT
    created_at: datetime = Field(default_factory=utc_now)
    state_changed_at: datetime = Field(default_factory=utc_now)
    last_seen_at: datetime = Field(default_factory=utc_now)
    device_check_started_at: datetime | None = None
    conversation_started_at: datetime | None = None
    generation_started_at: datetime | None = None
    conversation_phase: ConversationPhase = ConversationPhase.WAITING
    transcript: list[TranscriptEntry] = Field(default_factory=list)
    interview_state: InterviewState = Field(default_factory=InterviewState)
    capture_stats: CaptureStats = Field(default_factory=CaptureStats)
    generation_steps: list[GenerationStep] = Field(default_factory=list)
    worker_statuses: list[WorkerStatusPublic] = Field(default_factory=list)
    generation_events: list[GenerationEvent] = Field(default_factory=list)
    selected_episode_id: str | None = None
    selected_episode_name: str | None = None
    selected_effect_id: str | None = None
    selected_effect_name: str | None = None
    base_rarity: Rarity | None = None
    final_rarity: Rarity | None = None
    quality_profile: QualityProfile | None = None
    video_artifact: VideoArtifact | None = None
    error_code: str | None = None
    error_message: str | None = None
    error_detail: str | None = None
    failed_worker_role: WorkerRole | None = None
    failed_worker_phase: str | None = None
    completion_reason: str | None = None
    auto_finish_scheduled: bool = False
    model_switch_notice: str | None = None

    @property
    def visitor_char_count(self) -> int:
        return sum(
            len(entry.text.strip())
            for entry in self.transcript
            if entry.speaker == "visitor"
        )


class SessionPublic(BaseModel):
    session_id: str
    state: SessionState
    created_at: datetime
    state_changed_at: datetime
    elapsed_seconds: int
    conversation_elapsed_seconds: int
    generation_elapsed_seconds: int
    conversation_phase: ConversationPhase
    visitor_char_count: int
    latest_visitor_transcript: str | None
    target_transcript_chars: int
    minimum_transcript_chars: int
    conversation_time_limit_seconds: int
    generation_time_limit_seconds: int
    generation_steps: list[GenerationStep]
    worker_statuses: list[WorkerStatusPublic]
    generation_events: list[GenerationEvent]
    selected_episode_id: str | None
    selected_episode_name: str | None
    selected_effect_id: str | None
    selected_effect_name: str | None
    base_rarity: Rarity | None
    final_rarity: Rarity | None
    quality_profile: QualityProfile | None
    stage_models: dict[WorkerRole, str]
    allow_video_download: bool
    episode_mode: Literal["formal", "underground"]
    current_question_id: str | None
    current_question_text: str | None
    answer_count: int
    capture_stats: CaptureStats
    video_artifact: VideoArtifact | None
    error_code: str | None
    error_message: str | None
    error_detail: str | None
    failed_worker_role: WorkerRole | None
    failed_worker_phase: str | None
    completion_reason: str | None
    model_switch_notice: str | None


class ConsentRequest(BaseModel):
    voice_clone_consent: bool


class DeviceCheckReport(BaseModel):
    camera_width: int = Field(gt=0)
    camera_height: int = Field(gt=0)
    camera_fps: float | None = Field(default=None, ge=0)
    face_check_supported: bool = False
    face_detected: bool | None = None
    brightness: float | None = Field(default=None, ge=0, le=255)


class AnswerCompleteRequest(BaseModel):
    sequence: int = Field(ge=0)
    duration_ms: int = Field(ge=0)
    silence_reason: Literal["silence", "max_duration", "operator", "debug"]
    byte_count: int = Field(ge=0)


class AddUtteranceRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


class RuntimeOptions(BaseModel):
    models: list[ModelOption]
    profile_models: dict[QualityProfile, dict[WorkerRole, str]]
    episodes: list[EpisodeSummary]
    worker_groups: dict[WorkerGroup, list[WorkerRole]]
    limits: StaffLimits
    debug_mode: bool
    config_precedence: list[str]


class PreparationGroupStatus(BaseModel):
    group: WorkerGroup
    state: WorkerLifecycleState
    roles: list[WorkerStatusPublic] = Field(default_factory=list)
    error_code: str | None = None


class PreparationPublic(BaseModel):
    state: PreparationState
    message: str
    groups: list[PreparationGroupStatus] = Field(default_factory=list)
    retry_available: bool = True


class WorkerModelSpec(BaseModel):
    schema_version: str = SCHEMA_VERSION
    worker: WorkerRole
    backend: str
    catalog_id: str
    model_id: str
    model_revision: str
    dtype: str
    quantization: str
    device: str
    adapter_entrypoint: str
    model_path: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: int = Field(gt=0)
    fallback_model_id: str | None = None


class WorkerRequest(BaseModel):
    schema_version: str = SCHEMA_VERSION
    request_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    worker: WorkerRole
    session_id: str
    model: WorkerModelSpec
    deadline_seconds: int = Field(gt=0)
    input_paths: dict[str, str] = Field(default_factory=dict)
    output_dir: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkerProgressEvent(BaseModel):
    schema_version: str = SCHEMA_VERSION
    request_id: str
    worker: WorkerRole
    progress: float = Field(ge=0, le=1)
    message: str
    phase: str = "inference"
    detail: str | None = None
    created_at: datetime = Field(default_factory=utc_now)


class WorkerError(BaseModel):
    code: str
    category: Literal[
        "configuration",
        "load",
        "input",
        "timeout",
        "cancelled",
        "resource",
        "schema",
        "runtime",
    ]
    retryable: bool
    message: str


class WorkerMetrics(BaseModel):
    load_time_ms: int = 0
    processing_time_ms: int = 0
    unload_time_ms: int = 0
    peak_vram_mb: int = 0
    peak_cpu_memory_mb: int = 0


class WorkerResult(BaseModel):
    schema_version: str = SCHEMA_VERSION
    request_id: str
    worker: WorkerRole
    backend: str
    model_id: str
    model_revision: str
    implemented: bool = False
    output_paths: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    metrics: WorkerMetrics = Field(default_factory=WorkerMetrics)
    error: WorkerError | None = None


class WorkerHealth(BaseModel):
    schema_version: str = SCHEMA_VERSION
    worker: WorkerRole
    loaded: bool
    ready: bool
    backend: str | None = None
    model_id: str | None = None
    error_code: str | None = None
