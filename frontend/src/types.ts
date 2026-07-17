export type SessionState =
  | "consent"
  | "device_check"
  | "conversation"
  | "generating"
  | "review"
  | "error"
  | "stopped";

export type ConversationPhase =
  | "waiting"
  | "speaking"
  | "listening"
  | "thinking"
  | "closing";

export type Rarity = "R" | "SR" | "SSR" | "UR";
export type QualityProfile = "quality" | "balanced" | "fast" | "custom";
export type DebugTestMode = "normal" | "short";

export type WorkerRole =
  | "audio_preprocess_worker"
  | "streaming_asr_worker"
  | "final_asr_worker"
  | "interview_llm_worker"
  | "interview_tts_worker"
  | "interview_summary_worker"
  | "episode_selector"
  | "script_design_llm_worker"
  | "script_safety_review_worker"
  | "reference_frame_selector"
  | "voice_reference_selector"
  | "image_generation_worker"
  | "voice_clone_tts_worker"
  | "video_generation_worker"
  | "lip_sync_worker"
  | "video_postprocess_worker";

export type WorkerGroup =
  | "interview"
  | "material_preparation"
  | "generation"
  | "finishing";

export interface AppConfig {
  app_name: string;
  base_path: string;
  debug_mode: boolean;
  capture: {
    video_chunk_seconds: number;
    silence_seconds: number;
    speech_start_threshold: number;
    response_max_seconds: number;
    upload_retry_count: number;
    browser_queue_limit_mb: number;
    camera_stable_seconds: number;
    brightness_min: number;
    brightness_max: number;
  };
  notices: {
    privacy: string;
    fiction: string;
    implementation: string;
  };
}

export interface ModelOption {
  id: string;
  label: string;
  description: string;
  roles: WorkerRole[];
  backend: string;
  model_id: string;
  revision: string;
  dtype: string;
  quantization: string;
  device: string;
  enabled: boolean;
  is_stub: boolean;
}

export interface EpisodeOption {
  id: string;
  name: string;
  base_rarity: Rarity;
  formal_mode_allowed: boolean;
  public_demo_allowed: boolean;
  limited_only: boolean;
}

export interface RangeLimit {
  min: number;
  max: number;
  step: number;
}

export interface StaffSettings {
  generation_time_limit_seconds: number;
  quality_profile: QualityProfile;
  stage_models: Record<WorkerRole, string>;
  episode_mode: "formal" | "underground";
  episode_selection: "random" | "fixed";
  fixed_episode_id: string | null;
  auto_model_fallback: boolean;
  simple_video_fallback: boolean;
  allow_video_download: boolean;
  target_transcript_chars: number;
  minimum_transcript_chars: number;
  conversation_time_limit_seconds: number;
  debug_test_mode: DebugTestMode;
}

export interface RuntimeOptions {
  models: ModelOption[];
  profile_models: Record<Exclude<QualityProfile, "custom">, Record<WorkerRole, string>>;
  episodes: EpisodeOption[];
  worker_groups: Record<WorkerGroup, WorkerRole[]>;
  limits: {
    generation_time_limit_seconds: RangeLimit;
    target_transcript_chars: RangeLimit;
    minimum_transcript_chars: RangeLimit;
    conversation_time_limit_seconds: RangeLimit;
  };
  debug_mode: boolean;
  config_precedence: string[];
}

export interface SettingsResponse {
  settings: StaffSettings;
  options: RuntimeOptions;
}

export interface GenerationStep {
  id: string;
  label: string;
  status: "pending" | "current" | "completed" | "skipped" | "failed";
}

export interface WorkerStatus {
  role: WorkerRole;
  group: WorkerGroup;
  state:
    | "stopped"
    | "starting"
    | "loading"
    | "ready"
    | "running"
    | "cancelling"
    | "unloading"
    | "failed"
    | "skipped";
  phase: string | null;
  catalog_id: string | null;
  model_id: string | null;
  model_revision: string | null;
  backend: string | null;
  dtype: string | null;
  quantization: string | null;
  device: string | null;
  request_id: string | null;
  attempt: number;
  progress: number;
  message: string | null;
  detail: string | null;
  phase_started_at: string | null;
  updated_at: string | null;
  load_time_ms: number | null;
  processing_time_ms: number | null;
  peak_vram_mb: number | null;
  peak_cpu_memory_mb: number | null;
  error_code: string | null;
}

export interface GenerationEvent {
  event_id: string;
  created_at: string;
  role: WorkerRole | null;
  state: WorkerStatus["state"] | null;
  phase: string;
  progress: number;
  message: string;
  detail: string | null;
  model_id: string | null;
  backend: string | null;
  device: string | null;
  request_id: string | null;
  attempt: number;
  error_code: string | null;
}

export interface PreparationGroup {
  group: WorkerGroup;
  state: WorkerStatus["state"];
  roles: WorkerStatus[];
  error_code: string | null;
}

export interface Preparation {
  state: "loading" | "ready" | "failed" | "stopped";
  message: string;
  groups: PreparationGroup[];
  retry_available: boolean;
}

export interface CaptureStats {
  camera_permission: "unknown" | "granted" | "denied";
  microphone_permission: "unknown" | "granted" | "denied";
  camera_width: number | null;
  camera_height: number | null;
  camera_fps: number | null;
  face_check_supported: boolean;
  face_detected: boolean | null;
  brightness: number | null;
  video_chunk_count: number;
  audio_segment_count: number;
  uploaded_bytes: number;
  upload_failure_count: number;
  last_silence_reason: string | null;
  recording_started_at: string | null;
  recording_duration_seconds: number;
}

export type DebugArtifactKind = "image" | "audio" | "video" | "text";

export interface DebugArtifact {
  path: string;
  name: string;
  kind: DebugArtifactKind;
  size_bytes: number;
  text_preview: string | null;
  media_url: string;
}
export interface Session {
  session_id: string;
  state: SessionState;
  elapsed_seconds: number;
  conversation_elapsed_seconds: number;
  generation_elapsed_seconds: number;
  conversation_phase: ConversationPhase;
  visitor_char_count: number;
  latest_visitor_transcript: string | null;
  target_transcript_chars: number;
  minimum_transcript_chars: number;
  conversation_time_limit_seconds: number;
  generation_time_limit_seconds: number;
  generation_steps: GenerationStep[];
  worker_statuses: WorkerStatus[];
  generation_events: GenerationEvent[];
  selected_episode_id: string | null;
  selected_episode_name: string | null;
  selected_effect_id: string | null;
  selected_effect_name: string | null;
  base_rarity: Rarity | null;
  final_rarity: Rarity | null;
  quality_profile: QualityProfile | null;
  stage_models: Record<WorkerRole, string>;
  allow_video_download: boolean;
  episode_mode: "formal" | "underground";
  current_question_id: string | null;
  current_question_text: string | null;
  answer_count: number;
  capture_stats: CaptureStats;
  video_artifact: {
    implemented: boolean;
    media_url: string | null;
    metadata_path: string | null;
    message: string;
    ai_generated_label: string;
  } | null;
  error_code: string | null;
  error_message: string | null;
  error_detail: string | null;
  failed_worker_role: WorkerRole | null;
  failed_worker_phase: string | null;
  completion_reason: string | null;
  model_switch_notice: string | null;
}
