import {
  AlertOctagon,
  Check,
  Clock3,
  Cpu,
  Download,
  Gauge,
  RotateCcw,
  Save,
  Settings2,
  SlidersHorizontal,
  Sparkles,
  X
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import type {
  QualityProfile,
  RuntimeOptions,
  StaffSettings,
  WorkerGroup,
  WorkerRole
} from "../types";

interface Props {
  open: boolean;
  settings: StaffSettings;
  options: RuntimeOptions;
  busy: boolean;
  hasSession: boolean;
  onClose: () => void;
  onSave: (settings: StaffSettings) => Promise<void>;
  onResetSettings: () => Promise<void>;
  onResetDemo: () => Promise<void>;
  onEmergencyStop: () => Promise<void>;
}

interface RangeRowProps {
  label: string;
  value: number;
  unit: string;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
}

const roleLabels: Record<WorkerRole, string> = {
  audio_preprocess_worker: "音声前処理",
  streaming_asr_worker: "会話中の文字起こし",
  final_asr_worker: "確定文字起こし",
  interview_llm_worker: "インタビュー会話",
  interview_tts_worker: "インタビュー音声",
  interview_summary_worker: "会話の整理",
  episode_selector: "エピソード選択",
  script_design_llm_worker: "未来設計・台本",
  script_safety_review_worker: "公開用ジャッジ",
  reference_frame_selector: "人物フレーム選択",
  voice_reference_selector: "本人音声選択",
  image_generation_worker: "未来画像",
  voice_clone_tts_worker: "本人声の生成",
  video_generation_worker: "動画生成",
  lip_sync_worker: "口元同期",
  video_postprocess_worker: "動画仕上げ"
};

const groupLabels: Record<WorkerGroup, string> = {
  interview: "インタビュー",
  material_preparation: "設計と素材準備",
  generation: "画像と音声",
  finishing: "動画仕上げ"
};

function RangeRow({
  label,
  value,
  unit,
  min,
  max,
  step,
  onChange
}: RangeRowProps) {
  return (
    <label className="setting-field range-field">
      <span className="setting-label">
        <span>{label}</span>
        <strong>
          {value}
          {unit}
        </strong>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </label>
  );
}

export function SettingsDrawer({
  open,
  settings,
  options,
  busy,
  hasSession,
  onClose,
  onSave,
  onResetSettings,
  onResetDemo,
  onEmergencyStop
}: Props) {
  const [draft, setDraft] = useState(settings);

  useEffect(() => {
    setDraft(settings);
  }, [settings, open]);

  const availableEpisodes = useMemo(
    () =>
      options.episodes.filter((episode) => {
        if (draft.episode_mode === "underground") return true;
        return (
          episode.formal_mode_allowed &&
          episode.public_demo_allowed &&
          !episode.limited_only
        );
      }),
    [draft.episode_mode, options.episodes]
  );

  const update = <K extends keyof StaffSettings>(
    key: K,
    value: StaffSettings[K]
  ) => setDraft((current) => ({ ...current, [key]: value }));

  const chooseProfile = (profile: Exclude<QualityProfile, "custom">) => {
    setDraft((current) => ({
      ...current,
      quality_profile: profile,
      stage_models: { ...options.profile_models[profile] }
    }));
  };

  const chooseModel = (role: WorkerRole, modelId: string) => {
    setDraft((current) => ({
      ...current,
      quality_profile: "custom",
      stage_models: { ...current.stage_models, [role]: modelId }
    }));
  };

  const chooseEpisodeMode = (mode: StaffSettings["episode_mode"]) => {
    if (
      mode === "underground" &&
      draft.episode_mode !== "underground" &&
      !window.confirm("全エピソードを対象にし、公開用LLMジャッジを省略しますか？")
    ) {
      return;
    }
    setDraft((current) => ({
      ...current,
      episode_mode: mode,
      fixed_episode_id:
        current.episode_selection === "fixed" ? null : current.fixed_episode_id
    }));
  };

  const save = async () => {
    const normalized = {
      ...draft,
      minimum_transcript_chars: Math.min(
        draft.minimum_transcript_chars,
        draft.target_transcript_chars
      )
    };
    await onSave(normalized);
  };

  return (
    <>
      <button
        className={`drawer-scrim ${open ? "is-open" : ""}`}
        aria-hidden={!open}
        onClick={onClose}
        aria-label="設定を閉じる"
        tabIndex={open ? 0 : -1}
      />
      <aside
        className={`settings-drawer ${open ? "is-open" : ""}`}
        aria-hidden={!open}
        ref={(element) => element?.toggleAttribute("inert", !open)}
      >
        <header className="drawer-header">
          <div>
            <span className="eyebrow">OPERATOR</span>
            <h2>運営設定</h2>
          </div>
          <button className="icon-button" onClick={onClose} aria-label="閉じる">
            <X size={21} />
          </button>
        </header>

        <div className="drawer-content">
          {hasSession && (
            <div className="session-setting-note">
              現在のセッションには開始時の設定が適用されています
            </div>
          )}

          <section className="settings-section">
            <h3>
              <Clock3 size={17} />
              時間と会話
            </h3>
            <RangeRow
              label="動画生成の制限"
              value={draft.generation_time_limit_seconds}
              unit="秒"
              {...options.limits.generation_time_limit_seconds}
              onChange={(value) =>
                update("generation_time_limit_seconds", value)
              }
            />
            <RangeRow
              label="会話時間の上限"
              value={draft.conversation_time_limit_seconds}
              unit="秒"
              {...options.limits.conversation_time_limit_seconds}
              onChange={(value) =>
                update("conversation_time_limit_seconds", value)
              }
            />
            <RangeRow
              label="目標文字数"
              value={draft.target_transcript_chars}
              unit="字"
              {...options.limits.target_transcript_chars}
              onChange={(value) => update("target_transcript_chars", value)}
            />
            <RangeRow
              label="最低文字数"
              value={draft.minimum_transcript_chars}
              unit="字"
              {...options.limits.minimum_transcript_chars}
              max={Math.min(
                options.limits.minimum_transcript_chars.max,
                draft.target_transcript_chars
              )}
              onChange={(value) => update("minimum_transcript_chars", value)}
            />
          </section>

          <section className="settings-section">
            <h3>
              <Gauge size={17} />
              生成構成
            </h3>
            <div className="setting-field">
              <span className="setting-label">プリセット</span>
              <div className="segmented-control profile-control">
                {(["quality", "balanced", "fast"] as const).map((profile) => (
                  <button
                    key={profile}
                    className={draft.quality_profile === profile ? "active" : ""}
                    onClick={() => chooseProfile(profile)}
                  >
                    {profile === "quality"
                      ? "品質"
                      : profile === "balanced"
                        ? "標準"
                        : "高速"}
                  </button>
                ))}
              </div>
              {draft.quality_profile === "custom" && (
                <small>工程ごとのモデルを選択中</small>
              )}
            </div>

            <div className="setting-field">
              <span className="setting-label">エピソードモード</span>
              <div className="segmented-control">
                <button
                  className={draft.episode_mode === "formal" ? "active" : ""}
                  onClick={() => chooseEpisodeMode("formal")}
                >
                  フォーマル
                </button>
                <button
                  className={
                    draft.episode_mode === "underground" ? "active" : ""
                  }
                  onClick={() => chooseEpisodeMode("underground")}
                >
                  アングラ
                </button>
              </div>
            </div>

            <div className="setting-field">
              <span className="setting-label">選択方法</span>
              <div className="segmented-control">
                <button
                  className={
                    draft.episode_selection === "random" ? "active" : ""
                  }
                  onClick={() => update("episode_selection", "random")}
                >
                  ランダム
                </button>
                <button
                  className={
                    draft.episode_selection === "fixed" ? "active" : ""
                  }
                  onClick={() => update("episode_selection", "fixed")}
                >
                  固定
                </button>
              </div>
            </div>

            {draft.episode_selection === "fixed" && (
              <label className="setting-field">
                <span className="setting-label">固定エピソード</span>
                <select
                  value={draft.fixed_episode_id ?? ""}
                  onChange={(event) =>
                    update("fixed_episode_id", event.target.value || null)
                  }
                >
                  <option value="">選択してください</option>
                  {availableEpisodes.map((episode) => (
                    <option value={episode.id} key={episode.id}>
                      {episode.name} / {episode.base_rarity}
                    </option>
                  ))}
                </select>
              </label>
            )}

            <label className="toggle-row">
              <span>
                <strong>自動モデル切り替え</strong>
                <small>失敗時に登録済みの軽量モデルを一度試します</small>
              </span>
              <input
                type="checkbox"
                checked={draft.auto_model_fallback}
                onChange={(event) =>
                  update("auto_model_fallback", event.target.checked)
                }
              />
            </label>

            <label className="toggle-row">
              <span>
                <strong>簡易動画への切り替え</strong>
                <small>現在の初期設定ではオフです</small>
              </span>
              <input
                type="checkbox"
                checked={draft.simple_video_fallback}
                onChange={(event) =>
                  update("simple_video_fallback", event.target.checked)
                }
              />
            </label>

            <label className="toggle-row">
              <span>
                <strong>完成動画のダウンロード</strong>
                <small>オフのときは画面にも表示しません</small>
              </span>
              <input
                type="checkbox"
                checked={draft.allow_video_download}
                onChange={(event) =>
                  update("allow_video_download", event.target.checked)
                }
              />
            </label>
          </section>

          {options.debug_mode && (
            <section className="settings-section">
              <h3>
                <Sparkles size={17} />
                デバッグ
              </h3>
              <div className="setting-field">
                <span className="setting-label">確認フロー</span>
                <div className="segmented-control">
                  <button
                    className={
                      draft.debug_test_mode === "normal" ? "active" : ""
                    }
                    onClick={() => update("debug_test_mode", "normal")}
                  >
                    通常
                  </button>
                  <button
                    className={
                      draft.debug_test_mode === "short" ? "active" : ""
                    }
                    onClick={() => update("debug_test_mode", "short")}
                  >
                    短縮
                  </button>
                </div>
              </div>
            </section>
          )}

          <details className="advanced-settings">
            <summary>
              <Cpu size={17} />
              工程ごとのモデル
            </summary>
            <div className="advanced-settings-content">
              {(Object.entries(options.worker_groups) as [
                WorkerGroup,
                WorkerRole[]
              ][]).map(([group, roles]) => (
                <div className="model-group" key={group}>
                  <h4>{groupLabels[group]}</h4>
                  {roles.map((role) => {
                    const available = options.models.filter((model) =>
                      model.roles.includes(role)
                    );
                    const current = options.models.find(
                      (model) => model.id === draft.stage_models[role]
                    );
                    return (
                      <label className="setting-field compact-model" key={role}>
                        <span className="setting-label">{roleLabels[role]}</span>
                        <select
                          value={draft.stage_models[role] ?? ""}
                          onChange={(event) =>
                            chooseModel(role, event.target.value)
                          }
                        >
                          {available.map((model) => (
                            <option value={model.id} key={model.id}>
                              {model.label}
                            </option>
                          ))}
                        </select>
                        {current && (
                          <small>
                            {current.backend} / {current.device} / {current.revision}
                          </small>
                        )}
                      </label>
                    );
                  })}
                </div>
              ))}
            </div>
          </details>

          <section className="settings-section control-section">
            <h3>
              <SlidersHorizontal size={17} />
              デモ操作
            </h3>
            <button
              className="secondary-button full-width"
              disabled={busy}
              onClick={onResetSettings}
            >
              <RotateCcw size={17} />
              初期設定に戻す
            </button>
            <button
              className="secondary-button full-width"
              disabled={busy || !hasSession}
              onClick={onResetDemo}
            >
              <RotateCcw size={17} />
              デモをリセット
            </button>
            <button
              className="danger-button full-width"
              disabled={busy || !hasSession}
              onClick={onEmergencyStop}
            >
              <AlertOctagon size={17} />
              緊急停止
            </button>
          </section>
        </div>

        <footer className="drawer-footer">
          <button className="primary-button" disabled={busy} onClick={save}>
            {busy ? <Settings2 className="spin" size={18} /> : <Save size={18} />}
            設定を保存
          </button>
          <span className="saved-note">
            <Check size={14} />
            サーバー再起動後も保持
          </span>
        </footer>
      </aside>
    </>
  );
}
