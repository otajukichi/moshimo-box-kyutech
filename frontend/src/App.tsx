import {
  AlertTriangle,
  ArrowRight,
  BookOpenText,
  Camera,
  Check,
  CheckCircle2,
  CircleStop,
  Clock3,
  Download,
  Expand,
  FlaskConical,
  ImageOff,
  LoaderCircle,
  Menu,
  Mic2,
  MonitorCheck,
  Play,
  RotateCcw,
  ShieldCheck,
  Sparkles,
  Square,
  Trash2,
  UploadCloud,
  UserRound,
  Video,
  WandSparkles,
  XCircle
} from "lucide-react";
import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState
} from "react";
import { api, apiUrl } from "./api";
import {
  DemoMediaCapture,
  type AnswerRecordingResult,
  type UploadQueueState
} from "./capture";
import { SettingsDrawer } from "./components/SettingsDrawer";
import { TechnologyGuide } from "./components/TechnologyGuide";
import type {
  AppConfig,
  DebugArtifact,
  Preparation,
  RuntimeOptions,
  Session,
  StaffSettings,
  WorkerRole
} from "./types";

const consentItems = [
  "入力された個人情報は、今回の動画生成にのみ使用します",
  "取得した情報や生成途中のデータは、デモ終了後に削除します",
  "使用するモデルは学内サーバー上で動作し、外部サービスへ情報を送りません",
  "生成される内容はフィクションであり、実際の将来を示すものではありません",
  "不快または不適切な内容のおそれがある場合、運営側が中止します"
];

const interviewPreparationLabels: Partial<Record<WorkerRole, string>> = {
  audio_preprocess_worker: "音声入力",
  streaming_asr_worker: "文字起こし",
  interview_llm_worker: "会話AI",
  interview_tts_worker: "AI音声"
};

const roleLabels: Partial<Record<WorkerRole, string>> = {
  final_asr_worker: "確定文字起こし",
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

function formatTime(seconds: number) {
  const minutes = Math.floor(seconds / 60);
  const rest = seconds % 60;
  return `${String(minutes).padStart(2, "0")}:${String(rest).padStart(2, "0")}`;
}

function formatBytes(bytes: number) {
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}


function TopBar({
  title,
  onOpenSettings,
  onOpenTechnology
}: {
  title: string;
  onOpenSettings: () => void;
  onOpenTechnology: () => void;
}) {
  return (
    <header className="top-bar">
      <button
        className="icon-button menu-button"
        onClick={onOpenSettings}
        aria-label="運営設定を開く"
        title="運営設定"
      >
        <Menu size={22} />
      </button>
      <button
        className="icon-button technology-button"
        onClick={onOpenTechnology}
        aria-label="技術解説を開く"
        title="技術解説"
      >
        <BookOpenText size={21} />
      </button>
      <div className="mini-brand">
        <span className="mini-brand-mark">
          <WandSparkles size={17} />
        </span>
        <span>{title}</span>
      </div>
      <span className="lab-label">OKITA LAB.</span>
    </header>
  );
}

function FutureBoxVisual({ preparation }: { preparation?: Preparation }) {
  const interview = preparation?.groups.find(
    (group) => group.group === "interview"
  );
  const readyCount =
    interview?.roles.filter((worker) => worker.state === "ready").length ?? 0;
  const workerCount = interview?.roles.length ?? 0;
  const failed = preparation?.state === "failed";

  return (
    <div
      className={`future-box-visual ${preparation ? "future-box-preparing" : ""} ${failed ? "future-box-failed" : ""}`}
      aria-hidden="true"
    >
      <div className="future-box-shadow" />
      <div className="future-box">
        <div className="box-top">
          <WandSparkles size={31} />
        </div>
        <div className="box-screen">
          {failed ? (
            <AlertTriangle size={34} />
          ) : preparation ? (
            <LoaderCircle className="spin" size={34} />
          ) : (
            <Sparkles size={34} />
          )}
          <span>
            {failed ? "LINK ERROR" : preparation ? "CONNECTING" : "MESSAGE READY"}
          </span>
          {preparation && (
            <small>
              {readyCount} / {workerCount} WORKERS
            </small>
          )}
        </div>
        <div className="box-slot" />
        <div className="box-light" />
      </div>
      <div className="message-ticket">
        <span>FUTURE LINK</span>
        <strong>
          {failed ? "接続を確認してください" : preparation ? "AIを準備中" : "未来から通信中"}
        </strong>
      </div>
      <div className="signal-line signal-line-one" />
      <div className="signal-line signal-line-two" />
    </div>
  );
}

function PreparationPage({
  preparation,
  busy,
  onRetry
}: {
  preparation: Preparation;
  busy: boolean;
  onRetry: () => void;
}) {
  const interview = preparation.groups.find(
    (group) => group.group === "interview"
  );
  return (
    <main className="page title-page preparation-page">
      <section className="title-copy preparation-title-copy">
        <span className="eyebrow">KYUTECH OPEN CAMPUS</span>
        <h1>
          もしもボックス
          <br />
          九工大出張所
        </h1>
        <div className="preparation-status">
          <span className="preparation-kicker">SYSTEM PREPARATION</span>
          <h2>{preparation.message}</h2>
          <p>未来との会話に必要なAIを、先に研究室サーバーで起動しています。</p>
          <div className="preparation-workers">
            {interview?.roles.map((worker) => (
              <div className={`prep-worker prep-${worker.state}`} key={worker.role}>
                <span>
                  {worker.state === "ready" ? (
                    <Check size={15} />
                  ) : worker.state === "failed" ? (
                    <XCircle size={15} />
                  ) : (
                    <LoaderCircle className="spin" size={15} />
                  )}
                </span>
                <strong>
                  {interviewPreparationLabels[worker.role] ??
                    worker.role.replace("_worker", "")}
                </strong>
              </div>
            ))}
          </div>
          {preparation.state === "failed" && (
            <button className="primary-button" disabled={busy} onClick={onRetry}>
              <RotateCcw size={18} />
              再試行
            </button>
          )}
        </div>
      </section>
      <FutureBoxVisual preparation={preparation} />
      <div className="title-footer">
        <span>九州工業大学 大北研究室</span>
        <span>FICTIONAL FUTURE EXPERIENCE</span>
      </div>
    </main>
  );
}

function TitlePage({ onStart, busy }: { onStart: () => void; busy: boolean }) {
  return (
    <main className="page title-page">
      <section className="title-copy">
        <span className="eyebrow">KYUTECH OPEN CAMPUS</span>
        <h1>
          もしもボックス
          <br />
          九工大出張所
        </h1>
        <p>思いもしない未来の自分から、メッセージが届きます。</p>
        <button
          className="primary-button start-button"
          onClick={onStart}
          disabled={busy}
        >
          {busy ? (
            <LoaderCircle className="spin" size={20} />
          ) : (
            <Play size={20} fill="currentColor" />
          )}
          始める
          <ArrowRight size={19} />
        </button>
      </section>
      <FutureBoxVisual />
      <div className="title-footer">
        <span>九州工業大学 大北研究室</span>
        <span>FICTIONAL FUTURE EXPERIENCE</span>
      </div>
    </main>
  );
}

function ConsentPage({
  onConsent,
  busy
}: {
  onConsent: (voiceConsent: boolean) => void;
  busy: boolean;
}) {
  const [voiceConsent, setVoiceConsent] = useState(false);
  return (
    <main className="page content-page consent-page">
      <section className="content-header">
        <span className="section-icon yellow">
          <ShieldCheck size={28} />
        </span>
        <div>
          <span className="eyebrow">BEFORE WE START</span>
          <h1>
            未来からのメッセージを
            <br />
            受け取る前に
          </h1>
        </div>
      </section>
      <div className="consent-list">
        {consentItems.map((item, index) => (
          <div className="consent-item" key={item}>
            <span>{String(index + 1).padStart(2, "0")}</span>
            <p>{item}</p>
            <Check size={20} />
          </div>
        ))}
      </div>
      <label className="voice-consent-row">
        <input
          type="checkbox"
          checked={voiceConsent}
          onChange={(event) => setVoiceConsent(event.target.checked)}
        />
        <span>
          <strong>本人の声を模倣したAI音声の生成に同意します</strong>
          <small>インタビュー音声は今回の動画生成にのみ使用します</small>
        </span>
      </label>
      <footer className="page-actions">
        <p>
          <FlaskConical size={17} />
          大学内の研究デモとして実施します
        </p>
        <button
          className="primary-button"
          onClick={() => onConsent(voiceConsent)}
          disabled={busy || !voiceConsent}
        >
          {busy ? (
            <LoaderCircle className="spin" size={18} />
          ) : (
            <ShieldCheck size={18} />
          )}
          内容に同意して進む
        </button>
      </footer>
    </main>
  );
}

type FaceDetectorResult = { boundingBox: DOMRectReadOnly };
type FaceDetectorInstance = {
  detect: (source: HTMLVideoElement) => Promise<FaceDetectorResult[]>;
};
type FaceDetectorConstructor = new (options?: {
  fastMode?: boolean;
  maxDetectedFaces?: number;
}) => FaceDetectorInstance;

function DeviceCheckPage({
  sessionId,
  capture,
  config,
  busy,
  onReady,
  onDenied,
  onFailure
}: {
  sessionId: string;
  capture: DemoMediaCapture;
  config: AppConfig;
  busy: boolean;
  onReady: (report: {
    camera_width: number;
    camera_height: number;
    camera_fps: number | null;
    face_check_supported: boolean;
    face_detected: boolean | null;
    brightness: number | null;
  }) => Promise<void>;
  onDenied: () => Promise<void>;
  onFailure: (message: string) => void;
}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [attempt, setAttempt] = useState(0);
  const [state, setState] = useState<
    "requesting" | "checking" | "denied" | "failed"
  >("requesting");
  const [brightness, setBrightness] = useState<number | null>(null);
  const [faceDetected, setFaceDetected] = useState<boolean | null>(null);
  const [stableProgress, setStableProgress] = useState(0);
  const completedRef = useRef(false);

  useEffect(() => {
    let cancelled = false;
    let timer = 0;
    let detecting = false;
    let stableSince: number | null = null;
    completedRef.current = false;
    setState("requesting");
    setStableProgress(0);

    const start = async () => {
      try {
        const stream = await capture.openDevices();
        if (cancelled || !videoRef.current) return;
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
        capture.startVideo(sessionId);
        setState("checking");

        const videoTrack = stream.getVideoTracks()[0];
        const trackSettings = videoTrack.getSettings();
        const FaceDetectorApi = (
          window as unknown as { FaceDetector?: FaceDetectorConstructor }
        ).FaceDetector;
        const detector = FaceDetectorApi
          ? new FaceDetectorApi({ fastMode: true, maxDetectedFaces: 2 })
          : null;
        const canvas = document.createElement("canvas");
        canvas.width = 160;
        canvas.height = 90;
        const context = canvas.getContext("2d", { willReadFrequently: true });

        timer = window.setInterval(async () => {
          if (
            cancelled ||
            detecting ||
            !context ||
            !videoRef.current ||
            videoRef.current.readyState < 2
          ) {
            return;
          }
          detecting = true;
          try {
            context.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
            const pixels = context.getImageData(
              0,
              0,
              canvas.width,
              canvas.height
            ).data;
            let luminance = 0;
            for (let index = 0; index < pixels.length; index += 16) {
              luminance +=
                pixels[index] * 0.2126 +
                pixels[index + 1] * 0.7152 +
                pixels[index + 2] * 0.0722;
            }
            const sampled = pixels.length / 16;
            const currentBrightness = Math.round(luminance / sampled);
            setBrightness(currentBrightness);

            let faceOkay = true;
            let detected: boolean | null = null;
            if (detector) {
              const faces = await detector.detect(videoRef.current);
              detected = faces.length === 1;
              faceOkay = false;
              if (faces.length === 1) {
                const box = faces[0].boundingBox;
                const width = videoRef.current.videoWidth;
                const height = videoRef.current.videoHeight;
                const sizeRatio = box.width / width;
                const centerX = (box.x + box.width / 2) / width;
                const centerY = (box.y + box.height / 2) / height;
                faceOkay =
                  sizeRatio >= 0.16 &&
                  sizeRatio <= 0.72 &&
                  Math.abs(centerX - 0.5) < 0.22 &&
                  Math.abs(centerY - 0.46) < 0.26;
              }
              setFaceDetected(detected);
            }

            const lightOkay =
              currentBrightness >= config.capture.brightness_min &&
              currentBrightness <= config.capture.brightness_max;
            if (lightOkay && faceOkay) {
              if (stableSince === null) stableSince = performance.now();
              const stableSeconds = (performance.now() - stableSince) / 1000;
              setStableProgress(
                Math.min(1, stableSeconds / config.capture.camera_stable_seconds)
              );
              if (
                stableSeconds >= config.capture.camera_stable_seconds &&
                !completedRef.current
              ) {
                completedRef.current = true;
                window.clearInterval(timer);
                await onReady({
                  camera_width: trackSettings.width ?? videoRef.current.videoWidth,
                  camera_height:
                    trackSettings.height ?? videoRef.current.videoHeight,
                  camera_fps: trackSettings.frameRate ?? null,
                  face_check_supported: Boolean(detector),
                  face_detected: detected,
                  brightness: currentBrightness
                });
              }
            } else {
              stableSince = null;
              setStableProgress(0);
            }
          } finally {
            detecting = false;
          }
        }, 220);
      } catch (error) {
        if (cancelled) return;
        const denied =
          error instanceof DOMException &&
          ["NotAllowedError", "PermissionDeniedError"].includes(error.name);
        if (denied) {
          setState("denied");
          await onDenied();
        } else {
          setState("failed");
          onFailure(
            error instanceof Error
              ? error.message
              : "カメラとマイクを開始できません"
          );
        }
      }
    };

    void start();
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [
    attempt,
    capture,
    sessionId,
    config.capture.brightness_min,
    config.capture.brightness_max,
    config.capture.camera_stable_seconds
  ]);

  return (
    <main className="page device-check-page">
      <section className="device-copy">
        <span className="eyebrow">CAMERA &amp; MICROPHONE</span>
        <h1>カメラの方を向いてください</h1>
        <p>自然な姿勢のまま、顔が中央に見える位置で少しお待ちください。</p>
        <div className="device-checks">
          <div className={state !== "requesting" ? "check-ok" : ""}>
            <Camera size={18} />
            カメラ
          </div>
          <div className={state !== "requesting" ? "check-ok" : ""}>
            <Mic2 size={18} />
            マイク
          </div>
          <div
            className={
              brightness !== null &&
              brightness >= config.capture.brightness_min &&
              brightness <= config.capture.brightness_max
                ? "check-ok"
                : ""
            }
          >
            <Sparkles size={18} />
            明るさ
          </div>
          <div className={faceDetected !== false ? "check-ok" : ""}>
            <UserRound size={18} />
            顔の位置
          </div>
        </div>
        {state === "denied" && (
          <div className="permission-help">
            <AlertTriangle size={20} />
            <div>
              <strong>カメラとマイクを許可してください</strong>
              <span>Edgeのアドレスバー左側から許可し、再試行します。</span>
            </div>
            <button
              className="secondary-button"
              disabled={busy}
              onClick={() => setAttempt((value) => value + 1)}
            >
              <RotateCcw size={17} />
              再試行
            </button>
          </div>
        )}
      </section>
      <div className="device-preview-shell">
        <video ref={videoRef} muted playsInline className="device-video" />
        <div className="face-guide" aria-hidden="true" />
        <div className="device-progress">
          <span style={{ width: `${stableProgress * 100}%` }} />
        </div>
        <div className="device-live-label">
          <span /> LIVE CHECK
        </div>
      </div>
    </main>
  );
}

const phaseCopy = {
  waiting: ["準備しています", "少しだけお待ちください"],
  speaking: ["AIが話しています", "未来の話を一緒に探しています"],
  listening: ["あなたの声を聞いています", "話しやすいところから、自由にどうぞ"],
  thinking: ["会話を考えています", "次の話題を組み立てています"],
  closing: ["会話をまとめています", "まもなく未来のメッセージ作りへ進みます"]
} as const;

function ConversationPage({
  session,
  debugMode,
  capture,
  queueState,
  busy,
  onAiFinished,
  onAnswer,
  onFinish,
  onFailure
}: {
  session: Session;
  debugMode: boolean;
  capture: DemoMediaCapture;
  queueState: UploadQueueState;
  busy: boolean;
  onAiFinished: () => Promise<void>;
  onAnswer: (result: AnswerRecordingResult) => Promise<void>;
  onFinish: () => Promise<void>;
  onFailure: (message: string) => void;
}) {
  const [volume, setVolume] = useState(0);
  const videoRef = useRef<HTMLVideoElement>(null);
  const spokenQuestionRef = useRef<string | null>(null);
  const onAiFinishedRef = useRef(onAiFinished);
  onAiFinishedRef.current = onAiFinished;
  const recordingRef = useRef(false);
  const answerTaskRef = useRef<Promise<void> | null>(null);
  const finishTaskRef = useRef<Promise<void> | null>(null);
  const phase = phaseCopy[session.conversation_phase];
  const progress = Math.min(
    100,
    Math.round((session.visitor_char_count / session.target_transcript_chars) * 100)
  );
  const uploadsBlocked = ["blocked", "overflow"].includes(queueState.state);

  const finishConversation = useCallback((): Promise<void> => {
    if (finishTaskRef.current) return finishTaskRef.current;
    capture.stopAnswer("operator");
    const pendingAnswer = answerTaskRef.current;
    const task = (async () => {
      await pendingAnswer;
      await onFinish();
    })();
    finishTaskRef.current = task;
    void task.finally(() => {
      if (finishTaskRef.current === task) finishTaskRef.current = null;
    });
    return task;
  }, [capture, onFinish]);

  useEffect(() => {
    if (videoRef.current && capture.stream) {
      videoRef.current.srcObject = capture.stream;
      void videoRef.current.play();
    }
  }, [capture]);

  useEffect(() => {
    const questionId = session.current_question_id;
    if (
      session.conversation_phase !== "speaking" ||
      !questionId ||
      spokenQuestionRef.current === questionId ||
      uploadsBlocked
    ) {
      return;
    }
    spokenQuestionRef.current = questionId;
    const questionText =
      session.current_question_text ?? "少しお話を聞かせてください。";
    let cancelled = false;
    let finished = false;
    let observedPlayback = false;
    let watchdog = 0;
    let fallbackTimer = 0;
    let utterance: SpeechSynthesisUtterance | null = null;

    const finish = () => {
      if (cancelled || finished) return;
      finished = true;
      window.clearInterval(watchdog);
      window.clearTimeout(fallbackTimer);
      void onAiFinishedRef.current();
    };

    if (
      typeof SpeechSynthesisUtterance !== "undefined" &&
      "speechSynthesis" in window
    ) {
      window.speechSynthesis.cancel();
      utterance = new SpeechSynthesisUtterance(questionText);
      utterance.lang = "ja-JP";
      utterance.rate = 1;
      utterance.pitch = 1;
      const japaneseVoice = window.speechSynthesis
        .getVoices()
        .find((voice) => voice.lang.toLowerCase().startsWith("ja"));
      if (japaneseVoice) utterance.voice = japaneseVoice;
      utterance.onend = finish;
      utterance.onerror = finish;
      window.speechSynthesis.speak(utterance);

      watchdog = window.setInterval(() => {
        const synthesis = window.speechSynthesis;
        if (synthesis.speaking || synthesis.pending) {
          observedPlayback = true;
        } else if (observedPlayback) {
          finish();
        }
      }, 100);
      fallbackTimer = window.setTimeout(
        finish,
        Math.min(30_000, Math.max(6_000, questionText.length * 350))
      );
    } else {
      fallbackTimer = window.setTimeout(finish, 700);
    }

    return () => {
      cancelled = true;
      window.clearInterval(watchdog);
      window.clearTimeout(fallbackTimer);
      if (utterance) {
        utterance.onend = null;
        utterance.onerror = null;
      }
      if (!finished && spokenQuestionRef.current === questionId) {
        spokenQuestionRef.current = null;
      }
    };
  }, [
    session.conversation_phase,
    session.current_question_id,
    session.current_question_text,
    uploadsBlocked
  ]);

  useEffect(() => {
    if (
      session.conversation_phase !== "listening" ||
      recordingRef.current ||
      uploadsBlocked
    ) {
      return;
    }
    recordingRef.current = true;
    const task = capture
      .recordAnswer(session.session_id, setVolume)
      .then(onAnswer)
      .catch((error) => {
        onFailure(
          error instanceof Error ? error.message : "回答音声を保存できません"
        );
      })
      .finally(() => {
        recordingRef.current = false;
        setVolume(0);
      });
    answerTaskRef.current = task;
    void task.finally(() => {
      if (answerTaskRef.current === task) answerTaskRef.current = null;
    });
  }, [capture, onAnswer, onFailure, session, uploadsBlocked]);

  useEffect(() => {
    if (session.conversation_phase === "closing") {
      void finishConversation();
    }
  }, [finishConversation, session.conversation_phase]);

  return (
    <main className="page conversation-page">
      <div className="conversation-stage">
        <div
          className={`voice-core phase-${session.conversation_phase}`}
          style={{ transform: `scale(${1 + volume * 0.08})` }}
        >
          <div className="voice-wave">
            {[18, 31, 45, 25, 52, 37, 21].map((height, index) => (
              <span
                style={{
                  height: `${height + volume * 25}px`,
                  animationDelay: `${index * 90}ms`
                }}
                key={`${height}-${index}`}
              />
            ))}
          </div>
        </div>
        <div className="phase-copy">
          <span className="live-label">
            <span />
            {session.conversation_phase === "listening" ? "LISTENING" : "LIVE"}
          </span>
          <h1>{uploadsBlocked ? "保存接続を回復しています" : phase[0]}</h1>
          <p>
            {uploadsBlocked
              ? "データを安全に保存できるまで会話を一時停止します"
              : session.conversation_phase === "speaking" &&
                  session.current_question_text
                ? session.current_question_text
                : phase[1]}
          </p>
        </div>
      </div>

      <div className="camera-preview live-camera-preview">
        <video ref={videoRef} muted playsInline />
        <span>
          映像プレビュー
          <small>REC</small>
        </span>
        <Video size={16} />
      </div>

      <div className="conversation-status">
        <div className="status-number">
          <span>あなたの発話</span>
          <strong>{session.visitor_char_count}</strong>
          <small>/ {session.target_transcript_chars}字</small>
        </div>
        <div className="status-progress">
          <span style={{ width: `${progress}%` }} />
        </div>
        <div className="status-time">
          <Clock3 size={16} />
          {formatTime(session.conversation_elapsed_seconds)}
          <span>/ {formatTime(session.conversation_time_limit_seconds)}</span>
        </div>
        <div className="capture-counts">
          <UploadCloud size={14} />
          {session.capture_stats.video_chunk_count} chunks / {" "}
          {session.capture_stats.audio_segment_count} answers
        </div>
      </div>

      <div className="conversation-controls">
        <button
          className="icon-button"
          title="現在の発話を区切る"
          aria-label="現在の発話を区切る"
          disabled={session.conversation_phase !== "listening"}
          onClick={() => capture.stopAnswer("operator")}
        >
          <Square size={17} fill="currentColor" />
        </button>
        <button
          className="secondary-button"
          disabled={busy || session.conversation_phase === "closing"}
          onClick={() => void finishConversation()}
        >
          <CircleStop size={17} />
          会話を終了
        </button>
      </div>

      {debugMode && (
        <section className="debug-console">
          <div className="debug-console-title">
            <span>LATEST ASR</span>
            <small>
              {session.capture_stats.recording_duration_seconds}s / {" "}
              {formatBytes(session.capture_stats.uploaded_bytes)} / silence {" "}
              {session.capture_stats.last_silence_reason ?? "-"}
            </small>
          </div>
          <textarea
            value={session.latest_visitor_transcript ?? ""}
            readOnly
            aria-label="最新のASR文字起こし"
            placeholder="回答後にASRの文字起こしが表示されます"
            rows={3}
          />
          <div className="debug-transcript-state">
            <Mic2 size={17} />
            <span>
              {session.latest_visitor_transcript
                ? `${session.latest_visitor_transcript.length}字`
                : "待機中"}
            </span>
          </div>
        </section>
      )}
    </main>
  );
}

const generationMessages: Record<string, string> = {
  reflection: "会話の中にあった小さな手がかりを集めています。",
  "future-event": "たくさんの未来から、今回の物語を選んでいます。",
  "future-portrait": "未来の世界と、そこにいるあなたを組み立てています。",
  "message-video": "未来から届くメッセージの形に整えています。"
};

const workerStateLabels: Record<string, string> = {
  stopped: "待機中",
  starting: "プロセス起動中",
  loading: "モデル読込中",
  ready: "完了",
  running: "推論中",
  cancelling: "停止処理中",
  unloading: "モデル解放中",
  failed: "失敗",
  skipped: "省略"
};

const phaseLabels: Record<string, string> = {
  interview_release: "インタビュー用モデルの解放",
  worker_start: "ワーカープロセスの起動",
  model_load: "モデルの読み込み",
  healthcheck: "モデルの起動確認",
  inference: "モデルの推論",
  input_validation: "入力データの確認",
  output_validation: "出力構造の検証",
  model_unload: "モデルの解放",
  gpu_release: "GPUメモリの解放確認",
  failure_cleanup: "失敗後のモデル解放",
  retry: "同じモデルで再試行",
  model_fallback: "軽量モデルへの切り替え",
  completed: "処理完了",
  "script.future_world": "未来の世界を生成",
  "script.future_person": "未来の本人を生成",
  "script.positive_interpretation": "前向きな意味付けを生成",
  "script.narration_script": "未来からのメッセージを生成",
  "script.narration_revision": "メッセージの長さを調整",
  "script.visual_concept": "映像コンセプトを生成",
  "script.clothing": "未来の服装を生成",
  "script.background": "未来の背景を生成",
  "script.emotion": "表情と感情を生成",
  "script.image_prompt": "画像生成指示を生成",
  "script.video_prompt": "動画生成指示を生成",
  "script.voice_instruction": "音声演技指示を生成",
  "safety.verdict": "公開用ジャッジ",
  "safety.rewrite": "公開用メッセージへ修正",
  "safety.validation": "ジャッジ結果の検証"
};

function displayPhase(phase: string | null): string {
  if (!phase) return "待機中";
  if (phase.startsWith("step.")) return "工程を開始";
  return phaseLabels[phase] ?? phase;
}

function formatMilliseconds(value: number | null): string {
  if (value === null) return "-";
  if (value < 1000) return `${value} ms`;
  return `${(value / 1000).toFixed(1)} s`;
}

function eventTime(value: string): string {
  return new Date(value).toLocaleTimeString("ja-JP", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit"
  });
}

function GenerationPage({
  session,
  debugMode,
  busy,
  onComplete,
  onError
}: {
  session: Session;
  debugMode: boolean;
  busy: boolean;
  onComplete: () => Promise<void>;
  onError: () => Promise<void>;
}) {
  const current =
    session.generation_steps.find((step) => step.status === "current") ??
    session.generation_steps.at(-1);
  const detailedWorkers = session.worker_statuses.filter(
    (worker) => roleLabels[worker.role]
  );
  const activeStates = new Set(["starting", "loading", "running", "unloading"]);
  const activeWorker =
    detailedWorkers.find((worker) => activeStates.has(worker.state)) ??
    [...detailedWorkers]
      .reverse()
      .find((worker) => worker.state === "ready" || worker.state === "failed") ??
    null;
  const recentEvents = session.generation_events.slice(-12).reverse();
  const debugEvents = session.generation_events.slice(-40).reverse();

  return (
    <main className="page generation-page">
      <header className="generation-header">
        <span className="eyebrow">CREATING YOUR FUTURE</span>
        <h1>
          未来からのメッセージを
          <br />
          準備しています
        </h1>
        <p>{current ? generationMessages[current.id] : "準備を始めています。"}</p>
      </header>

      {activeWorker && (
        <section className={`current-operation worker-${activeWorker.state}`}>
          <div className="current-operation-heading">
            <div>
              <small>CURRENT OPERATION</small>
              <strong>{roleLabels[activeWorker.role]}</strong>
            </div>
            <span>{displayPhase(activeWorker.phase)}</span>
          </div>
          <p>{activeWorker.message ?? workerStateLabels[activeWorker.state]}</p>
          <div className="operation-progress" aria-hidden="true">
            <span style={{ width: `${Math.round(activeWorker.progress * 100)}%` }} />
          </div>
          <div className="current-operation-meta">
            <span>{activeWorker.model_id ?? "モデル未選択"}</span>
            <span>{Math.round(activeWorker.progress * 100)}%</span>
          </div>
        </section>
      )}

      <div className="generation-layout">
        <div className="generation-visual" aria-hidden="true">
          <div className="scan-frame">
            <div className="scan-window">
              <ImageOff size={42} />
              <span>FUTURE FRAME</span>
            </div>
            <div className="scan-line" />
          </div>
          <div className="data-strip">
            <span>PROFILE {session.quality_profile?.toUpperCase()}</span>
            <span>TARGET 20 SEC</span>
          </div>
        </div>

        <ol className="generation-steps">
          {session.generation_steps.map((step, index) => (
            <li className={`step-${step.status}`} key={step.id}>
              <span className="step-marker">
                {step.status === "completed" ? (
                  <Check size={19} />
                ) : step.status === "current" ? (
                  <LoaderCircle className="spin" size={19} />
                ) : step.status === "failed" ? (
                  <XCircle size={19} />
                ) : (
                  String(index + 1).padStart(2, "0")
                )}
              </span>
              <div>
                <small>
                  {step.status === "completed"
                    ? "完了"
                    : step.status === "current"
                      ? "処理中"
                      : step.status === "failed"
                        ? "失敗"
                        : "待機中"}
                </small>
                <strong>{step.label}</strong>
              </div>
            </li>
          ))}
        </ol>
      </div>

      {session.model_switch_notice && (
        <div className="model-switch-notice">
          <RotateCcw size={16} />
          軽量モデルへ自動で切り替えました
        </div>
      )}

      <details className="operator-progress" open>
        <summary>処理の詳細</summary>
        <div className="worker-progress-grid">
          {detailedWorkers.map((worker) => (
            <div className={`worker-progress worker-${worker.state}`} key={worker.role}>
              <div className="worker-progress-heading">
                <span>{roleLabels[worker.role]}</span>
                <small>{workerStateLabels[worker.state]}</small>
              </div>
              <strong>{displayPhase(worker.phase)}</strong>
              <p>{worker.message ?? "まだ開始していません"}</p>
              <div className="worker-progress-bar" aria-hidden="true">
                <span style={{ width: `${Math.round(worker.progress * 100)}%` }} />
              </div>
              <small className="worker-model-name">
                {worker.model_id ?? "未起動"}
                {worker.processing_time_ms !== null
                  ? ` / ${formatMilliseconds(worker.processing_time_ms)}`
                  : ""}
              </small>
            </div>
          ))}
        </div>

        <div className="generation-event-list">
          <h2>直近の処理</h2>
          {recentEvents.length === 0 ? (
            <p className="empty-events">処理開始を待っています</p>
          ) : (
            <ol>
              {recentEvents.map((event) => (
                <li key={event.event_id} className={event.error_code ? "event-error" : ""}>
                  <time>{eventTime(event.created_at)}</time>
                  <div>
                    <small>
                      {event.role ? roleLabels[event.role] : "生成全体"} / {displayPhase(event.phase)}
                    </small>
                    <strong>{event.message}</strong>
                    {event.detail && <p>{event.detail}</p>}
                  </div>
                </li>
              ))}
            </ol>
          )}
        </div>
      </details>

      {debugMode && (
        <details className="pipeline-debug-panel" open>
          <summary>DEBUG PIPELINE DIAGNOSTICS</summary>
          <div className="debug-worker-table">
            {detailedWorkers
              .filter((worker) => worker.model_id || worker.state !== "stopped")
              .map((worker) => (
                <dl key={worker.role}>
                  <dt>{roleLabels[worker.role]}</dt>
                  <dd>state: {worker.state}</dd>
                  <dd>phase: {worker.phase ?? "-"}</dd>
                  <dd>model: {worker.model_id ?? "-"}</dd>
                  <dd>backend: {worker.backend ?? "-"}</dd>
                  <dd>
                    runtime: {worker.device ?? "-"} / {worker.dtype ?? "-"} / {worker.quantization ?? "-"}
                  </dd>
                  <dd>attempt: {worker.attempt || "-"}</dd>
                  <dd>request: {worker.request_id ?? "-"}</dd>
                  <dd>
                    load {formatMilliseconds(worker.load_time_ms)} / inference {formatMilliseconds(worker.processing_time_ms)}
                  </dd>
                  <dd>
                    peak VRAM {worker.peak_vram_mb ?? "-"} MB / CPU {worker.peak_cpu_memory_mb ?? "-"} MB
                  </dd>
                  {worker.error_code && <dd className="debug-error-code">error: {worker.error_code}</dd>}
                  {worker.detail && <dd className="debug-detail">{worker.detail}</dd>}
                </dl>
              ))}
          </div>
          <div className="debug-event-stream">
            {debugEvents.map((event) => (
              <code key={event.event_id}>
                {eventTime(event.created_at)} [{event.role ?? "pipeline"}] {event.phase} {event.message}
                {event.error_code ? ` error=${event.error_code}` : ""}
              </code>
            ))}
          </div>
        </details>
      )}

      <footer className="generation-footer">
        <span>
          <Clock3 size={16} />
          生成経過 {formatTime(session.generation_elapsed_seconds)}
        </span>
        <p>正確な残り時間は表示していません</p>
      </footer>

      {debugMode && (
        <div className="debug-corner">
          <span>DEBUG CONTROLS</span>
          <button disabled={busy} onClick={onComplete}>
            <CheckCircle2 size={15} />
            生成を完了
          </button>
          <button disabled={busy} onClick={onError}>
            <XCircle size={15} />
            エラー表示
          </button>
        </div>
      )}
    </main>
  );
}

function DebugArtifactsPanel({
  sessionId,
  onRetentionChange
}: {
  sessionId: string;
  onRetentionChange?: (retained: boolean) => void;
}) {
  const [artifacts, setArtifacts] = useState<DebugArtifact[]>([]);
  const [loadState, setLoadState] = useState<
    "loading" | "ready" | "error"
  >("loading");

  useEffect(() => {
    let active = true;
    setLoadState("loading");
    api.debugArtifacts(sessionId)
      .then((response) => {
        if (!active) return;
        setArtifacts(response.artifacts);
        onRetentionChange?.(response.retained);
        setLoadState("ready");
      })
      .catch(() => {
        if (!active) return;
        onRetentionChange?.(false);
        setLoadState("error");
      });
    return () => {
      active = false;
    };
  }, [onRetentionChange, sessionId]);

  return (
    <details className="debug-artifacts-panel" open>
      <summary>DEBUG GENERATED ARTIFACTS ({artifacts.length})</summary>
      {loadState === "loading" && (
        <p className="debug-artifact-status">生成物を読み込んでいます...</p>
      )}
      {loadState === "error" && (
        <p className="debug-artifact-status error">生成物の一覧を取得できませんでした。</p>
      )}
      {loadState === "ready" && artifacts.length === 0 && (
        <p className="debug-artifact-status">確認できる生成物はまだありません。</p>
      )}
      {artifacts.length > 0 && (
        <div className="debug-artifact-grid">
          {artifacts.map((artifact) => {
            const mediaUrl = apiUrl(artifact.media_url);
            return (
              <article
                className={`debug-artifact-card artifact-${artifact.kind}`}
                key={artifact.path}
              >
                <header>
                  <strong>{artifact.name}</strong>
                  <span>{formatBytes(artifact.size_bytes)}</span>
                </header>
                <code>{artifact.path}</code>
                {artifact.kind === "image" && (
                  <img
                    src={mediaUrl}
                    alt={`デバッグ生成物 ${artifact.name}`}
                    loading="lazy"
                  />
                )}
                {artifact.kind === "audio" && (
                  <audio
                    controls
                    preload="metadata"
                    src={mediaUrl}
                    aria-label={artifact.name}
                  />
                )}
                {artifact.kind === "video" && (
                  <video controls preload="metadata" src={mediaUrl} />
                )}
                {artifact.kind === "text" && (
                  <details className="debug-text-preview">
                    <summary>内容を表示</summary>
                    <pre>{artifact.text_preview ?? "プレビューはありません"}</pre>
                  </details>
                )}
              </article>
            );
          })}
        </div>
      )}
    </details>
  );
}

function ReviewPage({
  session,
  debugMode,
  busy,
  onRegenerate,
  onFinish,
  onNotice
}: {
  session: Session;
  debugMode: boolean;
  busy: boolean;
  onRegenerate: () => Promise<void>;
  onFinish: () => Promise<void>;
  onNotice: (message: string) => void;
}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [videoPlaying, setVideoPlaying] = useState(false);
  const mediaUrl = session.video_artifact?.media_url
    ? apiUrl(session.video_artifact.media_url)
    : null;
  const hasVideo = Boolean(session.video_artifact?.implemented && mediaUrl);
  const usesFishAudio = session.stage_models.voice_clone_tts_worker?.startsWith(
    "fish-s2-pro"
  );

  useEffect(() => {
    setVideoPlaying(false);
  }, [mediaUrl]);

  const playVideo = async () => {
    const video = videoRef.current;
    if (!video) return;

    if (video.ended) video.currentTime = 0;
    try {
      await video.play();
    } catch {
      setVideoPlaying(false);
      onNotice("動画を再生できませんでした");
    }
  };

  return (
    <main className="page review-page">
      <header className="review-header">
        <span className="eyebrow">MESSAGE RECEIVED</span>
        <h1>
          未来のあなたから
          <br />
          メッセージが届きました
        </h1>
      </header>

      <div className="video-shell">
        {hasVideo && mediaUrl ? (
          <div className="video-stage video-stage-ready">
            <video
              ref={videoRef}
              src={mediaUrl}
              controls
              playsInline
              preload="metadata"
              controlsList={session.allow_video_download ? undefined : "nodownload"}
              disablePictureInPicture={!session.allow_video_download}
              onPlay={() => setVideoPlaying(true)}
              onPause={() => setVideoPlaying(false)}
              onEnded={() => setVideoPlaying(false)}
            />
            {!videoPlaying && (
              <button
                type="button"
                className="review-play-button"
                aria-label="動画を再生"
                onClick={() => void playVideo()}
              >
                <Play size={38} fill="currentColor" strokeWidth={2.5} />
              </button>
            )}
            <span className="ai-video-label">
              {session.video_artifact?.ai_generated_label ?? "AI生成映像"}
            </span>
          </div>
        ) : (
          <button
            className="video-stage"
            onClick={() =>
              onNotice(session.video_artifact?.message ?? "動画は未接続です")
            }
            aria-label="動画生成状況を確認"
          >
            <div className="video-placeholder-copy">
              <span className="play-button">
                <Play size={31} fill="currentColor" />
              </span>
              <strong>動画生成ワーカーは未接続です</strong>
              <small>収録・設定・ワーカー経路の確認は完了しています</small>
            </div>
            <span className="ai-video-label">AI生成映像</span>
            <span className="video-expand">
              <Expand size={17} />
            </span>
          </button>
        )}
        <div className="video-caption">
          <span>{session.selected_episode_name}</span>
          <span>{session.selected_effect_name}</span>
        </div>
        {usesFishAudio && (
          <div className="fish-attribution">Built with Fish Audio</div>
        )}
      </div>

      <div className="review-actions">
        {hasVideo && mediaUrl && session.allow_video_download && (
          <a className="secondary-button download-button" href={mediaUrl} download>
            <Download size={18} />
            動画を保存
          </a>
        )}
        <button className="secondary-button" disabled={busy} onClick={onRegenerate}>
          <RotateCcw size={18} />
          エピソードを再抽選して再生成
        </button>
        <button className="primary-button" disabled={busy} onClick={onFinish}>
          <Trash2 size={18} />
          終了してデータを削除
        </button>
      </div>

      {debugMode && <DebugArtifactsPanel sessionId={session.session_id} />}

      {session.final_rarity && (
        <div
          className={`rarity-corner rarity-${session.final_rarity.toLowerCase()}`}
        >
          <span>RARITY</span>
          <strong>{session.final_rarity}</strong>
          {session.base_rarity !== session.final_rarity && (
            <small>{session.base_rarity}から昇格</small>
          )}
        </div>
      )}
    </main>
  );
}

function SafeEndPage({
  session,
  stopped,
  debugMode,
  message,
  busy,
  onReset
}: {
  session: Session;
  stopped: boolean;
  debugMode: boolean;
  message: string | null;
  busy: boolean;
  onReset: () => Promise<void>;
}) {
  const failedWorker =
    session.worker_statuses.find(
      (worker) => worker.role === session.failed_worker_role
    ) ?? session.worker_statuses.find((worker) => worker.state === "failed") ?? null;
  const recentEvents = session.generation_events.slice(-12).reverse();
  const [artifactsRetained, setArtifactsRetained] = useState(false);

  return (
    <main className="page safe-end-page">
      <span className={`safe-end-icon ${stopped ? "stopped" : "error"}`}>
        {stopped ? <CircleStop size={38} /> : <AlertTriangle size={38} />}
      </span>
      <span className="eyebrow">{stopped ? "DEMO STOPPED" : "SAFE EXIT"}</span>
      <h1>{stopped ? "デモを停止しました" : "処理を安全に終了しました"}</h1>
      <p>{message ?? "一時データは削除されています。"}</p>

      {!stopped && (session.failed_worker_role || session.failed_worker_phase) && (
        <section className="failure-location">
          <div>
            <small>失敗した処理</small>
            <strong>
              {session.failed_worker_role
                ? roleLabels[session.failed_worker_role] ?? session.failed_worker_role
                : "生成パイプライン"}
            </strong>
          </div>
          <div>
            <small>停止した段階</small>
            <strong>{displayPhase(session.failed_worker_phase)}</strong>
          </div>
          {failedWorker?.message && <p>{failedWorker.message}</p>}
        </section>
      )}

      {debugMode && !stopped && (
        <details className="error-debug-panel" open>
          <summary>DEBUG ERROR DIAGNOSTICS</summary>
          <dl>
            <dt>error code</dt>
            <dd>{session.error_code ?? "-"}</dd>
            <dt>worker</dt>
            <dd>{session.failed_worker_role ?? "-"}</dd>
            <dt>phase</dt>
            <dd>{session.failed_worker_phase ?? "-"}</dd>
            <dt>model</dt>
            <dd>{failedWorker?.model_id ?? "-"}</dd>
            <dt>backend</dt>
            <dd>{failedWorker?.backend ?? "-"}</dd>
            <dt>runtime</dt>
            <dd>
              {failedWorker?.device ?? "-"} / {failedWorker?.dtype ?? "-"} / {failedWorker?.quantization ?? "-"}
            </dd>
            <dt>request</dt>
            <dd>{failedWorker?.request_id ?? "-"}</dd>
            <dt>attempt</dt>
            <dd>{failedWorker?.attempt || "-"}</dd>
            <dt>detail</dt>
            <dd>{session.error_detail ?? failedWorker?.detail ?? "-"}</dd>
          </dl>
          <div className="debug-event-stream error-events">
            {recentEvents.map((event) => (
              <code key={event.event_id}>
                {eventTime(event.created_at)} [{event.role ?? "pipeline"}] {event.phase} {event.message}
                {event.error_code ? ` error=${event.error_code}` : ""}
              </code>
            ))}
          </div>
        </details>
      )}

      {debugMode && !stopped && (
        <DebugArtifactsPanel
          sessionId={session.session_id}
          onRetentionChange={setArtifactsRetained}
        />
      )}

      <div
        className={`cleanup-confirmation ${
          debugMode && !stopped && artifactsRetained ? "retained" : ""
        }`}
      >
        <ShieldCheck size={19} />
        {debugMode && !stopped
          ? artifactsRetained
            ? "デバッグ確認のため一時データを保持しています。タイトルへ戻ると削除します"
            : "保持された生成データはありません"
          : "セッション内の取得情報と生成データを削除しました"}
      </div>
      <button className="primary-button" disabled={busy} onClick={onReset}>
        <RotateCcw size={18} />
        タイトルへ戻る
      </button>
    </main>
  );
}

const idleQueueState: UploadQueueState = {
  state: "idle",
  queuedBytes: 0,
  consecutiveFailures: 0
};

export default function App() {
  const [config, setConfig] = useState<AppConfig | null>(null);
  const [settings, setSettings] = useState<StaffSettings | null>(null);
  const [options, setOptions] = useState<RuntimeOptions | null>(null);
  const [preparation, setPreparation] = useState<Preparation | null>(null);
  const [session, setSession] = useState<Session | null>(null);
  const [capture, setCapture] = useState<DemoMediaCapture | null>(null);
  const [queueState, setQueueState] = useState<UploadQueueState>(idleQueueState);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [technologyOpen, setTechnologyOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [toast, setToast] = useState<string | null>(null);
  const [fatalError, setFatalError] = useState<string | null>(null);

  const run = useCallback(async (action: () => Promise<void>): Promise<void> => {
    setBusy(true);
    try {
      await action();
    } catch (error) {
      const message = error instanceof Error ? error.message : "処理に失敗しました";
      setToast(message);
    } finally {
      setBusy(false);
    }
  }, []);

  useEffect(() => {
    Promise.all([
      api.config(),
      api.settings(),
      api.runtimeStatus(),
      api.currentSession()
    ])
      .then(([appConfig, settingsResponse, runtimeResponse, sessionResponse]) => {
        const mediaCapture = new DemoMediaCapture(appConfig.capture);
        mediaCapture.setQueueStateListener(setQueueState);
        setCapture(mediaCapture);
        setConfig(appConfig);
        setSettings(settingsResponse.settings);
        setOptions(settingsResponse.options);
        setPreparation(runtimeResponse.preparation);
        setSession(sessionResponse.session);
      })
      .catch((error) => {
        setFatalError(error instanceof Error ? error.message : "起動に失敗しました");
      });
  }, []);

  useEffect(() => {
    const timer = window.setInterval(() => {
      void api
        .runtimeStatus()
        .then((response) => setPreparation(response.preparation))
        .catch(() => undefined);
      void api
        .currentSession()
        .then((response) => setSession(response.session))
        .catch(() => undefined);
    }, 1000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    if (!toast) return;
    const timer = window.setTimeout(() => setToast(null), 3600);
    return () => window.clearTimeout(timer);
  }, [toast]);

  useEffect(() => {
    if (!session || !capture) return;
    if (["generating", "review", "error", "stopped"].includes(session.state)) {
      window.speechSynthesis?.cancel();
      capture.discard();
    }
  }, [capture, session?.state]);

  useEffect(() => {
    if (!session || !capture) return;
    if (!["device_check", "conversation"].includes(session.state)) return;
    const sessionId = session.session_id;
    const handleUnload = () => {
      capture.discard();
      navigator.sendBeacon(api.abandonUrl(sessionId), new Blob([]));
    };
    window.addEventListener("beforeunload", handleUnload);
    return () => window.removeEventListener("beforeunload", handleUnload);
  }, [capture, session]);

  const title = config?.app_name ?? "もしもボックス九工大出張所";
  const debugMode = Boolean(config?.debug_mode);
  const isLoading = !config || !settings || !options || !preparation || !capture;
  const underground = (session?.episode_mode ?? settings?.episode_mode) === "underground";

  const page = useMemo(() => {
    if (fatalError) {
      return (
        <main className="page loading-page">
          <AlertTriangle size={32} />
          <span>{fatalError}</span>
        </main>
      );
    }
    if (isLoading) {
      return (
        <main className="page loading-page">
          <LoaderCircle className="spin" size={32} />
          <span>準備しています</span>
        </main>
      );
    }
    if (!session && preparation.state !== "ready") {
      return (
        <PreparationPage
          preparation={preparation}
          busy={busy}
          onRetry={() =>
            void run(async () => {
              const response = await api.retryRuntime();
              setPreparation(response.preparation);
            })
          }
        />
      );
    }
    if (!session) {
      return (
        <TitlePage
          busy={busy}
          onStart={() =>
            void run(async () => {
              const response = await api.startSession();
              setSession(response.session);
            })
          }
        />
      );
    }
    if (session.state === "consent") {
      return (
        <ConsentPage
          busy={busy}
          onConsent={(voiceConsent) =>
            void run(async () => {
              const response = await api.consent(
                session.session_id,
                voiceConsent
              );
              setSession(response.session);
            })
          }
        />
      );
    }
    if (session.state === "device_check") {
      return (
        <DeviceCheckPage
          sessionId={session.session_id}
          capture={capture}
          config={config}
          busy={busy}
          onReady={(report) =>
            run(async () => {
              const response = await api.deviceComplete(session.session_id, report);
              setSession(response.session);
            })
          }
          onDenied={() =>
            run(async () => {
              const response = await api.deviceDenied(session.session_id);
              setSession(response.session);
            })
          }
          onFailure={setToast}
        />
      );
    }
    if (session.state === "conversation") {
      return (
        <ConversationPage
          session={session}
          debugMode={debugMode}
          capture={capture}
          queueState={queueState}
          busy={busy}
          onAiFinished={() =>
            run(async () => {
              const response = await api.aiFinished(session.session_id);
              setSession(response.session);
            })
          }
          onAnswer={(result) =>
            run(async () => {
              const response = await api.answerComplete(session.session_id, {
                sequence: result.sequence,
                duration_ms: result.durationMs,
                silence_reason: result.reason,
                byte_count: result.byteCount
              });
              setSession(response.session);
              if (response.warning) setToast(response.warning);
            })
          }
          onFinish={() =>
            run(async () => {
              window.speechSynthesis?.cancel();
              capture.stopAnswer("operator");
              await capture.stopAll(true);
              const response = await api.finishConversation(session.session_id);
              setSession(response.session);
            })
          }
          onFailure={setToast}
        />
      );
    }
    if (session.state === "generating") {
      return (
        <GenerationPage
          session={session}
          debugMode={debugMode}
          busy={busy}
          onComplete={() =>
            run(async () => {
              const response = await api.forceGenerationComplete(
                session.session_id
              );
              setSession(response.session);
            })
          }
          onError={() =>
            run(async () => {
              const response = await api.debugError(session.session_id);
              setSession(response.session);
            })
          }
        />
      );
    }
    if (session.state === "review") {
      return (
        <ReviewPage
          session={session}
          debugMode={debugMode}
          busy={busy}
          onNotice={setToast}
          onRegenerate={() =>
            run(async () => {
              if (
                !window.confirm(
                  "エピソードと追加演出を再抽選し、最初から再生成しますか？"
                )
              ) {
                return;
              }
              const response = await api.regenerate(session.session_id);
              setSession(response.session);
              setToast("エピソードを再抽選し、再生成を開始しました");
            })
          }
          onFinish={() =>
            run(async () => {
              const response = await api.resetDemo();
              setSession(null);
              setPreparation(response.preparation);
              setToast("セッションデータを削除しました");
            })
          }
        />
      );
    }
    return (
      <SafeEndPage
        session={session}
        stopped={session.state === "stopped"}
        debugMode={debugMode}
        message={session.error_message}
        busy={busy}
        onReset={() =>
          run(async () => {
            const response = await api.resetDemo();
            setSession(null);
            setPreparation(response.preparation);
          })
        }
      />
    );
  }, [
    busy,
    capture,
    config,
    debugMode,
    fatalError,
    isLoading,
    preparation,
    queueState,
    run,
    session
  ]);

  return (
    <div
      className={`app ${debugMode ? "debug-mode" : ""} ${underground ? "underground-mode" : ""}`}
    >
      {debugMode && (
        <div className="debug-ribbon">
          <FlaskConical size={14} />
          DEBUG MODE
          <span>LOCAL AI PIPELINE / DEVELOPMENT BUILD</span>
        </div>
      )}
      <TopBar
        title={title}
        onOpenSettings={() => {
          setTechnologyOpen(false);
          setSettingsOpen(true);
        }}
        onOpenTechnology={() => {
          setSettingsOpen(false);
          setTechnologyOpen(true);
        }}
      />
      {page}
      {settings && options && (
        <SettingsDrawer
          open={settingsOpen}
          settings={settings}
          options={options}
          busy={busy}
          hasSession={Boolean(session)}
          onClose={() => setSettingsOpen(false)}
          onSave={(next) =>
            run(async () => {
              const response = await api.saveSettings(next);
              setSettings(response.settings);
              setOptions(response.options);
              setSettingsOpen(false);
              setToast("運営設定を保存しました");
            })
          }
          onResetSettings={() =>
            run(async () => {
              const response = await api.resetSettings();
              setSettings(response.settings);
              setOptions(response.options);
              setToast("初期設定に戻しました");
            })
          }
          onResetDemo={() =>
            run(async () => {
              if (!window.confirm("現在のデモを終了し、データを削除しますか？")) {
                return;
              }
              capture?.discard();
              const response = await api.resetDemo();
              setSession(null);
              setPreparation(response.preparation);
              setSettingsOpen(false);
              setToast("デモをリセットしました");
            })
          }
          onEmergencyStop={() =>
            run(async () => {
              if (!window.confirm("緊急停止して一時データを削除しますか？")) {
                return;
              }
              capture?.discard();
              const response = await api.emergencyStop();
              setSession(null);
              setPreparation(response.preparation);
              setSettingsOpen(false);
            })
          }
        />
      )}
      <TechnologyGuide
        open={technologyOpen}
        onClose={() => setTechnologyOpen(false)}
      />
      {session &&
        ["device_check", "conversation"].includes(session.state) &&
        queueState.state === "blocked" && (
        <div className="upload-toast">
          <UploadCloud size={17} />
          保存接続を再試行しています / {formatBytes(queueState.queuedBytes)}
        </div>
      )}
      {toast && <div className="toast">{toast}</div>}
    </div>
  );
}
