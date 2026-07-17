import { api, ApiError } from "./api";
import type { AppConfig } from "./types";

export type SilenceReason = "silence" | "max_duration" | "operator" | "debug";

export interface AnswerRecordingResult {
  sequence: number;
  durationMs: number;
  reason: SilenceReason;
  byteCount: number;
}

export interface UploadQueueState {
  state: "idle" | "uploading" | "blocked" | "overflow";
  queuedBytes: number;
  consecutiveFailures: number;
}

interface UploadItem {
  sessionId: string;
  kind: "video" | "audio";
  sequence: number;
  blob: Blob;
  resolve: () => void;
  reject: (error: Error) => void;
}

type CaptureConfig = AppConfig["capture"];

function recorderMime(kind: "video" | "audio"): string | undefined {
  const candidates =
    kind === "video"
      ? ["video/webm;codecs=vp8", "video/webm", "video/mp4"]
      : ["audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus"];
  return candidates.find((candidate) => MediaRecorder.isTypeSupported(candidate));
}

export class DemoMediaCapture {
  private config: CaptureConfig;
  private mediaStream: MediaStream | null = null;
  private videoRecorder: MediaRecorder | null = null;
  private answerRecorder: MediaRecorder | null = null;
  private answerStop: ((reason: SilenceReason) => void) | null = null;
  private answerSettled: Promise<void> | null = null;
  private videoStopped: Promise<void> | null = null;
  private videoSequence = 0;
  private audioSequence = 0;
  private uploadQueue: UploadItem[] = [];
  private queuedBytes = 0;
  private processingQueue = false;
  private consecutiveFailures = 0;
  private closed = false;
  private queueStateListener: ((state: UploadQueueState) => void) | null = null;

  constructor(config: CaptureConfig) {
    this.config = config;
  }

  setQueueStateListener(
    listener: ((state: UploadQueueState) => void) | null
  ): void {
    this.queueStateListener = listener;
    this.emitQueueState(this.processingQueue ? "uploading" : "idle");
  }

  get stream(): MediaStream | null {
    return this.mediaStream;
  }

  async openDevices(): Promise<MediaStream> {
    if (this.mediaStream?.active) return this.mediaStream;
    this.closed = false;
    this.mediaStream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 1280 },
        height: { ideal: 720 },
        frameRate: { ideal: 30, max: 30 },
        facingMode: "user"
      },
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
        channelCount: 1
      }
    });
    return this.mediaStream;
  }

  startVideo(sessionId: string): void {
    if (this.videoRecorder && this.videoRecorder.state !== "inactive") return;
    if (!this.mediaStream) throw new Error("カメラが準備されていません");
    const tracks = this.mediaStream.getVideoTracks();
    if (!tracks.length) throw new Error("カメラ映像を取得できません");
    const stream = new MediaStream(tracks);
    const mimeType = recorderMime("video");
    this.videoRecorder = new MediaRecorder(
      stream,
      mimeType ? { mimeType, videoBitsPerSecond: 2_500_000 } : undefined
    );
    this.videoRecorder.ondataavailable = (event) => {
      if (!event.data.size) return;
      const sequence = this.videoSequence++;
      void this.enqueueUpload(sessionId, "video", sequence, event.data).catch(
        () => undefined
      );
    };
    this.videoStopped = new Promise((resolve) => {
      if (this.videoRecorder) this.videoRecorder.onstop = () => resolve();
    });
    this.videoRecorder.start(this.config.video_chunk_seconds * 1000);
  }

  async recordAnswer(
    sessionId: string,
    onVolume?: (volume: number) => void
  ): Promise<AnswerRecordingResult> {
    if (!this.mediaStream) throw new Error("マイクが準備されていません");
    if (this.answerRecorder && this.answerRecorder.state !== "inactive") {
      throw new Error("音声をすでに収録しています");
    }
    const audioTrack = this.mediaStream.getAudioTracks()[0];
    if (!audioTrack) throw new Error("マイク音声を取得できません");

    const stream = new MediaStream([audioTrack]);
    const mimeType = recorderMime("audio");
    const recorder = new MediaRecorder(
      stream,
      mimeType ? { mimeType, audioBitsPerSecond: 96_000 } : undefined
    );
    this.answerRecorder = recorder;
    const chunks: Blob[] = [];
    const sequence = this.audioSequence++;
    const startedAt = performance.now();
    const audioContext = new AudioContext();
    const analyser = audioContext.createAnalyser();
    analyser.fftSize = 1024;
    const source = audioContext.createMediaStreamSource(stream);
    source.connect(analyser);
    const samples = new Uint8Array(analyser.fftSize);
    let speechFrames = 0;
    let speechStarted = false;
    let silenceStartedAt: number | null = null;
    let stopReason: SilenceReason = "max_duration";
    let timer = 0;

    const recording = new Promise<AnswerRecordingResult>((resolve, reject) => {
      const cleanup = () => {
        window.clearInterval(timer);
        source.disconnect();
        void audioContext.close();
        this.answerStop = null;
        this.answerRecorder = null;
        onVolume?.(0);
      };

      const stop = (reason: SilenceReason) => {
        if (recorder.state === "inactive") return;
        stopReason = reason;
        recorder.stop();
      };
      this.answerStop = stop;

      recorder.ondataavailable = (event) => {
        if (event.data.size) chunks.push(event.data);
      };
      recorder.onerror = () => {
        cleanup();
        reject(new Error("回答音声の収録に失敗しました"));
      };
      recorder.onstop = () => {
        const durationMs = Math.max(0, Math.round(performance.now() - startedAt));
        const blob = new Blob(chunks, {
          type: recorder.mimeType || "audio/webm"
        });
        cleanup();
        this.enqueueUpload(sessionId, "audio", sequence, blob)
          .then(() =>
            resolve({
              sequence,
              durationMs,
              reason: stopReason,
              byteCount: blob.size
            })
          )
          .catch(reject);
      };

      recorder.start(250);
      timer = window.setInterval(() => {
        analyser.getByteTimeDomainData(samples);
        let sum = 0;
        for (const sample of samples) {
          const centered = (sample - 128) / 128;
          sum += centered * centered;
        }
        const rms = Math.sqrt(sum / samples.length);
        onVolume?.(Math.min(1, rms / 0.18));
        const now = performance.now();
        const elapsed = (now - startedAt) / 1000;

        if (rms >= this.config.speech_start_threshold) {
          speechFrames += 1;
          silenceStartedAt = null;
          if (speechFrames >= 3) speechStarted = true;
        } else if (speechStarted) {
          if (silenceStartedAt === null) silenceStartedAt = now;
          if ((now - silenceStartedAt) / 1000 >= this.config.silence_seconds) {
            stop("silence");
          }
        }

        if (elapsed >= this.config.response_max_seconds) {
          stop("max_duration");
        }
      }, 50);
    });
    this.answerSettled = recording.then(
      () => undefined,
      () => undefined
    );
    return recording;
  }

  stopAnswer(reason: SilenceReason = "operator"): void {
    this.answerStop?.(reason);
  }

  async stopAll(waitForUploads = true): Promise<void> {
    const answerSettled = this.answerSettled;
    const videoStopped = this.videoStopped;
    this.stopAnswer("operator");
    if (this.videoRecorder && this.videoRecorder.state !== "inactive") {
      this.videoRecorder.stop();
    }
    if (waitForUploads) {
      await Promise.all([answerSettled, videoStopped].filter(Boolean));
      await this.flushUploads();
    }
    this.videoRecorder = null;
    this.answerSettled = null;
    this.videoStopped = null;
    this.mediaStream?.getTracks().forEach((track) => track.stop());
    this.mediaStream = null;
    this.closed = true;
    this.consecutiveFailures = 0;
    this.emitQueueState("idle");
  }

  async flushUploads(): Promise<void> {
    while (this.processingQueue || this.uploadQueue.length) {
      await new Promise((resolve) => window.setTimeout(resolve, 80));
    }
  }

  discard(reason = "セッションを終了しました"): void {
    this.closed = true;
    this.stopAnswer("operator");
    if (this.videoRecorder && this.videoRecorder.state !== "inactive") {
      this.videoRecorder.stop();
    }
    this.videoRecorder = null;
    this.answerSettled = null;
    this.videoStopped = null;
    this.mediaStream?.getTracks().forEach((track) => track.stop());
    this.mediaStream = null;
    this.rejectPendingUploads(new Error(reason));
    this.consecutiveFailures = 0;
    this.emitQueueState("idle");
  }

  private enqueueUpload(
    sessionId: string,
    kind: "video" | "audio",
    sequence: number,
    blob: Blob
  ): Promise<void> {
    if (this.closed) return Promise.reject(new Error("収録は終了しています"));
    const limitBytes = this.config.browser_queue_limit_mb * 1024 * 1024;
    if (this.queuedBytes + blob.size > limitBytes) {
      this.emitQueueState("overflow");
      return Promise.reject(
        new Error("未送信データが上限に達したため、収録を停止しました")
      );
    }
    this.queuedBytes += blob.size;
    const promise = new Promise<void>((resolve, reject) => {
      this.uploadQueue.push({
        sessionId,
        kind,
        sequence,
        blob,
        resolve,
        reject
      });
    });
    void this.processUploads();
    return promise;
  }

  private async processUploads(): Promise<void> {
    if (this.processingQueue) return;
    this.processingQueue = true;
    this.emitQueueState("uploading");
    try {
      while (this.uploadQueue.length && !this.closed) {
        const item = this.uploadQueue[0];
        try {
          await api.uploadMedia(
            item.sessionId,
            item.kind,
            item.sequence,
            item.blob
          );
          this.settleUpload(item);
          this.consecutiveFailures = 0;
        } catch (error) {
          if (error instanceof ApiError && error.status === 413) {
            this.settleUpload(item, error);
            this.consecutiveFailures = 0;
            continue;
          }
          if (
            this.closed ||
            (error instanceof ApiError && [404, 409].includes(error.status))
          ) {
            this.discard("保存受付が終了したため、未送信データを破棄しました");
            break;
          }
          this.consecutiveFailures += 1;
          if (this.consecutiveFailures >= this.config.upload_retry_count) {
            this.emitQueueState("blocked");
            await api.reportUploadFailure(item.sessionId).catch(() => undefined);
            await new Promise((resolve) => window.setTimeout(resolve, 2500));
          } else {
            await new Promise((resolve) =>
              window.setTimeout(resolve, 350 * this.consecutiveFailures)
            );
          }
        }
      }
    } finally {
      this.processingQueue = false;
      this.emitQueueState("idle");
    }
  }

  private settleUpload(item: UploadItem, error?: Error): void {
    const index = this.uploadQueue.indexOf(item);
    if (index < 0) return;
    this.uploadQueue.splice(index, 1);
    this.queuedBytes = Math.max(0, this.queuedBytes - item.blob.size);
    if (error) item.reject(error);
    else item.resolve();
  }

  private rejectPendingUploads(error: Error): void {
    for (const item of this.uploadQueue.splice(0)) {
      item.reject(error);
    }
    this.queuedBytes = 0;
  }

  private emitQueueState(state: UploadQueueState["state"]): void {
    this.queueStateListener?.({
      state,
      queuedBytes: this.queuedBytes,
      consecutiveFailures: this.consecutiveFailures
    });
  }
}
