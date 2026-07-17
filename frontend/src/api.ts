import type {
  AppConfig,
  DebugArtifact,
  Preparation,
  Session,
  SettingsResponse,
  StaffSettings
} from "./types";

export function apiUrl(path: string): string {
  const base = new URL("./", document.baseURI);
  return new URL(path.replace(/^\/+/, ""), base).toString();
}

export class ApiError extends Error {
  constructor(message: string, readonly status: number) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(apiUrl(path), {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {})
    }
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new ApiError(
      body.detail ?? `通信に失敗しました (${response.status})`,
      response.status
    );
  }
  return response.json() as Promise<T>;
}

export const api = {
  config: () => request<AppConfig>("api/config"),
  runtimeStatus: () =>
    request<{ preparation: Preparation }>("api/runtime/status"),
  retryRuntime: () =>
    request<{ preparation: Preparation }>("api/runtime/retry", {
      method: "POST"
    }),
  settings: () => request<SettingsResponse>("api/settings"),
  saveSettings: (settings: StaffSettings) =>
    request<SettingsResponse>("api/settings", {
      method: "PUT",
      body: JSON.stringify(settings)
    }),
  resetSettings: () =>
    request<SettingsResponse>("api/settings/reset", { method: "POST" }),
  currentSession: () =>
    request<{ session: Session | null }>("api/session/current"),
  startSession: () =>
    request<{ session: Session }>("api/session/start", { method: "POST" }),
  consent: (sessionId: string, voiceCloneConsent: boolean) =>
    request<{ session: Session }>(`api/session/${sessionId}/consent`, {
      method: "POST",
      body: JSON.stringify({ voice_clone_consent: voiceCloneConsent })
    }),
  deviceDenied: (sessionId: string) =>
    request<{ session: Session }>(
      `api/session/${sessionId}/device-check/denied`,
      { method: "POST" }
    ),
  deviceComplete: (
    sessionId: string,
    report: {
      camera_width: number;
      camera_height: number;
      camera_fps: number | null;
      face_check_supported: boolean;
      face_detected: boolean | null;
      brightness: number | null;
    }
  ) =>
    request<{ session: Session }>(
      `api/session/${sessionId}/device-check/complete`,
      { method: "POST", body: JSON.stringify(report) }
    ),
  uploadMedia: async (
    sessionId: string,
    kind: "video" | "audio",
    sequence: number,
    blob: Blob
  ) => {
    const query = new URLSearchParams({
      kind,
      sequence: String(sequence),
      mime_type: blob.type || "application/octet-stream"
    });
    const response = await fetch(
      apiUrl(`api/session/${sessionId}/media/chunk?${query}`),
      {
        method: "POST",
        headers: { "Content-Type": "application/octet-stream" },
        body: blob
      }
    );
    if (!response.ok) {
      throw new ApiError(
        `メディアの保存に失敗しました (${response.status})`,
        response.status
      );
    }
    return response.json() as Promise<{ ok: boolean; bytes: number }>;
  },
  reportUploadFailure: (sessionId: string) =>
    request<{ ok: boolean }>(
      `api/session/${sessionId}/media/upload-failure`,
      { method: "POST" }
    ),
  aiFinished: (sessionId: string) =>
    request<{ session: Session }>(
      `api/session/${sessionId}/conversation/ai-finished`,
      { method: "POST" }
    ),
  answerComplete: (
    sessionId: string,
    payload: {
      sequence: number;
      duration_ms: number;
      silence_reason: "silence" | "max_duration" | "operator" | "debug";
      byte_count: number;
    }
  ) =>
    request<{ session: Session; warning: string | null }>(
      `api/session/${sessionId}/conversation/answer-complete`,
      { method: "POST", body: JSON.stringify(payload) }
    ),
  addUtterance: (sessionId: string, text: string) =>
    request<{ session: Session }>(
      `api/session/${sessionId}/conversation/utterance`,
      { method: "POST", body: JSON.stringify({ text }) }
    ),
  finishConversation: (sessionId: string) =>
    request<{ session: Session }>(
      `api/session/${sessionId}/conversation/finish`,
      { method: "POST" }
    ),
  regenerate: (sessionId: string) =>
    request<{ session: Session }>(
      `api/session/${sessionId}/review/regenerate`,
      { method: "POST" }
    ),
  forceGenerationComplete: (sessionId: string) =>
    request<{ session: Session }>(
      `api/session/${sessionId}/generation/complete`,
      { method: "POST" }
    ),
  debugArtifacts: (sessionId: string) =>
    request<{ retained: boolean; artifacts: DebugArtifact[] }>(
      `api/session/${sessionId}/debug/artifacts`
    ),
  debugError: (sessionId: string) =>
    request<{ session: Session }>(`api/session/${sessionId}/debug/error`, {
      method: "POST"
    }),
  abandonUrl: (sessionId: string) =>
    apiUrl(`api/session/${sessionId}/abandon`),
  abandon: (sessionId: string) =>
    request<{ deleted: boolean }>(`api/session/${sessionId}/abandon`, {
      method: "POST"
    }),
  emergencyStop: () =>
    request<{ session: null; preparation: Preparation }>(
      "api/control/emergency-stop",
      { method: "POST" }
    ),
  resetDemo: () =>
    request<{ session: null; preparation: Preparation }>("api/control/reset", {
      method: "POST"
    })
};
