from __future__ import annotations

import asyncio
import json
import resource
import subprocess
import time
import wave
from pathlib import Path
from typing import Any, Callable

from ...schemas import (
    SCHEMA_VERSION,
    WorkerHealth,
    WorkerMetrics,
    WorkerModelSpec,
    WorkerProgressEvent,
    WorkerRequest,
    WorkerResult,
    WorkerRole,
)
from ..base import ProgressCallback, WorkerAdapter


SUPPORTED_ROLES = {
    WorkerRole.FINAL_ASR,
    WorkerRole.INTERVIEW_SUMMARY,
    WorkerRole.EPISODE_SELECTOR,
    WorkerRole.REFERENCE_FRAME_SELECTOR,
    WorkerRole.VOICE_REFERENCE_SELECTOR,
    WorkerRole.LIP_SYNC,
    WorkerRole.VIDEO_POSTPROCESS,
}


class VoiceReferenceSelectionError(RuntimeError):
    code = "voice_reference_selection_failed"
    phase = "voice_reference.normalization"


class PipelineUtilitiesAdapter(WorkerAdapter):
    def __init__(self, role: WorkerRole) -> None:
        if role not in SUPPORTED_ROLES:
            raise ValueError(f"unsupported_pipeline_utility_role: {role.value}")
        self.role = role
        self._model_spec: WorkerModelSpec | None = None
        self._cancelled: set[str] = set()

    async def load(self, model: WorkerModelSpec) -> WorkerHealth:
        if model.schema_version != SCHEMA_VERSION or model.worker != self.role:
            raise ValueError("worker_model_schema_mismatch")
        self._model_spec = model
        return await self.healthcheck()

    async def healthcheck(self) -> WorkerHealth:
        return WorkerHealth(
            worker=self.role,
            loaded=self._model_spec is not None,
            ready=self._model_spec is not None,
            backend=self._model_spec.backend if self._model_spec else None,
            model_id=self._model_spec.model_id if self._model_spec else None,
        )

    async def run(
        self,
        request: WorkerRequest,
        progress: ProgressCallback | None = None,
    ) -> WorkerResult:
        if request.schema_version != SCHEMA_VERSION:
            raise ValueError("worker_request_schema_mismatch")
        if request.worker != self.role:
            raise ValueError("worker_role_mismatch")
        if self._model_spec is None:
            raise RuntimeError("worker_not_loaded")
        if request.model.catalog_id != self._model_spec.catalog_id:
            raise ValueError("worker_loaded_model_mismatch")
        if request.request_id in self._cancelled:
            self._cancelled.discard(request.request_id)
            raise asyncio.CancelledError

        output_dir = Path(request.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if progress:
            phase = "inference"
            message = "入力データを確認しています"
            if self.role == WorkerRole.VOICE_REFERENCE_SELECTOR:
                phase = "voice_reference.candidate_scan"
                message = "録音した音声候補を確認しています"
            await progress(
                WorkerProgressEvent(
                    request_id=request.request_id,
                    worker=self.role,
                    progress=0.1,
                    message=message,
                    phase=phase,
                )
            )

        started = time.perf_counter()
        handler = self._handler()
        output_paths, metadata = await asyncio.to_thread(
            handler,
            request,
            output_dir,
        )
        if request.request_id in self._cancelled:
            self._cancelled.discard(request.request_id)
            raise asyncio.CancelledError

        if progress:
            phase = "completed"
            detail = None
            if self.role == WorkerRole.VOICE_REFERENCE_SELECTOR:
                phase = "voice_reference.completed"
                detail = (
                    f"候補 {metadata.get('candidate_count', 0)}件 / "
                    f"採用 {metadata.get('selected_count', 0)}件 / "
                    f"{metadata.get('duration_seconds', 0):.1f}秒"
                )
            await progress(
                WorkerProgressEvent(
                    request_id=request.request_id,
                    worker=self.role,
                    progress=1.0,
                    message="処理が完了しました",
                    phase=phase,
                    detail=detail,
                )
            )
        return WorkerResult(
            request_id=request.request_id,
            worker=self.role,
            backend=self._model_spec.backend,
            model_id=self._model_spec.model_id,
            model_revision=self._model_spec.model_revision,
            implemented=True,
            output_paths=output_paths,
            metadata=metadata,
            metrics=WorkerMetrics(
                processing_time_ms=int((time.perf_counter() - started) * 1000),
                peak_cpu_memory_mb=self._peak_cpu_memory_mb(),
            ),
        )

    def _handler(
        self,
    ) -> Callable[[WorkerRequest, Path], tuple[dict[str, str], dict[str, Any]]]:
        return {
            WorkerRole.FINAL_ASR: self._final_asr,
            WorkerRole.INTERVIEW_SUMMARY: self._interview_summary,
            WorkerRole.EPISODE_SELECTOR: self._episode_selector,
            WorkerRole.REFERENCE_FRAME_SELECTOR: self._reference_frame_selector,
            WorkerRole.VOICE_REFERENCE_SELECTOR: self._voice_reference_selector,
            WorkerRole.LIP_SYNC: self._lip_sync_passthrough,
            WorkerRole.VIDEO_POSTPROCESS: self._video_postprocess,
        }[self.role]

    @staticmethod
    def _read_json(path_value: str | None) -> Any:
        if not path_value:
            raise ValueError("required_json_input_missing")
        path = Path(path_value)
        if not path.is_file():
            raise FileNotFoundError(path)
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _write_json(path: Path, value: Any) -> None:
        path.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _final_asr(
        self,
        request: WorkerRequest,
        output_dir: Path,
    ) -> tuple[dict[str, str], dict[str, Any]]:
        transcript = self._read_json(request.input_paths.get("transcript"))
        visitor_entries = [
            item for item in transcript if item.get("speaker") == "visitor"
        ]
        payload = {
            "schema_version": SCHEMA_VERSION,
            "text": "\n".join(str(item.get("text", "")).strip() for item in visitor_entries),
            "segments": visitor_entries,
            "source": "turn_asr_transcript",
        }
        path = output_dir / "final-transcript.json"
        self._write_json(path, payload)
        return {"final_transcript": str(path)}, {"visitor_segment_count": len(visitor_entries)}

    def _interview_summary(
        self,
        request: WorkerRequest,
        output_dir: Path,
    ) -> tuple[dict[str, str], dict[str, Any]]:
        transcript = self._read_json(request.input_paths.get("transcript"))
        state = self._read_json(request.input_paths.get("interview_state"))
        utterances = [
            str(item.get("text", "")).strip()
            for item in transcript
            if item.get("speaker") == "visitor" and str(item.get("text", "")).strip()
        ]
        summary = "来場者本人の発話: " + " / ".join(utterances)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "summary": summary,
            "person_information": state.get("acquired_information", {}),
            "visitor_utterances": utterances,
        }
        path = output_dir / "interview-summary.json"
        self._write_json(path, payload)
        return {"interview_summary": str(path)}, {"summary_chars": len(summary)}

    def _episode_selector(
        self,
        request: WorkerRequest,
        output_dir: Path,
    ) -> tuple[dict[str, str], dict[str, Any]]:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "episode": request.metadata.get("episode"),
            "effect": request.metadata.get("effect"),
            "final_rarity": request.metadata.get("final_rarity"),
            "episode_mode": request.metadata.get("episode_mode"),
        }
        path = output_dir / "episode-selection.json"
        self._write_json(path, payload)
        return {"episode_selection": str(path)}, payload

    def _reference_frame_selector(
        self,
        request: WorkerRequest,
        output_dir: Path,
    ) -> tuple[dict[str, str], dict[str, Any]]:
        source_dir = Path(request.input_paths.get("video_chunks_dir", ""))
        candidates = sorted(path for path in source_dir.glob("*.*") if path.is_file())
        if not candidates:
            raise FileNotFoundError("reference_video_chunks_not_found")

        from PIL import Image, ImageFilter, ImageStat

        ffmpeg = str(request.model.parameters.get("ffmpeg_path", "/usr/bin/ffmpeg"))
        ffprobe = str(request.model.parameters.get("ffprobe_path", "/usr/bin/ffprobe"))
        scored: list[tuple[float, Path, dict[str, float]]] = []
        for index, video_path in enumerate(candidates):
            duration = self._duration_seconds(video_path, ffprobe)
            seek = max(0.0, min(duration * 0.5, max(0.0, duration - 0.05)))
            frame_path = output_dir / f"candidate-{index:03d}.png"
            command = [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                f"{seek:.3f}",
                "-i",
                str(video_path),
                "-frames:v",
                "1",
                str(frame_path),
            ]
            completed = subprocess.run(command, capture_output=True, text=True)
            if completed.returncode != 0 or not frame_path.is_file():
                continue
            with Image.open(frame_path) as image:
                rgb = image.convert("RGB")
                side = min(rgb.size)
                left = (rgb.width - side) // 2
                top = (rgb.height - side) // 2
                square = rgb.crop((left, top, left + side, top + side)).resize(
                    (768, 768),
                    Image.Resampling.LANCZOS,
                )
                gray = square.convert("L")
                brightness = float(ImageStat.Stat(gray).mean[0])
                contrast = float(ImageStat.Stat(gray).stddev[0])
                edges = gray.filter(ImageFilter.FIND_EDGES)
                sharpness = float(ImageStat.Stat(edges).var[0])
                brightness_score = max(0.0, 1.0 - abs(brightness - 128.0) / 128.0)
                score = brightness_score * 35.0 + min(contrast, 64.0) + min(sharpness / 20.0, 80.0)
                square.save(frame_path, format="PNG")
            scored.append(
                (
                    score,
                    frame_path,
                    {
                        "brightness": brightness,
                        "contrast": contrast,
                        "sharpness": sharpness,
                    },
                )
            )

        if not scored:
            raise RuntimeError("reference_frame_extraction_failed")
        score, selected, details = max(scored, key=lambda item: item[0])
        final_path = output_dir / "reference-frame.png"
        selected.replace(final_path)
        for _, candidate, _ in scored:
            if candidate != selected:
                candidate.unlink(missing_ok=True)
        metadata = {"score": score, **details, "candidate_count": len(scored)}
        self._write_json(output_dir / "reference-frame.json", metadata)
        return {"reference_image": str(final_path)}, metadata

    def _voice_reference_selector(
        self,
        request: WorkerRequest,
        output_dir: Path,
    ) -> tuple[dict[str, str], dict[str, Any]]:
        source_dir = Path(request.input_paths.get("audio_answers_dir", ""))
        audio_paths = sorted(path for path in source_dir.glob("*.*") if path.is_file())
        transcript = self._read_json(request.input_paths.get("transcript"))
        visitor_texts = [
            str(item.get("text", "")).strip()
            for item in transcript
            if item.get("speaker") == "visitor" and str(item.get("text", "")).strip()
        ]
        if not audio_paths:
            raise FileNotFoundError("voice_reference_audio_not_found")

        ffmpeg = str(request.model.parameters.get("ffmpeg_path", "/usr/bin/ffmpeg"))
        ffprobe = str(request.model.parameters.get("ffprobe_path", "/usr/bin/ffprobe"))
        normalized: list[tuple[float, float, Path, str, str, str]] = []
        diagnostics: list[dict[str, Any]] = []
        for index, audio_path in enumerate(audio_paths):
            source_duration = self._duration_seconds(audio_path, ffprobe)
            text = visitor_texts[index] if index < len(visitor_texts) else ""
            normalized_path = output_dir / f"reference-part-{index:03d}.wav"
            candidate_diagnostic: dict[str, Any] = {
                "file": audio_path.name,
                "bytes": audio_path.stat().st_size,
                "source_duration_seconds": (
                    round(source_duration, 3) if source_duration else None
                ),
                "attempts": [],
            }
            normalized_duration = 0.0
            normalization_method = ""
            filters = (
                (
                    "silence_trim",
                    "highpass=f=60,lowpass=f=12000,"
                    "silenceremove=start_periods=1:start_duration=0.1:"
                    "start_threshold=-45dB:stop_periods=1:stop_duration=1.0:"
                    "stop_threshold=-45dB,loudnorm=I=-20:TP=-2:LRA=11",
                ),
                (
                    "normalization_without_trim",
                    "highpass=f=60,lowpass=f=12000,"
                    "loudnorm=I=-20:TP=-2:LRA=11",
                ),
            )
            for method, audio_filter in filters:
                normalized_path.unlink(missing_ok=True)
                command = [
                    ffmpeg,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(audio_path),
                    "-vn",
                    "-af",
                    audio_filter,
                    "-ar",
                    "44100",
                    "-ac",
                    "1",
                    "-c:a",
                    "pcm_s16le",
                    str(normalized_path),
                ]
                completed = subprocess.run(command, capture_output=True, text=True)
                normalized_duration = (
                    self._duration_seconds(normalized_path, ffprobe)
                    if completed.returncode == 0 and normalized_path.is_file()
                    else 0.0
                )
                candidate_diagnostic["attempts"].append(
                    {
                        "method": method,
                        "return_code": completed.returncode,
                        "duration_seconds": round(normalized_duration, 3),
                        "stderr": completed.stderr.strip()[-240:] or None,
                    }
                )
                if normalized_duration >= 1.2:
                    normalization_method = method
                    break

            if normalized_duration < 1.2:
                normalized_path.unlink(missing_ok=True)
                candidate_diagnostic["result"] = "too_short_or_decode_failed"
                diagnostics.append(candidate_diagnostic)
                continue

            candidate_diagnostic["result"] = "accepted"
            candidate_diagnostic["normalization_method"] = normalization_method
            diagnostics.append(candidate_diagnostic)
            duration_score = min(normalized_duration, 20.0)
            text_score = min(len(text), 120) / 20.0
            normalized.append(
                (
                    duration_score + text_score,
                    normalized_duration,
                    normalized_path,
                    text,
                    str(audio_path),
                    normalization_method,
                )
            )

        if not normalized:
            summary_parts = []
            for item in diagnostics:
                last_attempt = item["attempts"][-1] if item["attempts"] else {}
                stderr = str(last_attempt.get("stderr") or "none").replace("\n", " ")
                summary_parts.append(
                    f"{item['file']}: source={item['source_duration_seconds']}, "
                    f"result={item['result']}, ffmpeg={last_attempt.get('return_code')}, "
                    f"decoded={last_attempt.get('duration_seconds')}, stderr={stderr[:160]}"
                )
            raise VoiceReferenceSelectionError(
                "参照音声を作成できませんでした: "
                f"候補={len(audio_paths)}件、採用=0件、"
                f"結果=[{'; '.join(summary_parts)}]"
            )

        normalized.sort(key=lambda item: item[0], reverse=True)
        chosen: list[tuple[float, float, Path, str, str, str]] = []
        total_duration = 0.0
        for item in normalized:
            if chosen and total_duration + item[1] > 28.0:
                continue
            chosen.append(item)
            total_duration += item[1]
            if total_duration >= 12.0 or len(chosen) >= 3:
                break

        reference_path = output_dir / "voice-reference.wav"
        self._concatenate_wav([item[2] for item in chosen], reference_path)
        prompt_text = " ".join(item[3] for item in chosen if item[3]).strip()
        if not prompt_text:
            raise RuntimeError("voice_reference_transcript_missing")
        payload = {
            "schema_version": SCHEMA_VERSION,
            "audio_path": str(reference_path),
            "prompt_text": prompt_text,
            "duration_seconds": self._duration_seconds(reference_path, ffprobe),
            "source_files": [item[4] for item in chosen],
            "selection_method": "duration_text_with_audio_normalization",
            "normalization_methods": [item[5] for item in chosen],
            "candidate_count": len(audio_paths),
            "accepted_candidate_count": len(normalized),
            "selected_count": len(chosen),
            "candidate_diagnostics": diagnostics,
        }
        metadata_path = output_dir / "voice-reference.json"
        self._write_json(metadata_path, payload)
        for _, _, candidate, _, _, _ in normalized:
            candidate.unlink(missing_ok=True)
        return {
            "voice_reference": str(metadata_path),
            "voice_reference_audio": str(reference_path),
        }, payload

    def _lip_sync_passthrough(
        self,
        request: WorkerRequest,
        output_dir: Path,
    ) -> tuple[dict[str, str], dict[str, Any]]:
        video_value = request.input_paths.get("generated_video")
        if not video_value or not Path(video_value).is_file():
            raise FileNotFoundError("generated_video_not_found")
        payload = {
            "schema_version": SCHEMA_VERSION,
            "method": "integrated_in_echomimic_v3",
            "video_path": video_value,
        }
        self._write_json(output_dir / "lip-sync.json", payload)
        return {"lip_synced_video": video_value}, payload

    def _video_postprocess(
        self,
        request: WorkerRequest,
        output_dir: Path,
    ) -> tuple[dict[str, str], dict[str, Any]]:
        video_value = request.input_paths.get("lip_synced_video") or request.input_paths.get(
            "generated_video"
        )
        audio_value = request.input_paths.get("generated_audio")
        session_dir_value = request.input_paths.get("session_dir")
        if not video_value or not Path(video_value).is_file():
            raise FileNotFoundError("postprocess_video_not_found")
        if not audio_value or not Path(audio_value).is_file():
            raise FileNotFoundError("postprocess_audio_not_found")
        if not session_dir_value:
            raise ValueError("postprocess_session_dir_required")

        ffmpeg = str(request.model.parameters.get("ffmpeg_path", "/usr/bin/ffmpeg"))
        final_path = Path(session_dir_value) / "output" / "future-message.mp4"
        final_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            video_value,
            "-i",
            audio_value,
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            "-movflags",
            "+faststart",
            "-metadata",
            "comment=AI-generated fictional future message",
            str(final_path),
        ]
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode != 0 or not final_path.is_file():
            raise RuntimeError(f"video_postprocess_failed: {completed.stderr[-500:]}")
        payload = {
            "schema_version": SCHEMA_VERSION,
            "video_path": str(final_path),
            "ai_generated": True,
        }
        self._write_json(output_dir / "video-postprocess.json", payload)
        return {"final_video": str(final_path)}, payload

    @staticmethod
    def _duration_seconds(path: Path, ffprobe: str) -> float:
        if path.suffix.lower() == ".wav":
            try:
                with wave.open(str(path), "rb") as source:
                    frame_rate = source.getframerate()
                    if frame_rate > 0:
                        return max(0.0, source.getnframes() / frame_rate)
            except (OSError, EOFError, wave.Error):
                pass

        command = [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode == 0:
            try:
                return max(0.0, float(completed.stdout.strip()))
            except ValueError:
                pass

        # MediaRecorder WebM often omits container duration metadata.
        # Packet timestamps still describe all decodable audio.
        packet_command = [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "packet=pts_time,duration_time",
            "-of",
            "csv=p=0",
            str(path),
        ]
        packet_result = subprocess.run(
            packet_command,
            capture_output=True,
            text=True,
        )
        if packet_result.returncode != 0:
            return 0.0

        duration = 0.0
        for line in packet_result.stdout.splitlines():
            values = line.rstrip(",").split(",")
            if not values:
                continue
            try:
                start = float(values[0])
                packet_duration = float(values[1]) if len(values) > 1 else 0.0
            except ValueError:
                continue
            duration = max(duration, start + packet_duration)
        return max(0.0, duration)

    @staticmethod
    def _concatenate_wav(paths: list[Path], output_path: Path) -> None:
        if not paths:
            raise ValueError("wav_parts_required")
        with wave.open(str(paths[0]), "rb") as first:
            params = first.getparams()
            frames = [first.readframes(first.getnframes())]
        silence = b"\x00" * int(params.framerate * 0.15) * params.sampwidth * params.nchannels
        for path in paths[1:]:
            with wave.open(str(path), "rb") as source:
                current = source.getparams()
                if (
                    current.nchannels != params.nchannels
                    or current.sampwidth != params.sampwidth
                    or current.framerate != params.framerate
                ):
                    raise ValueError("wav_reference_format_mismatch")
                frames.extend([silence, source.readframes(source.getnframes())])
        with wave.open(str(output_path), "wb") as target:
            target.setnchannels(params.nchannels)
            target.setsampwidth(params.sampwidth)
            target.setframerate(params.framerate)
            target.writeframes(b"".join(frames))

    @staticmethod
    def _peak_cpu_memory_mb() -> int:
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) // 1024

    async def cancel(self, request_id: str) -> None:
        self._cancelled.add(request_id)

    async def unload(self) -> None:
        self._model_spec = None
        self._cancelled.clear()


def create_worker(role: WorkerRole) -> WorkerAdapter:
    return PipelineUtilitiesAdapter(role)
