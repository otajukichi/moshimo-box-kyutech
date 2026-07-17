from __future__ import annotations

import asyncio
import gc
import json
import os
import resource
import subprocess
import time
from pathlib import Path
from typing import Any

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


SUPPORTED_ROLES = {WorkerRole.STREAMING_ASR, WorkerRole.FINAL_ASR}


class FasterWhisperAsrAdapter(WorkerAdapter):
    def __init__(self, role: WorkerRole) -> None:
        if role not in SUPPORTED_ROLES:
            raise ValueError(f"unsupported_faster_whisper_role: {role.value}")
        self.role = role
        self._model_spec: WorkerModelSpec | None = None
        self._model: Any | None = None
        self._cancelled: set[str] = set()
        self._gpu_baseline_mb = 0

    async def load(self, model: WorkerModelSpec) -> WorkerHealth:
        if model.schema_version != SCHEMA_VERSION or model.worker != self.role:
            raise ValueError("worker_model_schema_mismatch")
        source = model.model_path or model.model_id
        if model.model_path and not Path(model.model_path).is_dir():
            raise FileNotFoundError(f"asr_model_path_not_found: {model.model_path}")
        self._gpu_baseline_mb = (
            self._total_gpu_memory_mb()
            if model.device.lower().startswith("cuda")
            else 0
        )
        self._model = await asyncio.to_thread(self._create_model, source, model)
        self._model_spec = model
        return await self.healthcheck()

    @staticmethod
    def _create_model(source: str, model: WorkerModelSpec) -> Any:
        from faster_whisper import WhisperModel

        device = "cuda" if model.device.lower().startswith("cuda") else "cpu"
        kwargs: dict[str, Any] = {
            "device": device,
            "compute_type": FasterWhisperAsrAdapter._compute_type(model),
        }
        if device == "cuda":
            kwargs["device_index"] = FasterWhisperAsrAdapter._device_index(model.device)
        return WhisperModel(source, **kwargs)

    @staticmethod
    def _compute_type(model: WorkerModelSpec) -> str:
        quantization = model.quantization.lower()
        if quantization not in {"", "none"}:
            return quantization
        dtype = model.dtype.lower()
        if dtype in {"float16", "float32", "int8", "int8_float16", "int8_float32"}:
            return dtype
        return "float16" if model.device.lower().startswith("cuda") else "int8"

    @staticmethod
    def _device_index(device: str) -> int:
        _, separator, index = device.partition(":")
        return int(index) if separator and index.isdigit() else 0

    async def healthcheck(self) -> WorkerHealth:
        return WorkerHealth(
            worker=self.role,
            loaded=self._model is not None,
            ready=self._model is not None and self._model_spec is not None,
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
        if self._model is None or self._model_spec is None:
            raise RuntimeError("worker_not_loaded")
        if request.model.catalog_id != self._model_spec.catalog_id:
            raise ValueError("worker_loaded_model_mismatch")

        audio_value = request.input_paths.get("audio")
        if not audio_value:
            raise ValueError("asr_audio_input_required")
        audio_path = Path(audio_value)
        if not audio_path.is_file():
            raise FileNotFoundError(f"asr_audio_input_not_found: {audio_path}")
        output_dir = Path(request.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if request.request_id in self._cancelled:
            self._cancelled.discard(request.request_id)
            raise asyncio.CancelledError
        if progress:
            await progress(
                WorkerProgressEvent(
                    request_id=request.request_id,
                    worker=self.role,
                    progress=0.1,
                    message="回答音声を確認しています",
                )
            )

        started = time.perf_counter()
        peak_vram_mb = self._gpu_memory_mb()
        transcription = await asyncio.to_thread(
            self._transcribe,
            audio_path,
            request.metadata,
        )
        peak_vram_mb = max(peak_vram_mb, self._gpu_memory_mb())
        if request.request_id in self._cancelled:
            self._cancelled.discard(request.request_id)
            raise asyncio.CancelledError

        output_path = output_dir / "transcription.json"
        output_path.write_text(
            json.dumps(transcription, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        if progress:
            await progress(
                WorkerProgressEvent(
                    request_id=request.request_id,
                    worker=self.role,
                    progress=1.0,
                    message="文字起こしが完了しました",
                )
            )

        return WorkerResult(
            request_id=request.request_id,
            worker=self.role,
            backend=self._model_spec.backend,
            model_id=self._model_spec.model_id,
            model_revision=self._model_spec.model_revision,
            implemented=True,
            output_paths={"transcription": str(output_path)},
            metadata={
                "text": transcription["text"],
                "language": transcription["language"],
                "language_probability": transcription["language_probability"],
                "duration_seconds": transcription["duration_seconds"],
                "segment_count": len(transcription["segments"]),
            },
            metrics=WorkerMetrics(
                processing_time_ms=elapsed_ms,
                peak_vram_mb=peak_vram_mb,
                peak_cpu_memory_mb=self._peak_cpu_memory_mb(),
            ),
        )

    def _transcribe(self, audio_path: Path, metadata: dict[str, Any]) -> dict[str, Any]:
        assert self._model is not None
        language = str(metadata.get("language", "ja"))
        segments_iter, info = self._model.transcribe(
            str(audio_path),
            language=language,
            beam_size=max(1, int(metadata.get("beam_size", 5))),
            chunk_length=max(1, int(metadata.get("chunk_length", 15))),
            condition_on_previous_text=bool(
                metadata.get("condition_on_previous_text", False)
            ),
            vad_filter=bool(metadata.get("vad_filter", False)),
        )
        segments = list(segments_iter)
        text = "".join(str(segment.text).strip() for segment in segments).strip()
        return {
            "schema_version": SCHEMA_VERSION,
            "text": text,
            "language": getattr(info, "language", language),
            "language_probability": float(
                getattr(info, "language_probability", 0.0) or 0.0
            ),
            "duration_seconds": float(getattr(info, "duration", 0.0) or 0.0),
            "duration_after_vad_seconds": float(
                getattr(info, "duration_after_vad", 0.0) or 0.0
            ),
            "segments": [
                {
                    "id": int(getattr(segment, "id", index)),
                    "start": float(segment.start),
                    "end": float(segment.end),
                    "text": str(segment.text).strip(),
                    "avg_logprob": float(getattr(segment, "avg_logprob", 0.0) or 0.0),
                    "no_speech_prob": float(
                        getattr(segment, "no_speech_prob", 0.0) or 0.0
                    ),
                }
                for index, segment in enumerate(segments)
            ],
        }

    @staticmethod
    def _peak_cpu_memory_mb() -> int:
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) // 1024

    def _gpu_memory_mb(self) -> int:
        try:
            output = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,used_gpu_memory",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=3,
            )
        except (OSError, subprocess.SubprocessError):
            output = ""
        current_pid = str(os.getpid())
        values: list[int] = []
        for line in output.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) == 2 and parts[0] == current_pid and parts[1].isdigit():
                values.append(int(parts[1]))
        if values:
            return max(values)
        return max(0, self._total_gpu_memory_mb() - self._gpu_baseline_mb)

    @staticmethod
    def _total_gpu_memory_mb() -> int:
        try:
            output = subprocess.check_output(
                ["nvidia-smi"],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=3,
            )
        except (OSError, subprocess.SubprocessError):
            return 0
        import re

        values = [
            int(match.group(1))
            for match in re.finditer(r"(\d+)MiB\s*/\s*(\d+)MiB", output)
            if int(match.group(2)) >= 1024
        ]
        return max(values, default=0)

    async def cancel(self, request_id: str) -> None:
        self._cancelled.add(request_id)

    async def unload(self) -> None:
        self._model = None
        self._model_spec = None
        self._cancelled.clear()
        self._gpu_baseline_mb = 0
        await asyncio.to_thread(gc.collect)


def create_worker(role: WorkerRole) -> WorkerAdapter:
    return FasterWhisperAsrAdapter(role)
