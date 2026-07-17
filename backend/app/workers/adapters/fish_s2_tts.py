from __future__ import annotations

import asyncio
import gc
import json
import resource
import time
from pathlib import Path
from typing import Any

from ...contracts import ScriptDesignOutput
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


class FishS2TtsAdapter(WorkerAdapter):
    role = WorkerRole.VOICE_CLONE_TTS

    def __init__(self, role: WorkerRole) -> None:
        if role != self.role:
            raise ValueError(f"unsupported_fish_s2_role: {role.value}")
        self._model_spec: WorkerModelSpec | None = None
        self._engine: Any | None = None
        self._torch: Any | None = None
        self._cancelled: set[str] = set()

    async def load(self, model: WorkerModelSpec) -> WorkerHealth:
        if model.schema_version != SCHEMA_VERSION or model.worker != self.role:
            raise ValueError("worker_model_schema_mismatch")
        if not model.model_path:
            raise ValueError("fish_s2_model_path_required")
        checkpoint_path = Path(model.model_path)
        required = [
            checkpoint_path / "config.json",
            checkpoint_path / "codec.pth",
            checkpoint_path / "model.safetensors.index.json",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"fish_s2_model_files_missing: {missing}")
        self._engine, self._torch = await asyncio.to_thread(
            self._create_engine,
            checkpoint_path,
            model,
        )
        self._model_spec = model
        return await self.healthcheck()

    @staticmethod
    def _create_engine(checkpoint_path: Path, model: WorkerModelSpec) -> tuple[Any, Any]:
        import torch
        from fish_speech.inference_engine import TTSInferenceEngine
        from fish_speech.models.dac.inference import load_model as load_decoder_model
        from fish_speech.models.text2semantic.inference import launch_thread_safe_queue

        if model.device.lower().startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("fish_s2_cuda_unavailable")
        precision = torch.bfloat16 if model.dtype.lower() == "bfloat16" else torch.float16
        compile_model = bool(model.parameters.get("compile", False))
        llama_queue = launch_thread_safe_queue(
            checkpoint_path=str(checkpoint_path),
            device=model.device,
            precision=precision,
            compile=compile_model,
        )
        decoder_model = load_decoder_model(
            config_name="modded_dac_vq",
            checkpoint_path=str(checkpoint_path / "codec.pth"),
            device=model.device,
        )
        engine = TTSInferenceEngine(
            llama_queue=llama_queue,
            decoder_model=decoder_model,
            precision=precision,
            compile=compile_model,
        )
        return engine, torch

    async def healthcheck(self) -> WorkerHealth:
        return WorkerHealth(
            worker=self.role,
            loaded=self._engine is not None,
            ready=self._engine is not None and self._model_spec is not None,
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
        if self._engine is None or self._model_spec is None or self._torch is None:
            raise RuntimeError("worker_not_loaded")
        if request.model.catalog_id != self._model_spec.catalog_id:
            raise ValueError("worker_loaded_model_mismatch")
        if request.request_id in self._cancelled:
            self._cancelled.discard(request.request_id)
            raise asyncio.CancelledError

        script = ScriptDesignOutput.model_validate(
            self._read_json(request.input_paths.get("script_design"))
        )
        reference = self._read_json(request.input_paths.get("voice_reference"))
        reference_audio = Path(str(reference.get("audio_path", "")))
        prompt_text = str(reference.get("prompt_text", "")).strip()
        if not reference_audio.is_file() or not prompt_text:
            raise ValueError("fish_s2_valid_voice_reference_required")
        output_dir = Path(request.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if progress:
            await progress(
                WorkerProgressEvent(
                    request_id=request.request_id,
                    worker=self.role,
                    progress=0.1,
                    message="本人の声の特徴を読み取っています",
                )
            )
        started = time.perf_counter()
        if self._model_spec.device.lower().startswith("cuda"):
            self._torch.cuda.reset_peak_memory_stats()
        sample_rate, audio = await asyncio.to_thread(
            self._synthesize,
            script.narration_script,
            reference_audio,
            prompt_text,
            request,
        )
        if request.request_id in self._cancelled:
            self._cancelled.discard(request.request_id)
            raise asyncio.CancelledError

        import soundfile as sf

        output_path = output_dir / "future-voice.wav"
        sf.write(output_path, audio, sample_rate, subtype="PCM_16")
        duration_seconds = float(len(audio) / sample_rate) if sample_rate else 0.0
        metadata_path = output_dir / "future-voice.json"
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "audio_path": str(output_path),
            "sample_rate": sample_rate,
            "duration_seconds": duration_seconds,
            "reference_duration_seconds": reference.get("duration_seconds"),
            "voice_instruction": script.voice_instruction,
        }
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        peak_vram_mb = 0
        if self._model_spec.device.lower().startswith("cuda"):
            peak_vram_mb = int(self._torch.cuda.max_memory_allocated() / 1024 / 1024)
        if progress:
            await progress(
                WorkerProgressEvent(
                    request_id=request.request_id,
                    worker=self.role,
                    progress=1.0,
                    message="未来の本人の音声が完成しました",
                )
            )
        return WorkerResult(
            request_id=request.request_id,
            worker=self.role,
            backend=self._model_spec.backend,
            model_id=self._model_spec.model_id,
            model_revision=self._model_spec.model_revision,
            implemented=True,
            output_paths={
                "generated_audio": str(output_path),
                "generated_audio_metadata": str(metadata_path),
            },
            metadata=metadata,
            metrics=WorkerMetrics(
                processing_time_ms=int((time.perf_counter() - started) * 1000),
                peak_vram_mb=peak_vram_mb,
                peak_cpu_memory_mb=self._peak_cpu_memory_mb(),
            ),
        )

    def _synthesize(
        self,
        text: str,
        reference_audio: Path,
        prompt_text: str,
        request: WorkerRequest,
    ) -> tuple[int, Any]:
        assert self._engine is not None
        from fish_speech.utils.schema import ServeReferenceAudio, ServeTTSRequest

        reference = ServeReferenceAudio(
            audio=reference_audio.read_bytes(),
            text=prompt_text,
        )
        tts_request = ServeTTSRequest(
            text=text,
            references=[reference],
            max_new_tokens=int(request.model.parameters.get("max_new_tokens", 4096)),
            chunk_length=int(request.model.parameters.get("chunk_length", 300)),
            top_p=float(request.model.parameters.get("top_p", 0.8)),
            repetition_penalty=float(
                request.model.parameters.get("repetition_penalty", 1.1)
            ),
            temperature=float(request.model.parameters.get("temperature", 0.8)),
            seed=int(request.model.parameters.get("seed", 42)),
            use_memory_cache="off",
            streaming=False,
            format="wav",
        )
        final_audio: tuple[int, Any] | None = None
        for result in self._engine.inference(tts_request):
            if result.code == "error":
                raise RuntimeError(f"fish_s2_inference_failed: {result.error}")
            if result.code == "final" and result.audio is not None:
                final_audio = result.audio
        if final_audio is None:
            raise RuntimeError("fish_s2_no_audio_generated")
        return int(final_audio[0]), final_audio[1]

    @staticmethod
    def _read_json(path_value: str | None) -> Any:
        if not path_value:
            raise ValueError("fish_s2_input_path_required")
        path = Path(path_value)
        if not path.is_file():
            raise FileNotFoundError(path)
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _peak_cpu_memory_mb() -> int:
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) // 1024

    async def cancel(self, request_id: str) -> None:
        self._cancelled.add(request_id)

    async def unload(self) -> None:
        self._engine = None
        torch = self._torch
        self._torch = None
        self._model_spec = None
        self._cancelled.clear()
        await asyncio.to_thread(gc.collect)
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def create_worker(role: WorkerRole) -> WorkerAdapter:
    return FishS2TtsAdapter(role)
