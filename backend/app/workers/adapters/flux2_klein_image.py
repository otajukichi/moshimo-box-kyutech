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


class Flux2KleinImageAdapter(WorkerAdapter):
    role = WorkerRole.IMAGE_GENERATION

    def __init__(self, role: WorkerRole) -> None:
        if role != self.role:
            raise ValueError(f"unsupported_flux2_klein_role: {role.value}")
        self._model_spec: WorkerModelSpec | None = None
        self._pipeline: Any | None = None
        self._torch: Any | None = None
        self._cancelled: set[str] = set()

    async def load(self, model: WorkerModelSpec) -> WorkerHealth:
        if model.schema_version != SCHEMA_VERSION or model.worker != self.role:
            raise ValueError("worker_model_schema_mismatch")
        if not model.model_path or not Path(model.model_path).is_dir():
            raise FileNotFoundError(f"flux2_model_path_not_found: {model.model_path}")
        self._pipeline, self._torch = await asyncio.to_thread(
            self._create_pipeline,
            Path(model.model_path),
            model,
        )
        self._model_spec = model
        return await self.healthcheck()

    @staticmethod
    def _create_pipeline(model_path: Path, model: WorkerModelSpec) -> tuple[Any, Any]:
        import torch
        from diffusers import Flux2KleinPipeline

        if model.device.lower().startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("flux2_cuda_unavailable")
        dtype = torch.bfloat16 if model.dtype.lower() == "bfloat16" else torch.float16
        pipeline = Flux2KleinPipeline.from_pretrained(
            str(model_path),
            torch_dtype=dtype,
            local_files_only=True,
        )
        if bool(model.parameters.get("cpu_offload", False)):
            pipeline.enable_model_cpu_offload()
        else:
            pipeline.to(model.device)
        pipeline.set_progress_bar_config(disable=True)
        return pipeline, torch

    async def healthcheck(self) -> WorkerHealth:
        return WorkerHealth(
            worker=self.role,
            loaded=self._pipeline is not None,
            ready=self._pipeline is not None and self._model_spec is not None,
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
        if self._pipeline is None or self._torch is None or self._model_spec is None:
            raise RuntimeError("worker_not_loaded")
        if request.model.catalog_id != self._model_spec.catalog_id:
            raise ValueError("worker_loaded_model_mismatch")
        reference_value = request.input_paths.get("reference_image")
        script_value = request.input_paths.get("script_design")
        if not reference_value or not Path(reference_value).is_file():
            raise FileNotFoundError("flux2_reference_image_not_found")
        if not script_value or not Path(script_value).is_file():
            raise FileNotFoundError("flux2_script_design_not_found")
        script = ScriptDesignOutput.model_validate(
            json.loads(Path(script_value).read_text(encoding="utf-8"))
        )
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
                    message="未来の本人の姿を描いています",
                )
            )
        started = time.perf_counter()
        if self._model_spec.device.lower().startswith("cuda"):
            self._torch.cuda.reset_peak_memory_stats()
        image, prompt = await asyncio.to_thread(
            self._generate,
            Path(reference_value),
            script,
            request,
        )
        if request.request_id in self._cancelled:
            self._cancelled.discard(request.request_id)
            raise asyncio.CancelledError

        output_path = output_dir / "future-person.png"
        image.save(output_path, format="PNG")
        metadata_path = output_dir / "future-person.json"
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "image_path": str(output_path),
            "reference_image": reference_value,
            "width": image.width,
            "height": image.height,
            "prompt": prompt,
            "seed": int(request.model.parameters.get("seed", 42)),
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
                    message="未来の本人画像が完成しました",
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
                "future_image": str(output_path),
                "future_image_metadata": str(metadata_path),
            },
            metadata=metadata,
            metrics=WorkerMetrics(
                processing_time_ms=int((time.perf_counter() - started) * 1000),
                peak_vram_mb=peak_vram_mb,
                peak_cpu_memory_mb=self._peak_cpu_memory_mb(),
            ),
        )

    def _generate(
        self,
        reference_path: Path,
        script: ScriptDesignOutput,
        request: WorkerRequest,
    ) -> tuple[Any, str]:
        assert self._pipeline is not None
        assert self._torch is not None
        from PIL import Image

        reference = Image.open(reference_path).convert("RGB")
        prompt = (
            f"Edit the reference photo into this fictional future scene. {script.image_prompt} "
            "Keep the exact same person's recognizable facial identity, facial proportions, "
            "skin tone, and key facial features. Use a centered, front-facing chest-up portrait. "
            f"Clothing: {script.clothing}. Background: {script.background}. "
            f"Emotion: {script.emotion}. Camera: {script.camera}. "
            f"Avoid: {script.negative_prompt}. No text, logo, caption, or watermark."
        )
        device = "cuda" if self._model_spec and self._model_spec.device.startswith("cuda") else "cpu"
        generator = self._torch.Generator(device=device).manual_seed(
            int(request.model.parameters.get("seed", 42))
        )
        result = self._pipeline(
            image=reference,
            prompt=prompt,
            height=int(request.model.parameters.get("height", 768)),
            width=int(request.model.parameters.get("width", 768)),
            guidance_scale=float(request.model.parameters.get("guidance_scale", 1.0)),
            num_inference_steps=int(request.model.parameters.get("num_inference_steps", 4)),
            generator=generator,
            max_sequence_length=int(request.model.parameters.get("max_sequence_length", 512)),
        )
        return result.images[0], prompt

    @staticmethod
    def _peak_cpu_memory_mb() -> int:
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) // 1024

    async def cancel(self, request_id: str) -> None:
        self._cancelled.add(request_id)
        if self._pipeline is not None and hasattr(self._pipeline, "_interrupt"):
            self._pipeline._interrupt = True

    async def unload(self) -> None:
        self._pipeline = None
        torch = self._torch
        self._torch = None
        self._model_spec = None
        self._cancelled.clear()
        await asyncio.to_thread(gc.collect)
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def create_worker(role: WorkerRole) -> WorkerAdapter:
    return Flux2KleinImageAdapter(role)
