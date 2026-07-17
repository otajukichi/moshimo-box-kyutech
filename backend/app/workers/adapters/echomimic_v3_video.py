from __future__ import annotations

import asyncio
import json
import os
import resource
import subprocess
import sys
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


class EchoMimicV3VideoAdapter(WorkerAdapter):
    role = WorkerRole.VIDEO_GENERATION

    def __init__(self, role: WorkerRole) -> None:
        if role != self.role:
            raise ValueError(f"unsupported_echomimic_v3_role: {role.value}")
        self._model_spec: WorkerModelSpec | None = None
        self._cancelled: set[str] = set()
        self._process: asyncio.subprocess.Process | None = None

    async def load(self, model: WorkerModelSpec) -> WorkerHealth:
        if model.schema_version != SCHEMA_VERSION or model.worker != self.role:
            raise ValueError("worker_model_schema_mismatch")
        if model.backend == "musetalk-1.5":
            self._validate_musetalk(model)
            self._model_spec = model
            return await self.healthcheck()

        required = {
            "source_path": "infer_flash.py",
            "base_model_path": "config.json",
            "wav2vec_model_path": "config.json",
        }
        runner_path = Path(__file__).resolve().parents[4] / (
            "workers/runners/echomimic_v3_flash.py"
        )
        if not runner_path.is_file():
            raise FileNotFoundError(f"echomimic_v3_runner_missing: {runner_path}")
        missing: list[str] = []
        for key, marker in required.items():
            value = model.parameters.get(key)
            if not value or not (Path(str(value)) / marker).is_file():
                missing.append(f"{key}/{marker}")
        transformer_value = model.parameters.get("transformer_path")
        config_value = model.parameters.get("config_path")
        if not transformer_value or not Path(str(transformer_value)).is_file():
            missing.append("transformer_path")
        if not config_value or not Path(str(config_value)).is_file():
            missing.append("config_path")
        if missing:
            raise FileNotFoundError(f"echomimic_v3_files_missing: {missing}")
        self._model_spec = model
        return await self.healthcheck()

    @staticmethod
    def _validate_musetalk(model: WorkerModelSpec) -> None:
        source_value = model.parameters.get("source_path")
        source = Path(str(source_value)) if source_value else Path()
        required = [
            source / "scripts" / "inference.py",
            source / "models" / "sd-vae" / "config.json",
            source / "models" / "sd-vae" / "diffusion_pytorch_model.bin",
            source / "models" / "face-parse-bisent" / "79999_iter.pth",
            source / "models" / "face-parse-bisent" / "resnet18-5c106cde.pth",
            source / "models" / "dwpose" / "dw-ll_ucoco_384.pth",
        ]
        unet_value = model.parameters.get("unet_model_path")
        config_value = model.parameters.get("unet_config_path")
        whisper_value = model.parameters.get("whisper_dir")
        if unet_value:
            required.append(Path(str(unet_value)))
        if config_value:
            required.append(Path(str(config_value)))
        if whisper_value:
            required.extend(
                [
                    Path(str(whisper_value)) / "config.json",
                    Path(str(whisper_value)) / "pytorch_model.bin",
                    Path(str(whisper_value)) / "preprocessor_config.json",
                ]
            )
        missing = [str(path) for path in required if not path.is_file()]
        if not source_value or not unet_value or not config_value or not whisper_value:
            missing.append("required_model_parameter")
        if missing:
            raise FileNotFoundError(f"musetalk_files_missing: {missing}")

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
        image_value = request.input_paths.get("future_image")
        audio_value = request.input_paths.get("generated_audio")
        script_value = request.input_paths.get("script_design")
        engine = (
            "musetalk" if request.model.backend == "musetalk-1.5" else "echomimic_v3"
        )
        if not image_value or not Path(image_value).is_file():
            raise FileNotFoundError(f"{engine}_future_image_not_found")
        if not audio_value or not Path(audio_value).is_file():
            raise FileNotFoundError(f"{engine}_generated_audio_not_found")
        if not script_value or not Path(script_value).is_file():
            raise FileNotFoundError(f"{engine}_script_design_not_found")
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
                    progress=0.05,
                    message="未来の本人を動かすモデルを準備しています",
                )
            )

        started = time.perf_counter()
        command, expected_output = self._build_command(
            request,
            Path(image_value),
            Path(audio_value),
            script,
            output_dir,
        )
        source_path = Path(str(request.model.parameters["source_path"]))
        environment = os.environ.copy()
        existing_python_path = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            f"{source_path}:{existing_python_path}"
            if existing_python_path
            else str(source_path)
        )
        log_path = output_dir / f"{engine.replace('_', '-')}.log"
        peak_vram_mb = 0
        with log_path.open("wb") as log_handle:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=source_path,
                env=environment,
                stdout=log_handle,
                stderr=asyncio.subprocess.STDOUT,
            )
            self._process = process
            wait_task = asyncio.create_task(process.wait())
            try:
                while not wait_task.done():
                    if request.request_id in self._cancelled:
                        self._cancelled.discard(request.request_id)
                        await self._terminate_process()
                        raise asyncio.CancelledError
                    completed, _ = await asyncio.wait({wait_task}, timeout=1.0)
                    if completed:
                        break
                    peak_vram_mb = max(
                        peak_vram_mb,
                        self._gpu_memory_for_pid(process.pid),
                    )
                    if progress:
                        elapsed = time.perf_counter() - started
                        fraction = min(
                            0.9,
                            0.1
                            + elapsed / max(60.0, request.deadline_seconds) * 0.75,
                        )
                        await progress(
                            WorkerProgressEvent(
                                request_id=request.request_id,
                                worker=self.role,
                                progress=fraction,
                                message="音声に合わせてメッセージ映像を生成しています",
                            )
                        )
                return_code = await wait_task
            finally:
                if not wait_task.done():
                    wait_task.cancel()
                    await asyncio.gather(wait_task, return_exceptions=True)
                self._process = None
        output_text = log_path.read_text(encoding="utf-8", errors="replace")
        if return_code != 0:
            raise RuntimeError(f"{engine}_failed: {output_text[-2000:]}")
        if not expected_output.is_file():
            raise FileNotFoundError(f"{engine}_output_missing: {expected_output}")

        ffprobe = str(request.model.parameters.get("ffprobe_path", "/usr/bin/ffprobe"))
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "engine": request.model.backend,
            "video_path": str(expected_output),
            "duration_seconds": self._duration_seconds(expected_output, ffprobe),
            "prompt": script.video_prompt,
            "sample_size": request.model.parameters.get("sample_size", [512, 512]),
            "num_inference_steps": int(
                request.model.parameters.get("num_inference_steps", 8)
            ),
            "log_path": str(log_path),
        }
        metadata_path = output_dir / "future-video.json"
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if progress:
            await progress(
                WorkerProgressEvent(
                    request_id=request.request_id,
                    worker=self.role,
                    progress=1.0,
                    message="メッセージ映像が完成しました",
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
                "generated_video": str(expected_output),
                "generated_video_metadata": str(metadata_path),
            },
            metadata=metadata,
            metrics=WorkerMetrics(
                processing_time_ms=int((time.perf_counter() - started) * 1000),
                peak_vram_mb=peak_vram_mb,
                peak_cpu_memory_mb=self._peak_cpu_memory_mb(),
            ),
        )

    def _build_command(
        self,
        request: WorkerRequest,
        image_path: Path,
        audio_path: Path,
        script: ScriptDesignOutput,
        output_dir: Path,
    ) -> tuple[list[str], Path]:
        if request.model.backend == "musetalk-1.5":
            return self._build_musetalk_command(
                request,
                image_path,
                audio_path,
                output_dir,
            )

        parameters = request.model.parameters
        sample_size = parameters.get("sample_size", [512, 512])
        if not isinstance(sample_size, list) or len(sample_size) != 2:
            raise ValueError("echomimic_sample_size_must_have_two_values")
        prompt = (
            f"{script.video_prompt} The same fictional future scene is maintained. "
            "The person speaks directly to the camera with natural lip motion, subtle "
            "blinking, small head movement, and restrained hand gestures."
        )
        runner_path = Path(__file__).resolve().parents[4] / (
            "workers/runners/echomimic_v3_flash.py"
        )
        expected_output = output_dir / f"{image_path.stem}_output.mp4"
        command = [
            sys.executable,
            str(runner_path),
            "--image-path",
            str(image_path),
            "--audio-path",
            str(audio_path),
            "--output-path",
            str(expected_output),
            "--prompt",
            prompt,
            "--negative-prompt",
            script.negative_prompt,
            "--config-path",
            str(parameters["config_path"]),
            "--base-model-path",
            str(parameters["base_model_path"]),
            "--transformer-path",
            str(parameters["transformer_path"]),
            "--wav2vec-model-path",
            str(parameters["wav2vec_model_path"]),
            "--target-seconds",
            str(float(request.metadata.get("target_video_seconds", 20))),
            "--fps",
            str(int(parameters.get("fps", 25))),
            "--sample-size",
            str(int(sample_size[0])),
            str(int(sample_size[1])),
            "--chunk-frames",
            str(int(parameters.get("chunk_frames", 113))),
            "--overlap-frames",
            str(int(parameters.get("overlap_frames", 8))),
            "--num-inference-steps",
            str(int(parameters.get("num_inference_steps", 8))),
            "--sampler-name",
            str(parameters.get("sampler_name", "Flow_Unipc")),
            "--guidance-scale",
            str(float(parameters.get("guidance_scale", 5.0))),
            "--audio-guidance-scale",
            str(float(parameters.get("audio_guidance_scale", 2.0))),
            "--audio-scale",
            str(float(parameters.get("audio_scale", 1.0))),
            "--seed",
            str(int(parameters.get("seed", 43))),
            "--teacache-threshold",
            str(float(parameters.get("teacache_threshold", 0.1))),
            "--num-skip-start-steps",
            str(int(parameters.get("num_skip_start_steps", 5))),
            "--shift",
            str(float(parameters.get("shift", 5.0))),
            "--weight-dtype",
            str(parameters.get("weight_dtype", "bfloat16")),
            "--riflex-k",
            str(int(parameters.get("riflex_k", 6))),
        ]
        if bool(parameters.get("enable_riflex", True)):
            command.append("--enable-riflex")
        return command, expected_output

    @staticmethod
    def _build_musetalk_command(
        request: WorkerRequest,
        image_path: Path,
        audio_path: Path,
        output_dir: Path,
    ) -> tuple[list[str], Path]:
        import yaml

        parameters = request.model.parameters
        result_dir = output_dir / "musetalk-results"
        result_name = "future-video.mp4"
        expected_output = result_dir / "v15" / result_name
        inference_config = output_dir / "musetalk-inference.yaml"
        inference_config.write_text(
            yaml.safe_dump(
                {
                    "future_message": {
                        "video_path": str(image_path),
                        "audio_path": str(audio_path),
                        "result_name": result_name,
                    }
                },
                allow_unicode=True,
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        command = [
            sys.executable,
            "-m",
            "scripts.inference",
            "--inference_config",
            str(inference_config),
            "--result_dir",
            str(result_dir),
            "--unet_model_path",
            str(parameters["unet_model_path"]),
            "--unet_config",
            str(parameters["unet_config_path"]),
            "--whisper_dir",
            str(parameters["whisper_dir"]),
            "--version",
            "v15",
            "--ffmpeg_path",
            str(parameters.get("ffmpeg_path", "/usr/bin")),
            "--fps",
            str(int(parameters.get("fps", 25))),
            "--batch_size",
            str(int(parameters.get("batch_size", 16))),
            "--bbox_shift",
            str(int(parameters.get("bbox_shift", 0))),
            "--extra_margin",
            str(int(parameters.get("extra_margin", 10))),
            "--parsing_mode",
            str(parameters.get("parsing_mode", "jaw")),
            "--left_cheek_width",
            str(int(parameters.get("left_cheek_width", 90))),
            "--right_cheek_width",
            str(int(parameters.get("right_cheek_width", 90))),
        ]
        if bool(parameters.get("use_float16", True)):
            command.append("--use_float16")
        return command, expected_output

    async def cancel(self, request_id: str) -> None:
        self._cancelled.add(request_id)
        await self._terminate_process()

    async def _terminate_process(self) -> None:
        process = self._process
        if process is None or process.returncode is not None:
            return
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=5)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
        self._process = None

    async def unload(self) -> None:
        await self._terminate_process()
        self._model_spec = None
        self._cancelled.clear()

    @staticmethod
    def _duration_seconds(path: Path, ffprobe: str) -> float:
        completed = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            return 0.0
        try:
            return max(0.0, float(completed.stdout.strip()))
        except ValueError:
            return 0.0

    @staticmethod
    def _gpu_memory_for_pid(pid: int) -> int:
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
            return 0
        for line in output.splitlines():
            process_id, separator, memory = line.partition(",")
            if separator and process_id.strip() == str(pid):
                value = memory.strip()
                if value.isdigit():
                    return int(value)
        return 0

    @staticmethod
    def _peak_cpu_memory_mb() -> int:
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) // 1024


def create_worker(role: WorkerRole) -> WorkerAdapter:
    return EchoMimicV3VideoAdapter(role)
