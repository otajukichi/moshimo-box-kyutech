from __future__ import annotations

import argparse
import asyncio
import json
import platform
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.schemas import WorkerModelSpec, WorkerRequest, WorkerRole
from backend.app.workers.adapters.faster_whisper_asr import create_worker

MODEL_ID = "kotoba-tech/kotoba-whisper-v2.0-faster"
MODEL_REVISION = "f44edd35eaeb2274e85ac7b31fb2c6f59ff1c4bc"


def command_output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            command,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def audio_duration(path: Path) -> float:
    value = command_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
    )
    if value and value != "N/A":
        try:
            return float(value)
        except ValueError:
            pass

    import av

    duration = 0.0
    with av.open(str(path)) as container:
        for frame in container.decode(audio=0):
            sample_rate = int(frame.sample_rate or 0)
            if sample_rate <= 0:
                continue
            start = float(frame.time or 0.0)
            duration = max(duration, start + frame.samples / sample_rate)
    return duration


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


async def benchmark(args: argparse.Namespace) -> dict[str, object]:
    audio_path = args.audio.resolve()
    model_path = args.model_path.resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(audio_path)
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)

    spec = WorkerModelSpec(
        worker=WorkerRole.STREAMING_ASR,
        backend="faster-whisper",
        catalog_id="kotoba-whisper-v2-faster-fp16",
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        dtype=args.compute_type,
        quantization="none",
        device="cuda:0",
        adapter_entrypoint=(
            "backend.app.workers.adapters.faster_whisper_asr:create_worker"
        ),
        model_path=str(model_path),
        timeout_seconds=args.timeout,
    )
    duration_seconds = audio_duration(audio_path)
    worker = create_worker(WorkerRole.STREAMING_ASR)
    load_started = time.perf_counter()
    health = await worker.load(spec)
    load_time_ms = int((time.perf_counter() - load_started) * 1000)
    if not health.ready:
        raise RuntimeError("ASR worker did not become ready")

    runs: list[dict[str, object]] = []
    errors: list[str] = []
    with tempfile.TemporaryDirectory(prefix="moshimo-asr-benchmark-") as temporary:
        for run_index in range(args.runs):
            request = WorkerRequest(
                worker=WorkerRole.STREAMING_ASR,
                session_id="benchmark",
                model=spec,
                deadline_seconds=args.timeout,
                input_paths={"audio": str(audio_path)},
                output_dir=str(Path(temporary) / f"run-{run_index + 1}"),
                metadata={
                    "language": "ja",
                    "beam_size": args.beam_size,
                    "chunk_length": 15,
                    "condition_on_previous_text": False,
                    "vad_filter": False,
                },
            )
            try:
                result = await worker.run(request)
                processing_seconds = result.metrics.processing_time_ms / 1000
                runs.append(
                    {
                        "run": run_index + 1,
                        "processing_time_ms": result.metrics.processing_time_ms,
                        "real_time_factor": (
                            processing_seconds / duration_seconds
                            if duration_seconds > 0
                            else None
                        ),
                        "text_char_count": len(str(result.metadata.get("text", ""))),
                        "segment_count": result.metadata.get("segment_count", 0),
                        "language": result.metadata.get("language"),
                        "language_probability": result.metadata.get(
                            "language_probability", 0
                        ),
                        "peak_vram_mb": result.metrics.peak_vram_mb,
                        "peak_cpu_memory_mb": result.metrics.peak_cpu_memory_mb,
                    }
                )
            except Exception as exc:
                errors.append(type(exc).__name__)

    unload_started = time.perf_counter()
    await worker.unload()
    unload_time_ms = int((time.perf_counter() - unload_started) * 1000)

    import ctranslate2
    import faster_whisper

    return {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "model": {
            "catalog_id": spec.catalog_id,
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "model_path": str(model_path),
            "download_size_bytes": directory_size(model_path),
            "dtype": args.compute_type,
            "quantization": "none",
            "device": "cuda:0",
        },
        "license_review": {
            "faster_whisper_code": "MIT",
            "converted_weight_repository": "MIT",
            "source_model_repository": "Apache-2.0",
        },
        "input": {
            "filename": audio_path.name,
            "size_bytes": audio_path.stat().st_size,
            "duration_seconds": duration_seconds,
            "raw_transcript_stored": False,
        },
        "metrics": {
            "load_time_ms": load_time_ms,
            "unload_time_ms": unload_time_ms,
            "successful_runs": len(runs),
            "error_count": len(errors),
            "errors": errors,
            "runs": runs,
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "faster_whisper": faster_whisper.__version__,
            "ctranslate2": ctranslate2.__version__,
            "gpu": command_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total",
                    "--format=csv,noheader",
                ]
            ),
            "command": " ".join(sys.argv),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=ROOT_DIR / "models/asr/kotoba-whisper-v2.0-faster",
    )
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be at least 1")

    result = asyncio.run(benchmark(args))
    output = args.output or (
        ROOT_DIR
        / "benchmarks/results"
        / f"asr-kotoba-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
