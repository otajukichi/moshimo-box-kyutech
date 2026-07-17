from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from backend.app.schemas import WorkerModelSpec, WorkerRequest, WorkerRole
from backend.app.workers.adapters.faster_whisper_asr import (
    FasterWhisperAsrAdapter,
    create_worker,
)


class FakeWhisperModel:
    def transcribe(self, audio_path: str, **kwargs):
        assert audio_path.endswith("answer.webm")
        assert kwargs["language"] == "ja"
        assert kwargs["condition_on_previous_text"] is False
        segments = [
            SimpleNamespace(
                id=0,
                start=0.0,
                end=1.2,
                text="未来の技術に興味があります。",
                avg_logprob=-0.1,
                no_speech_prob=0.01,
            )
        ]
        info = SimpleNamespace(
            language="ja",
            language_probability=0.99,
            duration=1.2,
            duration_after_vad=1.1,
        )
        return iter(segments), info


def test_faster_whisper_adapter_contract(tmp_path, monkeypatch) -> None:
    async def scenario() -> None:
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        audio_path = tmp_path / "answer.webm"
        audio_path.write_bytes(b"fake-audio")
        output_dir = tmp_path / "output"
        spec = WorkerModelSpec(
            worker=WorkerRole.STREAMING_ASR,
            backend="faster-whisper",
            catalog_id="test-asr",
            model_id="test/model",
            model_revision="revision",
            dtype="float16",
            quantization="none",
            device="cuda:0",
            adapter_entrypoint=(
                "backend.app.workers.adapters.faster_whisper_asr:create_worker"
            ),
            model_path=str(model_dir),
            timeout_seconds=60,
        )
        worker = FasterWhisperAsrAdapter(WorkerRole.STREAMING_ASR)
        monkeypatch.setattr(
            worker,
            "_create_model",
            lambda source, model: FakeWhisperModel(),
        )
        monkeypatch.setattr(worker, "_total_gpu_memory_mb", lambda: 100)
        monkeypatch.setattr(worker, "_gpu_memory_mb", lambda: 512)

        health = await worker.load(spec)
        assert health.ready is True
        request = WorkerRequest(
            worker=WorkerRole.STREAMING_ASR,
            session_id="test-session",
            model=spec,
            deadline_seconds=60,
            input_paths={"audio": str(audio_path)},
            output_dir=str(output_dir),
            metadata={"language": "ja"},
        )
        result = await worker.run(request)

        assert result.implemented is True
        assert result.metadata["text"] == "未来の技術に興味があります。"
        assert result.metrics.peak_vram_mb == 512
        payload = json.loads((output_dir / "transcription.json").read_text())
        assert payload["language"] == "ja"
        assert payload["segments"][0]["start"] == 0.0

        await worker.unload()
        assert (await worker.healthcheck()).loaded is False

    asyncio.run(scenario())


def test_faster_whisper_factory_rejects_non_asr_role() -> None:
    with pytest.raises(ValueError, match="unsupported_faster_whisper_role"):
        create_worker(WorkerRole.IMAGE_GENERATION)
