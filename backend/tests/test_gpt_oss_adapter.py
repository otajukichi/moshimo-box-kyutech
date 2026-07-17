from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from backend.app.schemas import WorkerRole
from backend.app.workers.adapters.gpt_oss_vllm import (
    GptOssGenerationLlmAdapter,
    GptOssInterviewLlmAdapter,
    GptOssVllmRuntime,
    create_worker,
)


class FakeHarmonyMessage:
    def __init__(self, channel: str, content: object) -> None:
        self.channel = channel
        self.content = content

    def to_dict(self) -> dict[str, object]:
        return {"channel": self.channel, "content": self.content}


def test_harmony_parser_returns_only_final_channel() -> None:
    messages = [
        FakeHarmonyMessage("analysis", "internal reasoning"),
        FakeHarmonyMessage("final", [{"type": "text", "text": "公開する返答"}]),
    ]

    assert GptOssVllmRuntime.extract_final_message(messages) == "公開する返答"


def test_harmony_parser_never_exposes_analysis_fallback() -> None:
    with pytest.raises(RuntimeError, match="gpt_oss_final_message_missing"):
        GptOssVllmRuntime.extract_final_message(
            [FakeHarmonyMessage("analysis", "internal reasoning")]
        )


def test_factory_supports_interview_design_and_safety_roles() -> None:
    assert isinstance(
        create_worker(WorkerRole.INTERVIEW_LLM),
        GptOssInterviewLlmAdapter,
    )
    assert isinstance(
        create_worker(WorkerRole.SCRIPT_DESIGN_LLM),
        GptOssGenerationLlmAdapter,
    )
    assert isinstance(
        create_worker(WorkerRole.SCRIPT_SAFETY_REVIEW),
        GptOssGenerationLlmAdapter,
    )


def test_catalog_keeps_gpt_oss_text_only_and_in_a_separate_environment() -> None:
    root = Path(__file__).resolve().parents[2]
    catalog = yaml.safe_load(
        (root / "config" / "model-catalog.yaml").read_text(encoding="utf-8")
    )
    entry = next(
        item
        for item in catalog["models"]
        if item["id"] == "gpt-oss-20b-mxfp4-vllm"
    )

    assert entry["environment"] == "gpt-oss"
    assert entry["quantization"] == "mxfp4"
    assert entry["parameters"]["multimodal"] is False
    assert entry["parameters"]["gpu_memory_utilization"] < 0.8
    assert entry["parameters"]["planning_reasoning_effort"] == "low"
    assert entry["parameters"]["planning_min_new_tokens"] >= 512
