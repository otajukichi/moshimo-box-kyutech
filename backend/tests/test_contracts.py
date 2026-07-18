from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from backend.app.schemas import Episode, EpisodeEffect, WorkerRequest, WorkerRole
from backend.app.schemas import InterviewState, TranscriptEntry
from backend.app.workers.adapters.transformers_generation_llm import (
    NARRATION_STYLE_GUIDANCE,
    SCRIPT_SYSTEM_PROMPT,
    TransformersGenerationLlmAdapter,
)
from backend.app.workers.adapters.transformers_interview_llm import (
    TransformersInterviewLlmAdapter,
)

from backend.app.contracts import (
    InterviewTurnInput,
    InterviewTurnOutput,
    ScriptDesignInput,
    ScriptDesignOutput,
    ScriptSafetyReviewOutput,
)


def test_llm_contracts_export_json_schema() -> None:
    for contract in (
        InterviewTurnInput,
        InterviewTurnOutput,
        ScriptDesignInput,
        ScriptDesignOutput,
        ScriptSafetyReviewOutput,
    ):
        schema = contract.model_json_schema()
        assert schema["type"] == "object"
        assert schema["properties"]["schema_version"]

    output_schema = ScriptDesignOutput.model_json_schema()
    required = set(output_schema["required"])
    assert {
        "future_world",
        "future_person",
        "narration_script",
        "shot_plan",
        "image_prompt",
        "video_prompt",
        "fallback_plan",
    } <= required
    assert output_schema["properties"]["shot_plan"]["maxItems"] == 1


def test_rejected_script_requires_safe_correction() -> None:
    with pytest.raises(
        ValueError,
        match="corrected_output is required",
    ):
        ScriptSafetyReviewOutput(approved=False, reasons=["unsafe"])


def test_script_design_is_assembled_from_plain_stage_answers(
    fast_config,
    tmp_path,
    monkeypatch,
) -> None:
    transcript_path = tmp_path / "transcript.json"
    summary_path = tmp_path / "summary.json"
    (tmp_path / "output").mkdir()
    transcript_path.write_text(
        json.dumps(
            [{"speaker": "visitor", "text": "宇宙と植物を育てることが好きです。"}],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps({"summary": "宇宙と栽培技術に興味がある。"}, ensure_ascii=False),
        encoding="utf-8",
    )
    episode = Episode(
        id="space-farm",
        name="宇宙農園",
        content="軌道上の農園で食料を育てる。",
        base_rarity="SR",
        weight=1,
    )
    effect = EpisodeEffect(
        id="straight",
        name="穏やかな未来",
        content="未来の本人が現在の自分へ語る。",
        weight=1,
    )
    role = WorkerRole.SCRIPT_DESIGN_LLM
    model = fast_config.catalog.spec(role, "foundation-stub")
    request = WorkerRequest(
        worker=role,
        session_id="stage-test",
        model=model,
        deadline_seconds=60,
        input_paths={
            "transcript": str(transcript_path),
            "interview_summary": str(summary_path),
        },
        output_dir=str(tmp_path / "output"),
        metadata={
            "episode": episode.model_dump(mode="json"),
            "effect": effect.model_dump(mode="json"),
            "final_rarity": "SR",
            "episode_mode": "formal",
            "target_video_seconds": 20,
            "remaining_time_seconds": 60,
            "person_information": {"interest": "宇宙と植物"},
            "capabilities": {},
            "prohibited_expressions": [],
        },
    )
    answers = iter(
        [
            "軌道都市では小さな農園が暮らしの中心となり、研究と食事が自然につながっている。",
            "未来の本人は植物の状態を観察しながら、仲間と新しい栽培方法を試している。",
            "今の好奇心が、遠い未来の日常を少しずつ面白くしている。",
            "未来からのメッセージです。" * 8,
            "明るい研究農園で本人が穏やかに未来を伝える、親しみやすいSF映像。",
            "清潔感のある未来素材の作業ジャケット。",
            "窓の外に地球が見える、植物に囲まれた軌道農園。",
            "穏やかな自信と、再会を喜ぶ温かな表情。",
            "Preserve the exact identity of the reference person in a chest-up portrait inside a bright orbital greenhouse.",
            "The same person speaks naturally to camera with subtle blinking and small gestures while the camera remains nearly fixed.",
            "本人らしい話し方を残し、少しうれしそうに、落ち着いて語る。",
        ]
    )
    calls: list[str] = []

    def fake_generate(system_prompt: str, user_prompt: str, *, max_new_tokens: int) -> str:
        calls.append(user_prompt)
        return next(answers)

    adapter = TransformersGenerationLlmAdapter(role)
    monkeypatch.setattr(adapter, "_generate_text", fake_generate)
    events = []

    async def progress(event) -> None:
        events.append(event)

    output_paths, metadata = asyncio.run(
        adapter._run_script_design(request, tmp_path / "output", progress)
    )
    output = ScriptDesignOutput.model_validate_json(
        Path(output_paths["script_design"]).read_text(encoding="utf-8")
    )

    assert len(calls) == 11
    assert "文学的な比喩" in calls[3]
    assert "具体的な名詞や行動" in calls[3]
    assert "at least one natural hand gesture" in calls[8]
    assert "avoid a frozen body or mouth-only animation" in calls[9]
    assert isinstance(output.future_world, str)
    assert 80 <= len(output.narration_script) <= 110
    assert "緩やかな" in output.camera
    assert "上半身" in output.shot_plan[0].action
    assert "frozen pose" in output.negative_prompt
    assert len(output.shot_plan) == 1
    assert metadata["staged_field_count"] == 11
    assert any(event.phase == "script.future_world" for event in events)
    assert any(event.phase == "output_validation" for event in events)


def test_narration_prompt_requires_plain_video_message_language() -> None:
    assert "スマートフォンで近況を話すような口調" in SCRIPT_SYSTEM_PROMPT
    assert "比喩、ポエム調" in SCRIPT_SYSTEM_PROMPT
    assert "具体的な名詞や行動を最低一つ" in NARRATION_STYLE_GUIDANCE
    assert "悪い例" in NARRATION_STYLE_GUIDANCE
    assert "良い例" in NARRATION_STYLE_GUIDANCE


def test_poetic_narration_is_marked_for_revision() -> None:
    narration = (
        "あの日の好奇心が未来への扉を開き、新しい景色が待っています。"
        "可能性を信じて、一歩ずつ進んでください。"
    )

    reasons = TransformersGenerationLlmAdapter._narration_revision_reasons(
        narration
    )

    assert "詩的な定型句を含む" in reasons


def test_narration_padding_does_not_add_old_poetic_phrases() -> None:
    narration = TransformersGenerationLlmAdapter._normalize_narration(
        "今は軌道農園で、仲間と野菜の育て方を試しています。"
    )

    assert 80 <= len(narration) <= 110
    assert "未来の可能性" not in narration
    assert "どんな景色" not in narration
    assert "未来で会える日" not in narration


def test_interview_reply_preserves_a_valid_model_utterance() -> None:
    turn = InterviewTurnInput(
        transcript=[
            TranscriptEntry(
                speaker="ai",
                text="未来の自分に何か聞きたいことはありますか？",
            ),
            TranscriptEntry(
                speaker="visitor",
                text="宇宙で暮らせているか聞いてみたいです。",
            ),
        ],
        state=InterviewState(answer_count=1),
        target_transcript_chars=700,
        minimum_transcript_chars=400,
        conversation_time_limit_seconds=180,
        remaining_time_seconds=150,
    )
    raw = json.dumps(
        {
            "acquired_information": {"future_question": ["宇宙での暮らし"]},
            "asked_topics": ["future_question"],
            "next_topics": [],
            "next_utterance": "宇宙で暮らす自分を想像しているんだね。",
        },
        ensure_ascii=False,
    )

    output = TransformersInterviewLlmAdapter._parse_output(raw, turn)

    assert output.next_utterance == "宇宙で暮らす自分を想像しているんだね。"
    assert output.current_theme.value == "future_question"
    assert output.topic_depth == 1


def test_interview_vl_messages_are_text_only_multimodal_content() -> None:
    messages = TransformersInterviewLlmAdapter._text_only_vl_messages(
        [{"role": "user", "content": "未来の話をしよう"}]
    )

    assert messages == [
        {
            "role": "user",
            "content": [{"type": "text", "text": "未来の話をしよう"}],
        }
    ]
