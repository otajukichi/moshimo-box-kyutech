from __future__ import annotations

import json

from backend.app.contracts import InterviewTurnInput
from backend.app.schemas import InterviewState, TranscriptEntry
from backend.app.workers.adapters.transformers_interview_llm import (
    SYSTEM_PROMPT,
    TURN_INSTRUCTION,
    TransformersInterviewLlmAdapter,
)


def make_turn(
    transcript: list[TranscriptEntry],
    *,
    asked_topics: list[str] | None = None,
    acquired_information: dict[str, object] | None = None,
) -> InterviewTurnInput:
    return InterviewTurnInput(
        transcript=transcript,
        state=InterviewState(
            acquired_information=acquired_information or {},
            asked_topics=asked_topics or [],
            answer_count=sum(
                entry.speaker == "visitor" for entry in transcript
            ),
            visitor_char_count=sum(
                len(entry.text)
                for entry in transcript
                if entry.speaker == "visitor"
            ),
            elapsed_seconds=40,
        ),
        target_transcript_chars=700,
        minimum_transcript_chars=400,
        conversation_time_limit_seconds=180,
        remaining_time_seconds=140,
    )


def test_prompt_prioritizes_one_step_follow_up_over_new_angles() -> None:
    assert "同じ話題を複数ターン続けてよい" in SYSTEM_PROMPT
    assert "情報量より会話の自然さを優先" in SYSTEM_PROMPT
    assert "質問の角度は、景色、事件、発明" not in SYSTEM_PROMPT
    assert "原則として現在の話題を一段だけ深掘り" in TURN_INSTRUCTION
    assert "新しい切り口" not in TURN_INSTRUCTION


def test_same_game_topic_can_be_deepened_across_turns() -> None:
    turn = make_turn(
        [
            TranscriptEntry(
                speaker="ai",
                text="未来の自分に何か聞きたいことはありますか？",
            ),
            TranscriptEntry(speaker="visitor", text="ゲームが好きです。"),
            TranscriptEntry(
                speaker="ai",
                text="ゲームの話なら想像しやすそうだね。最近はどんなゲームをしている？",
            ),
            TranscriptEntry(
                speaker="visitor",
                text="友達と協力するゲームです。",
            ),
        ],
        asked_topics=["games"],
    )
    utterance = (
        "一緒に作戦を考えるタイプなんだね。"
        "遊んでいて一番盛り上がるのはどんな瞬間？"
    )
    raw = json.dumps(
        {
            "acquired_information": {
                "interests": ["友達と遊ぶ協力ゲーム"]
            },
            "asked_topics": ["games"],
            "next_topics": ["memorable-game-moment"],
            "next_utterance": utterance,
        },
        ensure_ascii=False,
    )

    output = TransformersInterviewLlmAdapter._parse_output(raw, turn)

    assert output.next_utterance == utterance
    assert output.asked_topics == ["games"]
    assert "未来から一つだけ持ち帰る" not in output.next_utterance


def test_multiple_questions_are_replaced_with_one_contextual_question() -> None:
    turn = make_turn(
        [TranscriptEntry(speaker="visitor", text="友達と協力するゲームです。")],
        asked_topics=["games"],
    )
    generated = (
        "協力ゲームなんだね。どんなゲームをしている？"
        "いつも誰と遊んでいる？"
    )

    shaped, _ = TransformersInterviewLlmAdapter._shape_reply(
        generated,
        turn,
        ["games"],
    )

    assert shaped != generated
    assert "ゲーム" in shaped
    assert TransformersInterviewLlmAdapter._question_count(shaped) == 1


def test_short_uncertain_answer_gets_an_easy_choice() -> None:
    turn = make_turn(
        [TranscriptEntry(speaker="visitor", text="特にないです。")],
        asked_topics=["future-message"],
    )

    fallback, topic = TransformersInterviewLlmAdapter._fallback_question(turn)

    assert "どちら" in fallback
    assert "便利" in fallback
    assert "冒険" in fallback
    assert topic == "easy-choice"
    assert TransformersInterviewLlmAdapter._question_count(fallback) == 1


def test_reusing_topic_identifier_does_not_replace_a_new_question() -> None:
    turn = make_turn(
        [
            TranscriptEntry(
                speaker="ai",
                text="最近はどんなゲームをしている？",
            ),
            TranscriptEntry(
                speaker="visitor",
                text="友達と協力するゲームです。",
            ),
        ],
        asked_topics=["games"],
    )
    generated = (
        "友達との協力が面白いんだね。"
        "作戦がうまくいったときは、どんな空気になる？"
    )

    shaped, replacement_topic = TransformersInterviewLlmAdapter._shape_reply(
        generated,
        turn,
        ["games"],
    )

    assert shaped == generated
    assert replacement_topic is None


def test_invalid_model_output_falls_back_to_latest_visitor_message() -> None:
    turn = make_turn(
        [TranscriptEntry(speaker="visitor", text="友達と協力するゲームです。")],
        asked_topics=["games"],
        acquired_information={"interests": ["ゲーム"]},
    )

    output = TransformersInterviewLlmAdapter._parse_output(
        "JSONではない応答",
        turn,
    )

    assert "ゲーム" in output.next_utterance
    assert TransformersInterviewLlmAdapter._question_count(
        output.next_utterance
    ) == 1
    assert output.asked_topics == ["games"]
    assert output.acquired_information == {"interests": ["ゲーム"]}


def test_nearly_identical_recent_question_uses_fallback() -> None:
    turn = make_turn(
        [
            TranscriptEntry(
                speaker="ai",
                text="ゲームが好きなんだね。最近はどんなゲームをしていますか？",
            ),
            TranscriptEntry(speaker="visitor", text="友達と遊ぶものです。"),
        ],
        asked_topics=["games"],
    )
    repeated = (
        "ゲームの話なんだね。最近はどんなゲームをしていますか？"
    )

    shaped, _ = TransformersInterviewLlmAdapter._shape_reply(
        repeated,
        turn,
        ["games"],
    )

    assert shaped != repeated
    assert TransformersInterviewLlmAdapter._question_count(shaped) == 1
