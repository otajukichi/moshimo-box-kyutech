from __future__ import annotations

import json

from backend.app.contracts import InterviewTurnInput
from backend.app.schemas import (
    InterviewState,
    InterviewTheme,
    SessionRecord,
    StaffSettings,
    TranscriptEntry,
)
from backend.app.session_store import SessionStore
from backend.app.workers.adapters.transformers_interview_llm import (
    SYSTEM_PROMPT,
    TURN_INSTRUCTION,
    TransformersInterviewLlmAdapter,
    build_turn_prompt,
    plan_interview_turn,
)


NEXT_THEME = {
    InterviewTheme.FUTURE_QUESTION: InterviewTheme.PRESENT_CONNECTION,
    InterviewTheme.PRESENT_CONNECTION: InterviewTheme.CONCRETE_EPISODE,
    InterviewTheme.CONCRETE_EPISODE: InterviewTheme.FUTURE_EXPANSION,
    InterviewTheme.FUTURE_EXPANSION: InterviewTheme.FUTURE_MESSAGE,
    InterviewTheme.FUTURE_MESSAGE: None,
}


def make_turn(
    transcript: list[TranscriptEntry],
    *,
    current_theme: InterviewTheme = InterviewTheme.FUTURE_QUESTION,
    topic_depth: int = 0,
    topic_complete: bool = False,
    next_anchor: InterviewTheme | None = None,
    asked_topics: list[str] | None = None,
    acquired_information: dict[str, object] | None = None,
) -> InterviewTurnInput:
    return InterviewTurnInput(
        transcript=transcript,
        state=InterviewState(
            acquired_information=acquired_information or {},
            asked_topics=asked_topics or [],
            current_theme=current_theme,
            topic_depth=topic_depth,
            topic_complete=topic_complete,
            next_anchor=(
                NEXT_THEME[current_theme]
                if next_anchor is None
                and current_theme != InterviewTheme.FUTURE_MESSAGE
                else next_anchor
            ),
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


def legacy_model_json(utterance: str, topic: str) -> str:
    return json.dumps(
        {
            "acquired_information": {
                "interests": ["モデルが推測した未確認情報"]
            },
            "asked_topics": [topic],
            "next_topics": [],
            "next_utterance": utterance,
        },
        ensure_ascii=False,
    )


def test_prompt_defines_present_day_interviewer_role() -> None:
    turn = make_turn(
        [
            TranscriptEntry(
                speaker="ai",
                text="未来の自分に何か聞きたいことはありますか？",
            ),
            TranscriptEntry(speaker="visitor", text="歯は何本残っていますか"),
        ]
    )

    prompt = build_turn_prompt(turn)

    assert "あなたは未来の本人ではありません" in SYSTEM_PROMPT
    assert "未来の本人ならどう答えるか" in SYSTEM_PROMPT
    assert "歯は何本残っていますか" in SYSTEM_PROMPT
    assert "読み上げる返答本文だけ" in TURN_INSTRUCTION
    assert '"mode":"transition"' in prompt
    assert '"target_theme":"present_connection"' in prompt
    assert '"future_question_captured":true' in prompt
    assert "JSONだけ" not in SYSTEM_PROMPT


def test_future_question_is_captured_once_then_moves_to_present() -> None:
    question = "歯は何本残っていますか"
    turn = make_turn(
        [
            TranscriptEntry(
                speaker="ai",
                text="未来の自分に何か聞きたいことはありますか？",
            ),
            TranscriptEntry(speaker="visitor", text=question),
        ]
    )
    utterance = (
        "かなり具体的な質問だね。それは未来の自分に預けておこう。"
        "今のあなたが最近よくしていることは何？"
    )

    output = TransformersInterviewLlmAdapter._parse_output(utterance, turn)

    assert output.next_utterance == utterance
    assert output.current_theme == InterviewTheme.FUTURE_QUESTION
    assert output.topic_depth == 1
    assert output.topic_complete is True
    assert output.next_anchor == InterviewTheme.PRESENT_CONNECTION
    assert output.acquired_information == {"future_questions": [question]}
    assert output.asked_topics[-1] == InterviewTheme.PRESENT_CONNECTION.value


def test_short_asr_fragment_is_not_recorded_or_interpreted() -> None:
    turn = make_turn(
        [
            TranscriptEntry(speaker="visitor", text="歯は何本残っていますか"),
            TranscriptEntry(
                speaker="ai",
                text="今のあなたが最近よくしていることは何？",
            ),
            TranscriptEntry(speaker="visitor", text="いい山"),
        ],
        current_theme=InterviewTheme.FUTURE_QUESTION,
        topic_depth=1,
        topic_complete=True,
        next_anchor=InterviewTheme.PRESENT_CONNECTION,
        acquired_information={
            "future_questions": ["歯は何本残っていますか"]
        },
    )

    plan = plan_interview_turn(turn)
    prompt = build_turn_prompt(turn)
    output = TransformersInterviewLlmAdapter._parse_output(
        "了解、もっと簡単に聞くね。最近よくやっていることはある？",
        turn,
    )

    assert "いい山" not in prompt
    assert "短く曖昧な音声認識結果" in prompt
    assert plan.current_theme == InterviewTheme.PRESENT_CONNECTION
    assert plan.mode == "support"
    assert plan.topic_depth == 0
    assert plan.interesting_detail is None
    assert output.acquired_information == {
        "future_questions": ["歯は何本残っていますか"]
    }
    assert "いい山" not in json.dumps(
        output.acquired_information,
        ensure_ascii=False,
    )


def test_confusion_uses_repair_mode_without_inventing_emotion() -> None:
    turn = make_turn(
        [
            TranscriptEntry(
                speaker="ai",
                text="今のあなたが最近よくしていることは何？",
            ),
            TranscriptEntry(
                speaker="visitor",
                text="何を言ってるか分からないです",
            ),
        ],
        current_theme=InterviewTheme.PRESENT_CONNECTION,
        next_anchor=InterviewTheme.CONCRETE_EPISODE,
    )

    plan = plan_interview_turn(turn)
    output = TransformersInterviewLlmAdapter._parse_output("", turn)

    assert plan.mode == "repair"
    assert plan.question_theme == InterviewTheme.PRESENT_CONNECTION
    assert output.next_utterance.startswith("ごめん")
    assert "不思議な気持ち" not in output.next_utterance
    assert output.acquired_information == {}


def test_same_present_topic_is_deepened_before_switching() -> None:
    turn = make_turn(
        [TranscriptEntry(speaker="visitor", text="ゲームが好きです。")],
        current_theme=InterviewTheme.PRESENT_CONNECTION,
        next_anchor=InterviewTheme.CONCRETE_EPISODE,
    )
    utterance = "ゲームをしていて、いちばん夢中になるのはどんな瞬間？"

    output = TransformersInterviewLlmAdapter._parse_output(utterance, turn)

    assert output.next_utterance == utterance
    assert output.current_theme == InterviewTheme.PRESENT_CONNECTION
    assert output.topic_depth == 1
    assert output.topic_complete is False
    assert output.acquired_information == {
        "present_details": ["ゲームが好きです。"]
    }


def test_concrete_second_answer_opens_next_fixed_theme() -> None:
    turn = make_turn(
        [
            TranscriptEntry(speaker="visitor", text="ゲームが好きです。"),
            TranscriptEntry(
                speaker="ai",
                text="いちばん夢中になるのはどんな瞬間？",
            ),
            TranscriptEntry(
                speaker="visitor",
                text="友達と協力して作戦が成功した瞬間です。",
            ),
        ],
        current_theme=InterviewTheme.PRESENT_CONNECTION,
        topic_depth=1,
        next_anchor=InterviewTheme.CONCRETE_EPISODE,
        acquired_information={"present_details": ["ゲームが好きです。"]},
    )

    plan = plan_interview_turn(turn)

    assert plan.mode == "transition"
    assert plan.question_theme == InterviewTheme.CONCRETE_EPISODE
    assert plan.topic_complete is True
    assert plan.topic_depth == 2


def test_answer_to_transition_question_starts_new_theme_depth() -> None:
    turn = make_turn(
        [
            TranscriptEntry(
                speaker="ai",
                text="最近いちばん印象に残った場面は？",
            ),
            TranscriptEntry(
                speaker="visitor",
                text="学園祭で友達と展示を作りました。",
            ),
        ],
        current_theme=InterviewTheme.PRESENT_CONNECTION,
        topic_depth=2,
        topic_complete=True,
        next_anchor=InterviewTheme.CONCRETE_EPISODE,
    )

    plan = plan_interview_turn(turn)

    assert plan.current_theme == InterviewTheme.CONCRETE_EPISODE
    assert plan.topic_depth == 1
    assert plan.mode == "follow_up"


def test_non_question_model_reply_is_kept_verbatim() -> None:
    turn = make_turn(
        [TranscriptEntry(speaker="visitor", text="展示を作りました。")],
        current_theme=InterviewTheme.CONCRETE_EPISODE,
        next_anchor=InterviewTheme.FUTURE_EXPANSION,
    )
    utterance = "展示をみんなで完成させた場面、もう少し聞いてみたいな。"

    output = TransformersInterviewLlmAdapter._parse_output(utterance, turn)

    assert output.next_utterance == utterance


def test_legacy_json_only_contributes_spoken_utterance() -> None:
    turn = make_turn(
        [TranscriptEntry(speaker="visitor", text="友達とゲームをしています。")],
        current_theme=InterviewTheme.PRESENT_CONNECTION,
        next_anchor=InterviewTheme.CONCRETE_EPISODE,
    )
    utterance = "友達と遊んでいて、最近いちばん笑ったのはどんな場面？"

    output = TransformersInterviewLlmAdapter._parse_output(
        legacy_model_json(utterance, "present_connection"),
        turn,
    )

    assert output.next_utterance == utterance
    assert "モデルが推測した未確認情報" not in json.dumps(
        output.acquired_information,
        ensure_ascii=False,
    )
    assert output.acquired_information == {
        "present_details": ["友達とゲームをしています。"]
    }


def test_broken_json_recovers_next_utterance() -> None:
    turn = make_turn(
        [TranscriptEntry(speaker="visitor", text="友達と協力するゲームです。")],
        current_theme=InterviewTheme.PRESENT_CONNECTION,
        next_anchor=InterviewTheme.CONCRETE_EPISODE,
    )
    raw = '{"next_utterance":"協力がうまくいった場面をもっと聞きたい。", "asked_topics": ['

    output = TransformersInterviewLlmAdapter._parse_output(raw, turn)

    assert output.next_utterance == "協力がうまくいった場面をもっと聞きたい。"


def test_empty_output_uses_current_theme_only() -> None:
    turn = make_turn(
        [TranscriptEntry(speaker="visitor", text="ゲームが好きです。")],
        current_theme=InterviewTheme.PRESENT_CONNECTION,
        next_anchor=InterviewTheme.CONCRETE_EPISODE,
    )

    output = TransformersInterviewLlmAdapter._parse_output("", turn)

    assert output.next_utterance == "最近よくやっていることは何？"
    assert "未来の自分なら" not in output.next_utterance


def test_worker_failure_recovery_stays_in_current_role(tmp_path) -> None:
    store = SessionStore(tmp_path)
    session = SessionRecord(
        session_id="test-session",
        settings_snapshot=StaffSettings(),
        transcript=[
            TranscriptEntry(
                speaker="visitor",
                text="ゲームで友達と協力するのが好きです。",
            )
        ],
        interview_state=InterviewState(
            current_theme=InterviewTheme.PRESENT_CONNECTION,
            topic_depth=1,
            topic_complete=False,
            next_anchor=InterviewTheme.CONCRETE_EPISODE,
            answer_count=2,
        ),
    )

    store._set_recovery_question(session)

    assert session.interview_state.current_theme == InterviewTheme.PRESENT_CONNECTION
    assert session.interview_state.current_question_text == (
        "最近よくやっていることは何？"
    )
    assert "未来の自分なら" not in (
        session.interview_state.current_question_text or ""
    )
