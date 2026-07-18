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
                if next_anchor is None and current_theme != InterviewTheme.FUTURE_MESSAGE
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


def model_json(utterance: str, topic: str) -> str:
    return json.dumps(
        {
            "acquired_information": {},
            "asked_topics": [topic],
            "next_topics": [],
            "next_utterance": utterance,
        },
        ensure_ascii=False,
    )


def test_prompt_uses_system_managed_themes() -> None:
    turn = make_turn(
        [TranscriptEntry(speaker="visitor", text="ゲームが好きです。")],
        current_theme=InterviewTheme.PRESENT_CONNECTION,
        next_anchor=InterviewTheme.CONCRETE_EPISODE,
    )

    prompt = build_turn_prompt(turn)

    assert "一つのテーマは原則2〜3回答" in SYSTEM_PROMPT
    assert "次の話題を自由に選んではいけません" in SYSTEM_PROMPT
    assert "conversation_plan はシステムが決めた" in TURN_INSTRUCTION
    assert '"mode":"follow_up"' in prompt
    assert '"target_theme":"present_connection"' in prompt


def test_same_topic_is_deepened_before_switching() -> None:
    turn = make_turn(
        [TranscriptEntry(speaker="visitor", text="ゲームが好きです。")],
        current_theme=InterviewTheme.PRESENT_CONNECTION,
        next_anchor=InterviewTheme.CONCRETE_EPISODE,
    )
    utterance = (
        "ゲームのどんなところに引かれるのか気になる。"
        "遊んでいて時間を忘れるのはどんな瞬間？"
    )

    output = TransformersInterviewLlmAdapter._parse_output(
        model_json(utterance, "present_connection"),
        turn,
    )

    assert output.next_utterance == utterance
    assert output.current_theme == InterviewTheme.PRESENT_CONNECTION
    assert output.topic_depth == 1
    assert output.topic_complete is False
    assert output.next_anchor == InterviewTheme.CONCRETE_EPISODE


def test_concrete_second_answer_opens_the_next_fixed_theme() -> None:
    turn = make_turn(
        [
            TranscriptEntry(speaker="visitor", text="ゲームが好きです。"),
            TranscriptEntry(
                speaker="ai",
                text="遊んでいて時間を忘れるのはどんな瞬間？",
            ),
            TranscriptEntry(
                speaker="visitor",
                text="友達と協力して作戦が成功した瞬間です。",
            ),
        ],
        current_theme=InterviewTheme.PRESENT_CONNECTION,
        topic_depth=1,
        next_anchor=InterviewTheme.CONCRETE_EPISODE,
    )
    utterance = (
        "作戦が噛み合った瞬間は盛り上がりそう。"
        "最近いちばん印象に残った試合では、何が起きた？"
    )

    output = TransformersInterviewLlmAdapter._parse_output(
        model_json(utterance, "concrete_episode"),
        turn,
    )

    assert output.next_utterance == utterance
    assert output.current_theme == InterviewTheme.PRESENT_CONNECTION
    assert output.topic_depth == 2
    assert output.topic_complete is True
    assert output.next_anchor == InterviewTheme.CONCRETE_EPISODE
    assert "concrete_episode" in output.asked_topics


def test_answer_to_transition_question_starts_new_theme_depth() -> None:
    turn = make_turn(
        [
            TranscriptEntry(
                speaker="ai",
                text="最近いちばん印象に残った出来事は何だった？",
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
    assert plan.question_theme == InterviewTheme.CONCRETE_EPISODE


def test_non_question_model_reply_is_kept_verbatim() -> None:
    turn = make_turn(
        [TranscriptEntry(speaker="visitor", text="展示を作りました。")],
        current_theme=InterviewTheme.CONCRETE_EPISODE,
        next_anchor=InterviewTheme.FUTURE_EXPANSION,
    )
    utterance = "展示をみんなで完成させた場面、もう少し聞いてみたいな。"

    output = TransformersInterviewLlmAdapter._parse_output(
        model_json(utterance, "concrete_episode"),
        turn,
    )

    assert output.next_utterance == utterance


def test_multiple_questions_are_not_replaced_after_generation() -> None:
    turn = make_turn(
        [TranscriptEntry(speaker="visitor", text="ゲームが好きです。")],
        current_theme=InterviewTheme.PRESENT_CONNECTION,
        next_anchor=InterviewTheme.CONCRETE_EPISODE,
    )
    utterance = "どんなゲームが好き？誰と遊ぶことが多い？"

    output = TransformersInterviewLlmAdapter._parse_output(
        model_json(utterance, "present_connection"),
        turn,
    )

    assert output.next_utterance == utterance


def test_broken_json_recovers_next_utterance_instead_of_fixed_question() -> None:
    turn = make_turn(
        [TranscriptEntry(speaker="visitor", text="友達と協力するゲームです。")],
        current_theme=InterviewTheme.PRESENT_CONNECTION,
        next_anchor=InterviewTheme.CONCRETE_EPISODE,
    )
    raw = '{"next_utterance":"協力がうまくいった場面をもっと聞きたい。", "asked_topics": ['

    output = TransformersInterviewLlmAdapter._parse_output(raw, turn)

    assert output.next_utterance == "協力がうまくいった場面をもっと聞きたい。"


def test_plain_model_prose_is_used_as_the_utterance() -> None:
    turn = make_turn(
        [TranscriptEntry(speaker="visitor", text="研究でロボットを作っています。")],
        current_theme=InterviewTheme.CONCRETE_EPISODE,
        next_anchor=InterviewTheme.FUTURE_EXPANSION,
    )
    raw = "ロボットが初めて動いた瞬間には、きっと特別な空気があったんだろうな。"

    output = TransformersInterviewLlmAdapter._parse_output(raw, turn)

    assert output.next_utterance == raw


def test_empty_output_uses_only_the_last_resort_current_theme_prompt() -> None:
    turn = make_turn(
        [TranscriptEntry(speaker="visitor", text="ゲームが好きです。")],
        current_theme=InterviewTheme.PRESENT_CONNECTION,
        next_anchor=InterviewTheme.CONCRETE_EPISODE,
    )

    output = TransformersInterviewLlmAdapter._parse_output("", turn)

    assert "ゲーム" in output.next_utterance
    assert output.current_theme == InterviewTheme.PRESENT_CONNECTION
    assert "未来から一つ持ち帰れる" not in output.next_utterance


def test_short_uncertain_answer_uses_support_mode() -> None:
    turn = make_turn(
        [TranscriptEntry(speaker="visitor", text="特にないです。")],
    )

    plan = plan_interview_turn(turn)
    fallback, topic = TransformersInterviewLlmAdapter._fallback_question(turn)

    assert plan.mode == "support"
    assert plan.question_theme == InterviewTheme.FUTURE_QUESTION
    assert "どちら" in fallback
    assert topic == InterviewTheme.FUTURE_QUESTION.value


def test_worker_failure_recovery_does_not_jump_to_an_unrelated_question(
    tmp_path,
) -> None:
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
    assert "今つい時間を使ってしまうもの" in (
        session.interview_state.current_question_text or ""
    )
    assert "未来から一つ持ち帰れる" not in (
        session.interview_state.current_question_text or ""
    )
