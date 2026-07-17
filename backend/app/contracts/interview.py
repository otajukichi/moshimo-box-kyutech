from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from ..schemas import SCHEMA_VERSION, InterviewState, TranscriptEntry


class InterviewTurnInput(BaseModel):
    schema_version: str = SCHEMA_VERSION
    transcript: list[TranscriptEntry] = Field(default_factory=list)
    state: InterviewState
    target_transcript_chars: int = Field(gt=0)
    minimum_transcript_chars: int = Field(gt=0)
    conversation_time_limit_seconds: int = Field(gt=0)
    remaining_time_seconds: int = Field(ge=0)
    thinking_enabled: bool = False


class InterviewTurnOutput(BaseModel):
    schema_version: str = SCHEMA_VERSION
    acquired_information: dict[str, Any] = Field(default_factory=dict)
    asked_topics: list[str] = Field(default_factory=list)
    next_topics: list[str] = Field(default_factory=list)
    visitor_char_count: int = Field(ge=0)
    elapsed_seconds: int = Field(ge=0)
    should_end: bool = False
    end_reason: Literal[
        "target_transcript_reached",
        "conversation_time_limit",
        "operator_finished",
        "insufficient_input",
        "continue",
    ] = "continue"
    next_utterance: str = Field(min_length=1)
