from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from ..schemas import (
    SCHEMA_VERSION,
    Episode,
    EpisodeEffect,
    Rarity,
    TranscriptEntry,
)


class GenerationCapabilities(BaseModel):
    image_model: dict[str, Any] = Field(default_factory=dict)
    video_model: dict[str, Any] = Field(default_factory=dict)
    voice_model: dict[str, Any] = Field(default_factory=dict)


class ScriptDesignInput(BaseModel):
    schema_version: str = SCHEMA_VERSION
    transcript: list[TranscriptEntry] = Field(default_factory=list)
    interview_summary: str
    person_information: dict[str, Any] = Field(default_factory=dict)
    episode: Episode
    effect: EpisodeEffect
    final_rarity: Rarity
    episode_mode: Literal["formal", "underground"]
    target_video_seconds: int = Field(default=20, gt=0)
    capabilities: GenerationCapabilities = Field(default_factory=GenerationCapabilities)
    prohibited_expressions: list[str] = Field(default_factory=list)
    remaining_time_seconds: int = Field(gt=0)


class ShotPlanItem(BaseModel):
    shot_id: str
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    composition: str
    action: str
    narration: str = ""
    transition: str = "cut"

    @model_validator(mode="after")
    def validate_timing(self) -> "ShotPlanItem":
        if self.end_seconds <= self.start_seconds:
            raise ValueError("end_seconds must be greater than start_seconds")
        return self


class ScriptDesignOutput(BaseModel):
    schema_version: str = SCHEMA_VERSION
    source_visual_observation: str = ""
    future_world: str
    future_person: str
    positive_interpretation: str
    visual_concept: str
    clothing: str
    background: str
    camera: str
    emotion: str
    narration_script: str = Field(min_length=80, max_length=110)
    shot_plan: list[ShotPlanItem] = Field(min_length=1, max_length=1)
    image_prompt: str
    negative_prompt: str
    video_prompt: str
    voice_instruction: str
    safety_notes: list[str] = Field(default_factory=list)
    fallback_plan: str


class ScriptSafetyReviewOutput(BaseModel):
    schema_version: str = SCHEMA_VERSION
    approved: bool
    reasons: list[str] = Field(default_factory=list)
    corrected_output: ScriptDesignOutput | None = None

    @model_validator(mode="after")
    def require_safe_correction(self) -> "ScriptSafetyReviewOutput":
        if not self.approved and self.corrected_output is None:
            raise ValueError("corrected_output is required when a script is rejected")
        return self
