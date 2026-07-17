from __future__ import annotations

import random
from pathlib import Path

import yaml

from .schemas import (
    Episode,
    EpisodeEffect,
    EpisodeSummary,
    Rarity,
    SCHEMA_VERSION,
    StaffSettings,
)


RARITY_ORDER = [Rarity.R, Rarity.SR, Rarity.SSR, Rarity.UR]


def upgrade_rarity(base: Rarity, steps: int) -> Rarity:
    index = min(len(RARITY_ORDER) - 1, RARITY_ORDER.index(base) + steps)
    return RARITY_ORDER[index]


class EpisodeRepository:
    def __init__(
        self,
        source_dir: Path,
        effects_path: Path,
        rarity_weights: dict[Rarity, float],
    ) -> None:
        self.source_dir = source_dir
        self.effects_path = effects_path
        self.rarity_weights = rarity_weights
        self._episodes = self._load_episodes()
        self._effects = self._load_effects()

    def _load_episodes(self) -> list[Episode]:
        episodes: list[Episode] = []
        for path in sorted(self.source_dir.glob("*.yaml")):
            with path.open("r", encoding="utf-8") as handle:
                raw = yaml.safe_load(handle) or {}
            episode = Episode.model_validate(raw)
            if episode.schema_version != SCHEMA_VERSION:
                raise ValueError(f"未対応のエピソードschema_versionです: {path}")
            episodes.append(episode)
        if not episodes:
            raise ValueError(f"エピソードが見つかりません: {self.source_dir}")
        return episodes

    def _load_effects(self) -> list[EpisodeEffect]:
        with self.effects_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        if raw.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("未対応の追加演出schema_versionです")
        return [EpisodeEffect.model_validate(item) for item in raw.get("effects", [])]

    def reload(self) -> None:
        self._episodes = self._load_episodes()
        self._effects = self._load_effects()

    def summaries(self) -> list[EpisodeSummary]:
        return [
            EpisodeSummary(
                id=item.id,
                name=item.name,
                base_rarity=item.base_rarity,
                formal_mode_allowed=item.formal_mode_allowed,
                public_demo_allowed=item.public_demo_allowed,
                limited_only=item.limited_only,
            )
            for item in self._episodes
            if item.enabled
        ]

    def get_episode(self, episode_id: str) -> Episode:
        for episode in self._episodes:
            if episode.id == episode_id:
                return episode
        raise ValueError(f"エピソードが見つかりません: {episode_id}")

    def get_effect(self, effect_id: str) -> EpisodeEffect:
        for effect in self._effects:
            if effect.id == effect_id:
                return effect
        raise ValueError(f"追加演出が見つかりません: {effect_id}")

    def eligible(self, settings: StaffSettings) -> list[Episode]:
        episodes = [item for item in self._episodes if item.enabled]
        if settings.episode_mode == "formal":
            episodes = [
                item
                for item in episodes
                if item.formal_mode_allowed
                and item.public_demo_allowed
                and not item.limited_only
            ]
        return episodes

    def eligible_effects(self, settings: StaffSettings) -> list[EpisodeEffect]:
        effects = [item for item in self._effects if item.enabled]
        if settings.episode_mode == "formal":
            effects = [item for item in effects if item.formal_mode_allowed]
        return effects

    def select_episode(
        self,
        settings: StaffSettings,
        rng: random.Random | None = None,
    ) -> Episode:
        eligible = self.eligible(settings)
        if not eligible:
            raise ValueError("現在の設定で選択できるエピソードがありません")

        if settings.episode_selection == "fixed":
            for episode in eligible:
                if episode.id == settings.fixed_episode_id:
                    return episode
            raise ValueError("固定エピソードが現在の条件では利用できません")

        generator = rng or random.Random()
        available_rarities = [
            rarity for rarity in RARITY_ORDER if any(item.base_rarity == rarity for item in eligible)
        ]
        selected_rarity = generator.choices(
            available_rarities,
            weights=[self.rarity_weights.get(rarity, 0) for rarity in available_rarities],
            k=1,
        )[0]
        candidates = [item for item in eligible if item.base_rarity == selected_rarity]
        return generator.choices(
            candidates,
            weights=[episode.weight for episode in candidates],
            k=1,
        )[0]

    def select_effect(
        self,
        settings: StaffSettings,
        rng: random.Random | None = None,
    ) -> EpisodeEffect:
        eligible = self.eligible_effects(settings)
        if not eligible:
            raise ValueError("現在の設定で選択できる追加演出がありません")
        generator = rng or random.Random()
        return generator.choices(
            eligible,
            weights=[effect.weight for effect in eligible],
            k=1,
        )[0]

    def select(
        self,
        settings: StaffSettings,
        rng: random.Random | None = None,
    ) -> tuple[Episode, EpisodeEffect, Rarity]:
        generator = rng or random.Random()
        episode = self.select_episode(settings, generator)
        effect = self.select_effect(settings, generator)
        return episode, effect, upgrade_rarity(
            episode.base_rarity,
            effect.rarity_upgrade_steps,
        )
