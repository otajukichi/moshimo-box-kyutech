from __future__ import annotations

from pathlib import Path

from backend.app.episodes import EpisodeRepository, upgrade_rarity
from backend.app.schemas import Rarity, StaffSettings


class FirstChoiceRandom:
    def __init__(self) -> None:
        self.recorded_weights: list[list[float]] = []

    def choices(self, population, weights, k):
        self.recorded_weights.append(list(weights))
        return [population[0]]


def repository(project_root: Path) -> EpisodeRepository:
    return EpisodeRepository(
        project_root / "config" / "episodes",
        project_root / "config" / "effects.yaml",
        {
            Rarity.R: 50,
            Rarity.SR: 30,
            Rarity.SSR: 16,
            Rarity.UR: 4,
        },
    )


def test_formal_filter_uses_only_public_safe_episodes(project_root: Path) -> None:
    repo = repository(project_root)
    eligible = repo.eligible(StaffSettings())

    assert eligible
    assert all(episode.formal_mode_allowed for episode in eligible)
    assert all(episode.public_demo_allowed for episode in eligible)
    assert all(not episode.limited_only for episode in eligible)
    assert "biohazard-survivor" not in {episode.id for episode in eligible}


def test_underground_mode_uses_all_enabled_episodes(project_root: Path) -> None:
    repo = repository(project_root)
    settings = StaffSettings(episode_mode="underground")

    assert "biohazard-survivor" in {episode.id for episode in repo.eligible(settings)}


def test_fixed_episode_selection_and_effect_upgrade(project_root: Path) -> None:
    repo = repository(project_root)
    settings = StaffSettings(
        episode_selection="fixed",
        fixed_episode_id="digital-existence",
    )

    episode = repo.select_episode(settings)

    assert episode.id == "digital-existence"
    assert episode.base_rarity == Rarity.SSR
    assert upgrade_rarity(Rarity.SSR, 1) == Rarity.UR


def test_random_selection_uses_rarity_then_episode_weights(project_root: Path) -> None:
    repo = repository(project_root)
    random_source = FirstChoiceRandom()

    episode = repo.select_episode(StaffSettings(), rng=random_source)

    assert episode.base_rarity == Rarity.R
    assert random_source.recorded_weights[0] == [50, 30, 16]
    assert random_source.recorded_weights[1]
