from __future__ import annotations

from dataclasses import dataclass, field
import threading
from typing import Any

from .learning import RewardLearner

LEVEL_XP_STEP = 100
XP_PER_PRIZE = 25

GEN_1_GYM_LEADERS: tuple[tuple[str, str, str], ...] = (
    ("brock", "Brock", "Rock"),
    ("misty", "Misty", "Water"),
    ("lt_surge", "Lt. Surge", "Electric"),
    ("erika", "Erika", "Grass"),
    ("koga", "Koga", "Poison"),
    ("sabrina", "Sabrina", "Psychic"),
    ("blaine", "Blaine", "Fire"),
    ("giovanni", "Giovanni", "Ground"),
)


def trainer_experience_for_game(damage_dealt: int, prizes_taken: int) -> int:
    return max(0, damage_dealt // 10) + max(0, prizes_taken) * XP_PER_PRIZE


def level_for_experience(experience: int) -> int:
    return max(1, (max(0, experience) // LEVEL_XP_STEP) + 1)


@dataclass
class TrainerProfile:
    trainer_id: str
    name: str
    specialty: str
    learner: RewardLearner = field(default_factory=RewardLearner, repr=False)
    experience: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def gain_experience(self, damage_dealt: int, prizes_taken: int) -> int:
        gained = trainer_experience_for_game(damage_dealt, prizes_taken)
        with self._lock:
            self.experience += gained
        return gained

    def snapshot(self, selected: bool = False) -> dict[str, Any]:
        with self._lock:
            experience = self.experience

        level = level_for_experience(experience)
        level_floor = (level - 1) * LEVEL_XP_STEP
        next_level_xp = level * LEVEL_XP_STEP
        return {
            "id": self.trainer_id,
            "name": self.name,
            "specialty": self.specialty,
            "level": level,
            "experience": experience,
            "xp_into_level": experience - level_floor,
            "xp_to_next_level": max(0, next_level_xp - experience),
            "selected": selected,
        }


class TrainerStore:
    def __init__(self) -> None:
        self.default_trainer_id = GEN_1_GYM_LEADERS[0][0]
        self._profiles = {
            trainer_id: TrainerProfile(trainer_id=trainer_id, name=name, specialty=specialty)
            for trainer_id, name, specialty in GEN_1_GYM_LEADERS
        }

    def get(self, trainer_id: str) -> TrainerProfile | None:
        return self._profiles.get(trainer_id)

    def snapshots(self, selected_id: str | None = None) -> list[dict[str, Any]]:
        return [
            profile.snapshot(selected=profile.trainer_id == selected_id)
            for profile in self._profiles.values()
        ]
