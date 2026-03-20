from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import threading
from typing import Any

from .learning import RewardLearner

LEVEL_XP_STEP = 100
XP_PER_PRIZE = 25
DEFAULT_TRAINER_STATE_PATH = Path(__file__).resolve().parents[2] / "trainer_progress.txt"

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

    def export_state(self) -> dict[str, Any]:
        with self._lock:
            experience = self.experience
        return {
            "id": self.trainer_id,
            "name": self.name,
            "specialty": self.specialty,
            "experience": experience,
            "learner": self.learner.export_state(),
        }

    def load_state(self, payload: dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            return
        with self._lock:
            self.experience = int(payload.get("experience", self.experience))
        self.learner.load_state(payload.get("learner", {}))


class TrainerStore:
    def __init__(self, state_path: Path | None = None) -> None:
        self.default_trainer_id = GEN_1_GYM_LEADERS[0][0]
        self.state_path = state_path or DEFAULT_TRAINER_STATE_PATH
        self._lock = threading.Lock()
        self._profiles = {
            trainer_id: TrainerProfile(trainer_id=trainer_id, name=name, specialty=specialty)
            for trainer_id, name, specialty in GEN_1_GYM_LEADERS
        }
        self.load_from_disk()

    def get(self, trainer_id: str) -> TrainerProfile | None:
        return self._profiles.get(trainer_id)

    def snapshots(self, selected_id: str | None = None) -> list[dict[str, Any]]:
        return [
            profile.snapshot(selected=profile.trainer_id == selected_id)
            for profile in self._profiles.values()
        ]

    def export_state(self) -> dict[str, Any]:
        return {
            "version": 1,
            "trainers": [profile.export_state() for profile in self._profiles.values()],
        }

    def save_to_disk(self) -> None:
        with self._lock:
            payload = json.dumps(self.export_state(), indent=2, sort_keys=True)
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = self.state_path.with_suffix(f"{self.state_path.suffix}.tmp")
            temporary_path.write_text(payload, encoding="utf-8")
            temporary_path.replace(self.state_path)

    def load_from_disk(self) -> None:
        if not self.state_path.exists():
            return

        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return

        if not isinstance(payload, dict):
            return

        for trainer_state in payload.get("trainers", []):
            if not isinstance(trainer_state, dict):
                continue
            trainer_id = trainer_state.get("id")
            if not isinstance(trainer_id, str):
                continue
            trainer = self._profiles.get(trainer_id)
            if trainer is None:
                continue
            trainer.load_state(trainer_state)
