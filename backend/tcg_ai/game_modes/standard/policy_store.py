from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import threading
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_STANDARD_POLICY_STATE_PATH = PROJECT_ROOT / "standard_policy_progress.json"


@dataclass
class OpenerPolicyStats:
    resolved_samples: int = 0
    wins: int = 0
    total_terminal_reward: float = 0.0

    @property
    def average_terminal_reward(self) -> float:
        if self.resolved_samples <= 0:
            return 0.0
        return self.total_terminal_reward / self.resolved_samples

    @property
    def win_rate(self) -> float:
        if self.resolved_samples <= 0:
            return 0.0
        return self.wins / self.resolved_samples

    def record_outcome(self, terminal_reward: float, did_win: bool) -> None:
        self.resolved_samples += 1
        self.total_terminal_reward += terminal_reward
        if did_win:
            self.wins += 1

    def export_state(self) -> dict[str, Any]:
        return {
            "resolved_samples": self.resolved_samples,
            "wins": self.wins,
            "total_terminal_reward": round(self.total_terminal_reward, 6),
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            **self.export_state(),
            "average_terminal_reward": round(self.average_terminal_reward, 6),
            "win_rate": round(self.win_rate, 6),
        }


class StandardPolicyStore:
    def __init__(self, state_path: Path | None = None) -> None:
        self.state_path = state_path or DEFAULT_STANDARD_POLICY_STATE_PATH
        self._lock = threading.Lock()
        self._stats: dict[str, dict[str, dict[str, OpenerPolicyStats]]] = {}
        self.load_from_disk()

    def stats_for_deck(self, trainer_id: str, ai_deck_id: str) -> dict[str, OpenerPolicyStats]:
        with self._lock:
            deck_stats = self._stats.get(trainer_id, {}).get(ai_deck_id, {})
            return {
                card_id: OpenerPolicyStats(
                    resolved_samples=stats.resolved_samples,
                    wins=stats.wins,
                    total_terminal_reward=stats.total_terminal_reward,
                )
                for card_id, stats in deck_stats.items()
            }

    def record_opener_outcome(
        self,
        trainer_id: str,
        ai_deck_id: str,
        chosen_card_id: str,
        terminal_reward: float,
        did_win: bool,
    ) -> None:
        with self._lock:
            trainer_stats = self._stats.setdefault(trainer_id, {})
            deck_stats = trainer_stats.setdefault(ai_deck_id, {})
            stats = deck_stats.setdefault(chosen_card_id, OpenerPolicyStats())
            stats.record_outcome(terminal_reward=terminal_reward, did_win=did_win)
        self.save_to_disk()

    def export_state(self) -> dict[str, Any]:
        with self._lock:
            return {
                "version": 1,
                "trainers": {
                    trainer_id: {
                        deck_id: {
                            card_id: stats.export_state()
                            for card_id, stats in deck_stats.items()
                        }
                        for deck_id, deck_stats in trainer_stats.items()
                    }
                    for trainer_id, trainer_stats in self._stats.items()
                },
            }

    def load_from_disk(self) -> None:
        if not self.state_path.exists():
            return

        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return

        if not isinstance(payload, dict):
            return

        raw_trainers = payload.get("trainers")
        if not isinstance(raw_trainers, dict):
            return

        loaded: dict[str, dict[str, dict[str, OpenerPolicyStats]]] = {}
        for trainer_id, trainer_payload in raw_trainers.items():
            if not isinstance(trainer_id, str) or not isinstance(trainer_payload, dict):
                continue
            trainer_stats: dict[str, dict[str, OpenerPolicyStats]] = {}
            for deck_id, deck_payload in trainer_payload.items():
                if not isinstance(deck_id, str) or not isinstance(deck_payload, dict):
                    continue
                opener_stats: dict[str, OpenerPolicyStats] = {}
                for card_id, stats_payload in deck_payload.items():
                    if not isinstance(card_id, str) or not isinstance(stats_payload, dict):
                        continue
                    opener_stats[card_id] = OpenerPolicyStats(
                        resolved_samples=int(stats_payload.get("resolved_samples", 0)),
                        wins=int(stats_payload.get("wins", 0)),
                        total_terminal_reward=float(stats_payload.get("total_terminal_reward", 0.0)),
                    )
                trainer_stats[deck_id] = opener_stats
            loaded[trainer_id] = trainer_stats

        with self._lock:
            self._stats = loaded

    def save_to_disk(self) -> None:
        payload = json.dumps(self.export_state(), indent=2, sort_keys=True)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.state_path.with_suffix(f"{self.state_path.suffix}.tmp")
        temporary_path.write_text(payload, encoding="utf-8")
        temporary_path.replace(self.state_path)
