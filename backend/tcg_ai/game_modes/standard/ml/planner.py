from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from math import inf
from typing import Any

from ..engine import action_id_for, apply_action_for_player, list_legal_actions
from ..models import GameState
from .evaluator import evaluate_state, score_action_prior


@dataclass(frozen=True)
class PlannerConfig:
    max_depth: int = 3
    beam_width: int = 6
    opponent_branch_width: int = 3
    include_opponent_turn: bool = True


class StandardTurnPlanner:
    def __init__(self, config: PlannerConfig | None = None) -> None:
        self.config = config or PlannerConfig()
        self._nodes_evaluated = 0

    def plan(
        self,
        state: GameState,
        acting_player_index: int,
        legal_actions: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        legal_actions = legal_actions or list_legal_actions(state, player_index=acting_player_index)
        if not legal_actions:
            raise ValueError("Planner needs at least one legal action.")

        baseline_score = evaluate_state(state, acting_player_index)
        self._nodes_evaluated = 0
        ranked_actions = self._rank_actions(state, acting_player_index, legal_actions, acting_player_index)
        candidates: list[dict[str, Any]] = []
        for action in ranked_actions[: self.config.beam_width]:
            simulated_state = deepcopy(state)
            apply_action_for_player(simulated_state, action, acting_player_index)
            score, line = self._search(simulated_state, acting_player_index, depth=1)
            candidates.append(
                {
                    "action": action,
                    "action_id": _safe_action_id(action),
                    "score": round(score, 6),
                    "delta": round(score - baseline_score, 6),
                    "line": [_safe_action_id(action), *line],
                }
            )

        best = max(candidates, key=lambda candidate: (candidate["score"], candidate["action_id"]))
        return {
            "chosen_action": best["action"],
            "chosen_action_id": best["action_id"],
            "planned_action_sequence": best["line"],
            "diagnostics": {
                "planner": "beam_minimax",
                "baseline_score": round(baseline_score, 6),
                "nodes_evaluated": self._nodes_evaluated,
                "max_depth": self.config.max_depth,
                "beam_width": self.config.beam_width,
                "top_candidates": [
                    {
                        "action_id": candidate["action_id"],
                        "score": candidate["score"],
                        "delta": candidate["delta"],
                        "line": candidate["line"],
                    }
                    for candidate in sorted(
                        candidates,
                        key=lambda candidate: (-candidate["score"], candidate["action_id"]),
                    )[:3]
                ],
            },
        }

    def _search(
        self,
        state: GameState,
        root_player_index: int,
        depth: int,
    ) -> tuple[float, list[str]]:
        self._nodes_evaluated += 1
        if state.winner is not None or depth >= self.config.max_depth:
            return evaluate_state(state, root_player_index), []

        acting_player_index = state.current_player
        if acting_player_index != root_player_index and not self.config.include_opponent_turn:
            return evaluate_state(state, root_player_index), []

        legal_actions = list_legal_actions(state, player_index=acting_player_index)
        if not legal_actions:
            return evaluate_state(state, root_player_index), []

        ranked_actions = self._rank_actions(
            state,
            acting_player_index,
            legal_actions,
            root_player_index,
        )
        branch_width = (
            self.config.beam_width
            if acting_player_index == root_player_index
            else self.config.opponent_branch_width
        )
        best_score = -inf if acting_player_index == root_player_index else inf
        best_line: list[str] = []
        for action in ranked_actions[:branch_width]:
            simulated_state = deepcopy(state)
            apply_action_for_player(simulated_state, action, acting_player_index)
            score, line = self._search(simulated_state, root_player_index, depth + 1)
            if acting_player_index == root_player_index:
                if score > best_score:
                    best_score = score
                    best_line = [_safe_action_id(action), *line]
            else:
                if score < best_score:
                    best_score = score
                    best_line = [_safe_action_id(action), *line]

        return round(best_score, 6), best_line

    def _rank_actions(
        self,
        state: GameState,
        acting_player_index: int,
        legal_actions: list[dict[str, Any]],
        root_player_index: int,
    ) -> list[dict[str, Any]]:
        baseline_score = evaluate_state(state, root_player_index)
        ranked: list[tuple[float, str, dict[str, Any]]] = []
        for action in legal_actions:
            simulated_state = deepcopy(state)
            apply_action_for_player(simulated_state, action, acting_player_index)
            score = evaluate_state(simulated_state, root_player_index)
            prior = score_action_prior(state, acting_player_index, action)
            ranked.append(
                (
                    round(score + prior * 0.1 + (score - baseline_score) * 0.3, 6),
                    _safe_action_id(action),
                    action,
                )
            )
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return [action for _, _, action in ranked]


def _safe_action_id(action: dict[str, Any]) -> str:
    try:
        return action_id_for(action)
    except Exception:
        return str(action)
