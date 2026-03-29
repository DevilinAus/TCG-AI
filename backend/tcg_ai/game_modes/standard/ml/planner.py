from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from math import inf
from typing import Any

from ..engine import action_id_for, apply_action_for_player, list_legal_actions
from ..models import GameState
from .oracle import (
    HeuristicPolicyValueOracle,
    PolicyValueOracle,
    PolicyValueRequest,
)


@dataclass(frozen=True)
class PlannerConfig:
    max_depth: int = 3
    beam_width: int = 6
    opponent_branch_width: int = 3
    include_opponent_turn: bool = True


@dataclass(frozen=True)
class RankedAction:
    action: dict[str, Any]
    action_id: str
    rank_score: float
    prior: float
    one_step_score: float
    continuation_score: float
    successor_state: GameState
    successor_legal_actions: list[dict[str, Any]]


class StandardTurnPlanner:
    def __init__(
        self,
        config: PlannerConfig | None = None,
        oracle: PolicyValueOracle | None = None,
    ) -> None:
        self.config = config or PlannerConfig()
        self.oracle = oracle or HeuristicPolicyValueOracle()
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

        self._nodes_evaluated = 0
        baseline_score, ranked_actions = self._rank_actions(
            state,
            acting_player_index,
            legal_actions,
            acting_player_index,
        )
        candidates: list[dict[str, Any]] = []
        for ranked_action in ranked_actions[: self.config.beam_width]:
            score, line = self._search(
                ranked_action.successor_state,
                acting_player_index,
                depth=1,
                legal_actions=ranked_action.successor_legal_actions,
                current_score=ranked_action.one_step_score,
            )
            candidates.append(
                {
                    "action": ranked_action.action,
                    "action_id": ranked_action.action_id,
                    "score": round(score, 6),
                    "delta": round(score - baseline_score, 6),
                    "line": [ranked_action.action_id, *line],
                    "rank_score": round(ranked_action.rank_score, 6),
                    "prior": round(ranked_action.prior, 6),
                    "one_step_score": round(ranked_action.one_step_score, 6),
                    "continuation_score": round(ranked_action.continuation_score, 6),
                }
            )

        best_score = max(candidate["score"] for candidate in candidates)
        best = next(candidate for candidate in candidates if candidate["score"] == best_score)
        policy_target_scores = _build_policy_target_scores(
            legal_action_ids=[ranked_action.action_id for ranked_action in ranked_actions],
            candidates=candidates,
        )
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
                "policy_target_scores": policy_target_scores,
            },
        }

    def _search(
        self,
        state: GameState,
        root_player_index: int,
        depth: int,
        legal_actions: list[dict[str, Any]] | None = None,
        current_score: float | None = None,
    ) -> tuple[float, list[str]]:
        self._nodes_evaluated += 1
        if state.winner is not None or depth >= self.config.max_depth:
            return (
                self._evaluate_state_value(
                    state,
                    state.current_player,
                    root_player_index,
                    legal_actions=legal_actions,
                    current_score=current_score,
                ),
                [],
            )

        acting_player_index = state.current_player
        if acting_player_index != root_player_index and not self.config.include_opponent_turn:
            return (
                self._evaluate_state_value(
                    state,
                    acting_player_index,
                    root_player_index,
                    legal_actions=legal_actions,
                    current_score=current_score,
                ),
                [],
            )

        legal_actions = legal_actions or list_legal_actions(state, player_index=acting_player_index)
        if not legal_actions:
            return (
                self._evaluate_state_value(
                    state,
                    acting_player_index,
                    root_player_index,
                    legal_actions=legal_actions,
                    current_score=current_score,
                ),
                [],
            )

        _, ranked_actions = self._rank_actions(
            state,
            acting_player_index,
            legal_actions,
            root_player_index,
            baseline_score=current_score,
        )
        branch_width = (
            self.config.beam_width
            if acting_player_index == root_player_index
            else self.config.opponent_branch_width
        )
        best_score = -inf if acting_player_index == root_player_index else inf
        best_line: list[str] = []
        for ranked_action in ranked_actions[:branch_width]:
            score, line = self._search(
                ranked_action.successor_state,
                root_player_index,
                depth + 1,
                legal_actions=ranked_action.successor_legal_actions,
                current_score=ranked_action.one_step_score,
            )
            if acting_player_index == root_player_index:
                if score > best_score:
                    best_score = score
                    best_line = [ranked_action.action_id, *line]
            else:
                if score < best_score:
                    best_score = score
                    best_line = [ranked_action.action_id, *line]

        return round(best_score, 6), best_line

    def _rank_actions(
        self,
        state: GameState,
        acting_player_index: int,
        legal_actions: list[dict[str, Any]],
        root_player_index: int,
        baseline_score: float | None = None,
    ) -> tuple[float, list[RankedAction]]:
        oracle_result = self._evaluate_requests(
            [
                PolicyValueRequest(
                    state=state,
                    acting_player_index=acting_player_index,
                    root_player_index=root_player_index,
                    legal_actions=legal_actions,
                )
            ]
        )[0]
        baseline_score = float(oracle_result.value) if baseline_score is None else float(baseline_score)
        ranked: list[RankedAction] = []
        simulated_actions: list[tuple[dict[str, Any], str, GameState, int, list[dict[str, Any]]]] = []
        for action in legal_actions:
            simulated_state = deepcopy(state)
            apply_action_for_player(simulated_state, action, acting_player_index)
            action_id = _safe_action_id(action)
            next_player_index = simulated_state.current_player
            next_legal_actions = list_legal_actions(
                simulated_state,
                player_index=next_player_index,
            )
            simulated_actions.append(
                (
                    action,
                    action_id,
                    simulated_state,
                    next_player_index,
                    next_legal_actions,
                )
            )
        one_step_requests = [
            PolicyValueRequest(
                state=simulated_state,
                acting_player_index=next_player_index,
                root_player_index=root_player_index,
                legal_actions=next_legal_actions,
            )
            for _, _, simulated_state, next_player_index, next_legal_actions in simulated_actions
        ]
        one_step_results = self._evaluate_requests(one_step_requests)
        for (
            action,
            action_id,
            simulated_state,
            _next_player_index,
            _next_legal_actions,
        ), one_step_result in zip(simulated_actions, one_step_results):
            one_step_score = float(one_step_result.value)
            prior = float(oracle_result.action_priors.get(action_id, 0.0))
            continuation_score = self._same_turn_continuation_score(
                simulated_state,
                acting_player_index=acting_player_index,
                root_player_index=root_player_index,
                max_steps=2,
                current_score=one_step_score,
            )
            rank_score = round(
                one_step_score
                + prior * 8.0
                + (one_step_score - baseline_score) * 0.3
                + self._continuation_rank_bonus(one_step_score, continuation_score),
                6,
            )
            ranked.append(
                RankedAction(
                    action=action,
                    action_id=action_id,
                    rank_score=rank_score,
                    prior=prior,
                    one_step_score=one_step_score,
                    continuation_score=continuation_score,
                    successor_state=simulated_state,
                    successor_legal_actions=_next_legal_actions,
                )
            )
        ranked.sort(key=lambda item: (-item.rank_score, item.action_id))
        return baseline_score, ranked

    def _same_turn_continuation_score(
        self,
        state: GameState,
        *,
        acting_player_index: int,
        root_player_index: int,
        max_steps: int,
        current_score: float | None = None,
    ) -> float:
        best_score = (
            float(current_score)
            if current_score is not None
            else self._evaluate_state_value(state, state.current_player, root_player_index)
        )
        if max_steps <= 0 or state.winner is not None or state.current_player != acting_player_index:
            return best_score

        legal_actions = list_legal_actions(state, player_index=acting_player_index)
        if not legal_actions:
            return best_score

        tactical_followups = [
            action
            for action in legal_actions
            if str(action.get("type", "")) in {
                "attack",
                "evolve",
                "play_energy",
                "retreat",
                "bench_basic",
                "play_basic_to_active",
            }
        ]
        ordered_followups = sorted(
            tactical_followups,
            key=lambda action: (
                -_same_turn_followup_priority(action),
                _safe_action_id(action),
            ),
        )[:6]
        followup_states: list[tuple[dict[str, Any], GameState, int, list[dict[str, Any]]]] = []
        for followup in ordered_followups:
            simulated_state = deepcopy(state)
            apply_action_for_player(simulated_state, followup, acting_player_index)
            next_player_index = simulated_state.current_player
            next_legal_actions = list_legal_actions(
                simulated_state,
                player_index=next_player_index,
            )
            followup_states.append(
                (
                    followup,
                    simulated_state,
                    next_player_index,
                    next_legal_actions,
                )
            )
        followup_requests = [
            PolicyValueRequest(
                state=simulated_state,
                acting_player_index=next_player_index,
                root_player_index=root_player_index,
                legal_actions=next_legal_actions,
            )
            for _, simulated_state, next_player_index, next_legal_actions in followup_states
        ]
        followup_results = self._evaluate_requests(followup_requests)
        for (_, simulated_state, _next_player_index, _next_legal_actions), followup_result in zip(
            followup_states,
            followup_results,
        ):
            followup_score = self._same_turn_continuation_score(
                simulated_state,
                acting_player_index=acting_player_index,
                root_player_index=root_player_index,
                max_steps=max_steps - 1,
                current_score=float(followup_result.value),
            )
            if followup_score > best_score:
                best_score = followup_score
        return round(best_score, 6)

    @staticmethod
    def _continuation_rank_bonus(one_step_score: float, continuation_score: float) -> float:
        if continuation_score >= 9_000.0:
            return 1_000.0
        if continuation_score <= -9_000.0:
            return -120.0
        delta = continuation_score - one_step_score
        if delta <= 0:
            return 0.0
        return min(24.0, delta * 0.45)

    def _evaluate_state_value(
        self,
        state: GameState,
        acting_player_index: int,
        root_player_index: int,
        *,
        legal_actions: list[dict[str, Any]] | None = None,
        current_score: float | None = None,
    ) -> float:
        if current_score is not None:
            return float(current_score)
        legal_actions = legal_actions or list_legal_actions(state, player_index=acting_player_index)
        result = self._evaluate_requests(
            [
                PolicyValueRequest(
                    state=state,
                    acting_player_index=acting_player_index,
                    root_player_index=root_player_index,
                    legal_actions=legal_actions,
                )
            ]
        )[0]
        return float(result.value)

    def _evaluate_requests(
        self,
        requests: list[PolicyValueRequest],
    ):
        if not requests:
            return []
        return self.oracle.evaluate_batch(requests)


def _safe_action_id(action: dict[str, Any]) -> str:
    try:
        return action_id_for(action)
    except Exception:
        return str(action)


def _same_turn_followup_priority(action: dict[str, Any]) -> int:
    priorities = {
        "attack": 60,
        "evolve": 50,
        "play_energy": 40,
        "retreat": 30,
        "bench_basic": 20,
        "play_basic_to_active": 15,
    }
    return priorities.get(str(action.get("type", "")), 0)


def _build_policy_target_scores(
    *,
    legal_action_ids: list[str],
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    candidate_by_id = {
        str(candidate["action_id"]): candidate
        for candidate in candidates
    }
    if candidate_by_id:
        floor_score = min(float(candidate["score"]) for candidate in candidates) - 5.0
    else:
        floor_score = -5.0
    targets: list[dict[str, Any]] = []
    for action_id in legal_action_ids:
        candidate = candidate_by_id.get(action_id)
        if candidate is not None:
            targets.append(
                {
                    "action_id": action_id,
                    "score": round(float(candidate["score"]), 6),
                    "source": "search",
                }
            )
            continue
        targets.append(
            {
                "action_id": action_id,
                "score": round(floor_score, 6),
                "source": "pruned",
            }
        )
    return targets
