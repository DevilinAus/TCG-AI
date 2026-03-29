from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass
from math import inf
from typing import Any

from ..action_analysis import analyze_legal_actions, planner_adjustment_for_analysis
from ..engine import action_id_for, apply_action_for_player, list_legal_actions
from ..models import GameState
from .oracle import (
    HeuristicPolicyValueOracle,
    PolicyValueOracle,
    PolicyValueRequest,
)
from .profiling import (
    DecisionProfile,
    current_profile,
    observe_max,
    record_counter,
    time_metric,
    use_profile,
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
    successor_action_analysis_by_id: dict[str, dict[str, Any]]
    analysis: dict[str, Any]


class StandardTurnPlanner:
    def __init__(
        self,
        config: PlannerConfig | None = None,
        oracle: PolicyValueOracle | None = None,
    ) -> None:
        self.config = config or PlannerConfig()
        self.oracle = oracle or HeuristicPolicyValueOracle()
        self._nodes_evaluated = 0
        self._analysis_cache: dict[tuple[int, int, tuple[str, ...]], dict[str, dict[str, Any]]] = {}

    def plan(
        self,
        state: GameState,
        acting_player_index: int,
        legal_actions: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        legal_actions = legal_actions or list_legal_actions(state, player_index=acting_player_index)
        if not legal_actions:
            raise ValueError("Planner needs at least one legal action.")

        profile = current_profile() or DecisionProfile()
        profile_context = use_profile(profile) if current_profile() is None else nullcontext(profile)
        with profile_context:
            self._nodes_evaluated = 0
            self._analysis_cache = {}
            record_counter("planner.plan.calls")
            observe_max("planner.max_legal_actions", len(legal_actions))
            observe_max("planner.max_depth_config", self.config.max_depth)
            observe_max("planner.max_beam_width_config", self.config.beam_width)
            observe_max("planner.max_opponent_branch_width_config", self.config.opponent_branch_width)
            with time_metric("planner.plan_total"):
                root_analysis = self._analysis_for_state(
                    state,
                    acting_player_index=acting_player_index,
                    legal_actions=legal_actions,
                    profile_metric="planner.root_analysis",
                )
                baseline_score, ranked_actions = self._rank_actions(
                    state,
                    acting_player_index,
                    legal_actions,
                    acting_player_index,
                    analysis_by_action_id=root_analysis,
                )
                candidates: list[dict[str, Any]] = []
                with time_metric("planner.root_search_total"):
                    for ranked_action in ranked_actions[: self.config.beam_width]:
                        score, line = self._search(
                            ranked_action.successor_state,
                            acting_player_index,
                            depth=1,
                            legal_actions=ranked_action.successor_legal_actions,
                            analysis_by_action_id=ranked_action.successor_action_analysis_by_id,
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
                                "reason_tags": list(ranked_action.analysis.get("reason_tags") or []),
                                "reason_summary": ranked_action.analysis.get("reason_summary"),
                                "penalty_breakdown": dict(ranked_action.analysis.get("penalty_breakdown") or {}),
                                "dominance_context": dict(ranked_action.analysis.get("dominance_context") or {}),
                            }
                        )

                best_score = max(candidate["score"] for candidate in candidates)
                best = next(candidate for candidate in candidates if candidate["score"] == best_score)
                policy_target_scores = _build_policy_target_scores(
                    legal_action_ids=[ranked_action.action_id for ranked_action in ranked_actions],
                    candidates=candidates,
                )
            performance_profile = profile.snapshot()
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
                        "reason_tags": candidate["reason_tags"],
                        "reason_summary": candidate["reason_summary"],
                        "penalty_breakdown": candidate["penalty_breakdown"],
                        "dominance_context": candidate["dominance_context"],
                    }
                    for candidate in sorted(
                        candidates,
                        key=lambda candidate: (-candidate["score"], candidate["action_id"]),
                    )[:3]
                ],
                "reason_tags": best["reason_tags"],
                "reason_summary": best["reason_summary"],
                "penalty_breakdown": best["penalty_breakdown"],
                "dominance_context": best["dominance_context"],
                "policy_target_scores": policy_target_scores,
                "performance_profile": performance_profile,
            },
        }

    def _search(
        self,
        state: GameState,
        root_player_index: int,
        depth: int,
        legal_actions: list[dict[str, Any]] | None = None,
        analysis_by_action_id: dict[str, dict[str, Any]] | None = None,
        current_score: float | None = None,
    ) -> tuple[float, list[str]]:
        record_counter("planner.search.calls")
        record_counter(f"planner.search.depth_{depth}.calls")
        observe_max("planner.max_depth_reached", depth)
        self._nodes_evaluated += 1
        record_counter("planner.nodes_evaluated")
        if state.winner is not None or depth >= self.config.max_depth:
            return (
                self._evaluate_state_value(
                    state,
                    state.current_player,
                    root_player_index,
                    legal_actions=legal_actions,
                    analysis_by_action_id=analysis_by_action_id,
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
                    analysis_by_action_id=analysis_by_action_id,
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
                    analysis_by_action_id=analysis_by_action_id,
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
            analysis_by_action_id=analysis_by_action_id,
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
                analysis_by_action_id=ranked_action.successor_action_analysis_by_id,
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
        analysis_by_action_id: dict[str, dict[str, Any]] | None = None,
    ) -> tuple[float, list[RankedAction]]:
        record_counter("planner.rank_actions.calls")
        observe_max("planner.rank_actions.max_legal_actions", len(legal_actions))
        with time_metric("planner.rank_actions.total"):
            if analysis_by_action_id is None:
                analysis_by_action_id = self._analysis_for_state(
                    state,
                    acting_player_index=acting_player_index,
                    legal_actions=legal_actions,
                    profile_metric="planner.rank_actions.analysis",
                )
            oracle_result = self._evaluate_requests(
                [
                    PolicyValueRequest(
                        state=state,
                        acting_player_index=acting_player_index,
                        root_player_index=root_player_index,
                        legal_actions=legal_actions,
                        action_analysis_by_id=analysis_by_action_id,
                    )
                ]
            )[0]
            baseline_score = float(oracle_result.value) if baseline_score is None else float(baseline_score)
            ranked: list[RankedAction] = []
            simulated_actions: list[
                tuple[dict[str, Any], str, GameState, int, list[dict[str, Any]], dict[str, dict[str, Any]]]
            ] = []
            for action in legal_actions:
                record_counter("planner.simulated_actions")
                with time_metric("planner.simulation.deepcopy"):
                    simulated_state = deepcopy(state)
                with time_metric("planner.simulation.apply_action"):
                    apply_action_for_player(simulated_state, action, acting_player_index)
                action_id = _safe_action_id(action)
                next_player_index = simulated_state.current_player
                with time_metric("planner.simulation.legal_actions"):
                    next_legal_actions = list_legal_actions(
                        simulated_state,
                        player_index=next_player_index,
                    )
                next_analysis_by_id = self._analysis_for_state(
                    simulated_state,
                    acting_player_index=next_player_index,
                    legal_actions=next_legal_actions,
                    profile_metric="planner.simulation.analysis",
                )
                simulated_actions.append(
                    (
                        action,
                        action_id,
                        simulated_state,
                        next_player_index,
                        next_legal_actions,
                        next_analysis_by_id,
                    )
                )
            one_step_requests = [
                PolicyValueRequest(
                    state=simulated_state,
                    acting_player_index=next_player_index,
                    root_player_index=root_player_index,
                    legal_actions=next_legal_actions,
                    action_analysis_by_id=next_analysis_by_id,
                )
                for _, _, simulated_state, next_player_index, next_legal_actions, next_analysis_by_id in simulated_actions
            ]
            one_step_results = self._evaluate_requests(one_step_requests)
            for (
                action,
                action_id,
                simulated_state,
                _next_player_index,
                _next_legal_actions,
                next_analysis_by_id,
            ), one_step_result in zip(simulated_actions, one_step_results):
                one_step_score = float(one_step_result.value)
                prior = float(oracle_result.action_priors.get(action_id, 0.0))
                analysis = analysis_by_action_id.get(action_id, {})
                planner_adjustment, penalty_breakdown = planner_adjustment_for_analysis(analysis)
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
                    + self._continuation_rank_bonus(one_step_score, continuation_score)
                    + planner_adjustment,
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
                        successor_action_analysis_by_id=next_analysis_by_id,
                        analysis={
                            **analysis,
                            "penalty_breakdown": penalty_breakdown,
                            "planner_adjustment": planner_adjustment,
                        },
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
        record_counter("planner.same_turn_continuation.calls")
        observe_max("planner.same_turn_continuation.max_steps", max_steps)
        with time_metric("planner.same_turn_continuation.total"):
            best_score = (
                float(current_score)
                if current_score is not None
                else self._evaluate_state_value(state, state.current_player, root_player_index)
            )
            if max_steps <= 0 or state.winner is not None or state.current_player != acting_player_index:
                return best_score

            with time_metric("planner.same_turn_continuation.legal_actions"):
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
            observe_max("planner.same_turn_continuation.max_followups", len(ordered_followups))
            followup_states: list[tuple[dict[str, Any], GameState, int, list[dict[str, Any]]]] = []
            for followup in ordered_followups:
                record_counter("planner.same_turn_continuation.followups")
                with time_metric("planner.same_turn_continuation.deepcopy"):
                    simulated_state = deepcopy(state)
                with time_metric("planner.same_turn_continuation.apply_action"):
                    apply_action_for_player(simulated_state, followup, acting_player_index)
                next_player_index = simulated_state.current_player
                with time_metric("planner.same_turn_continuation.legal_actions"):
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
                    action_analysis_by_id=self._analysis_for_state(
                        simulated_state,
                        acting_player_index=next_player_index,
                        legal_actions=next_legal_actions,
                        profile_metric="planner.same_turn_continuation.analysis",
                    ),
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
        analysis_by_action_id: dict[str, dict[str, Any]] | None = None,
        current_score: float | None = None,
    ) -> float:
        if current_score is not None:
            record_counter("planner.evaluate_state_value.cached")
            return float(current_score)
        with time_metric("planner.evaluate_state_value.total"):
            if legal_actions is None:
                with time_metric("planner.evaluate_state_value.legal_actions"):
                    legal_actions = list_legal_actions(state, player_index=acting_player_index)
            result = self._evaluate_requests(
                [
                    PolicyValueRequest(
                        state=state,
                        acting_player_index=acting_player_index,
                        root_player_index=root_player_index,
                        legal_actions=legal_actions,
                        action_analysis_by_id=analysis_by_action_id,
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
        record_counter("planner.oracle.calls")
        record_counter("planner.oracle.requests", len(requests))
        observe_max("planner.oracle.max_batch_size", len(requests))
        with time_metric("planner.oracle.total"):
            return self.oracle.evaluate_batch(requests)

    def _analysis_for_state(
        self,
        state: GameState,
        *,
        acting_player_index: int,
        legal_actions: list[dict[str, Any]],
        profile_metric: str,
    ) -> dict[str, dict[str, Any]]:
        action_ids = tuple(_safe_action_id(action) for action in legal_actions)
        cache_key = (id(state), acting_player_index, action_ids)
        cached = self._analysis_cache.get(cache_key)
        if cached is not None:
            record_counter("planner.analysis_cache_hits")
            return cached
        record_counter("planner.analysis_cache_misses")
        with time_metric(profile_metric):
            analysis = analyze_legal_actions(
                state,
                acting_player_index=acting_player_index,
                legal_actions=legal_actions,
            )
        self._analysis_cache[cache_key] = analysis
        return analysis


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
