from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Protocol

from ..engine import action_id_for
from ..models import GameState
from .evaluator import evaluate_state, score_action_prior
from .knowledge_state import serialize_knowledge_actions, serialize_knowledge_state
from .neural_policy import PolicyValueBackend
from .profiling import observe_max, record_counter, set_metadata, time_metric


@dataclass(frozen=True)
class PolicyValueRequest:
    state: GameState
    acting_player_index: int
    root_player_index: int
    legal_actions: list[dict[str, object]]
    action_analysis_by_id: dict[str, dict[str, object]] | None = None


@dataclass(frozen=True)
class PolicyValueResult:
    value: float
    action_priors: dict[str, float]
    diagnostics: dict[str, object]


class PolicyValueOracle(Protocol):
    def evaluate_batch(self, requests: list[PolicyValueRequest]) -> list[PolicyValueResult]:
        raise NotImplementedError


class HeuristicPolicyValueOracle:
    def evaluate_batch(self, requests: list[PolicyValueRequest]) -> list[PolicyValueResult]:
        return [_evaluate_request(request) for request in requests]


class BackendPolicyValueOracle:
    def __init__(self, backend: PolicyValueBackend | None = None) -> None:
        self.backend = backend or PolicyValueBackend()

    def evaluate_batch(self, requests: list[PolicyValueRequest]) -> list[PolicyValueResult]:
        record_counter("oracle.evaluate_batch.calls")
        record_counter("oracle.requests", len(requests))
        observe_max("oracle.max_batch_size", len(requests))
        set_metadata("oracle.backend", self.backend.status.backend)
        payload = []
        with time_metric("oracle.evaluate_batch.total"):
            for request in requests:
                with time_metric("oracle.serialize_state"):
                    belief_state = serialize_knowledge_state(
                        request.state,
                        perspective_player_index=request.acting_player_index,
                    )
                with time_metric("oracle.serialize_actions"):
                    legal_actions = serialize_knowledge_actions(
                        request.state,
                        acting_player_index=request.acting_player_index,
                        legal_actions=request.legal_actions,
                        analysis_by_action_id=request.action_analysis_by_id,
                    )
                payload.append(
                    {
                        "acting_player_index": request.acting_player_index,
                        "root_player_index": request.root_player_index,
                        "belief_state": belief_state,
                        "legal_actions": legal_actions,
                    }
                )
            with time_metric("oracle.backend.evaluate_batch"):
                responses = self.backend.evaluate_batch(payload)
        return [
            PolicyValueResult(
                value=float(response.get("value", 0.0)),
                action_priors={
                    str(action_id): float(prior)
                    for action_id, prior in (response.get("action_priors") or {}).items()
                },
                diagnostics=dict(response.get("diagnostics") or {}),
            )
            for response in responses
        ]


def _evaluate_request(request: PolicyValueRequest) -> PolicyValueResult:
    logits: list[tuple[str, float]] = []
    for action in request.legal_actions:
        action_id = action_id_for(action)
        logits.append(
            (
                action_id,
                float(score_action_prior(request.state, request.acting_player_index, action)),
            )
        )
    priors = _softmax(logits)
    return PolicyValueResult(
        value=round(evaluate_state(request.state, request.root_player_index), 6),
        action_priors=priors,
        diagnostics={"source": "heuristic_oracle"},
    )


def _softmax(logits: list[tuple[str, float]]) -> dict[str, float]:
    if not logits:
        return {}
    max_logit = max(score for _, score in logits)
    weights = [(action_id, exp(score - max_logit)) for action_id, score in logits]
    total = sum(weight for _, weight in weights)
    if total <= 0:
        uniform = 1.0 / len(weights)
        return {action_id: uniform for action_id, _ in weights}
    return {
        action_id: round(weight / total, 6)
        for action_id, weight in weights
    }
