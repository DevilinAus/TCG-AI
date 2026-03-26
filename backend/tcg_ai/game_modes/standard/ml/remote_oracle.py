from __future__ import annotations

import json
from typing import Any
from urllib import error, request as urllib_request

from .knowledge_state import serialize_knowledge_actions, serialize_knowledge_state
from .oracle import PolicyValueOracle, PolicyValueRequest, PolicyValueResult


class RemotePolicyValueOracleError(RuntimeError):
    """Raised when the remote policy/value worker cannot provide predictions."""


class RemotePolicyValueOracle(PolicyValueOracle):
    def __init__(
        self,
        *,
        batch_eval_url: str,
        timeout_ms: int,
        api_token: str | None = None,
        session_id: str | None = None,
    ) -> None:
        self.batch_eval_url = batch_eval_url
        self.timeout_ms = timeout_ms
        self.api_token = api_token
        self.session_id = session_id

    def evaluate_batch(self, requests: list[PolicyValueRequest]) -> list[PolicyValueResult]:
        payload = {
            "schema_version": 1,
            "session_id": self.session_id,
            "evaluations": [
                {
                    "acting_player_index": request.acting_player_index,
                    "root_player_index": request.root_player_index,
                    "belief_state": serialize_knowledge_state(
                        request.state,
                        perspective_player_index=request.acting_player_index,
                    ),
                    "legal_actions": serialize_knowledge_actions(
                        request.state,
                        acting_player_index=request.acting_player_index,
                        legal_actions=request.legal_actions,
                    ),
                }
                for request in requests
            ],
        }
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_token:
            headers["X-Standard-ML-Token"] = self.api_token
        http_request = urllib_request.Request(
            self.batch_eval_url,
            data=body,
            headers=headers,
            method="POST",
        )
        try:
            with urllib_request.urlopen(http_request, timeout=self.timeout_ms / 1000) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except (TimeoutError, error.URLError, json.JSONDecodeError, OSError) as exc:
            raise RemotePolicyValueOracleError(str(exc)) from exc

        if not isinstance(response_payload, dict):
            raise RemotePolicyValueOracleError("Remote batch-eval returned a malformed payload.")
        raw_evaluations = response_payload.get("evaluations")
        if not isinstance(raw_evaluations, list) or len(raw_evaluations) != len(requests):
            raise RemotePolicyValueOracleError("Remote batch-eval returned the wrong number of results.")

        results: list[PolicyValueResult] = []
        for evaluation in raw_evaluations:
            if not isinstance(evaluation, dict):
                raise RemotePolicyValueOracleError("Remote batch-eval result was malformed.")
            action_priors = evaluation.get("action_priors", {})
            if not isinstance(action_priors, dict):
                action_priors = {}
            diagnostics = evaluation.get("diagnostics", {})
            if not isinstance(diagnostics, dict):
                diagnostics = {}
            results.append(
                PolicyValueResult(
                    value=float(evaluation.get("value", 0.0) or 0.0),
                    action_priors={
                        str(action_id): float(prior)
                        for action_id, prior in action_priors.items()
                    },
                    diagnostics=diagnostics,
                )
            )
        return results
