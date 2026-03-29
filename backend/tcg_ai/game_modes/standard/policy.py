from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from time import perf_counter
from typing import Any
from urllib import error, request as urllib_request
from urllib.parse import urlsplit, urlunsplit

from ...logging_utils import get_logger
from .decision_payload import build_decision_request
from .engine import action_id_for, apply_action_for_player, card_definition, list_legal_actions
from .ml.canonical_state import serialize_state
from .ml.planner import PlannerConfig
from .models import GameState
from .policy_store import OpenerPolicyStats, StandardPolicyStore

DEFAULT_REMOTE_TIMEOUT_MS = 1_800_000
DEFAULT_EXPLORATION_RATE = 0.20
DEFAULT_MIN_EXPLORATION_RATE = 0.05
FULL_STATE_REQUEST_SCHEMA_VERSION = 2
logger = get_logger(__name__)


@dataclass(frozen=True)
class StandardPolicyConfig:
    remote_enabled: bool = False
    remote_url: str | None = None
    remote_batch_eval_url: str | None = None
    remote_timeout_ms: int = DEFAULT_REMOTE_TIMEOUT_MS
    remote_api_token: str | None = None
    exploration_rate: float = DEFAULT_EXPLORATION_RATE
    min_exploration_rate: float = DEFAULT_MIN_EXPLORATION_RATE

    @classmethod
    def from_env(cls) -> "StandardPolicyConfig":
        raw_enabled = os.environ.get("TCG_AI_STANDARD_REMOTE_ENABLED", "")
        remote_enabled = raw_enabled.lower() in {"1", "true", "yes", "on"}
        remote_url = os.environ.get("TCG_AI_STANDARD_REMOTE_URL")
        remote_batch_eval_url = os.environ.get("TCG_AI_STANDARD_REMOTE_BATCH_EVAL_URL")
        remote_api_token = os.environ.get("TCG_AI_STANDARD_REMOTE_API_TOKEN")
        try:
            remote_timeout_ms = int(
                os.environ.get("TCG_AI_STANDARD_REMOTE_TIMEOUT_MS", str(DEFAULT_REMOTE_TIMEOUT_MS))
            )
        except ValueError:
            remote_timeout_ms = DEFAULT_REMOTE_TIMEOUT_MS
        return cls(
            remote_enabled=remote_enabled and bool(remote_url),
            remote_url=remote_url,
            remote_batch_eval_url=remote_batch_eval_url,
            remote_timeout_ms=max(100, remote_timeout_ms),
            remote_api_token=remote_api_token or None,
        )

    def resolved_remote_batch_eval_url(self) -> str | None:
        if self.remote_batch_eval_url:
            return self.remote_batch_eval_url
        if not self.remote_url:
            return None
        if self.remote_url.endswith("/decision"):
            return f"{self.remote_url[:-len('/decision')]}/batch-eval"
        return f"{self.remote_url.rstrip('/')}/batch-eval"

    def resolved_remote_ready_url(self) -> str | None:
        worker_root = self._resolved_remote_worker_root()
        if worker_root is None:
            return None
        path = worker_root.path.rstrip("/")
        return urlunsplit(worker_root._replace(path=f"{path}/readyz" if path else "/readyz"))

    def _resolved_remote_worker_root(self):
        candidate_url = self.remote_batch_eval_url or self.remote_url
        if not candidate_url:
            return None
        parsed = urlsplit(candidate_url)
        path = parsed.path or ""
        for suffix in (
            "/api/standard-ml/decision",
            "/api/standard-ml/batch-eval",
            "/api/standard-ml/outcome",
            "/decision",
            "/batch-eval",
            "/outcome",
        ):
            if path.endswith(suffix):
                path = path[: -len(suffix)]
                break
        else:
            path = path.rstrip("/")
        return parsed._replace(path=path, query="", fragment="")


@dataclass(frozen=True)
class DecisionRequest:
    state: GameState
    acting_player_index: int
    decision_type: str
    decision_id: str
    legal_actions: list[dict[str, Any]]
    payload: dict[str, Any]


@dataclass(frozen=True)
class DecisionResult:
    chosen_action: dict[str, Any]
    action_id: str
    source: str
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PendingDecisionTrace:
    trace_id: str
    decision_id: str
    decision_type: str
    trainer_id: str
    ai_deck_id: str
    chosen_card_id: str
    chosen_action_id: str
    turn_number: int
    source: str


@dataclass(frozen=True)
class DecisionOutcome:
    terminal_reward: float
    winner: int | None
    learner_player_index: int = 1


@dataclass
class StandardDecisionRuntime:
    provider: "StandardDecisionProvider"
    _decision_counter: int = 0

    def next_decision_id(
        self,
        decision_type: str,
        turn_number: int,
        player_index: int,
    ) -> str:
        self._decision_counter += 1
        return (
            f"{self.provider.session_id}:{decision_type}:"
            f"turn{turn_number}:p{player_index}:d{self._decision_counter}"
        )


class StandardRemoteDecisionError(RuntimeError):
    """Raised when the remote policy cannot provide a valid action."""


class StandardDecisionProvider:
    provider_type = "base"

    def __init__(
        self,
        *,
        trainer_id: str,
        ai_deck_id: str,
        session_id: str,
        policy_store: StandardPolicyStore,
        config: StandardPolicyConfig,
    ) -> None:
        self.trainer_id = trainer_id
        self.ai_deck_id = ai_deck_id
        self.session_id = session_id
        self.policy_store = policy_store
        self.config = config
        self.pending_traces: dict[str, PendingDecisionTrace] = {}
        self.last_decision: dict[str, Any] | None = None

    def choose_action(self, request: DecisionRequest) -> DecisionResult:
        raise NotImplementedError

    def record_pending(self, trace: PendingDecisionTrace) -> None:
        self.pending_traces[trace.trace_id] = trace

    def finalize_outcome(self, trace_id: str, outcome: DecisionOutcome) -> bool:
        trace = self.pending_traces.pop(trace_id, None)
        if trace is None:
            return False
        self.policy_store.record_opener_outcome(
            trainer_id=trace.trainer_id,
            ai_deck_id=trace.ai_deck_id,
            chosen_card_id=trace.chosen_card_id,
            terminal_reward=outcome.terminal_reward,
            did_win=outcome.winner == outcome.learner_player_index,
        )
        return True

    def current_exploration_rate(self) -> float:
        deck_stats = self.policy_store.stats_for_deck(self.trainer_id, self.ai_deck_id)
        total_resolved = sum(stats.resolved_samples for stats in deck_stats.values())
        if total_resolved <= 0:
            return self.config.exploration_rate
        decayed_rate = self.config.exploration_rate / (total_resolved ** 0.5)
        return max(self.config.min_exploration_rate, round(decayed_rate, 6))

    def snapshot(self) -> dict[str, Any]:
        deck_stats = self.policy_store.stats_for_deck(self.trainer_id, self.ai_deck_id)
        return {
            "provider_type": self.provider_type,
            "remote_enabled": bool(self.config.remote_enabled and self.config.remote_url),
            "current_exploration_rate": round(self.current_exploration_rate(), 6),
            "pending_traces": [
                {
                    "trace_id": trace.trace_id,
                    "decision_id": trace.decision_id,
                    "decision_type": trace.decision_type,
                    "chosen_card_id": trace.chosen_card_id,
                    "chosen_action_id": trace.chosen_action_id,
                    "turn_number": trace.turn_number,
                    "source": trace.source,
                }
                for trace in sorted(self.pending_traces.values(), key=lambda item: item.trace_id)
            ],
            "opener_stats": {
                card_id: stats.snapshot()
                for card_id, stats in sorted(deck_stats.items(), key=lambda item: item[0])
            },
            "last_decision": self.last_decision,
        }

    def _remember_decision(self, request: DecisionRequest, result: DecisionResult) -> None:
        chosen_card_id = _chosen_card_id_for_action(request.state, result.chosen_action)
        self.last_decision = {
            "decision_id": request.decision_id,
            "decision_type": request.decision_type,
            "chosen_action_id": result.action_id,
            "chosen_card_id": chosen_card_id,
            "source": result.source,
            "diagnostics": result.diagnostics,
        }


class LocalStandardDecisionProvider(StandardDecisionProvider):
    provider_type = "local"

    def choose_action(self, request: DecisionRequest) -> DecisionResult:
        if not request.legal_actions:
            raise ValueError("Standard decision request has no legal actions.")

        deck_stats = self.policy_store.stats_for_deck(self.trainer_id, self.ai_deck_id)
        candidate_stats = {
            action_id_for(action): deck_stats.get(_action_card_id(request.state, action), OpenerPolicyStats())
            for action in request.legal_actions
        }
        has_resolved_data = any(stats.resolved_samples > 0 for stats in candidate_stats.values())
        should_explore = (
            not has_resolved_data
            or request.state.rng.random() < self.current_exploration_rate()
        )
        if should_explore:
            chosen_action = request.state.rng.choice(request.legal_actions)
            result = DecisionResult(
                chosen_action=chosen_action,
                action_id=action_id_for(chosen_action),
                source="local",
                diagnostics={
                    "selection_mode": "explore",
                    "resolved_candidates": sum(
                        1 for stats in candidate_stats.values() if stats.resolved_samples > 0
                    ),
                },
            )
            self._remember_decision(request, result)
            return result

        ranked_actions = sorted(
            request.legal_actions,
            key=lambda action: _exploit_sort_key(
                deck_stats.get(_action_card_id(request.state, action), OpenerPolicyStats()),
                _action_card_id(request.state, action),
            ),
        )
        chosen_action = ranked_actions[0]
        chosen_card_id = _action_card_id(request.state, chosen_action)
        chosen_stats = deck_stats.get(chosen_card_id, OpenerPolicyStats())
        result = DecisionResult(
            chosen_action=chosen_action,
            action_id=action_id_for(chosen_action),
            source="local",
            diagnostics={
                "selection_mode": "exploit",
                "resolved_samples": chosen_stats.resolved_samples,
                "average_terminal_reward": round(chosen_stats.average_terminal_reward, 6),
                "win_rate": round(chosen_stats.win_rate, 6),
            },
        )
        self._remember_decision(request, result)
        return result


class RemoteStandardDecisionProvider(StandardDecisionProvider):
    provider_type = "remote"

    def choose_action(self, request: DecisionRequest) -> DecisionResult:
        if not self.config.remote_enabled or not self.config.remote_url:
            raise StandardRemoteDecisionError("Remote Standard policy is disabled.")

        body = json.dumps(request.payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.config.remote_api_token:
            headers["X-Standard-ML-Token"] = self.config.remote_api_token
        http_request = urllib_request.Request(
            self.config.remote_url,
            data=body,
            headers=headers,
            method="POST",
        )
        start_time = perf_counter()
        logger.info(
            "remote decision request start session=%s decision_id=%s type=%s turn=%s player=%s url=%s timeout_ms=%s legal_action_count=%s",
            self.session_id,
            request.decision_id,
            request.decision_type,
            request.state.turn_number,
            request.acting_player_index,
            self.config.remote_url,
            self.config.remote_timeout_ms,
            len(request.legal_actions),
        )
        try:
            with urllib_request.urlopen(
                http_request,
                timeout=self.config.remote_timeout_ms / 1000,
            ) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (TimeoutError, error.URLError, json.JSONDecodeError, OSError) as exc:
            elapsed_ms = round((perf_counter() - start_time) * 1000, 1)
            logger.error(
                "remote decision request failed session=%s decision_id=%s type=%s timeout_ms=%s elapsed_ms=%s error=%s",
                self.session_id,
                request.decision_id,
                request.decision_type,
                self.config.remote_timeout_ms,
                elapsed_ms,
                exc,
            )
            raise StandardRemoteDecisionError(str(exc)) from exc

        if not isinstance(payload, dict):
            elapsed_ms = round((perf_counter() - start_time) * 1000, 1)
            logger.error(
                "remote decision request malformed session=%s decision_id=%s elapsed_ms=%s payload_type=%s",
                self.session_id,
                request.decision_id,
                elapsed_ms,
                type(payload).__name__,
            )
            raise StandardRemoteDecisionError("Remote Standard policy returned a malformed payload.")
        if payload.get("decision_id") != request.decision_id:
            elapsed_ms = round((perf_counter() - start_time) * 1000, 1)
            logger.error(
                "remote decision request wrong-id session=%s decision_id=%s elapsed_ms=%s returned_decision_id=%s",
                self.session_id,
                request.decision_id,
                elapsed_ms,
                payload.get("decision_id"),
            )
            raise StandardRemoteDecisionError("Remote Standard policy returned the wrong decision ID.")

        chosen_action_id = payload.get("chosen_action_id")
        if not isinstance(chosen_action_id, str) or not chosen_action_id:
            elapsed_ms = round((perf_counter() - start_time) * 1000, 1)
            logger.error(
                "remote decision request missing-action session=%s decision_id=%s elapsed_ms=%s payload=%s",
                self.session_id,
                request.decision_id,
                elapsed_ms,
                payload,
            )
            raise StandardRemoteDecisionError("Remote Standard policy omitted chosen_action_id.")

        action_by_id = {
            action_id_for(action): action for action in request.legal_actions
        }
        chosen_action = action_by_id.get(chosen_action_id)
        if chosen_action is None:
            elapsed_ms = round((perf_counter() - start_time) * 1000, 1)
            logger.error(
                "remote decision request illegal-action session=%s decision_id=%s elapsed_ms=%s chosen_action_id=%s legal_action_ids=%s",
                self.session_id,
                request.decision_id,
                elapsed_ms,
                chosen_action_id,
                sorted(action_by_id),
            )
            raise StandardRemoteDecisionError("Remote Standard policy returned an illegal action.")

        diagnostics = payload.get("diagnostics")
        if not isinstance(diagnostics, dict):
            diagnostics = {}
        elapsed_ms = round((perf_counter() - start_time) * 1000, 1)
        logger.info(
            "remote decision request accepted session=%s decision_id=%s type=%s elapsed_ms=%s chosen_action_id=%s planned_action_sequence=%s reason=%s",
            self.session_id,
            request.decision_id,
            request.decision_type,
            elapsed_ms,
            chosen_action_id,
            payload.get("planned_action_sequence"),
            diagnostics.get("reason_summary"),
        )
        result = DecisionResult(
            chosen_action=chosen_action,
            action_id=chosen_action_id,
            source="remote",
            diagnostics=diagnostics,
        )
        self._remember_decision(request, result)
        return result


class FallbackStandardDecisionProvider(StandardDecisionProvider):
    provider_type = "fallback"

    def __init__(
        self,
        *,
        trainer_id: str,
        ai_deck_id: str,
        session_id: str,
        policy_store: StandardPolicyStore,
        config: StandardPolicyConfig,
    ) -> None:
        super().__init__(
            trainer_id=trainer_id,
            ai_deck_id=ai_deck_id,
            session_id=session_id,
            policy_store=policy_store,
            config=config,
        )
        self._local = LocalStandardDecisionProvider(
            trainer_id=trainer_id,
            ai_deck_id=ai_deck_id,
            session_id=session_id,
            policy_store=policy_store,
            config=config,
        )
        self._remote = RemoteStandardDecisionProvider(
            trainer_id=trainer_id,
            ai_deck_id=ai_deck_id,
            session_id=session_id,
            policy_store=policy_store,
            config=config,
        )

    def choose_action(self, request: DecisionRequest) -> DecisionResult:
        if self.config.remote_enabled and self.config.remote_url:
            result = self._remote.choose_action(request)
            self._remember_decision(request, result)
            return result

        result = self._local.choose_action(request)
        self._remember_decision(request, result)
        return result


def initialize_session(
    state: GameState,
    *,
    session_id: str,
    trainer_id: str,
    ai_deck_id: str | None,
    policy_store: StandardPolicyStore | None = None,
    policy_config: StandardPolicyConfig | None = None,
    **_: Any,
) -> StandardDecisionRuntime | None:
    if ai_deck_id is None:
        return None

    runtime = StandardDecisionRuntime(
        provider=FallbackStandardDecisionProvider(
            trainer_id=trainer_id,
            ai_deck_id=ai_deck_id,
            session_id=session_id,
            policy_store=policy_store or StandardPolicyStore(),
            config=policy_config or StandardPolicyConfig(),
        )
    )
    legal_actions = [
        action
        for action in list_legal_actions(state, player_index=1)
        if action["type"] == "play_basic_to_active"
    ]
    if not legal_actions:
        return runtime

    decision_id = runtime.next_decision_id("opening_active", state.turn_number, 1)
    decision_request = DecisionRequest(
        state=state,
        acting_player_index=1,
        decision_type="opening_active",
        decision_id=decision_id,
        legal_actions=legal_actions,
        payload=build_decision_request(
            state,
            session_id=session_id,
            decision_id=decision_id,
            decision_type="opening_active",
            acting_player_index=1,
            ai_trainer_id=trainer_id,
            ai_deck_id=ai_deck_id,
            legal_actions=legal_actions,
        ),
    )
    decision_result = runtime.provider.choose_action(decision_request)
    chosen_card_id = _action_card_id(state, decision_result.chosen_action)
    runtime.provider.record_pending(
        PendingDecisionTrace(
            trace_id=decision_id,
            decision_id=decision_id,
            decision_type="opening_active",
            trainer_id=trainer_id,
            ai_deck_id=ai_deck_id,
            chosen_card_id=chosen_card_id,
            chosen_action_id=decision_result.action_id,
            turn_number=state.turn_number,
            source=decision_result.source,
        )
    )
    apply_action_for_player(state, decision_result.chosen_action, 1)
    return runtime


def runtime_snapshot(runtime: StandardDecisionRuntime | None) -> dict[str, Any]:
    if runtime is None:
        return {}
    return runtime.provider.snapshot()


def build_turn_action_request(
    state: GameState,
    *,
    runtime: StandardDecisionRuntime,
    acting_player_index: int,
    legal_actions: list[dict[str, Any]],
) -> DecisionRequest:
    decision_id = runtime.next_decision_id("turn_action", state.turn_number, acting_player_index)
    payload = {
        "schema_version": FULL_STATE_REQUEST_SCHEMA_VERSION,
        "decision_id": decision_id,
        "decision_type": "turn_action",
        "session_id": runtime.provider.session_id,
        "turn_number": state.turn_number,
        "acting_player_index": acting_player_index,
        "search_config": _serialize_planner_config(PlannerConfig()),
        "state": serialize_state(state),
    }
    return DecisionRequest(
        state=state,
        acting_player_index=acting_player_index,
        decision_type="turn_action",
        decision_id=decision_id,
        legal_actions=legal_actions,
        payload=payload,
    )


def _action_card_id(state: GameState, action: dict[str, Any]) -> str:
    if action["type"] != "play_basic_to_active":
        raise ValueError(f"Unsupported Standard policy action type: {action['type']}")
    return card_definition(state, action["hand_card_id"]).card_id


def _chosen_card_id_for_action(state: GameState, action: dict[str, Any]) -> str | None:
    if action.get("type") == "play_basic_to_active":
        return _action_card_id(state, action)
    hand_card_id = action.get("hand_card_id")
    if isinstance(hand_card_id, str):
        try:
            return card_definition(state, hand_card_id).card_id
        except Exception:
            return None
    return None


def _serialize_planner_config(config: PlannerConfig) -> dict[str, Any]:
    return {
        "max_depth": int(config.max_depth),
        "beam_width": int(config.beam_width),
        "opponent_branch_width": int(config.opponent_branch_width),
        "include_opponent_turn": bool(config.include_opponent_turn),
    }


def _exploit_sort_key(stats: OpenerPolicyStats, card_id: str) -> tuple[float, float, int, str]:
    return (
        -round(stats.average_terminal_reward, 6),
        -round(stats.win_rate, 6),
        -stats.resolved_samples,
        card_id,
    )
