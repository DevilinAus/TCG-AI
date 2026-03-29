from __future__ import annotations

from dataclasses import dataclass
from math import exp, tanh
import os
from pathlib import Path
import pickle
from typing import Any

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover - torch is optional in the test environment
    torch = None
    nn = None


MAX_BENCH_SIZE = 5
POKEMON_SLOT_VECTOR_SIZE = 14
STATE_VECTOR_SIZE = 204
ACTION_VECTOR_SIZE = 72
DEFAULT_HIDDEN_SIZE = 128
ENCODER_VERSION = 3
DEFAULT_CHECKPOINT_PATH = Path(__file__).resolve().parents[5] / "standard_ml_data" / "champion.pt"
_TORCH_BASE = nn.Module if nn is not None else object


@dataclass(frozen=True)
class PolicyValueBackendStatus:
    backend: str
    model_loaded: bool
    checkpoint_path: str | None


class ActionConditionedPolicyValueNet(_TORCH_BASE):  # type: ignore[misc]
    def __init__(self, state_dim: int = STATE_VECTOR_SIZE, action_dim: int = ACTION_VECTOR_SIZE) -> None:
        if nn is None:
            raise RuntimeError("PyTorch is not available.")
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, DEFAULT_HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(DEFAULT_HIDDEN_SIZE, DEFAULT_HIDDEN_SIZE),
            nn.ReLU(),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, DEFAULT_HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(DEFAULT_HIDDEN_SIZE, DEFAULT_HIDDEN_SIZE),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(DEFAULT_HIDDEN_SIZE * 2, 1)
        self.value_head = nn.Linear(DEFAULT_HIDDEN_SIZE, 1)

    def forward(self, state_vector: torch.Tensor, action_vectors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        state_hidden = self.state_encoder(state_vector)
        action_hidden = self.action_encoder(action_vectors)
        if state_hidden.ndim == 1:
            repeated_state = state_hidden.unsqueeze(0).expand(action_hidden.shape[0], -1)
            policy_logits = self.policy_head(torch.cat([repeated_state, action_hidden], dim=1)).squeeze(-1)
            value = torch.tanh(self.value_head(state_hidden)).squeeze(-1) * 100.0
            return policy_logits, value
        if state_hidden.ndim == 2:
            repeated_state = state_hidden.unsqueeze(1).expand(-1, action_hidden.shape[1], -1)
            policy_logits = self.policy_head(
                torch.cat([repeated_state, action_hidden], dim=-1)
            ).squeeze(-1)
            value = torch.tanh(self.value_head(state_hidden)).squeeze(-1) * 100.0
            return policy_logits, value
        raise ValueError("Unsupported ActionConditionedPolicyValueNet input shape.")


class PolicyValueBackend:
    def __init__(self, checkpoint_path: Path | None = None) -> None:
        self.checkpoint_path = checkpoint_path or Path(
            os.environ.get("TCG_AI_STANDARD_MODEL_CHECKPOINT", DEFAULT_CHECKPOINT_PATH)
        )
        self._model = None
        self._state_dim = STATE_VECTOR_SIZE
        self._action_dim = ACTION_VECTOR_SIZE
        self._status = PolicyValueBackendStatus(
            backend="heuristic",
            model_loaded=False,
            checkpoint_path=str(self.checkpoint_path) if self.checkpoint_path else None,
        )
        self._try_load_model()

    @property
    def status(self) -> PolicyValueBackendStatus:
        return self._status

    def evaluate_batch(self, evaluations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self._model is None or torch is None:
            return [self._heuristic_evaluation(evaluation) for evaluation in evaluations]
        return self._model_evaluate_batch(evaluations)

    def _try_load_model(self) -> None:
        if torch is None or nn is None or self.checkpoint_path is None or not self.checkpoint_path.exists():
            return
        device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = load_trusted_checkpoint(self.checkpoint_path, map_location=device)
        state_dim, action_dim = infer_checkpoint_model_dimensions(checkpoint)
        model = ActionConditionedPolicyValueNet(state_dim=state_dim, action_dim=action_dim)
        state_dict = checkpoint.get("state_dict") if isinstance(checkpoint, dict) else checkpoint
        model.load_state_dict(state_dict)
        model.eval()
        model.to(device)
        self._model = model
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._status = PolicyValueBackendStatus(
            backend=f"torch:{device}",
            model_loaded=True,
            checkpoint_path=str(self.checkpoint_path),
        )

    def _model_evaluation(self, evaluation: dict[str, Any]) -> dict[str, Any]:
        belief_state = evaluation.get("belief_state", {})
        state_vector = torch.tensor(
            encode_state_vector(belief_state, vector_size=self._state_dim),
            dtype=torch.float32,
            device=next(self._model.parameters()).device,
        )
        legal_actions = list(evaluation.get("legal_actions", []))
        if not legal_actions:
            with torch.no_grad():
                state_hidden = self._model.state_encoder(state_vector)
                value = torch.tanh(self._model.value_head(state_hidden)).squeeze(-1) * 100.0
            return {
                "value": round(float(value.detach().cpu().item()), 6),
                "action_priors": {},
                "diagnostics": {
                    "backend": self._status.backend,
                    "model_loaded": self._status.model_loaded,
                },
            }
        action_vectors = torch.tensor(
            [
                encode_action_vector(action, belief_state=belief_state, vector_size=self._action_dim)
                for action in legal_actions
            ],
            dtype=torch.float32,
            device=state_vector.device,
        )
        with torch.no_grad():
            policy_logits, value = self._model(state_vector, action_vectors)
        priors = _softmax_from_logits(
            [float(logit) for logit in policy_logits.detach().cpu().tolist()],
            [str(action.get("action_id", "")) for action in legal_actions],
        )
        return {
            "value": round(float(value.detach().cpu().item()), 6),
            "action_priors": priors,
            "diagnostics": {
                "backend": self._status.backend,
                "model_loaded": self._status.model_loaded,
            },
        }

    def _model_evaluate_batch(self, evaluations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not evaluations:
            return []

        model = self._model
        device = next(model.parameters()).device
        encoded_state_vectors: list[list[float]] = []
        action_ids_by_evaluation: list[list[str]] = []
        encoded_action_vectors: list[list[list[float]]] = []
        max_action_count = 0

        for evaluation in evaluations:
            belief_state = evaluation.get("belief_state", {})
            encoded_state_vectors.append(
                encode_state_vector(belief_state, vector_size=self._state_dim)
            )
            legal_actions = list(evaluation.get("legal_actions", []))
            action_ids = [str(action.get("action_id", "")) for action in legal_actions]
            action_vectors = [
                encode_action_vector(action, belief_state=belief_state, vector_size=self._action_dim)
                for action in legal_actions
            ]
            action_ids_by_evaluation.append(action_ids)
            encoded_action_vectors.append(action_vectors)
            max_action_count = max(max_action_count, len(action_vectors))

        state_tensor = torch.tensor(
            encoded_state_vectors,
            dtype=torch.float32,
            device=device,
        )

        with torch.no_grad():
            if max_action_count > 0:
                padded_action_vectors = [
                    action_vectors
                    + ([[0.0] * self._action_dim] * (max_action_count - len(action_vectors)))
                    for action_vectors in encoded_action_vectors
                ]
                action_tensor = torch.tensor(
                    padded_action_vectors,
                    dtype=torch.float32,
                    device=device,
                )
                policy_logits, values = model(state_tensor, action_tensor)
                policy_logits_rows = policy_logits.detach().cpu().tolist()
            else:
                state_hidden = model.state_encoder(state_tensor)
                values = torch.tanh(model.value_head(state_hidden)).squeeze(-1) * 100.0
                policy_logits_rows = []

        value_rows = values.detach().cpu().tolist()
        results: list[dict[str, Any]] = []
        for index, action_ids in enumerate(action_ids_by_evaluation):
            priors = {}
            if action_ids:
                logits = [float(logit) for logit in policy_logits_rows[index][: len(action_ids)]]
                priors = _softmax_from_logits(logits, action_ids)
            results.append(
                {
                    "value": round(float(value_rows[index]), 6),
                    "action_priors": priors,
                    "diagnostics": {
                        "backend": self._status.backend,
                        "model_loaded": self._status.model_loaded,
                    },
                }
            )
        return results

    def _heuristic_evaluation(self, evaluation: dict[str, Any]) -> dict[str, Any]:
        belief_state = evaluation.get("belief_state", {})
        legal_actions = list(evaluation.get("legal_actions", []))
        value = _heuristic_value_from_belief_state(belief_state)
        priors = _heuristic_action_priors(legal_actions)
        return {
            "value": round(value, 6),
            "action_priors": priors,
            "diagnostics": {
                "backend": self._status.backend,
                "model_loaded": self._status.model_loaded,
            },
        }


def load_trusted_checkpoint(checkpoint_path: Path, *, map_location: str | torch.device) -> Any:
    if torch is None:
        raise RuntimeError("PyTorch is not available.")
    try:
        return torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location=map_location)
    except pickle.UnpicklingError as exc:
        if "Weights only load failed" not in str(exc):
            raise
        return torch.load(checkpoint_path, map_location=map_location, weights_only=False)


def infer_checkpoint_model_dimensions(checkpoint: Any) -> tuple[int, int]:
    if isinstance(checkpoint, dict):
        model_config = checkpoint.get("model_config")
        if isinstance(model_config, dict):
            state_dim = int(model_config.get("state_dim", 0) or 0)
            action_dim = int(model_config.get("action_dim", 0) or 0)
            if state_dim > 0 and action_dim > 0:
                return state_dim, action_dim
        state_dict = checkpoint.get("state_dict")
    else:
        state_dict = checkpoint

    if isinstance(state_dict, dict):
        state_weight = state_dict.get("state_encoder.0.weight")
        action_weight = state_dict.get("action_encoder.0.weight")
        if state_weight is not None and action_weight is not None:
            try:
                return int(state_weight.shape[1]), int(action_weight.shape[1])
            except Exception:
                pass
    return STATE_VECTOR_SIZE, ACTION_VECTOR_SIZE


def _heuristic_value_from_belief_state(belief_state: dict[str, Any]) -> float:
    players = belief_state.get("players", [{}, {}])
    player = players[0] if len(players) > 0 else {}
    opponent = players[1] if len(players) > 1 else {}
    derived = belief_state.get("derived_features", {})
    value = 0.0
    value += (6 - float(player.get("prize_count", 6))) * 30.0
    value -= (6 - float(opponent.get("prize_count", 6))) * 30.0
    value += _player_remaining_hp(player) * 0.16
    value -= _player_remaining_hp(opponent) * 0.16
    value += _player_energy_total(player) * 2.4
    value -= _player_energy_total(opponent) * 2.4
    value += len(player.get("bench", [])) * 4.0
    value -= len(opponent.get("bench", [])) * 4.0
    value += float(player.get("hand_count", 0)) * 1.2
    value -= float(opponent.get("hand_count", 0)) * 0.9
    if derived.get("player_active_likely_knockout_next_turn"):
        value -= 18.0
    if derived.get("opponent_active_likely_knockout_next_turn"):
        value += 18.0
    value += max(0, 4 - int(derived.get("player_active_turns_until_ready", 4) or 4)) * 2.5
    value -= max(0, 4 - int(derived.get("opponent_active_turns_until_ready", 4) or 4)) * 2.5
    return value


def _heuristic_action_priors(legal_actions: list[dict[str, Any]]) -> dict[str, float]:
    logits: list[tuple[str, float]] = []
    for action in legal_actions:
        action_id = str(action.get("action_id", ""))
        action_type = str(action.get("type", ""))
        score = {
            "attack": 5.0,
            "play_supporter": 3.0,
            "play_item": 2.4,
            "play_energy": 2.0,
            "evolve": 1.8,
            "bench_basic": 1.3,
            "retreat": 0.8,
            "play_basic_to_active": 3.2,
            "end_setup": 0.4,
            "end_turn": -1.5,
            "mulligan": -1.0,
        }.get(action_type, 0.0)
        score += len(action.get("search_selection", [])) * 0.6
        score += len(action.get("discard_from_hand", [])) * -0.2
        if action.get("consumes_supporter_for_turn"):
            score += 0.2
        if action.get("consumes_attachment_for_turn"):
            score += 0.3
        source_card = action.get("source_card") or {}
        effect_specs = source_card.get("effect_specs", []) if isinstance(source_card, dict) else []
        for effect_spec in effect_specs:
            if effect_spec.get("effect_type") == "draw":
                score += float(effect_spec.get("count", 0) or 0) * 0.15
            if effect_spec.get("effect_type") == "search_deck":
                score += 0.5
        logits.append((action_id, score))
    return _softmax_logits(logits)


def encode_state_vector(
    belief_state: dict[str, Any],
    *,
    vector_size: int = STATE_VECTOR_SIZE,
) -> list[float]:
    players = belief_state.get("players", [{}, {}])
    player = players[0] if len(players) > 0 else {}
    opponent = players[1] if len(players) > 1 else {}
    derived = belief_state.get("derived_features", {})
    vector = [
        _cap(float(belief_state.get("turn_number", 1)), 20) / 20.0,
        1.0 if belief_state.get("current_player") == belief_state.get("perspective_player_index") else 0.0,
        1.0 if belief_state.get("starting_player") == belief_state.get("perspective_player_index") else 0.0,
        1.0 if belief_state.get("winner") == belief_state.get("perspective_player_index") else 0.0,
        1.0 if belief_state.get("winner") == (1 - int(belief_state.get("perspective_player_index", 0))) else 0.0,
        1.0 if belief_state.get("setup_phase") is not None else 0.0,
        1.0 if derived.get("player_active_likely_knockout_next_turn") else 0.0,
        1.0 if derived.get("opponent_active_likely_knockout_next_turn") else 0.0,
        _cap(float(derived.get("player_energy_at_risk_on_active", 0)), 6) / 6.0,
        _cap(float(derived.get("opponent_energy_at_risk_on_active", 0)), 6) / 6.0,
    ]
    vector.extend(_encode_player_summary(player, derived.get("player_board_investment")))
    vector.extend(_encode_player_slots(player))
    vector.extend(_encode_player_summary(opponent, derived.get("opponent_board_investment")))
    vector.extend(_encode_player_slots(opponent))
    return _fit_vector_size(vector, vector_size)


def encode_action_vector(
    action: dict[str, Any],
    *,
    belief_state: dict[str, Any] | None = None,
    vector_size: int = ACTION_VECTOR_SIZE,
) -> list[float]:
    belief_state = belief_state if isinstance(belief_state, dict) else {}
    source_card = action.get("source_card") or {}
    action_type = str(action.get("type", ""))
    target = action.get("target") if isinstance(action.get("target"), dict) else {}
    effect_tags = {str(tag) for tag in action.get("effect_tags", []) if isinstance(tag, str)}
    resource_costs = action.get("resource_costs") if isinstance(action.get("resource_costs"), dict) else {}
    expected_state_delta = (
        action.get("expected_state_delta") if isinstance(action.get("expected_state_delta"), dict) else {}
    )
    target_pokemon = _resolve_target_pokemon(belief_state, action)
    player = _player_payload(belief_state, 0)
    active = player.get("active") if isinstance(player, dict) else None
    best_ready_before = _best_turns_until_ready(player)
    target_ready_before = _turns_until_ready_for_pokemon(target_pokemon)
    target_energy_before = _attached_energy_count_for_pokemon(target_pokemon)
    target_retreat_cost = _retreat_cost_for_pokemon(target_pokemon)
    cheapest_attack_cost = _pokemon_cheapest_attack_cost(target_pokemon)
    cheapest_remaining_cost = _pokemon_cheapest_remaining_cost(target_pokemon)
    projected_remaining_cost = _project_remaining_cost_after_action(action, target_pokemon)
    projected_energy = _project_energy_after_action(action, target_pokemon)
    projected_excess_energy = max(0.0, projected_energy - cheapest_attack_cost) if cheapest_attack_cost > 0 else 0.0
    projected_attack_live = 1.0 if _action_creates_live_attack(action, target_pokemon) else 0.0
    active_ready_before = _turns_until_ready_for_pokemon(active)
    active_ready_after = _project_active_turns_after_action(belief_state, action)
    target_player_relative = _relative_target_player_index(belief_state, target.get("player_index"))
    retreat_target_ready = (
        _turns_until_ready_for_pokemon(target_pokemon)
        if action_type in {"retreat", "promote"}
        else None
    )
    vector = [
        1.0 if action_type == "attack" else 0.0,
        1.0 if action_type == "play_supporter" else 0.0,
        1.0 if action_type == "play_item" else 0.0,
        1.0 if action_type == "play_energy" else 0.0,
        1.0 if action_type == "evolve" else 0.0,
        1.0 if action_type == "bench_basic" else 0.0,
        1.0 if action_type == "play_basic_to_active" else 0.0,
        1.0 if action_type == "end_turn" else 0.0,
        1.0 if action_type == "end_setup" else 0.0,
        1.0 if action_type == "retreat" else 0.0,
        1.0 if action_type == "promote" else 0.0,
        1.0 if action_type == "mulligan" else 0.0,
        1.0 if source_card.get("kind") == "pokemon" else 0.0,
        1.0 if source_card.get("kind") == "trainer" else 0.0,
        1.0 if source_card.get("kind") == "energy" else 0.0,
        1.0 if source_card.get("is_basic") else 0.0,
        1.0 if source_card.get("stage") == "stage1" else 0.0,
        1.0 if source_card.get("stage") == "stage2" else 0.0,
        1.0 if source_card.get("is_basic_energy") else 0.0,
        _cap(float(source_card.get("prize_card_value", 0) or 0), 3) / 3.0,
        1.0 if target.get("zone") == "active" else 0.0,
        1.0 if target.get("zone") == "bench" else 0.0,
        1.0 if target_player_relative == 0 else 0.0,
        1.0 if target_player_relative == 1 else 0.0,
        1.0 if target.get("bench_index") == 0 else 0.0,
        1.0 if target.get("bench_index") == 1 else 0.0,
        1.0 if target.get("bench_index") == 2 else 0.0,
        1.0 if target.get("bench_index") == 3 else 0.0,
        1.0 if target.get("bench_index") == 4 else 0.0,
        _cap(float(resource_costs.get("discard_from_hand_count", 0)), 3) / 3.0,
        _cap(float(resource_costs.get("discard_attached_energy_count", 0)), 4) / 4.0,
        _cap(float(resource_costs.get("recover_from_discard_count", 0)), 3) / 3.0,
        _cap(float(resource_costs.get("search_selection_count", 0)), 2) / 2.0,
        _cap(float(resource_costs.get("attack_energy_cost", 0)), 5) / 5.0,
        _cap(float(resource_costs.get("retreat_energy_cost", 0)), 4) / 4.0,
        1.0 if resource_costs.get("hand_card_count", 0) else 0.0,
        _cap(float(expected_state_delta.get("cards_drawn_known", 0) or 0), 5) / 5.0,
        1.0 if action.get("consumes_supporter_for_turn") else 0.0,
        1.0 if action.get("consumes_attachment_for_turn") else 0.0,
        1.0 if action.get("consumes_retreat_for_turn") else 0.0,
        _normalize_signed(float(expected_state_delta.get("hand_count_delta_known", 0) or 0), limit=7),
        _normalize_signed(float(expected_state_delta.get("bench_count_delta", 0) or 0), limit=2),
        _normalize_signed(float(expected_state_delta.get("discard_count_delta_known", 0) or 0), limit=4),
        1.0 if expected_state_delta.get("active_changes") else 0.0,
        1.0 if expected_state_delta.get("turn_ends") else 0.0,
        1.0 if "damage" in effect_tags else 0.0,
        1.0 if "draw" in effect_tags or "draw_cards" in effect_tags else 0.0,
        1.0 if "search_deck" in effect_tags else 0.0,
        1.0 if "attach_energy" in effect_tags else 0.0,
        1.0 if "evolution" in effect_tags else 0.0,
        1.0 if "switch_active" in effect_tags else 0.0,
        1.0 if "promotion" in effect_tags else 0.0,
        1.0 if "bench_development" in effect_tags else 0.0,
        1.0 if "active_development" in effect_tags else 0.0,
        1.0 if "recover_from_discard" in effect_tags else 0.0,
        1.0 if "hidden_information" in effect_tags else 0.0,
        1.0 if "hand_gain" in effect_tags else 0.0,
        _cap(float(target_energy_before), 6) / 6.0,
        _turn_ratio_or_zero(target_ready_before),
        _cap(float(target_retreat_cost), 4) / 4.0,
        1.0 if _pokemon_can_attack_now(target_pokemon) else 0.0,
        projected_attack_live,
        1.0 if _improves_turns_until_ready(active_ready_before, active_ready_after) else 0.0,
        1.0
        if target_player_relative == 0 and _improves_best_attacker_readiness(best_ready_before, projected_remaining_cost)
        else 0.0,
        1.0 if action_type == "play_energy" and target.get("zone") == "active" else 0.0,
        1.0 if action_type == "play_energy" and target.get("zone") == "bench" else 0.0,
        _cap(projected_excess_energy, 3) / 3.0,
        1.0 if projected_energy >= max(1, target_retreat_cost) and target_retreat_cost > 0 else 0.0,
        1.0 if action_type in {"retreat", "promote"} and _pokemon_can_attack_now(target_pokemon) else 0.0,
        _turn_ratio_or_zero(retreat_target_ready),
        1.0 if _search_selection_contains(action, kind="pokemon") else 0.0,
        1.0 if _search_selection_contains(action, kind="pokemon", is_basic=True) else 0.0,
    ]
    return _fit_vector_size(vector, vector_size)


def _encode_player_summary(player: dict[str, Any], board_investment: Any) -> list[float]:
    known_prizes = player.get("known_prize_cards_unordered", [])
    return [
        _cap(float(player.get("deck_count", 0)), 60) / 60.0,
        _cap(float(player.get("hand_count", 0)), 15) / 15.0,
        _cap(float(player.get("discard_count", 0)), 60) / 60.0,
        _cap(float(player.get("prize_count", 6)), 6) / 6.0,
        _cap(float(len(known_prizes)), 6) / 6.0,
        1.0 if player.get("deck_inspected_this_game") else 0.0,
        1.0 if player.get("supporter_played_this_turn") else 0.0,
        1.0 if player.get("energy_attached_this_turn") else 0.0,
        1.0 if player.get("retreated_this_turn") else 0.0,
        _cap(float(len(player.get("bench", []))), MAX_BENCH_SIZE) / float(MAX_BENCH_SIZE),
        _cap(_player_energy_total(player), 18) / 18.0,
        _cap(_player_remaining_hp(player), 900) / 900.0,
        _cap(float(board_investment or 0), 30) / 30.0,
    ]


def _encode_player_slots(player: dict[str, Any]) -> list[float]:
    vector: list[float] = []
    vector.extend(_encode_pokemon_slot(player.get("active")))
    bench = list(player.get("bench", []))
    for index in range(MAX_BENCH_SIZE):
        pokemon = bench[index] if index < len(bench) else None
        vector.extend(_encode_pokemon_slot(pokemon))
    return vector


def _encode_pokemon_slot(pokemon: Any) -> list[float]:
    if not isinstance(pokemon, dict):
        return [0.0] * POKEMON_SLOT_VECTOR_SIZE
    hp = float(pokemon.get("hp", 0) or 0)
    remaining_hp = float(pokemon.get("remaining_hp", 0) or 0)
    card = pokemon.get("card") if isinstance(pokemon.get("card"), dict) else {}
    return [
        1.0,
        _cap(remaining_hp, 350) / 350.0,
        _pokemon_hp_ratio(pokemon),
        _cap(float(pokemon.get("attached_energy_count", 0) or 0), 6) / 6.0,
        _cap(float(pokemon.get("retreat_cost", 0) or 0), 4) / 4.0,
        _turn_ratio(pokemon.get("turns_until_ready")),
        _cap(float(_pokemon_cheapest_attack_cost(pokemon)), 5) / 5.0,
        _cap(float(_pokemon_cheapest_remaining_cost(pokemon)), 5) / 5.0,
        _cap(float(_pokemon_max_attack_damage(pokemon)), 250) / 250.0,
        _cap(float(card.get("prize_card_value", 0) or 0), 3) / 3.0,
        1.0 if _pokemon_can_attack_now(pokemon) else 0.0,
        1.0 if card.get("is_basic") else 0.0,
        1.0 if card.get("stage") == "stage1" else 0.0,
        1.0 if card.get("stage") == "stage2" else 0.0,
    ]


def _player_remaining_hp(player: dict[str, Any]) -> float:
    total = 0.0
    active = player.get("active")
    bench = player.get("bench", [])
    for pokemon in [active, *bench]:
        if not isinstance(pokemon, dict):
            continue
        total += float(pokemon.get("remaining_hp", 0) or 0)
    return total


def _player_energy_total(player: dict[str, Any]) -> float:
    total = 0.0
    active = player.get("active")
    bench = player.get("bench", [])
    for pokemon in [active, *bench]:
        if not isinstance(pokemon, dict):
            continue
        total += float(pokemon.get("attached_energy_count", 0) or 0)
    return total


def _pokemon_hp_ratio(pokemon: dict[str, Any]) -> float:
    hp = float(pokemon.get("hp", 0) or 0)
    remaining = float(pokemon.get("remaining_hp", 0) or 0)
    if hp <= 0:
        return 0.0
    return remaining / hp


def _pokemon_energy_ratio(pokemon: dict[str, Any]) -> float:
    return _cap(float(pokemon.get("attached_energy_count", 0) or 0), 5) / 5.0


def _turn_ratio(turns_until_ready: Any) -> float:
    if turns_until_ready is None:
        return 1.0
    return _cap(float(turns_until_ready), 5) / 5.0


def _turn_ratio_or_zero(turns_until_ready: Any) -> float:
    if turns_until_ready is None:
        return 0.0
    return _turn_ratio(turns_until_ready)


def _normalize_signed(value: float, *, limit: int) -> float:
    capped = max(float(-limit), min(float(limit), value))
    return capped / float(limit)


def _fit_vector_size(vector: list[float], vector_size: int) -> list[float]:
    if len(vector) >= vector_size:
        return vector[:vector_size]
    return [*vector, *([0.0] * (vector_size - len(vector)))]


def _player_payload(belief_state: dict[str, Any], relative_index: int) -> dict[str, Any]:
    players = belief_state.get("players", [])
    if isinstance(players, list) and 0 <= relative_index < len(players):
        player = players[relative_index]
        if isinstance(player, dict):
            return player
    return {}


def _relative_target_player_index(belief_state: dict[str, Any], target_player_index: Any) -> int | None:
    perspective_player_index = belief_state.get("perspective_player_index")
    if isinstance(target_player_index, int) and isinstance(perspective_player_index, int):
        if target_player_index == perspective_player_index:
            return 0
        if target_player_index == 1 - perspective_player_index:
            return 1
    if isinstance(target_player_index, int) and target_player_index in {0, 1}:
        return target_player_index
    return None


def _resolve_target_pokemon(belief_state: dict[str, Any], action: dict[str, Any]) -> dict[str, Any] | None:
    action_type = str(action.get("type", ""))
    target = action.get("target") if isinstance(action.get("target"), dict) else {}
    players = belief_state.get("players", [])
    if not isinstance(players, list):
        return None

    relative_player_index = _relative_target_player_index(belief_state, target.get("player_index"))
    if relative_player_index is None and action_type in {"play_energy", "bench_basic", "play_basic_to_active", "evolve", "retreat", "promote"}:
        relative_player_index = 0
    if relative_player_index is None or not 0 <= relative_player_index < len(players):
        return None
    player = players[relative_player_index]
    if not isinstance(player, dict):
        return None

    if action_type == "promote":
        bench_index = target.get("bench_index")
        if isinstance(bench_index, int):
            bench = player.get("bench", [])
            if isinstance(bench, list) and 0 <= bench_index < len(bench) and isinstance(bench[bench_index], dict):
                return bench[bench_index]
        return None

    zone = target.get("zone")
    if zone == "active":
        active = player.get("active")
        return active if isinstance(active, dict) else None
    if zone == "bench":
        bench_index = target.get("bench_index")
        bench = player.get("bench", [])
        if isinstance(bench, list) and isinstance(bench_index, int) and 0 <= bench_index < len(bench):
            pokemon = bench[bench_index]
            return pokemon if isinstance(pokemon, dict) else None
    return None


def _turns_until_ready_for_pokemon(pokemon: Any) -> int | None:
    if not isinstance(pokemon, dict):
        return None
    value = pokemon.get("turns_until_ready")
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    return None


def _pokemon_cheapest_attack_cost(pokemon: Any) -> int:
    if not isinstance(pokemon, dict):
        return 0
    attacks = pokemon.get("attacks", [])
    if not isinstance(attacks, list) or not attacks:
        return 0
    costs = [int(attack.get("cost", 0) or 0) for attack in attacks if isinstance(attack, dict)]
    return min(costs) if costs else 0


def _pokemon_cheapest_remaining_cost(pokemon: Any) -> int:
    if not isinstance(pokemon, dict):
        return 0
    attacks = pokemon.get("attacks", [])
    if not isinstance(attacks, list) or not attacks:
        return 0
    remaining_costs = [int(attack.get("remaining_cost", 0) or 0) for attack in attacks if isinstance(attack, dict)]
    return min(remaining_costs) if remaining_costs else 0


def _pokemon_max_attack_damage(pokemon: Any) -> int:
    if not isinstance(pokemon, dict):
        return 0
    max_damage = 0
    for attack in pokemon.get("attacks", []):
        if not isinstance(attack, dict):
            continue
        damage_digits = "".join(character for character in str(attack.get("damage", "")) if character.isdigit())
        max_damage = max(max_damage, int(damage_digits or 0))
    return max_damage


def _pokemon_can_attack_now(pokemon: Any) -> bool:
    if not isinstance(pokemon, dict):
        return False
    attacks = pokemon.get("attacks", [])
    if not isinstance(attacks, list):
        return False
    return any(int(attack.get("remaining_cost", 0) or 0) <= 0 for attack in attacks if isinstance(attack, dict))


def _attached_energy_count_for_pokemon(pokemon: Any) -> int:
    if not isinstance(pokemon, dict):
        return 0
    return int(pokemon.get("attached_energy_count", 0) or 0)


def _retreat_cost_for_pokemon(pokemon: Any) -> int:
    if not isinstance(pokemon, dict):
        return 0
    return int(pokemon.get("retreat_cost", 0) or 0)


def _project_remaining_cost_after_action(action: dict[str, Any], target_pokemon: Any) -> int | None:
    if not isinstance(target_pokemon, dict):
        return None
    current_remaining = _pokemon_cheapest_remaining_cost(target_pokemon)
    if action.get("type") == "play_energy":
        return max(0, current_remaining - 1)
    if action.get("type") in {"retreat", "promote"}:
        return _turns_until_ready_for_pokemon(target_pokemon)
    return current_remaining


def _project_energy_after_action(action: dict[str, Any], target_pokemon: Any) -> int:
    current_energy = _attached_energy_count_for_pokemon(target_pokemon)
    if action.get("type") == "play_energy":
        return current_energy + 1
    return current_energy


def _action_creates_live_attack(action: dict[str, Any], target_pokemon: Any) -> bool:
    if action.get("type") == "attack":
        return True
    if not isinstance(target_pokemon, dict):
        return False
    current_remaining = _pokemon_cheapest_remaining_cost(target_pokemon)
    projected_remaining = _project_remaining_cost_after_action(action, target_pokemon)
    if projected_remaining is None:
        return False
    return current_remaining > 0 and projected_remaining == 0


def _project_active_turns_after_action(belief_state: dict[str, Any], action: dict[str, Any]) -> int | None:
    player = _player_payload(belief_state, 0)
    active = player.get("active") if isinstance(player, dict) else None
    if action.get("type") == "retreat":
        return _turns_until_ready_for_pokemon(_resolve_target_pokemon(belief_state, action))
    if action.get("type") == "play_energy":
        target = action.get("target") if isinstance(action.get("target"), dict) else {}
        if target.get("zone") == "active":
            return _project_remaining_cost_after_action(action, active)
    return _turns_until_ready_for_pokemon(active)


def _best_turns_until_ready(player: dict[str, Any]) -> int | None:
    pokemon_entries: list[Any] = [player.get("active")]
    bench = player.get("bench", [])
    if isinstance(bench, list):
        pokemon_entries.extend(bench)
    ready_values = [
        value
        for value in (_turns_until_ready_for_pokemon(pokemon) for pokemon in pokemon_entries)
        if value is not None
    ]
    return min(ready_values) if ready_values else None


def _improves_turns_until_ready(before: int | None, after: int | None) -> bool:
    if before is None or after is None:
        return False
    return after < before


def _improves_best_attacker_readiness(best_before: int | None, projected_target_ready: int | None) -> bool:
    if best_before is None or projected_target_ready is None:
        return False
    return projected_target_ready < best_before


def _search_selection_contains(action: dict[str, Any], *, kind: str, is_basic: bool | None = None) -> bool:
    for card in action.get("search_selection", []):
        if not isinstance(card, dict):
            continue
        if card.get("kind") != kind:
            continue
        if is_basic is not None and bool(card.get("is_basic")) != is_basic:
            continue
        return True
    return False


def _cap(value: float, limit: int) -> float:
    return max(0.0, min(float(limit), value))


def _softmax_logits(logits: list[tuple[str, float]]) -> dict[str, float]:
    if not logits:
        return {}
    max_logit = max(score for _, score in logits)
    weights = [(action_id, exp(score - max_logit)) for action_id, score in logits]
    total = sum(weight for _, weight in weights)
    if total <= 0:
        uniform = 1.0 / len(weights)
        return {action_id: uniform for action_id, _ in weights}
    return {action_id: round(weight / total, 6) for action_id, weight in weights}


def _softmax_from_logits(logits: list[float], action_ids: list[str]) -> dict[str, float]:
    return _softmax_logits(list(zip(action_ids, logits, strict=False)))
