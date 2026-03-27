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


STATE_VECTOR_SIZE = 32
ACTION_VECTOR_SIZE = 24
DEFAULT_HIDDEN_SIZE = 128
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
        return [self._model_evaluation(evaluation) for evaluation in evaluations]

    def _try_load_model(self) -> None:
        if torch is None or nn is None or self.checkpoint_path is None or not self.checkpoint_path.exists():
            return
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = ActionConditionedPolicyValueNet()
        checkpoint = load_trusted_checkpoint(self.checkpoint_path, map_location=device)
        state_dict = checkpoint.get("state_dict") if isinstance(checkpoint, dict) else checkpoint
        model.load_state_dict(state_dict)
        model.eval()
        model.to(device)
        self._model = model
        self._status = PolicyValueBackendStatus(
            backend=f"torch:{device}",
            model_loaded=True,
            checkpoint_path=str(self.checkpoint_path),
        )

    def _model_evaluation(self, evaluation: dict[str, Any]) -> dict[str, Any]:
        state_vector = torch.tensor(
            encode_state_vector(evaluation.get("belief_state", {})),
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
            [encode_action_vector(action) for action in legal_actions],
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


def encode_state_vector(belief_state: dict[str, Any]) -> list[float]:
    players = belief_state.get("players", [{}, {}])
    player = players[0] if len(players) > 0 else {}
    opponent = players[1] if len(players) > 1 else {}
    derived = belief_state.get("derived_features", {})
    active = player.get("active") or {}
    opponent_active = opponent.get("active") or {}
    vector = [
        _cap(float(belief_state.get("turn_number", 1)), 20) / 20.0,
        1.0 if belief_state.get("current_player") == belief_state.get("perspective_player_index") else 0.0,
        1.0 if belief_state.get("starting_player") == belief_state.get("perspective_player_index") else 0.0,
        1.0 if belief_state.get("winner") == belief_state.get("perspective_player_index") else 0.0,
        1.0 if belief_state.get("winner") == (1 - int(belief_state.get("perspective_player_index", 0))) else 0.0,
        1.0 if belief_state.get("setup_phase") is not None else 0.0,
        _cap(float(player.get("deck_count", 0)), 60) / 60.0,
        _cap(float(player.get("hand_count", 0)), 10) / 10.0,
        _cap(float(player.get("discard_count", 0)), 60) / 60.0,
        _cap(float(player.get("prize_count", 6)), 6) / 6.0,
        1.0 if player.get("deck_inspected_this_game") else 0.0,
        1.0 if player.get("supporter_played_this_turn") else 0.0,
        1.0 if player.get("energy_attached_this_turn") else 0.0,
        _pokemon_hp_ratio(active),
        _pokemon_energy_ratio(active),
        _turn_ratio(active.get("turns_until_ready")),
        _cap(float(len(player.get("bench", []))), 5) / 5.0,
        _player_energy_total(player) / 12.0,
        _player_remaining_hp(player) / 900.0,
        _cap(float(opponent.get("deck_count", 0)), 60) / 60.0,
        _cap(float(opponent.get("hand_count", 0)), 10) / 10.0,
        _cap(float(opponent.get("discard_count", 0)), 60) / 60.0,
        _cap(float(opponent.get("prize_count", 6)), 6) / 6.0,
        1.0 if opponent.get("supporter_played_this_turn") else 0.0,
        1.0 if opponent.get("energy_attached_this_turn") else 0.0,
        _pokemon_hp_ratio(opponent_active),
        _pokemon_energy_ratio(opponent_active),
        _turn_ratio(opponent_active.get("turns_until_ready")),
        _cap(float(len(opponent.get("bench", []))), 5) / 5.0,
        _player_energy_total(opponent) / 12.0,
        _player_remaining_hp(opponent) / 900.0,
        -1.0 if derived.get("player_active_likely_knockout_next_turn") else 0.0,
        1.0 if derived.get("opponent_active_likely_knockout_next_turn") else 0.0,
    ]
    return vector[:STATE_VECTOR_SIZE]


def encode_action_vector(action: dict[str, Any]) -> list[float]:
    source_card = action.get("source_card") or {}
    action_type = str(action.get("type", ""))
    vector = [
        1.0 if action_type == "attack" else 0.0,
        1.0 if action_type == "play_supporter" else 0.0,
        1.0 if action_type == "play_item" else 0.0,
        1.0 if action_type == "play_energy" else 0.0,
        1.0 if action_type == "evolve" else 0.0,
        1.0 if action_type == "bench_basic" else 0.0,
        1.0 if action_type == "play_basic_to_active" else 0.0,
        1.0 if action_type == "end_turn" else 0.0,
        1.0 if source_card.get("kind") == "pokemon" else 0.0,
        1.0 if source_card.get("kind") == "trainer" else 0.0,
        1.0 if source_card.get("kind") == "energy" else 0.0,
        1.0 if source_card.get("is_basic") else 0.0,
        1.0 if source_card.get("stage") == "stage1" else 0.0,
        1.0 if source_card.get("stage") == "stage2" else 0.0,
        1.0 if action.get("target", {}).get("zone") == "active" else 0.0,
        1.0 if action.get("target", {}).get("zone") == "bench" else 0.0,
        _cap(float(len(action.get("discard_from_hand", []))), 3) / 3.0,
        _cap(float(len(action.get("search_selection", []))), 2) / 2.0,
        1.0 if action.get("consumes_supporter_for_turn") else 0.0,
        1.0 if action.get("consumes_attachment_for_turn") else 0.0,
        1.0 if action.get("reveals_hidden_cards") else 0.0,
        1.0 if "supporter" in source_card.get("card_tags", []) else 0.0,
        1.0 if "item" in source_card.get("card_tags", []) else 0.0,
        1.0 if action.get("target", {}).get("player_index") == 1 else 0.0,
    ]
    return vector[:ACTION_VECTOR_SIZE]


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
