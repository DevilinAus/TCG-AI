from __future__ import annotations

from typing import Any

from ..cards import load_deck_cards
from ..decision_payload import SCHEMA_VERSION as LEGACY_SCHEMA_VERSION
from ..engine import action_id_for, list_legal_actions
from .canonical_state import SCHEMA_VERSION as FULL_STATE_SCHEMA_VERSION, deserialize_state
from .experience import StandardExperienceStore
from .planner import PlannerConfig, StandardTurnPlanner


class StandardMlService:
    def __init__(self, experience_store: StandardExperienceStore | None = None) -> None:
        self.experience_store = experience_store or StandardExperienceStore()

    def choose_action(self, payload: dict[str, Any]) -> dict[str, Any]:
        if _is_full_state_payload(payload):
            response = self._choose_full_state_action(payload)
        else:
            response = self._choose_legacy_action(payload)
        self.experience_store.record_decision(payload, response)
        return response

    def record_outcome(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.experience_store.record_outcome(payload)
        return {
            "ok": True,
            "schema_version": FULL_STATE_SCHEMA_VERSION,
        }

    def _choose_full_state_action(self, payload: dict[str, Any]) -> dict[str, Any]:
        state = deserialize_state(payload["state"])
        acting_player_index = int(payload.get("acting_player_index", state.current_player))
        legal_actions = list_legal_actions(state, player_index=acting_player_index)
        config = _planner_config_from_payload(payload.get("search_config"))
        planner = StandardTurnPlanner(config=config)
        decision = planner.plan(
            state,
            acting_player_index=acting_player_index,
            legal_actions=legal_actions,
        )
        return {
            "schema_version": FULL_STATE_SCHEMA_VERSION,
            "decision_id": payload.get("decision_id"),
            "decision_type": payload.get("decision_type", "turn_action"),
            "chosen_action_id": decision["chosen_action_id"],
            "planned_action_sequence": decision["planned_action_sequence"],
            "diagnostics": decision["diagnostics"],
        }

    def _choose_legacy_action(self, payload: dict[str, Any]) -> dict[str, Any]:
        legal_actions = payload.get("legal_actions", [])
        if not isinstance(legal_actions, list) or not legal_actions:
            raise ValueError("Legacy ML request is missing legal_actions.")

        hand_by_instance_id = {
            card["instance_id"]: card
            for card in payload.get("player_private_state", {}).get("hand", [])
            if isinstance(card, dict) and isinstance(card.get("instance_id"), str)
        }
        deck_card_stats = {
            card.card_id: card for card in load_deck_cards(str(payload.get("ai_deck_id", "")))
        }
        scored_actions: list[tuple[float, str, dict[str, Any], dict[str, Any]]] = []
        for action in legal_actions:
            action_id = str(action.get("action_id") or "")
            source = action.get("source", {})
            instance_id = source.get("instance_id") if isinstance(source, dict) else None
            card_view = hand_by_instance_id.get(instance_id, {})
            card_id = card_view.get("card_id")
            deck_card = deck_card_stats.get(card_id)
            score = 0.0
            if deck_card is not None:
                score += float(deck_card.hp or 0) * 0.12
                score += max((_legacy_attack_score(attack.damage) - attack.cost * 1.5) for attack in deck_card.attacks) if deck_card.attacks else 0.0
                if deck_card.is_basic:
                    score += 4.0
            scored_actions.append(
                (
                    round(score, 6),
                    action_id,
                    action,
                    {
                        "card_id": card_id,
                        "selection_mode": "legacy_heuristic",
                        "score": round(score, 6),
                    },
                )
            )

        _, chosen_action_id, _, diagnostics = max(scored_actions, key=lambda item: (item[0], item[1]))
        return {
            "schema_version": LEGACY_SCHEMA_VERSION,
            "decision_id": payload.get("decision_id"),
            "chosen_action_id": chosen_action_id,
            "diagnostics": diagnostics,
        }


def _is_full_state_payload(payload: dict[str, Any]) -> bool:
    state = payload.get("state")
    return isinstance(state, dict) and isinstance(state.get("players"), list) and "cards" in state


def _planner_config_from_payload(payload: Any) -> PlannerConfig:
    if not isinstance(payload, dict):
        return PlannerConfig()
    return PlannerConfig(
        max_depth=max(1, int(payload.get("max_depth", 3))),
        beam_width=max(1, int(payload.get("beam_width", 6))),
        opponent_branch_width=max(1, int(payload.get("opponent_branch_width", 3))),
        include_opponent_turn=bool(payload.get("include_opponent_turn", True)),
    )


def _legacy_attack_score(damage_text: str) -> float:
    digits = "".join(character for character in str(damage_text) if character.isdigit())
    return float(digits or 0)
