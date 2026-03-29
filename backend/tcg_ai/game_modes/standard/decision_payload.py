from __future__ import annotations

from typing import Any

from .action_analysis import analyze_legal_actions
from .action_metadata import build_action_metadata
from .engine import action_id_for, card_definition, get_top_card_definition
from .models import GameState, PlayerState, PokemonInPlay

SCHEMA_VERSION = 1


def build_decision_request(
    state: GameState,
    *,
    session_id: str,
    decision_id: str,
    decision_type: str,
    acting_player_index: int,
    ai_trainer_id: str,
    ai_deck_id: str,
    legal_actions: list[dict[str, Any]],
) -> dict[str, Any]:
    analysis_by_action_id = analyze_legal_actions(
        state,
        acting_player_index=acting_player_index,
        legal_actions=legal_actions,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "decision_id": decision_id,
        "decision_type": decision_type,
        "game_mode": "standard",
        "session_id": session_id,
        "turn_number": state.turn_number,
        "acting_player_index": acting_player_index,
        "ai_trainer_id": ai_trainer_id,
        "ai_deck_id": ai_deck_id,
        "public_state": _serialize_public_state(state),
        "player_private_state": {
            "hand": [
                _serialize_card_instance(state, instance_id)
                for instance_id in state.players[acting_player_index].hand
            ],
        },
        "legal_actions": [
            _serialize_legal_action(
                state,
                acting_player_index,
                action,
                analysis=analysis_by_action_id.get(action_id_for(action)),
            )
            for action in legal_actions
        ],
    }


def _serialize_public_state(state: GameState) -> dict[str, Any]:
    return {
        "turn_number": state.turn_number,
        "current_player": state.current_player,
        "starting_player": state.starting_player,
        "setup_phase": state.setup_phase,
        "pending_promotion_for": state.pending_promotion_for,
        "winner": state.winner,
        "players": [
            _serialize_public_player_state(state, player_index)
            for player_index in range(len(state.players))
        ],
    }


def _serialize_public_player_state(state: GameState, player_index: int) -> dict[str, Any]:
    player = state.players[player_index]
    return {
        "index": player_index,
        "name": player.name,
        "deck_name": player.deck_name,
        "element": player.element,
        "deck_count": len(player.deck),
        "discard_count": len(player.discard),
        "discard": [_serialize_card_instance(state, instance_id) for instance_id in player.discard],
        "prize_cards_remaining": len(player.prizes),
        "active_missing": player.active is None,
        "active": _serialize_public_pokemon(state, player.active),
        "bench": [_serialize_public_pokemon(state, pokemon) for pokemon in player.bench],
    }


def _serialize_public_pokemon(
    state: GameState,
    pokemon: PokemonInPlay | None,
) -> dict[str, Any] | None:
    if pokemon is None:
        return None

    top_card = get_top_card_definition(state, pokemon)
    if top_card is None:
        return None

    hp = top_card.hp or 0
    return {
        "card": _serialize_card_instance(state, pokemon.stack[-1]),
        "stack": [_serialize_card_instance(state, instance_id) for instance_id in pokemon.stack],
        "attached_energy_count": len(pokemon.attached_energy),
        "attached_energy": [
            _serialize_card_instance(state, instance_id) for instance_id in pokemon.attached_energy
        ],
        "lingering_effects": [
            _serialize_lingering_effect(effect)
            for effect in pokemon.lingering_effects
        ],
        "damage": pokemon.damage,
        "hp": hp,
        "remaining_hp": max(0, hp - pokemon.damage),
        "retreat_cost": int(top_card.retreat_cost or 0),
        "attacks": [
            {
                "name": attack.name,
                "cost": attack.cost,
                "damage": attack.damage,
                "effect": attack.effect,
                "text": attack.text,
                "effect_specs": [
                    _serialize_attack_effect_spec(effect_spec)
                    for effect_spec in attack.effect_specs
                ],
            }
            for attack in top_card.attacks
        ],
    }


def _serialize_card_instance(state: GameState, instance_id: str) -> dict[str, Any]:
    card = card_definition(state, instance_id)
    return {
        "instance_id": instance_id,
        "card_id": card.card_id,
        "name": card.name,
        "kind": card.kind,
        "stage": card.stage,
        "is_basic": card.is_basic,
        "is_basic_energy": card.is_basic_energy,
        "element": card.element,
        "image_url": card.image_url,
        "prize_card_value": card.prize_card_value,
        "retreat_cost": card.retreat_cost,
    }


def _serialize_legal_action(
    state: GameState,
    player_index: int,
    action: dict[str, Any],
    *,
    analysis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "action_id": action_id_for(action),
        "type": action["type"],
        "label": action["label"],
    }
    source = _serialize_action_source(state, player_index, action)
    target = _serialize_action_target(state, player_index, action)
    if source is not None:
        payload["source"] = source
    if target is not None:
        payload["target"] = target
    payload.update(build_action_metadata(state, player_index, action))
    if isinstance(analysis, dict):
        payload.update(
            {
                "tactical_outcomes": dict(analysis.get("tactical_outcomes") or {}),
                "resolution_facts": dict(analysis.get("resolution_facts") or {}),
                "intent_tags": list(analysis.get("intent_tags") or []),
                "quality_flags": list(analysis.get("quality_flags") or []),
                "reason_tags": list(analysis.get("reason_tags") or []),
                "reason_summary": analysis.get("reason_summary"),
                "dominance_context": dict(analysis.get("dominance_context") or {}),
                "penalty_breakdown": dict(analysis.get("penalty_breakdown") or {}),
            }
        )
    return payload


def _serialize_action_source(
    state: GameState,
    player_index: int,
    action: dict[str, Any],
) -> dict[str, Any] | None:
    if action["type"] in {"play_basic_to_active", "play_energy", "evolve", "play_supporter", "play_item"}:
        instance_id = action["hand_card_id"]
        card = card_definition(state, instance_id)
        return {
            "player_index": player_index,
            "zone": "hand",
            "instance_id": instance_id,
            "card_id": card.card_id,
            "name": card.name,
        }
    if action["type"] == "promote":
        bench_index = action["bench_index"]
        pokemon = state.players[player_index].bench[bench_index]
        instance_id = pokemon.stack[-1]
        card = card_definition(state, instance_id)
        return {
            "player_index": player_index,
            "zone": "bench",
            "bench_index": bench_index,
            "instance_id": instance_id,
            "card_id": card.card_id,
            "name": card.name,
        }
    if action["type"] == "retreat":
        player = state.players[player_index]
        instance_id = player.active.stack[-1] if player.active is not None and player.active.stack else None
        return {
            "player_index": player_index,
            "zone": "active",
            "instance_id": instance_id,
            "bench_index": None,
            "name": "Active Pokemon",
        }
    if action["type"] == "attack":
        player = state.players[player_index]
        instance_id = player.active.stack[-1] if player.active is not None and player.active.stack else None
        return {
            "player_index": player_index,
            "zone": "active",
            "instance_id": instance_id,
            "bench_index": None,
            "name": "Active Pokemon",
        }
    return None


def _serialize_action_target(
    state: GameState,
    player_index: int,
    action: dict[str, Any],
) -> dict[str, Any] | None:
    if action["type"] == "play_basic_to_active":
        return {
            "player_index": player_index,
            "zone": "active",
            "instance_id": None,
            "bench_index": None,
            "name": "Active Spot",
        }
    if action["type"] == "promote":
        return {
            "player_index": player_index,
            "zone": "active",
            "instance_id": None,
            "bench_index": None,
            "name": "Active Spot",
        }
    if action["type"] == "retreat":
        player = state.players[player_index]
        bench_index = action["target_bench_index"]
        pokemon = player.bench[bench_index]
        return {
            "player_index": player_index,
            "zone": "bench",
            "instance_id": pokemon.stack[-1] if pokemon.stack else None,
            "bench_index": bench_index,
            "name": "Bench Pokemon",
        }
    if action["type"] in {"play_energy", "evolve", "play_supporter", "play_item"}:
        player = state.players[player_index]
        if action["target_zone"] == "active":
            instance_id = player.active.stack[-1] if player.active is not None and player.active.stack else None
            return {
                "player_index": player_index,
                "zone": "active",
                "instance_id": instance_id,
                "bench_index": None,
                "name": "Active Pokemon",
            }
        if action["target_zone"] == "bench":
            bench_index = action["target_bench_index"]
            pokemon = player.bench[bench_index]
            return {
                "player_index": player_index,
                "zone": "bench",
                "instance_id": pokemon.stack[-1] if pokemon.stack else None,
                "bench_index": bench_index,
                "name": "Bench Pokemon",
            }
    if action["type"] == "attack":
        return _serialize_attack_target_ref(state, player_index, action)
    return None


def _serialize_attack_effect_spec(effect_spec: Any) -> dict[str, Any]:
    return {
        "effect_type": effect_spec.effect_type,
        "amount": effect_spec.amount,
        "count": effect_spec.count,
        "count_mode": effect_spec.count_mode,
        "source_zone": effect_spec.source_zone,
        "destination_zone": effect_spec.destination_zone,
        "destination_position": effect_spec.destination_position,
        "target_player": effect_spec.target_player,
        "target_zone": effect_spec.target_zone,
        "selection_count": effect_spec.selection_count,
        "choose_count": effect_spec.choose_count,
        "search_filters": list(effect_spec.search_filters),
        "energy_type": effect_spec.energy_type,
        "optional": effect_spec.optional,
        "shuffle_destination": effect_spec.shuffle_destination,
        "revealed_to": effect_spec.revealed_to,
        "changes_hidden_information": effect_spec.changes_hidden_information,
        "bonus_damage": effect_spec.bonus_damage,
        "condition": effect_spec.condition,
        "duration": effect_spec.duration,
    }


def _serialize_lingering_effect(effect: Any) -> dict[str, Any]:
    return {
        "effect_type": effect.effect_type,
        "source_player": effect.source_player,
        "expires_end_of_player_turn": effect.expires_end_of_player_turn,
        "activation_turn": effect.activation_turn,
        "condition": effect.condition,
        "blocked_attack_index": effect.blocked_attack_index,
    }


def _serialize_attack_target_ref(
    state: GameState,
    player_index: int,
    action: dict[str, Any],
) -> dict[str, Any] | None:
    target_player_index = int(action.get("target_player_index", 1 - player_index))
    player = state.players[target_player_index]
    target_zone = action.get("target_zone", "active")
    if target_zone == "active":
        instance_id = player.active.stack[-1] if player.active is not None and player.active.stack else None
        return {
            "player_index": target_player_index,
            "zone": "active",
            "instance_id": instance_id,
            "bench_index": None,
            "name": "Opposing Active Pokemon" if target_player_index != player_index else "Active Pokemon",
        }
    if target_zone == "bench":
        bench_index = action.get("target_bench_index")
        if not isinstance(bench_index, int):
            return None
        pokemon = player.bench[bench_index]
        return {
            "player_index": target_player_index,
            "zone": "bench",
            "instance_id": pokemon.stack[-1] if pokemon.stack else None,
            "bench_index": bench_index,
            "name": "Bench Pokemon",
        }
    return None
