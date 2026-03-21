from __future__ import annotations

from typing import Any

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
            _serialize_legal_action(state, acting_player_index, action) for action in legal_actions
        ],
    }


def _serialize_public_state(state: GameState) -> dict[str, Any]:
    return {
        "turn_number": state.turn_number,
        "current_player": state.current_player,
        "setup_phase": state.setup_phase,
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
        "prize_cards_remaining": player.prize_cards_remaining,
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
        "damage": pokemon.damage,
        "hp": hp,
        "remaining_hp": max(0, hp - pokemon.damage),
        "attacks": [
            {
                "name": attack.name,
                "cost": attack.cost,
                "damage": attack.damage,
                "effect": attack.effect,
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
        "element": card.element,
        "image_url": card.image_url,
    }


def _serialize_legal_action(
    state: GameState,
    player_index: int,
    action: dict[str, Any],
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
    return payload


def _serialize_action_source(
    state: GameState,
    player_index: int,
    action: dict[str, Any],
) -> dict[str, Any] | None:
    if action["type"] in {"play_basic_to_active", "play_energy"}:
        instance_id = action["hand_card_id"]
        card = card_definition(state, instance_id)
        return {
            "player_index": player_index,
            "zone": "hand",
            "instance_id": instance_id,
            "card_id": card.card_id,
            "name": card.name,
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
    if action["type"] == "play_energy":
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
    return None
