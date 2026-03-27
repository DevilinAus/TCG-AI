from __future__ import annotations

from typing import Any

from ..action_metadata import build_action_metadata
from ..engine import action_id_for, card_definition, get_top_card_definition
from ..models import GameState, PokemonInPlay

SCHEMA_VERSION = 1


def serialize_knowledge_state(
    state: GameState,
    *,
    perspective_player_index: int,
) -> dict[str, Any]:
    player = state.players[perspective_player_index]
    opponent_index = 1 - perspective_player_index
    opponent = state.players[opponent_index]
    player_active = _serialize_visible_pokemon(state, player.active)
    opponent_active = _serialize_visible_pokemon(state, opponent.active)
    return {
        "schema_version": SCHEMA_VERSION,
        "turn_number": state.turn_number,
        "current_player": state.current_player,
        "starting_player": state.starting_player,
        "winner": state.winner,
        "setup_phase": state.setup_phase,
        "perspective_player_index": perspective_player_index,
        "players": [
            _serialize_private_player_state(state, perspective_player_index),
            _serialize_public_player_state(state, opponent_index),
        ],
        "derived_features": {
            "player_active_turns_until_ready": _turns_until_ready(player_active),
            "opponent_active_turns_until_ready": _turns_until_ready(opponent_active),
            "player_active_likely_knockout_next_turn": _likely_knockout_next_turn(
                defender=player_active,
                attacker=opponent_active,
            ),
            "opponent_active_likely_knockout_next_turn": _likely_knockout_next_turn(
                defender=opponent_active,
                attacker=player_active,
            ),
            "player_energy_at_risk_on_active": _attached_energy_count(player_active),
            "opponent_energy_at_risk_on_active": _attached_energy_count(opponent_active),
            "player_board_investment": _board_investment(_serialize_private_player_state(state, perspective_player_index)),
            "opponent_board_investment": _board_investment(_serialize_public_player_state(state, opponent_index)),
        },
    }


def serialize_knowledge_actions(
    state: GameState,
    *,
    acting_player_index: int,
    legal_actions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        _serialize_knowledge_action(state, acting_player_index, action)
        for action in legal_actions
    ]


def _serialize_private_player_state(
    state: GameState,
    player_index: int,
) -> dict[str, Any]:
    player = state.players[player_index]
    return {
        "index": player_index,
        "name": player.name,
        "deck_name": player.deck_name,
        "element": player.element,
        "deck_count": len(player.deck),
        "hand_count": len(player.hand),
        "hand": [_serialize_card_instance(state, instance_id) for instance_id in player.hand],
        "discard": [_serialize_card_instance(state, instance_id) for instance_id in player.discard],
        "discard_count": len(player.discard),
        "prize_count": len(player.prizes),
        "known_prize_cards_unordered": [
            _serialize_card_instance(state, instance_id) for instance_id in player.prizes
        ] if player.deck_inspected_this_game else [],
        "deck_inspected_this_game": player.deck_inspected_this_game,
        "supporter_played_this_turn": player.supporter_played_this_turn,
        "energy_attached_this_turn": player.energy_attached_this_turn,
        "retreated_this_turn": player.retreated_this_turn,
        "active": _serialize_visible_pokemon(state, player.active),
        "bench": [_serialize_visible_pokemon(state, pokemon) for pokemon in player.bench],
    }


def _serialize_public_player_state(
    state: GameState,
    player_index: int,
) -> dict[str, Any]:
    player = state.players[player_index]
    return {
        "index": player_index,
        "name": player.name,
        "deck_name": player.deck_name,
        "element": player.element,
        "deck_count": len(player.deck),
        "hand_count": len(player.hand),
        "discard": [_serialize_card_instance(state, instance_id) for instance_id in player.discard],
        "discard_count": len(player.discard),
        "prize_count": len(player.prizes),
        "supporter_played_this_turn": player.supporter_played_this_turn,
        "energy_attached_this_turn": player.energy_attached_this_turn,
        "retreated_this_turn": player.retreated_this_turn,
        "active": _serialize_visible_pokemon(state, player.active),
        "bench": [_serialize_visible_pokemon(state, pokemon) for pokemon in player.bench],
    }


def _serialize_knowledge_action(
    state: GameState,
    acting_player_index: int,
    action: dict[str, Any],
) -> dict[str, Any]:
    source_card = None
    hand_card_id = action.get("hand_card_id")
    if isinstance(hand_card_id, str):
        source_card = _serialize_card_instance(state, hand_card_id)

    action_type = str(action.get("type", ""))
    discard_ids = [instance_id for instance_id in action.get("discard_from_hand_ids", []) if isinstance(instance_id, str)]
    retreat_discard_ids = [
        instance_id for instance_id in action.get("discard_attached_energy_ids", []) if isinstance(instance_id, str)
    ]
    recover_ids = [instance_id for instance_id in action.get("recover_from_discard_ids", []) if isinstance(instance_id, str)]
    search_ids = [instance_id for instance_id in action.get("search_deck_ids", []) if isinstance(instance_id, str)]
    payload = {
        "action_id": action_id_for(action),
        "type": action_type,
        "label": str(action.get("label", "")),
        "source_card": source_card,
        "target": {
            "player_index": action.get("target_player_index", acting_player_index),
            "zone": action.get("target_zone"),
            "bench_index": action.get("target_bench_index"),
        },
        "discard_from_hand": [_serialize_card_instance(state, instance_id) for instance_id in discard_ids],
        "discard_attached_energy": [_serialize_card_instance(state, instance_id) for instance_id in retreat_discard_ids],
        "recover_from_discard": [_serialize_card_instance(state, instance_id) for instance_id in recover_ids],
        "search_selection": [_serialize_card_instance(state, instance_id) for instance_id in search_ids],
        "blocked_attack_index": action.get("blocked_attack_index"),
        "consumes_supporter_for_turn": action_type == "play_supporter",
        "consumes_attachment_for_turn": action_type == "play_energy",
        "consumes_retreat_for_turn": action_type == "retreat",
        "reveals_hidden_cards": bool(search_ids),
        "card_tags": list(source_card.get("card_tags", [])) if isinstance(source_card, dict) else [],
        "effect_specs": list(source_card.get("effect_specs", [])) if isinstance(source_card, dict) else [],
    }
    payload.update(build_action_metadata(state, acting_player_index, action))
    return payload


def _serialize_visible_pokemon(
    state: GameState,
    pokemon: PokemonInPlay | None,
) -> dict[str, Any] | None:
    if pokemon is None:
        return None
    top_card = get_top_card_definition(state, pokemon)
    if top_card is None:
        return None
    hp = int(top_card.hp or 0)
    attacks = [
        {
            "name": attack.name,
            "cost": attack.cost,
            "damage": attack.damage,
            "remaining_cost": max(0, int(attack.cost) - len(pokemon.attached_energy)),
        }
        for attack in top_card.attacks
    ]
    return {
        "card": _serialize_card_instance(state, pokemon.stack[-1]),
        "damage": pokemon.damage,
        "hp": hp,
        "remaining_hp": max(0, hp - pokemon.damage),
        "retreat_cost": int(top_card.retreat_cost or 0),
        "attached_energy_count": len(pokemon.attached_energy),
        "attached_energy": [_serialize_card_instance(state, instance_id) for instance_id in pokemon.attached_energy],
        "attacks": attacks,
        "turns_until_ready": _turns_until_ready_from_attacks(attacks),
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
        "card_tags": list(card.card_tags),
        "prize_card_value": card.prize_card_value,
        "effect_specs": [
            {
                "effect_type": effect_spec.effect_type,
                "count": effect_spec.count,
                "count_mode": effect_spec.count_mode,
                "source_zone": effect_spec.source_zone,
                "destination_zone": effect_spec.destination_zone,
                "search_filters": list(effect_spec.search_filters),
                "optional": effect_spec.optional,
                "shuffle_destination": effect_spec.shuffle_destination,
                "changes_hidden_information": effect_spec.changes_hidden_information,
            }
            for effect_spec in card.effect_specs
        ],
    }


def _turns_until_ready(pokemon: dict[str, Any] | None) -> int | None:
    if not isinstance(pokemon, dict):
        return None
    return _turns_until_ready_from_attacks(pokemon.get("attacks", []))


def _turns_until_ready_from_attacks(attacks: list[dict[str, Any]]) -> int | None:
    if not attacks:
        return None
    remaining_costs = [int(attack.get("remaining_cost", 0) or 0) for attack in attacks]
    return min(remaining_costs) if remaining_costs else None


def _likely_knockout_next_turn(
    *,
    defender: dict[str, Any] | None,
    attacker: dict[str, Any] | None,
) -> bool:
    if not isinstance(defender, dict) or not isinstance(attacker, dict):
        return False
    if attacker.get("turns_until_ready") not in {0, 1}:
        return False
    defender_hp = int(defender.get("remaining_hp", 0) or 0)
    max_damage = 0
    for attack in attacker.get("attacks", []):
        if int(attack.get("remaining_cost", 0) or 0) > 1:
            continue
        damage_digits = "".join(character for character in str(attack.get("damage", "")) if character.isdigit())
        max_damage = max(max_damage, int(damage_digits or 0))
    return max_damage >= defender_hp > 0


def _attached_energy_count(pokemon: dict[str, Any] | None) -> int:
    if not isinstance(pokemon, dict):
        return 0
    return int(pokemon.get("attached_energy_count", 0) or 0)


def _board_investment(player_payload: dict[str, Any]) -> int:
    active = player_payload.get("active")
    bench = player_payload.get("bench", [])
    total_energy = _attached_energy_count(active)
    total_energy += sum(_attached_energy_count(pokemon) for pokemon in bench if isinstance(pokemon, dict))
    total_hp = 0
    for pokemon in [active, *bench]:
        if not isinstance(pokemon, dict):
            continue
        total_hp += int(pokemon.get("remaining_hp", 0) or 0)
    return total_energy * 10 + total_hp
