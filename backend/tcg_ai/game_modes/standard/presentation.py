from __future__ import annotations

from typing import Any

from .engine import action_id_for, card_definition, get_top_card_definition, list_legal_actions
from .models import EffectOption, EffectSpec, GameState, PlayerState, PokemonInPlay

FACEDOWN_CARD_ASSET_URL = "/assets/cards/shared/card-back.png"


def facedown_card_image_url() -> str:
    return FACEDOWN_CARD_ASSET_URL


def serialize_state(
    state: GameState,
    session_id: str,
    viewer: int = 0,
    ai_learning: dict[str, Any] | None = None,
) -> dict[str, Any]:
    raw_actions = list_legal_actions(state) if state.current_player == viewer else []
    action_views = [_serialize_action(state, action) for action in raw_actions]
    return {
        "session_id": session_id,
        "seed": state.seed,
        "turn_number": state.turn_number,
        "current_player": state.current_player,
        "setup_phase": state.setup_phase,
        "winner": state.winner,
        "pending_promotion_for": state.pending_promotion_for,
        "human_player": viewer,
        "matchup_label": f"{state.players[0].deck_name} vs {state.players[1].deck_name}",
        "shared_assets": {
            "face_down_card_image_url": facedown_card_image_url(),
        },
        "log": _serialize_log_entries(state.log[-30:], state=state, viewer=viewer),
        "players": [
            _serialize_player_state(state, index, viewer, action_views)
            for index in range(len(state.players))
        ],
        "legal_actions": action_views,
        "ai_learning": ai_learning or {},
    }


def _serialize_player_state(
    state: GameState,
    player_index: int,
    viewer: int,
    action_views: list[dict[str, Any]],
) -> dict[str, Any]:
    player = state.players[player_index]
    hand = [
        _serialize_hand_card(state, player_index, instance_id, action_views)
        for instance_id in player.hand
        if player_index == viewer
    ]
    return {
        "index": player_index,
        "name": player.name,
        "deck_name": player.deck_name,
        "element": player.element,
        "hand_count": len(player.hand),
        "hand": hand,
        "deck_count": len(player.deck),
        "deck_pile": {
            "count": len(player.deck),
            "face_down": True,
        },
        "deck_cards": [
            _serialize_card_instance(state, instance_id) for instance_id in player.deck
        ] if player_index == viewer else [],
        "discard_count": len(player.discard),
        "discard_top": _serialize_discard_top(state, player),
        "discard_cards": [
            _serialize_card_instance(state, instance_id) for instance_id in player.discard
        ] if player_index == viewer else [],
        "energy_count": _count_attached_energy(player),
        "energy_zone": [],
        "energy_attachment_available": not player.energy_attached_this_turn,
        "prize_tokens_remaining": len(player.prizes),
        "prize_pile": {
            "count": len(player.prizes),
            "face_down": True,
            "image_url": None,
            "known_cards": [
                _serialize_card_instance(state, instance_id) for instance_id in player.prizes
            ] if player_index == viewer and player.deck_inspected_this_game else [],
        },
        "requires_promotion": state.pending_promotion_for == player_index,
        "active": _serialize_pokemon(
            state,
            player.active,
            player_index,
            "active",
            None,
            action_views,
            viewer=viewer,
        ),
        "bench": [
            _serialize_pokemon(
                state,
                pokemon,
                player_index,
                "bench",
                bench_index,
                action_views,
                viewer=viewer,
            )
            for bench_index, pokemon in enumerate(player.bench)
        ],
    }


def _serialize_hand_card(
    state: GameState,
    player_index: int,
    instance_id: str,
    action_views: list[dict[str, Any]],
) -> dict[str, Any]:
    action_types = sorted(
        {
            action["type"]
            for action in action_views
            if _ref_matches(
                action.get("source"),
                {"player_index": player_index, "zone": "hand", "instance_id": instance_id},
            )
        }
    )
    return {
        **_serialize_card_instance(state, instance_id),
        "playable": bool(action_types),
        "action_types": action_types,
    }


def _serialize_discard_top(state: GameState, player: PlayerState) -> dict[str, Any] | None:
    if not player.discard:
        return None
    return _serialize_card_instance(state, player.discard[-1])


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
        "card_tags": list(card.card_tags),
        "rules_text": list(card.rules_text),
        "is_basic_energy": card.is_basic_energy,
        "prize_card_value": card.prize_card_value,
        "retreat_cost": card.retreat_cost,
        "effect_specs": [_serialize_effect_spec(effect_spec) for effect_spec in card.effect_specs],
    }


def _serialize_pokemon(
    state: GameState,
    pokemon: PokemonInPlay | None,
    player_index: int,
    zone: str,
    bench_index: int | None,
    action_views: list[dict[str, Any]],
    viewer: int,
) -> dict[str, Any] | None:
    if pokemon is None:
        return None

    top_card = get_top_card_definition(state, pokemon)
    if top_card is None:
        return None

    top_instance_id = pokemon.stack[-1]
    ref = {
        "player_index": player_index,
        "zone": zone,
        "instance_id": top_instance_id,
        "bench_index": bench_index,
    }
    source_action_types = sorted(
        {
            action["type"]
            for action in action_views
            if _ref_matches(action.get("source"), ref)
        }
    )
    target_action_types = sorted(
        {
            action["type"]
            for action in action_views
            if _ref_matches(action.get("target"), ref)
        }
    )
    if _should_hide_opening_active(state, viewer, player_index, zone):
        return {
            "instance_id": top_instance_id,
            "card_id": None,
            "image_url": facedown_card_image_url(),
            "name": "Face-down Active Pokemon",
            "kind": "pokemon",
            "element": None,
            "stage": None,
            "hp": 0,
            "damage": 0,
            "remaining_hp": 0,
            "ref": ref,
        "stack": [],
        "attacks": [],
        "attached_energy_count": 0,
        "attached_energy": [],
        "source_action_types": [],
        "target_action_types": [],
        "interactive": False,
        "can_attack": False,
            "requires_promotion": False,
            "face_down": True,
        }
    hp = top_card.hp or 0
    return {
        "instance_id": top_instance_id,
        "card_id": top_card.card_id,
        "image_url": top_card.image_url,
        "name": top_card.name,
        "kind": top_card.kind,
        "element": top_card.element,
        "stage": top_card.stage,
        "hp": hp,
        "damage": pokemon.damage,
        "remaining_hp": max(0, hp - pokemon.damage),
        "retreat_cost": top_card.retreat_cost,
        "ref": ref,
        "stack": [_serialize_card_instance(state, instance_id) for instance_id in pokemon.stack],
        "attached_energy_count": len(pokemon.attached_energy),
        "attached_energy": [
            _serialize_card_instance(state, instance_id) for instance_id in pokemon.attached_energy
        ],
        "lingering_effects": [
            _serialize_attack_effect_state(effect)
            for effect in pokemon.lingering_effects
        ],
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
        "source_action_types": source_action_types,
        "target_action_types": target_action_types,
        "interactive": bool(source_action_types or target_action_types),
        "can_attack": "attack" in source_action_types,
        "requires_promotion": "promote" in source_action_types,
        "face_down": False,
    }


def _should_hide_opening_active(
    state: GameState,
    viewer: int,
    player_index: int,
    zone: str,
) -> bool:
    return (
        zone == "active"
        and player_index != viewer
        and viewer == 0
        and state.setup_phase is not None
    )


def _serialize_action(state: GameState, action: dict[str, Any]) -> dict[str, Any]:
    player_index = state.current_player
    view = {
        "action_id": action_id_for(action),
        "type": action["type"],
        "label": action["label"],
        "action": action,
    }
    source = _serialize_action_source(state, player_index, action)
    target = _serialize_action_target(state, player_index, action)
    if source is not None:
        view["source"] = source
    if target is not None:
        view["target"] = target
    if action["type"] in {"play_supporter", "play_item"} and isinstance(action.get("hand_card_id"), str):
        card = card_definition(state, action["hand_card_id"])
        view["card_tags"] = list(card.card_tags)
        view["effect_specs"] = [_serialize_effect_spec(effect_spec) for effect_spec in card.effect_specs]
        view["changes_hidden_information"] = any(
            effect_spec.changes_hidden_information for effect_spec in card.effect_specs
        )
    return view


def _serialize_action_source(
    state: GameState,
    player_index: int,
    action: dict[str, Any],
) -> dict[str, Any] | None:
    if action["type"] in {
        "play_basic_to_active",
        "bench_basic",
        "play_energy",
        "evolve",
        "play_supporter",
        "play_item",
    }:
        instance_id = action["hand_card_id"]
        card = _serialize_card_instance(state, instance_id)
        return {
            "player_index": player_index,
            "zone": "hand",
            **card,
        }
    if action["type"] == "promote":
        bench_index = action["bench_index"]
        pokemon = state.players[player_index].bench[bench_index]
        instance_id = pokemon.stack[-1]
        card = _serialize_card_instance(state, instance_id)
        return {
            "player_index": player_index,
            "zone": "bench",
            "bench_index": bench_index,
            **card,
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
    if action["type"] in {"mulligan", "end_setup", "end_turn"}:
        return {
            "player_index": player_index,
            "zone": "system",
            "name": action["label"],
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
    if action["type"] in {"play_energy", "evolve", "play_supporter", "play_item"} and "target_zone" in action:
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


def _serialize_effect_spec(effect_spec: EffectSpec) -> dict[str, Any]:
    return {
        "effect_type": effect_spec.effect_type,
        "count": effect_spec.count,
        "count_mode": effect_spec.count_mode,
        "source_zone": effect_spec.source_zone,
        "destination_zone": effect_spec.destination_zone,
        "destination_position": effect_spec.destination_position,
        "target_player": effect_spec.target_player,
        "selection_count": effect_spec.selection_count,
        "choose_count": effect_spec.choose_count,
        "search_filters": list(effect_spec.search_filters),
        "options": [_serialize_effect_option(option) for option in effect_spec.options],
        "optional": effect_spec.optional,
        "shuffle_destination": effect_spec.shuffle_destination,
        "exclude_source_card": effect_spec.exclude_source_card,
        "revealed_to": effect_spec.revealed_to,
        "changes_hidden_information": effect_spec.changes_hidden_information,
    }


def _serialize_effect_option(option: EffectOption) -> dict[str, Any]:
    return {
        "option_id": option.option_id,
        "label": option.label,
        "effect_specs": [_serialize_effect_spec(effect_spec) for effect_spec in option.effect_specs],
    }


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


def _serialize_attack_effect_state(effect: Any) -> dict[str, Any]:
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


def _serialize_log_entries(
    entries: list[str],
    *,
    state: GameState,
    viewer: int,
) -> list[dict[str, str]]:
    return [_serialize_log_entry(entry, state=state, viewer=viewer) for entry in entries]


def _serialize_log_entry(
    entry: str,
    *,
    state: GameState,
    viewer: int,
) -> dict[str, str]:
    entry = _redact_hidden_log_text(entry, state=state, viewer=viewer)
    side = "system"
    if entry.startswith("You ") or entry.startswith("Your "):
        side = "human"
    elif entry.startswith("AI ") or entry.startswith("AI's "):
        side = "ai"

    kind = "system"
    if "drew" in entry:
        kind = "draw"

    return {"text": entry, "side": side, "kind": kind}


def _redact_hidden_log_text(
    entry: str,
    *,
    state: GameState,
    viewer: int,
) -> str:
    if viewer < 0 or viewer >= len(state.players):
        return entry
    for player_index, player in enumerate(state.players):
        if player_index == viewer:
            continue
        prefix = f"{player.name} drew "
        if not entry.startswith(prefix):
            continue
        remainder = entry[len(prefix):].strip()
        if _is_public_draw_log_remainder(remainder):
            return entry
        return f"{player.name} drew a card."
    return entry


def _is_public_draw_log_remainder(remainder: str) -> bool:
    if not remainder:
        return False
    if remainder.startswith("a card"):
        return True
    if remainder.startswith("no card"):
        return True
    return remainder[0].isdigit()


def _ref_matches(reference: dict[str, Any] | None, expected: dict[str, Any]) -> bool:
    if reference is None:
        return False
    for key, value in expected.items():
        if value is None:
            continue
        if reference.get(key) != value:
            return False
    return True


def _count_attached_energy(player: PlayerState) -> int:
    count = len(player.active.attached_energy) if player.active is not None else 0
    count += sum(len(pokemon.attached_energy) for pokemon in player.bench)
    return count
