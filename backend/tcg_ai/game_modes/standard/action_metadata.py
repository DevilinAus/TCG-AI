from __future__ import annotations

from typing import Any

from .engine import card_definition, get_top_card_definition
from .models import AttackDefinition, EffectSpec, GameState


def build_action_metadata(
    state: GameState,
    player_index: int,
    action: dict[str, Any],
) -> dict[str, Any]:
    return {
        "card_instance_id": _action_card_instance_id(state, player_index, action),
        "effect_tags": _effect_tags_for_action(state, player_index, action),
        "resource_costs": _resource_costs_for_action(state, player_index, action),
        "expected_state_delta": _expected_state_delta_for_action(state, player_index, action),
    }


def _action_card_instance_id(
    state: GameState,
    player_index: int,
    action: dict[str, Any],
) -> str | None:
    hand_card_id = action.get("hand_card_id")
    if isinstance(hand_card_id, str):
        return hand_card_id

    action_type = str(action.get("type", ""))
    player = state.players[player_index]
    if action_type in {"attack", "retreat"}:
        if player.active is None or not player.active.stack:
            return None
        return player.active.stack[-1]
    if action_type == "promote":
        bench_index = action.get("bench_index")
        if not isinstance(bench_index, int) or not 0 <= bench_index < len(player.bench):
            return None
        pokemon = player.bench[bench_index]
        return pokemon.stack[-1] if pokemon.stack else None
    return None


def _effect_tags_for_action(
    state: GameState,
    player_index: int,
    action: dict[str, Any],
) -> list[str]:
    action_type = str(action.get("type", ""))
    tags: list[str] = []
    _append_unique(tags, action_type)

    tag_map = {
        "attack": "damage",
        "bench_basic": "bench_development",
        "play_basic_to_active": "active_development",
        "play_energy": "attach_energy",
        "evolve": "evolution",
        "promote": "promotion",
        "retreat": "switch_active",
        "end_turn": "turn_end",
        "end_setup": "turn_setup_complete",
    }
    mapped = tag_map.get(action_type)
    if mapped is not None:
        _append_unique(tags, mapped)

    source_card = _source_card_definition_for_action(state, player_index, action)
    if source_card is not None:
        for card_tag in source_card.card_tags:
            _append_unique(tags, card_tag)

    for effect_spec in _effect_specs_for_action(state, player_index, action):
        _append_unique(tags, effect_spec.effect_type)
        if effect_spec.changes_hidden_information:
            _append_unique(tags, "hidden_information")
        if effect_spec.destination_zone == "bench":
            _append_unique(tags, "bench_development")
        if effect_spec.destination_zone == "hand":
            _append_unique(tags, "hand_gain")

    if action.get("discard_from_hand_ids"):
        _append_unique(tags, "discard_from_hand")
    if action.get("discard_attached_energy_ids"):
        _append_unique(tags, "discard_attached_energy")
    if action.get("recover_from_discard_ids"):
        _append_unique(tags, "recover_from_discard")
    if action.get("search_deck_ids"):
        _append_unique(tags, "search_selection")

    return tags


def _resource_costs_for_action(
    state: GameState,
    player_index: int,
    action: dict[str, Any],
) -> dict[str, int]:
    attack = _attack_definition_for_action(state, player_index, action)
    retreat_cost = 0
    if str(action.get("type", "")) == "retreat":
        player = state.players[player_index]
        active = get_top_card_definition(state, player.active)
        retreat_cost = int(active.retreat_cost or 0) if active is not None else 0

    return {
        "hand_card_count": 1 if isinstance(action.get("hand_card_id"), str) else 0,
        "discard_from_hand_count": len(action.get("discard_from_hand_ids") or []),
        "discard_attached_energy_count": len(action.get("discard_attached_energy_ids") or []),
        "recover_from_discard_count": len(action.get("recover_from_discard_ids") or []),
        "search_selection_count": len(action.get("search_deck_ids") or []),
        "attack_energy_cost": int(attack.cost) if attack is not None else 0,
        "retreat_energy_cost": retreat_cost,
        "supporter_turn_cost": 1 if action.get("type") == "play_supporter" else 0,
        "attachment_turn_cost": 1 if action.get("type") == "play_energy" else 0,
        "retreat_turn_cost": 1 if action.get("type") == "retreat" else 0,
    }


def _expected_state_delta_for_action(
    state: GameState,
    player_index: int,
    action: dict[str, Any],
) -> dict[str, int | bool | None]:
    action_type = str(action.get("type", ""))
    player = state.players[player_index]
    search_to_hand_count = _search_count_for_destination(state, player_index, action, "hand")
    search_to_bench_count = _search_count_for_destination(state, player_index, action, "bench")
    recover_count = len(action.get("recover_from_discard_ids") or [])
    discard_from_hand_count = len(action.get("discard_from_hand_ids") or [])
    discard_attached_energy_count = len(action.get("discard_attached_energy_ids") or [])
    draw_count = _draw_count_for_action(state, player_index, action)

    hand_count_delta_known = 0
    if isinstance(action.get("hand_card_id"), str):
        hand_count_delta_known -= 1
    hand_count_delta_known -= discard_from_hand_count
    hand_count_delta_known += recover_count
    hand_count_delta_known += search_to_hand_count
    if draw_count is not None:
        hand_count_delta_known += draw_count
    if _shuffles_remaining_hand_into_deck(state, player_index, action):
        remaining_hand_after_play = len(player.hand) - (1 if isinstance(action.get("hand_card_id"), str) else 0)
        hand_count_delta_known -= max(0, remaining_hand_after_play)

    bench_count_delta = 0
    if action_type == "bench_basic":
        bench_count_delta += 1
    if action_type == "promote":
        bench_count_delta -= 1
    bench_count_delta += search_to_bench_count

    discard_count_delta_known = discard_from_hand_count + discard_attached_energy_count
    if action_type in {"play_supporter", "play_item"}:
        discard_count_delta_known += 1

    active_changes = action_type in {"play_basic_to_active", "promote", "retreat"} or any(
        effect_spec.effect_type == "switch_active_with_bench"
        for effect_spec in _effect_specs_for_action(state, player_index, action)
    )

    return {
        "hand_count_delta_known": hand_count_delta_known,
        "bench_count_delta": bench_count_delta,
        "discard_count_delta_known": discard_count_delta_known,
        "active_changes": active_changes,
        "turn_ends": action_type in {"attack", "end_turn"},
        "supporter_flag_set": action_type == "play_supporter",
        "attachment_flag_set": action_type == "play_energy",
        "retreat_flag_set": action_type == "retreat",
        "cards_drawn_known": draw_count,
        "reveals_hidden_cards": bool(action.get("search_deck_ids")),
    }


def _source_card_definition_for_action(
    state: GameState,
    player_index: int,
    action: dict[str, Any],
):
    instance_id = _action_card_instance_id(state, player_index, action)
    if not isinstance(instance_id, str):
        return None
    return card_definition(state, instance_id)


def _effect_specs_for_action(
    state: GameState,
    player_index: int,
    action: dict[str, Any],
) -> tuple[EffectSpec, ...] | tuple[Any, ...]:
    action_type = str(action.get("type", ""))
    if action_type in {"play_supporter", "play_item"}:
        source_card = _source_card_definition_for_action(state, player_index, action)
        return source_card.effect_specs if source_card is not None else ()
    attack = _attack_definition_for_action(state, player_index, action)
    if attack is not None:
        return attack.effect_specs
    return ()


def _attack_definition_for_action(
    state: GameState,
    player_index: int,
    action: dict[str, Any],
) -> AttackDefinition | None:
    if str(action.get("type", "")) != "attack":
        return None
    attack_index = action.get("attack_index")
    if not isinstance(attack_index, int):
        return None
    player = state.players[player_index]
    active_card = get_top_card_definition(state, player.active)
    if active_card is None or not 0 <= attack_index < len(active_card.attacks):
        return None
    return active_card.attacks[attack_index]


def _search_count_for_destination(
    state: GameState,
    player_index: int,
    action: dict[str, Any],
    destination_zone: str,
) -> int:
    selected_count = len(action.get("search_deck_ids") or [])
    if selected_count <= 0:
        return 0
    for effect_spec in _effect_specs_for_action(state, player_index, action):
        if getattr(effect_spec, "effect_type", None) != "search_deck":
            continue
        if getattr(effect_spec, "destination_zone", None) == destination_zone:
            return selected_count
    return 0


def _draw_count_for_action(
    state: GameState,
    player_index: int,
    action: dict[str, Any],
) -> int | None:
    total_draw = 0
    found_any = False
    for effect_spec in _effect_specs_for_action(state, player_index, action):
        effect_type = getattr(effect_spec, "effect_type", None)
        if effect_type not in {"draw", "draw_cards"}:
            continue
        found_any = True
        total_draw += int(getattr(effect_spec, "count", None) or getattr(effect_spec, "amount", 0) or 0)
    if str(action.get("type", "")) != "attack" and not found_any:
        return 0
    return total_draw if found_any else None


def _shuffles_remaining_hand_into_deck(
    state: GameState,
    player_index: int,
    action: dict[str, Any],
) -> bool:
    return any(
        getattr(effect_spec, "effect_type", None) == "shuffle_zone_into_deck"
        and getattr(effect_spec, "source_zone", None) == "hand"
        and getattr(effect_spec, "destination_zone", None) == "deck"
        for effect_spec in _effect_specs_for_action(state, player_index, action)
    )


def _append_unique(values: list[str], value: str | None) -> None:
    if not isinstance(value, str) or not value or value in values:
        return
    values.append(value)
