from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
from typing import Any

from .engine import card_definition, get_top_card_definition, list_legal_actions
from .models import GameState, PlayerState, PokemonInPlay

PROJECT_ROOT = Path(__file__).resolve().parents[4]
CARD_MANIFEST_PATH = (
    PROJECT_ROOT / "frontend" / "assets" / "cards" / "my-first-battle" / "manifest.json"
)
CARD_ASSET_BASE_URL = "/assets/cards/my-first-battle"
FACEDOWN_CARD_ASSET_URL = "/assets/cards/shared/card-back.svg"


@lru_cache(maxsize=1)
def _load_card_manifest() -> dict[str, str]:
    payload = json.loads(CARD_MANIFEST_PATH.read_text(encoding="utf-8"))
    return payload.get("cards", {})


def card_image_url(card_id: str) -> str | None:
    filename = _load_card_manifest().get(card_id)
    if not filename:
        return None
    return f"{CARD_ASSET_BASE_URL}/{filename}"


def facedown_card_image_url() -> str:
    return FACEDOWN_CARD_ASSET_URL


def serialize_state(
    state: GameState,
    session_id: str,
    viewer: int = 0,
    ai_learning: dict[str, Any] | None = None,
) -> dict[str, Any]:
    raw_actions = list_legal_actions(state) if state.current_player == viewer else []
    action_views = [_serialize_legal_action(state, action) for action in raw_actions]
    return {
        "session_id": session_id,
        "seed": state.seed,
        "turn_number": state.turn_number,
        "current_player": state.current_player,
        "winner": state.winner,
        "pending_promotion_for": state.pending_promotion_for,
        "human_player": viewer,
        "matchup_label": f"{state.players[0].deck_name} vs {state.players[1].deck_name}",
        "shared_assets": {
            "face_down_card_image_url": facedown_card_image_url(),
        },
        "log": _serialize_log_entries(state.log[-30:]),
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
        "discard_count": len(player.discard),
        "discard_top": _serialize_discard_top(state, player),
        "energy_count": len(player.energy_zone),
        "energy_zone": [_serialize_card_instance(state, instance_id) for instance_id in player.energy_zone],
        "prize_tokens_remaining": player.prize_tokens_remaining,
        "prize_pile": {
            "count": player.prize_tokens_remaining,
            "face_down": True,
        },
        "requires_promotion": state.pending_promotion_for == player_index,
        "active": _serialize_pokemon(
            state,
            player.active,
            player_index,
            "active",
            None,
            action_views,
        ),
        "bench": [
            _serialize_pokemon(state, pokemon, player_index, "bench", bench_index, action_views)
            for bench_index, pokemon in enumerate(player.bench)
        ],
    }


def _serialize_hand_card(
    state: GameState,
    player_index: int,
    instance_id: str,
    action_views: list[dict[str, Any]],
) -> dict[str, Any]:
    card = _serialize_card_instance(state, instance_id)
    action_types = sorted(
        {
            action["type"]
            for action in action_views
            if _ref_matches(action.get("source"), {"player_index": player_index, "zone": "hand", "instance_id": instance_id})
        }
    )
    return {
        **card,
        "playable": bool(action_types),
        "action_types": action_types,
    }


def _serialize_discard_top(state: GameState, player: PlayerState) -> dict[str, Any] | None:
    if not player.discard:
        return None
    return _serialize_card_instance(state, player.discard[-1])


def _serialize_pokemon(
    state: GameState,
    pokemon: PokemonInPlay | None,
    player_index: int,
    zone: str,
    bench_index: int | None,
    action_views: list[dict[str, Any]],
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
    hp = top_card.hp or 0
    return {
        "instance_id": top_instance_id,
        "card_id": top_card.card_id,
        "image_url": card_image_url(top_card.card_id),
        "name": top_card.name,
        "kind": top_card.kind,
        "element": top_card.element,
        "stage": top_card.stage,
        "hp": hp,
        "damage": pokemon.damage,
        "remaining_hp": max(0, hp - pokemon.damage),
        "ref": ref,
        "stack": [_serialize_card_instance(state, instance_id) for instance_id in pokemon.stack],
        "attacks": [
            {
                "name": attack.name,
                "cost": attack.cost,
                "damage": attack.damage,
                "effect": attack.effect,
            }
            for attack in top_card.attacks
        ],
        "source_action_types": source_action_types,
        "target_action_types": target_action_types,
        "interactive": bool(source_action_types or target_action_types),
        "can_attack": "attack" in source_action_types,
        "requires_promotion": "promote" in source_action_types,
    }


def _serialize_card_instance(state: GameState, instance_id: str) -> dict[str, Any]:
    card = card_definition(state, instance_id)
    return {
        "instance_id": instance_id,
        "card_id": card.card_id,
        "name": card.name,
        "kind": card.kind,
        "stage": card.stage,
        "element": card.element,
        "image_url": card_image_url(card.card_id),
    }


def _serialize_legal_action(state: GameState, action: dict[str, Any]) -> dict[str, Any]:
    player_index = state.current_player
    action_type = action["type"]
    view = {
        "action_id": _action_id(action),
        "type": action_type,
        "label": action["label"],
        "action": action,
    }

    source = _serialize_action_source(state, player_index, action)
    target = _serialize_action_target(state, player_index, action)
    if source is not None:
        view["source"] = source
    if target is not None:
        view["target"] = target
    if action_type == "attack":
        active_card = get_top_card_definition(state, state.players[player_index].active)
        if active_card is not None:
            attack = active_card.attacks[action["attack_index"]]
            view["attack"] = {
                "name": attack.name,
                "cost": attack.cost,
                "damage": attack.damage,
                "effect": attack.effect,
            }
    return view


def _serialize_action_source(
    state: GameState, player_index: int, action: dict[str, Any]
) -> dict[str, Any] | None:
    player = state.players[player_index]
    action_type = action["type"]

    if action_type in {"bench_basic", "play_energy", "evolve", "play_potion", "play_switch"}:
        instance_id = action["hand_card_id"]
        card = card_definition(state, instance_id)
        return {
            "player_index": player_index,
            "zone": "hand",
            "instance_id": instance_id,
            "card_id": card.card_id,
            "name": card.name,
        }

    if action_type == "attack":
        active = player.active
        if active is None:
            return None
        top_instance_id = active.stack[-1]
        top_card = card_definition(state, top_instance_id)
        return {
            "player_index": player_index,
            "zone": "active",
            "instance_id": top_instance_id,
            "card_id": top_card.card_id,
            "name": top_card.name,
        }

    if action_type == "promote":
        bench_index = action["bench_index"]
        pokemon = player.bench[bench_index]
        top_instance_id = pokemon.stack[-1]
        top_card = card_definition(state, top_instance_id)
        return {
            "player_index": player_index,
            "zone": "bench",
            "bench_index": bench_index,
            "instance_id": top_instance_id,
            "card_id": top_card.card_id,
            "name": top_card.name,
        }

    if action_type == "end_turn":
        return {
            "player_index": player_index,
            "zone": "system",
            "instance_id": None,
            "card_id": None,
            "name": "Turn Controls",
        }

    return None


def _serialize_action_target(
    state: GameState, player_index: int, action: dict[str, Any]
) -> dict[str, Any] | None:
    player = state.players[player_index]
    opponent = state.players[1 - player_index]
    action_type = action["type"]

    if action_type == "play_energy":
        return {
            "player_index": player_index,
            "zone": "energy",
            "bench_index": None,
            "instance_id": None,
            "card_id": None,
            "name": "Shared Energy",
        }

    if action_type in {"evolve", "play_potion"}:
        return _ref_from_target_string(state, player_index, player, action["target"])

    if action_type == "play_switch":
        bench_index = action["bench_index"]
        pokemon = player.bench[bench_index]
        return _pokemon_ref(state, player_index, "bench", pokemon, bench_index)

    if action_type == "promote":
        return {
            "player_index": player_index,
            "zone": "active",
            "bench_index": None,
            "instance_id": player.active.stack[-1] if player.active else None,
        }

    if action_type == "attack" and opponent.active is not None:
        return _pokemon_ref(state, 1 - player_index, "active", opponent.active, None)

    return None


def _ref_from_target_string(
    state: GameState,
    player_index: int,
    player: PlayerState,
    target: str,
) -> dict[str, Any]:
    if target == "active":
        if player.active is None:
            return {"player_index": player_index, "zone": "active", "bench_index": None, "instance_id": None}
        return _pokemon_ref(state, player_index, "active", player.active, None)

    _, _, raw_index = target.partition(":")
    bench_index = int(raw_index)
    return _pokemon_ref(state, player_index, "bench", player.bench[bench_index], bench_index)


def _pokemon_ref(
    state: GameState,
    player_index: int,
    zone: str,
    pokemon: PokemonInPlay,
    bench_index: int | None,
) -> dict[str, Any]:
    top_instance_id = pokemon.stack[-1]
    top_card = card_definition(state, top_instance_id)
    return {
        "player_index": player_index,
        "zone": zone,
        "bench_index": bench_index,
        "instance_id": top_instance_id,
        "card_id": top_card.card_id,
        "name": top_card.name,
    }


def _action_id(action: dict[str, Any]) -> str:
    action_type = action["type"]
    if action_type == "bench_basic":
        return f"bench_basic:{action['hand_card_id']}"
    if action_type == "play_energy":
        return f"play_energy:{action['hand_card_id']}"
    if action_type == "evolve":
        return f"evolve:{action['hand_card_id']}:{action['target']}"
    if action_type == "play_potion":
        return f"play_potion:{action['hand_card_id']}:{action['target']}"
    if action_type == "play_switch":
        return f"play_switch:{action['hand_card_id']}:{action['bench_index']}"
    if action_type == "attack":
        return f"attack:{action['attack_index']}"
    if action_type == "promote":
        return f"promote:{action['bench_index']}"
    return action_type


def _serialize_log_entries(entries: list[str]) -> list[dict[str, str]]:
    return [_serialize_log_entry(entry) for entry in entries]


def _serialize_log_entry(entry: str) -> dict[str, str]:
    side = "system"
    if entry.startswith("You ") or entry.startswith("Your "):
        side = "human"
    elif entry.startswith("AI ") or entry.startswith("AI's "):
        side = "ai"

    kind = "system"
    if "Knocked Out" in entry:
        kind = "ko"
    elif "wins" in entry:
        kind = "result"
    elif "used" in entry:
        kind = "attack"
    elif "drew" in entry:
        kind = "draw"
    elif "promoted" in entry:
        kind = "promotion"
    elif entry.startswith("Turn "):
        kind = "turn"

    return {"text": entry, "side": side, "kind": kind}


def _ref_matches(reference: dict[str, Any] | None, expected: dict[str, Any]) -> bool:
    if reference is None:
        return False
    for key, value in expected.items():
        if value is None:
            continue
        if reference.get(key) != value:
            return False
    return True
