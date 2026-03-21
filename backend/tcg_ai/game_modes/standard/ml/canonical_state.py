from __future__ import annotations

from typing import Any

from ..models import AttackDefinition, CardDefinition, CardInstance, GameState, PlayerState, PokemonInPlay
import random

SCHEMA_VERSION = 3


def serialize_state(state: GameState) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "seed": state.seed,
        "rng_state": _encode_jsonable(state.rng.getstate()),
        "turn_number": state.turn_number,
        "current_player": state.current_player,
        "winner": state.winner,
        "setup_phase": state.setup_phase,
        "log": list(state.log),
        "cards": {
            instance_id: {
                "instance_id": card.instance_id,
                "card_id": card.card_id,
                "owner": card.owner,
            }
            for instance_id, card in state.cards.items()
        },
        "card_definitions": {
            card_id: _serialize_card_definition(definition)
            for card_id, definition in state.card_definitions.items()
        },
        "players": [_serialize_player(player) for player in state.players],
    }


def deserialize_state(payload: dict[str, Any]) -> GameState:
    rng = random.Random()
    rng_state = payload.get("rng_state")
    if rng_state is not None:
        rng.setstate(_decode_jsonable(rng_state))
    elif "seed" in payload:
        rng.seed(int(payload["seed"]))

    cards = {
        instance_id: CardInstance(
            instance_id=str(card_payload["instance_id"]),
            card_id=str(card_payload["card_id"]),
            owner=int(card_payload["owner"]),
        )
        for instance_id, card_payload in payload.get("cards", {}).items()
    }
    card_definitions = {
        card_id: _deserialize_card_definition(card_payload)
        for card_id, card_payload in payload.get("card_definitions", {}).items()
    }
    players = [_deserialize_player(player_payload) for player_payload in payload.get("players", [])]
    return GameState(
        cards=cards,
        card_definitions=card_definitions,
        players=players,
        current_player=int(payload["current_player"]),
        rng=rng,
        turn_number=int(payload.get("turn_number", 1)),
        winner=payload.get("winner"),
        log=[str(entry) for entry in payload.get("log", [])],
        seed=int(payload.get("seed", 0)),
        setup_phase=payload.get("setup_phase"),
    )


def _serialize_card_definition(definition: CardDefinition) -> dict[str, Any]:
    return {
        "card_id": definition.card_id,
        "name": definition.name,
        "kind": definition.kind,
        "element": definition.element,
        "stage": definition.stage,
        "is_basic": definition.is_basic,
        "hp": definition.hp,
        "image_url": definition.image_url,
        "attacks": [
            {
                "name": attack.name,
                "cost": attack.cost,
                "damage": attack.damage,
                "effect": attack.effect,
                "text": attack.text,
            }
            for attack in definition.attacks
        ],
    }


def _deserialize_card_definition(payload: dict[str, Any]) -> CardDefinition:
    return CardDefinition(
        card_id=str(payload["card_id"]),
        name=str(payload["name"]),
        kind=str(payload["kind"]),
        element=payload.get("element"),
        stage=payload.get("stage"),
        is_basic=bool(payload.get("is_basic", False)),
        hp=None if payload.get("hp") is None else int(payload["hp"]),
        image_url=payload.get("image_url"),
        attacks=tuple(
            AttackDefinition(
                name=str(attack["name"]),
                cost=int(attack["cost"]),
                damage=str(attack.get("damage", "")),
                effect=str(attack.get("effect", "none")),
                text=str(attack.get("text", "")),
            )
            for attack in payload.get("attacks", [])
        ),
    )


def _serialize_player(player: PlayerState) -> dict[str, Any]:
    return {
        "name": player.name,
        "deck_name": player.deck_name,
        "element": player.element,
        "deck": list(player.deck),
        "hand": list(player.hand),
        "discard": list(player.discard),
        "active": _serialize_pokemon(player.active),
        "bench": [_serialize_pokemon(pokemon) for pokemon in player.bench],
        "prize_cards_remaining": player.prize_cards_remaining,
        "mulligans_taken": player.mulligans_taken,
        "supporter_played_this_turn": player.supporter_played_this_turn,
        "energy_attached_this_turn": player.energy_attached_this_turn,
    }


def _deserialize_player(payload: dict[str, Any]) -> PlayerState:
    return PlayerState(
        name=str(payload["name"]),
        deck_name=str(payload["deck_name"]),
        element=str(payload["element"]),
        deck=[str(card_id) for card_id in payload.get("deck", [])],
        hand=[str(card_id) for card_id in payload.get("hand", [])],
        discard=[str(card_id) for card_id in payload.get("discard", [])],
        active=_deserialize_pokemon(payload.get("active")),
        bench=[_deserialize_pokemon(pokemon) for pokemon in payload.get("bench", [])],
        prize_cards_remaining=int(payload.get("prize_cards_remaining", 6)),
        mulligans_taken=int(payload.get("mulligans_taken", 0)),
        supporter_played_this_turn=bool(payload.get("supporter_played_this_turn", False)),
        energy_attached_this_turn=bool(payload.get("energy_attached_this_turn", False)),
    )


def _serialize_pokemon(pokemon: PokemonInPlay | None) -> dict[str, Any] | None:
    if pokemon is None:
        return None
    return {
        "stack": list(pokemon.stack),
        "damage": pokemon.damage,
        "attached_energy": list(pokemon.attached_energy),
    }


def _deserialize_pokemon(payload: dict[str, Any] | None) -> PokemonInPlay | None:
    if payload is None:
        return None
    return PokemonInPlay(
        stack=[str(instance_id) for instance_id in payload.get("stack", [])],
        damage=int(payload.get("damage", 0)),
        attached_energy=[str(instance_id) for instance_id in payload.get("attached_energy", [])],
    )


def _encode_jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return {"__tuple__": [_encode_jsonable(item) for item in value]}
    if isinstance(value, list):
        return [_encode_jsonable(item) for item in value]
    return value


def _decode_jsonable(value: Any) -> Any:
    if isinstance(value, dict) and "__tuple__" in value:
        return tuple(_decode_jsonable(item) for item in value["__tuple__"])
    if isinstance(value, list):
        return [_decode_jsonable(item) for item in value]
    return value
