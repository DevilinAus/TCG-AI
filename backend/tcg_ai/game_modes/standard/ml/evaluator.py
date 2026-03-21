from __future__ import annotations

import re
from typing import Any

from ..engine import card_definition, get_top_card_definition
from ..models import GameState, PokemonInPlay

_DAMAGE_PATTERN = re.compile(r"\d+")


def evaluate_state(state: GameState, perspective_player: int) -> float:
    if state.winner == perspective_player:
        return 10_000.0
    if state.winner is not None:
        return -10_000.0

    player_score = _evaluate_player_board(state, perspective_player)
    opponent_score = _evaluate_player_board(state, 1 - perspective_player)
    tempo_bonus = 1.5 if state.current_player == perspective_player else -1.5
    return round(player_score - opponent_score + tempo_bonus, 6)


def score_action_prior(
    state: GameState,
    player_index: int,
    action: dict[str, Any],
) -> float:
    action_type = str(action.get("type", ""))
    priorities = {
        "attack": 45.0,
        "play_supporter": 28.0,
        "play_item": 18.0,
        "play_energy": 16.0,
        "evolve": 15.0,
        "bench_basic": 12.0,
        "retreat": 8.0,
        "play_basic_to_active": 30.0,
        "end_setup": 4.0,
        "end_turn": -12.0,
        "mulligan": -8.0,
    }
    score = priorities.get(action_type, 0.0)

    hand_card_id = action.get("hand_card_id")
    if isinstance(hand_card_id, str) and hand_card_id in state.cards:
        card = card_definition(state, hand_card_id)
        score += _card_board_value(card.hp, card.attacks)
        if card.is_basic:
            score += 6.0

    attack_index = action.get("attack_index")
    if isinstance(attack_index, int):
        active = state.players[player_index].active
        active_card = get_top_card_definition(state, active)
        if active_card is not None and 0 <= attack_index < len(active_card.attacks):
            attack = active_card.attacks[attack_index]
            score += _attack_damage_value(attack.damage) * 0.4
            score -= attack.cost * 1.5

    return round(score, 6)


def _evaluate_player_board(state: GameState, player_index: int) -> float:
    player = state.players[player_index]
    active = player.active
    bench = player.bench
    hp_total = _remaining_hp(state, active) + sum(_remaining_hp(state, pokemon) for pokemon in bench)
    pressure = _attack_pressure(state, active) + sum(_attack_pressure(state, pokemon) for pokemon in bench)
    attached_energy_total = _attached_energy_count(active) + sum(
        _attached_energy_count(pokemon) for pokemon in bench
    )
    basics_in_hand = sum(
        1 for instance_id in player.hand if card_definition(state, instance_id).is_basic
    )
    return (
        (6 - player.prize_cards_remaining) * 30.0
        + hp_total * 0.16
        + pressure * 0.35
        + attached_energy_total * 2.4
        + len(bench) * 4.0
        + len(player.hand) * 1.2
        + basics_in_hand * 1.5
        - len(player.deck) * 0.02
    )


def _remaining_hp(state: GameState, pokemon: PokemonInPlay | None) -> int:
    if pokemon is None:
        return 0
    top_card = get_top_card_definition(state, pokemon)
    if top_card is None or top_card.hp is None:
        return 0
    return max(0, top_card.hp - pokemon.damage)


def _attack_pressure(state: GameState, pokemon: PokemonInPlay | None) -> float:
    top_card = get_top_card_definition(state, pokemon)
    if top_card is None:
        return 0.0
    if not top_card.attacks:
        return 0.0
    attached_energy = _attached_energy_count(pokemon)
    return max(_attack_readiness_value(attack, attached_energy) for attack in top_card.attacks)


def _card_board_value(hp: int | None, attacks: tuple[Any, ...]) -> float:
    max_damage = 0.0
    min_cost = 5.0
    for attack in attacks:
        max_damage = max(max_damage, _attack_damage_value(getattr(attack, "damage", "")))
        min_cost = min(min_cost, float(getattr(attack, "cost", 5)))
    hp_value = float(hp or 0) * 0.12
    return hp_value + max_damage * 0.3 - min_cost * 1.5


def _attack_damage_value(damage_text: str) -> float:
    matches = _DAMAGE_PATTERN.findall(str(damage_text))
    if not matches:
        return 0.0
    return float(sum(int(match) for match in matches))


def _attack_readiness_value(attack: Any, attached_energy: int) -> float:
    base_value = _card_board_value(None, (attack,))
    attack_cost = max(0, int(getattr(attack, "cost", 0) or 0))
    if attack_cost == 0:
        return base_value
    readiness_ratio = min(1.0, attached_energy / attack_cost)
    return base_value * (0.28 + readiness_ratio * 0.72)


def _attached_energy_count(pokemon: PokemonInPlay | None) -> int:
    if pokemon is None:
        return 0
    return len(pokemon.attached_energy)
