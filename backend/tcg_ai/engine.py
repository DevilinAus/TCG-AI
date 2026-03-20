from __future__ import annotations

from collections import defaultdict
import random
from typing import Any

from .cards import (
    CARD_DEFINITIONS,
    DEFAULT_HUMAN_DECK_ID,
    DECK_DEFINITIONS,
    DeckDefinition,
    paired_deck_id_for,
)
from .models import CardDefinition, CardInstance, GameState, PlayerState, PokemonInPlay

BENCH_LIMIT = 3
HEAL_AMOUNT = 30


def create_game(
    seed: int | None = None,
    human_first: bool = True,
    ai_name: str = "AI",
    human_deck_id: str = DEFAULT_HUMAN_DECK_ID,
) -> GameState:
    """Create a new game for the selected My First Battle deck pairing."""
    if seed is None:
        seed = random.randint(1, 999_999)

    rng = random.Random(seed)
    human_deck = DECK_DEFINITIONS[human_deck_id]
    ai_deck = DECK_DEFINITIONS[paired_deck_id_for(human_deck_id)]
    human_player, human_cards = _build_player_state("You", human_deck, 0, rng)
    ai_player, ai_cards = _build_player_state(ai_name, ai_deck, 1, rng)

    current_player = 0 if human_first else 1
    state = GameState(
        cards={**human_cards, **ai_cards},
        players=[human_player, ai_player],
        current_player=current_player,
        rng=rng,
        seed=seed,
    )
    state.log.append(f"New game started: {human_deck.name} vs {ai_deck.name}.")
    _begin_turn(state, current_player)
    return state


def list_legal_actions(state: GameState) -> list[dict[str, Any]]:
    if state.winner is not None:
        return []

    player_index = state.current_player
    player = state.players[player_index]

    if state.pending_promotion_for is not None:
        if state.pending_promotion_for != player_index:
            return []
        actions: list[dict[str, Any]] = []
        for bench_index, pokemon in enumerate(player.bench):
            actions.append(
                {
                    "type": "promote",
                    "bench_index": bench_index,
                    "label": f"Promote {_pokemon_name(state, pokemon)}",
                }
            )
        return actions

    actions = []

    if len(player.bench) < BENCH_LIMIT:
        for instance_id in player.hand:
            card = card_definition(state, instance_id)
            if card.kind == "pokemon" and card.stage == "basic":
                actions.append(
                    {
                        "type": "bench_basic",
                        "hand_card_id": instance_id,
                        "label": f"Bench {card.name}",
                    }
                )

    if not player.energy_played_this_turn:
        for instance_id in player.hand:
            card = card_definition(state, instance_id)
            if card.kind == "energy":
                actions.append(
                    {
                        "type": "play_energy",
                        "hand_card_id": instance_id,
                        "label": f"Play {card.name} to the shared Energy spot",
                    }
                )

    for instance_id in player.hand:
        card = card_definition(state, instance_id)
        if card.kind != "pokemon" or card.stage != "stage1":
            continue

        if _can_evolve(state, player, player.active, card):
            base_name = _pokemon_name(state, player.active)
            actions.append(
                {
                    "type": "evolve",
                    "hand_card_id": instance_id,
                    "target": "active",
                    "label": f"Evolve active {base_name} into {card.name}",
                }
            )

        for bench_index, pokemon in enumerate(player.bench):
            if _can_evolve(state, player, pokemon, card):
                base_name = _pokemon_name(state, pokemon)
                actions.append(
                    {
                        "type": "evolve",
                        "hand_card_id": instance_id,
                        "target": f"bench:{bench_index}",
                        "label": f"Evolve benched {base_name} into {card.name}",
                    }
                )

    for instance_id in player.hand:
        card = card_definition(state, instance_id)
        if card.kind != "trainer":
            continue

        if card.trainer_effect == "heal_30":
            if player.active and player.active.damage > 0:
                actions.append(
                    {
                        "type": "play_potion",
                        "hand_card_id": instance_id,
                        "target": "active",
                        "label": f"Use Potion on active {_pokemon_name(state, player.active)}",
                    }
                )
            for bench_index, pokemon in enumerate(player.bench):
                if pokemon.damage > 0:
                    actions.append(
                        {
                            "type": "play_potion",
                            "hand_card_id": instance_id,
                            "target": f"bench:{bench_index}",
                            "label": f"Use Potion on benched {_pokemon_name(state, pokemon)}",
                        }
                    )

        if card.trainer_effect == "switch_active" and player.active and player.bench:
            for bench_index, pokemon in enumerate(player.bench):
                actions.append(
                    {
                        "type": "play_switch",
                        "hand_card_id": instance_id,
                        "bench_index": bench_index,
                        "label": f"Switch with benched {_pokemon_name(state, pokemon)}",
                    }
                )

    active_card = get_top_card_definition(state, player.active)
    if active_card is not None:
        for attack_index, attack in enumerate(active_card.attacks):
            if len(player.energy_zone) >= attack.cost:
                actions.append(
                    {
                        "type": "attack",
                        "attack_index": attack_index,
                        "label": f"Attack with {attack.name}",
                    }
                )

    actions.append({"type": "end_turn", "label": "End turn"})
    return actions


def apply_action(state: GameState, action: dict[str, Any]) -> GameState:
    legal_actions = list_legal_actions(state)
    if action not in legal_actions:
        raise ValueError("Illegal action.")

    action_type = action["type"]

    if action_type == "promote":
        _promote_from_bench(state, action["bench_index"])
        return state

    player = state.players[state.current_player]

    if action_type == "bench_basic":
        card_id = _remove_from_hand(player, action["hand_card_id"])
        card = card_definition(state, card_id)
        player.bench.append(PokemonInPlay(stack=[card_id], entered_play_turn=state.turn_number))
        state.log.append(f"{player.name} benched {card.name}.")
        return state

    if action_type == "play_energy":
        card_id = _remove_from_hand(player, action["hand_card_id"])
        card = card_definition(state, card_id)
        player.energy_zone.append(card_id)
        player.energy_played_this_turn = True
        state.log.append(f"{player.name} played {card.name} to the shared Energy spot.")
        return state

    if action_type == "evolve":
        card_id = _remove_from_hand(player, action["hand_card_id"])
        card = card_definition(state, card_id)
        target = _resolve_target_slot(player, action["target"])
        previous_name = _pokemon_name(state, target)
        target.stack.append(card_id)
        state.log.append(f"{player.name} evolved {previous_name} into {card.name}.")
        return state

    if action_type == "play_potion":
        card_id = _remove_from_hand(player, action["hand_card_id"])
        target = _resolve_target_slot(player, action["target"])
        healed = min(HEAL_AMOUNT, target.damage)
        target.damage = max(0, target.damage - HEAL_AMOUNT)
        player.discard.append(card_id)
        state.log.append(f"{player.name} used Potion and healed {healed} damage.")
        return state

    if action_type == "play_switch":
        card_id = _remove_from_hand(player, action["hand_card_id"])
        bench_index = action["bench_index"]
        bench_pokemon = player.bench.pop(bench_index)
        previous_active = player.active
        player.active = bench_pokemon
        if previous_active is not None:
            player.bench.append(previous_active)
        player.discard.append(card_id)
        state.log.append(f"{player.name} used Switch and promoted {_pokemon_name(state, player.active)}.")
        return state

    if action_type == "attack":
        _execute_attack(state, action["attack_index"])
        return state

    if action_type == "end_turn":
        state.log.append(f"{player.name} ended the turn without attacking.")
        _advance_turn(state)
        return state

    raise ValueError(f"Unsupported action type: {action_type}")


def serialize_state(state: GameState, viewer: int = 0) -> dict[str, Any]:
    return {
        "seed": state.seed,
        "turn_number": state.turn_number,
        "current_player": state.current_player,
        "winner": state.winner,
        "pending_promotion_for": state.pending_promotion_for,
        "human_player": viewer,
        "log": state.log[-20:],
        "players": [
            _serialize_player_state(state, index, viewer) for index in range(len(state.players))
        ],
        "legal_actions": list_legal_actions(state) if state.current_player == viewer else [],
    }


def get_top_card_definition(state: GameState, pokemon: PokemonInPlay | None) -> CardDefinition | None:
    if pokemon is None or not pokemon.stack:
        return None
    return card_definition(state, pokemon.stack[-1])


def card_definition(state: GameState, instance_id: str) -> CardDefinition:
    return CARD_DEFINITIONS[state.cards[instance_id].card_id]


def _build_player_state(
    name: str, deck_definition: DeckDefinition, owner: int, rng: random.Random
) -> tuple[PlayerState, dict[str, CardInstance]]:
    counters: dict[str, int] = defaultdict(int)
    cards: dict[str, CardInstance] = {}
    deck_ids: list[str] = []

    for card_id, count in deck_definition.entries:
        for _ in range(count):
            counters[card_id] += 1
            instance_id = f"p{owner}_{card_id}_{counters[card_id]}"
            cards[instance_id] = CardInstance(instance_id=instance_id, card_id=card_id, owner=owner)
            deck_ids.append(instance_id)

    starter_card = _pop_first_matching(deck_ids, cards, deck_definition.starter_pokemon)
    rng.shuffle(deck_ids)
    hand = []
    for _ in range(3):
        if deck_ids:
            hand.append(deck_ids.pop(0))

    player = PlayerState(
        name=name,
        deck_name=deck_definition.name,
        element=deck_definition.element,
        deck=deck_ids,
        hand=hand,
        energy_zone=[],
        active=PokemonInPlay(stack=[starter_card], entered_play_turn=0),
    )
    return player, cards


def _begin_turn(state: GameState, player_index: int) -> None:
    state.current_player = player_index
    state.turn_number += 1
    player = state.players[player_index]
    player.turns_taken += 1
    player.energy_played_this_turn = False
    state.log.append(f"Turn {state.turn_number}: {_player_possessive(player.name)} turn.")
    if player.deck:
        drawn = player.deck.pop(0)
        player.hand.append(drawn)
        state.log.append(f"{player.name} drew {card_definition(state, drawn).name}.")
    else:
        state.log.append(f"{player.name} has no cards left to draw and skips the draw step.")


def _advance_turn(state: GameState) -> None:
    if state.winner is not None:
        return
    _begin_turn(state, 1 - state.current_player)


def _execute_attack(state: GameState, attack_index: int) -> None:
    attacker_index = state.current_player
    defender_index = 1 - attacker_index
    attacker = state.players[attacker_index]
    defender = state.players[defender_index]

    active_card = get_top_card_definition(state, attacker.active)
    defending_card = get_top_card_definition(state, defender.active)
    if active_card is None or defending_card is None or defender.active is None:
        raise ValueError("Attack requires both players to have an active Pokemon.")

    attack = active_card.attacks[attack_index]
    damage = attack.damage

    if attack.effect == "coin_flip_bonus_20":
        coin = _flip_coin(state)
        if coin == "heads":
            damage += 20
            state.log.append(f"{attacker.name} flipped heads, so {attack.name} does 20 extra damage.")
        else:
            state.log.append(f"{attacker.name} flipped tails, so {attack.name} stays at {damage} damage.")
    elif attack.effect == "coin_flip_bonus_30":
        coin = _flip_coin(state)
        if coin == "heads":
            damage += 30
            state.log.append(f"{attacker.name} flipped heads, so {attack.name} does 30 extra damage.")
        else:
            state.log.append(f"{attacker.name} flipped tails, so {attack.name} stays at {damage} damage.")
    elif attack.effect == "coin_flip_bonus_40":
        coin = _flip_coin(state)
        if coin == "heads":
            damage += 40
            state.log.append(f"{attacker.name} flipped heads, so {attack.name} does 40 extra damage.")
        else:
            state.log.append(f"{attacker.name} flipped tails, so {attack.name} stays at {damage} damage.")
    elif attack.effect == "coin_flip_fail":
        coin = _flip_coin(state)
        if coin == "tails":
            damage = 0
            state.log.append(f"{attacker.name} flipped tails, so {attack.name} does nothing.")
        else:
            state.log.append(f"{attacker.name} flipped heads, so {attack.name} lands.")
    elif attack.effect == "bonus_per_benched_matching_element_20":
        benched_matches = _count_benched_matching_element(state, attacker_index, active_card.element)
        if benched_matches:
            damage += benched_matches * 20
            state.log.append(
                f"{attack.name} gains {benched_matches * 20} bonus damage from matching benched Pokemon."
            )

    defender.active.damage += damage
    state.log.append(
        f"{_player_possessive(attacker.name)} {active_card.name} used {attack.name} for {damage} damage."
    )

    if attack.effect == "heal_self_10" and attacker.active is not None:
        healed = min(10, attacker.active.damage)
        attacker.active.damage = max(0, attacker.active.damage - 10)
        if healed:
            state.log.append(f"{active_card.name} healed {healed} damage.")
    elif attack.effect == "heal_self_20" and attacker.active is not None:
        healed = min(20, attacker.active.damage)
        attacker.active.damage = max(0, attacker.active.damage - 20)
        if healed:
            state.log.append(f"{active_card.name} healed {healed} damage.")

    if defender.active.damage >= defending_card.hp:
        _resolve_knock_out(state, attacker_index, defender_index)
        return

    _advance_turn(state)


def _resolve_knock_out(state: GameState, attacker_index: int, defender_index: int) -> None:
    attacker = state.players[attacker_index]
    defender = state.players[defender_index]
    knocked_out = defender.active
    if knocked_out is None:
        return

    knocked_out_name = _pokemon_name(state, knocked_out)
    defender.discard.extend(knocked_out.stack)
    defender.active = None
    attacker.prize_tokens_remaining = max(0, attacker.prize_tokens_remaining - 1)
    state.log.append(f"{knocked_out_name} was Knocked Out.")
    state.log.append(f"{attacker.name} took a Prize token. {attacker.prize_tokens_remaining} remaining.")

    if attacker.prize_tokens_remaining == 0:
        state.winner = attacker_index
        state.log.append(f"{attacker.name} wins by taking all Prize tokens.")
        return

    if not defender.bench:
        state.winner = attacker_index
        state.log.append(f"{attacker.name} wins because {defender.name} has no Pokemon left in play.")
        return

    state.current_player = defender_index
    state.pending_promotion_for = defender_index
    state.turn_starts_after_promotion = True
    state.log.append(f"{defender.name} must choose a new Active Pokemon.")


def _promote_from_bench(state: GameState, bench_index: int) -> None:
    player = state.players[state.current_player]
    player.active = player.bench.pop(bench_index)
    state.log.append(f"{player.name} promoted {_pokemon_name(state, player.active)}.")
    state.pending_promotion_for = None
    if state.turn_starts_after_promotion:
        state.turn_starts_after_promotion = False
        _begin_turn(state, state.current_player)


def _flip_coin(state: GameState) -> str:
    return state.rng.choice(["heads", "tails"])


def _remove_from_hand(player: PlayerState, instance_id: str) -> str:
    player.hand.remove(instance_id)
    return instance_id


def _can_evolve(
    state: GameState,
    player: PlayerState,
    pokemon: PokemonInPlay | None,
    evolution_card: CardDefinition,
) -> bool:
    if pokemon is None:
        return False
    if player.turns_taken <= 1:
        return False
    if pokemon.entered_play_turn >= state.turn_number:
        return False
    base_card = get_top_card_definition(state, pokemon)
    return base_card is not None and base_card.name == evolution_card.evolves_from


def _resolve_target_slot(player: PlayerState, target: str) -> PokemonInPlay:
    if target == "active":
        if player.active is None:
            raise ValueError("No active Pokemon.")
        return player.active
    prefix, _, value = target.partition(":")
    if prefix != "bench":
        raise ValueError(f"Unsupported target: {target}")
    return player.bench[int(value)]


def _pokemon_name(state: GameState, pokemon: PokemonInPlay | None) -> str:
    card = get_top_card_definition(state, pokemon)
    return card.name if card else "Unknown Pokemon"


def _player_possessive(name: str) -> str:
    if name == "You":
        return "Your"
    return f"{name}'s"


def _count_benched_matching_element(
    state: GameState,
    player_index: int,
    element: str | None,
) -> int:
    if not element:
        return 0

    count = 0
    for pokemon in state.players[player_index].bench:
        benched_card = get_top_card_definition(state, pokemon)
        if benched_card is not None and benched_card.element == element:
            count += 1
    return count


def _pop_first_matching(
    deck_ids: list[str], cards: dict[str, CardInstance], card_id: str
) -> str:
    for index, instance_id in enumerate(deck_ids):
        if cards[instance_id].card_id == card_id:
            return deck_ids.pop(index)
    raise ValueError(f"Could not find required card {card_id} in deck.")


def _serialize_player_state(state: GameState, player_index: int, viewer: int) -> dict[str, Any]:
    player = state.players[player_index]
    return {
        "index": player_index,
        "name": player.name,
        "deck_name": player.deck_name,
        "element": player.element,
        "hand_count": len(player.hand),
        "hand": [
            _serialize_hand_card(state, instance_id)
            for instance_id in player.hand
            if player_index == viewer
        ],
        "deck_count": len(player.deck),
        "discard_count": len(player.discard),
        "discard_top": _serialize_discard_top(state, player),
        "energy_count": len(player.energy_zone),
        "prize_tokens_remaining": player.prize_tokens_remaining,
        "active": _serialize_pokemon(state, player.active),
        "bench": [_serialize_pokemon(state, pokemon) for pokemon in player.bench],
    }


def _serialize_hand_card(state: GameState, instance_id: str) -> dict[str, Any]:
    card = card_definition(state, instance_id)
    return {
        "instance_id": instance_id,
        "name": card.name,
        "kind": card.kind,
        "stage": card.stage,
    }


def _serialize_discard_top(state: GameState, player: PlayerState) -> str | None:
    if not player.discard:
        return None
    return card_definition(state, player.discard[-1]).name


def _serialize_pokemon(state: GameState, pokemon: PokemonInPlay | None) -> dict[str, Any] | None:
    if pokemon is None:
        return None

    top_card = get_top_card_definition(state, pokemon)
    if top_card is None:
        return None

    hp = top_card.hp or 0
    return {
        "name": top_card.name,
        "stage": top_card.stage,
        "hp": hp,
        "damage": pokemon.damage,
        "remaining_hp": max(0, hp - pokemon.damage),
        "stack": [card_definition(state, instance_id).name for instance_id in pokemon.stack],
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
