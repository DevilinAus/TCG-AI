from __future__ import annotations

from collections import defaultdict
import random
import re
from typing import Any

from .cards import (
    DEFAULT_HUMAN_DECK_ID,
    DECK_DEFINITIONS,
    DeckCardDefinition,
    DeckDefinition,
    load_deck_cards,
    paired_deck_id_for,
)
from .models import (
    AttackDefinition,
    CardDefinition,
    CardInstance,
    EffectSpec,
    GameState,
    LingeringEffect,
    PlayerState,
    PokemonInPlay,
)

OPENING_HAND_SIZE = 7
BENCH_LIMIT = 5
_DAMAGE_PATTERN = re.compile(r"\d+")
_DYNAMIC_ACTION_FIELDS = frozenset({"discard_from_hand_ids", "search_result_ids"})


def create_game(
    seed: int | None = None,
    human_first: bool = True,
    ai_name: str = "AI",
    human_deck_id: str = DEFAULT_HUMAN_DECK_ID,
) -> GameState:
    del human_first
    if seed is None:
        seed = random.randint(1, 999_999)

    rng = random.Random(seed)
    human_deck = DECK_DEFINITIONS[human_deck_id]
    ai_deck = DECK_DEFINITIONS[paired_deck_id_for(human_deck_id)]
    human_player, human_instances, human_definitions = _build_player_state("You", human_deck, 0, rng)
    ai_player, ai_instances, ai_definitions = _build_player_state(ai_name, ai_deck, 1, rng)
    _ensure_deck_has_basic(ai_player, ai_instances, ai_definitions)
    while not _hand_has_basic(ai_player, ai_instances, ai_definitions):
        _redraw_opening_hand(ai_player, rng)

    state = GameState(
        cards={**human_instances, **ai_instances},
        card_definitions={**human_definitions, **ai_definitions},
        players=[human_player, ai_player],
        current_player=0,
        rng=rng,
        starting_player=0,
        seed=seed,
    )
    state.log.append(f"New Standard game started: {human_deck.name} vs {ai_deck.name}.")
    state.log.append(f"You drew {len(human_player.hand)} cards for your opening hand.")
    if _hand_has_basic(human_player, human_instances, human_definitions):
        state.log.append("Your opening hand contains a Basic Pokemon.")
    else:
        state.log.append("Your opening hand has no Basic Pokemon. Take a mulligan to redraw 7 cards.")
    state.log.append(f"{ai_name} drew {len(ai_player.hand)} cards for the opening hand.")
    if ai_player.mulligans_taken:
        state.log.append(
            f"{ai_name} mulliganed {ai_player.mulligans_taken} time"
            f"{'' if ai_player.mulligans_taken == 1 else 's'} before finding a Basic Pokemon."
        )
    state.log.append("Choose your Active Pokemon from the opening hand to continue setup.")
    return state


def choose_action(state: GameState, player_index: int, learner: Any | None = None) -> dict[str, Any] | None:
    del learner
    if state.current_player != player_index or state.winner is not None:
        return None

    legal_actions = list_legal_actions(state, player_index=player_index)
    if not legal_actions:
        return None

    if player_index == 1 and state.setup_phase is None:
        from .ml.planner import StandardTurnPlanner

        planner = StandardTurnPlanner()
        decision = planner.plan(
            state,
            acting_player_index=player_index,
            legal_actions=legal_actions,
        )
        return decision["chosen_action"]
    return legal_actions[0]


def list_legal_actions(
    state: GameState,
    player_index: int | None = None,
) -> list[dict[str, Any]]:
    if player_index is None:
        player_index = state.current_player
    player = state.players[player_index]
    if state.setup_phase == "choose_active":
        if player.active is not None:
            return []
        basic_actions = [
            {
                "type": "play_basic_to_active",
                "hand_card_id": instance_id,
                "label": f"Play {card_definition(state, instance_id).name} as your Active Pokemon",
            }
            for instance_id in player.hand
            if card_definition(state, instance_id).is_basic
        ]
        if basic_actions:
            return basic_actions

        if player_index == 0:
            return [
                {
                    "type": "mulligan",
                    "label": "Okay",
                }
            ]
        return []

    if state.setup_phase == "awaiting_end_setup":
        if player_index == 0 and player.active is not None:
            actions = _list_bench_basic_actions(state, player_index)
            actions.append(
                {
                    "type": "end_setup",
                    "label": "End Setup",
                }
            )
            return actions
        return []

    if player.active is None:
        basic_actions = [
            {
                "type": "play_basic_to_active",
                "hand_card_id": instance_id,
                "label": f"Play {card_definition(state, instance_id).name} as your Active Pokemon",
            }
            for instance_id in player.hand
            if card_definition(state, instance_id).is_basic
        ]
        if basic_actions:
            return basic_actions

        return []
    actions = _list_bench_basic_actions(state, player_index)
    actions.extend(_list_energy_attachment_actions(state, player_index))
    actions.extend(_list_evolution_actions(state, player_index))
    actions.extend(_list_trainer_actions(state, player_index))
    actions.extend(_list_attack_actions(state, player_index))
    actions.append(
        {
            "type": "end_turn",
            "label": "End Turn",
        }
    )
    return actions


def apply_action(state: GameState, action: dict[str, Any]) -> GameState:
    return apply_action_for_player(state, action, state.current_player)


def apply_action_for_player(
    state: GameState,
    action: dict[str, Any],
    player_index: int,
) -> GameState:
    legal_actions = list_legal_actions(state, player_index=player_index)
    resolved_action = _resolve_submitted_action(state, player_index, action, legal_actions)
    if resolved_action is None:
        raise ValueError("Illegal action.")
    action = resolved_action

    player = state.players[player_index]
    actor_name = "You" if player_index == 0 else player.name

    if action["type"] == "mulligan":
        _redraw_opening_hand(player, state.rng)
        state.log.append(f"{actor_name} had no Basic Pokemon in the opening hand, so 7 cards were redrawn.")
        if _hand_has_basic(player, state.cards, state.card_definitions):
            if player_index == 0:
                state.log.append("Your new opening hand contains a Basic Pokemon.")
            else:
                state.log.append(f"{player.name}'s new opening hand contains a Basic Pokemon.")
        else:
            if player_index == 0:
                state.log.append("Your new opening hand still has no Basic Pokemon. Mulligan again.")
            else:
                state.log.append(f"{player.name}'s new opening hand still has no Basic Pokemon. Mulligan again.")
        return state

    if action["type"] == "play_basic_to_active":
        card_id = _remove_from_hand(player, action["hand_card_id"])
        entered_play_turn = 0 if state.setup_phase is not None else player.turns_taken
        player.active = PokemonInPlay(stack=[card_id], entered_play_turn=entered_play_turn)
        card_name = card_definition(state, card_id).name
        if player_index == 0:
            state.log.append(f"You chose {card_name} as your Active Pokemon.")
            if state.setup_phase == "choose_active":
                state.setup_phase = "awaiting_end_setup"
                state.log.append("Opening setup is ready. End setup to begin your first turn.")
        else:
            state.log.append(f"{player.name} chose {card_name} as the Active Pokemon.")
        return state

    if action["type"] == "bench_basic":
        card_id = _remove_from_hand(player, action["hand_card_id"])
        entered_play_turn = 0 if state.setup_phase is not None else player.turns_taken
        player.bench.append(PokemonInPlay(stack=[card_id], entered_play_turn=entered_play_turn))
        card_name = card_definition(state, card_id).name
        if player_index == 0 and state.setup_phase == "awaiting_end_setup":
            state.log.append(f"You benched {card_name} during setup.")
        elif player_index == 0:
            state.log.append(f"You benched {card_name}.")
        elif state.setup_phase is not None:
            state.log.append(f"{player.name} benched {card_name} during setup.")
        else:
            state.log.append(f"{player.name} benched {card_name}.")
        return state

    if action["type"] == "play_energy":
        pokemon = _resolve_energy_target(player, action)
        energy_card_id = _remove_from_hand(player, action["hand_card_id"])
        pokemon.attached_energy.append(energy_card_id)
        player.energy_attached_this_turn = True
        energy_card = card_definition(state, energy_card_id)
        target_name = _energy_target_name(state, player, action)
        state.log.append(f"{actor_name} attached {energy_card.name} to {target_name}.")
        return state

    if action["type"] == "evolve":
        card_id = _remove_from_hand(player, action["hand_card_id"])
        card = card_definition(state, card_id)
        target = _resolve_board_target(player, action)
        previous_name = _board_target_name(state, target)
        target.stack.append(card_id)
        target.entered_play_turn = player.turns_taken
        state.log.append(f"{actor_name} evolved {previous_name} into {card.name}.")
        return state

    if action["type"] in {"play_supporter", "play_item"}:
        card_id = _remove_from_hand(player, action["hand_card_id"])
        card = card_definition(state, card_id)
        state.log.append(f"{actor_name} played {card.name}.")
        _resolve_effect_specs(state, player_index, card.effect_specs, action=action)
        player.discard.append(card_id)
        if "supporter" in card.card_tags:
            player.supporter_played_this_turn = True
        return state

    if action["type"] == "attack":
        if player.active is None:
            raise ValueError("Cannot attack without an Active Pokemon.")

        attacker_card = get_top_card_definition(state, player.active)
        if attacker_card is None:
            raise ValueError("Attacking Pokemon has no card definition.")

        attack_index = action.get("attack_index")
        if not isinstance(attack_index, int) or not 0 <= attack_index < len(attacker_card.attacks):
            raise ValueError("Attack index is out of range.")
        attack = attacker_card.attacks[attack_index]

        state.log.append(f"{actor_name} used {attack.name}.")
        _resolve_attack(state, player_index, attack, action)
        _resolve_knockouts_after_attack(state, attacker_index=player_index)
        if state.winner is not None:
            return state

        _advance_turn_after_attack(state, player_index)
        return state

    if action["type"] == "end_setup":
        state.setup_phase = None
        _start_turn(state, state.current_player)
        state.log.append("Opening setup is complete. Turn 1 begins. You go first.")
        _draw_turn_card(state, state.current_player)
        return state

    if action["type"] == "end_turn":
        next_player_index = 1 - player_index
        _expire_lingering_effects(state, ending_player_index=player_index)
        _start_turn(state, next_player_index)
        if player_index == 0:
            state.log.append("You ended the turn.")
            state.log.append(f"Turn {state.turn_number}: {state.players[1].name}'s turn.")
        else:
            state.turn_number += 1
            state.log.append(f"{player.name} ended the turn.")
            state.log.append(f"Turn {state.turn_number}: Your turn.")
        _draw_turn_card(state, next_player_index)
        return state

    raise ValueError(f"Unsupported Standard action type: {action['type']}")


def action_id_for(action: dict[str, Any]) -> str:
    if action["type"] == "bench_basic":
        return f"bench_basic:{action['hand_card_id']}"
    if action["type"] == "play_basic_to_active":
        return f"play_basic_to_active:{action['hand_card_id']}"
    if action["type"] == "play_energy":
        target_zone = action["target_zone"]
        if target_zone == "bench":
            return f"play_energy:{action['hand_card_id']}:bench:{action['target_bench_index']}"
        return f"play_energy:{action['hand_card_id']}:active"
    if action["type"] == "evolve":
        target_zone = action["target_zone"]
        if target_zone == "bench":
            return f"evolve:{action['hand_card_id']}:bench:{action['target_bench_index']}"
        return f"evolve:{action['hand_card_id']}:active"
    if action["type"] == "play_item":
        target_zone = action.get("target_zone")
        if target_zone == "bench":
            return f"play_item:{action['hand_card_id']}:bench:{action['target_bench_index']}"
        if target_zone == "active":
            return f"play_item:{action['hand_card_id']}:active"
        return f"play_item:{action['hand_card_id']}"
    if action["type"] == "play_supporter":
        target_zone = action.get("target_zone")
        if target_zone == "bench":
            return f"play_supporter:{action['hand_card_id']}:bench:{action['target_bench_index']}"
        if target_zone == "active":
            return f"play_supporter:{action['hand_card_id']}:active"
        return f"play_supporter:{action['hand_card_id']}"
    if action["type"] == "attack":
        suffix_parts = [f"attack:{action['attack_index']}"]
        target_player_index = action.get("target_player_index")
        if action.get("target_zone") == "bench":
            suffix_parts.append(f"p{target_player_index}:bench:{action['target_bench_index']}")
        elif action.get("target_zone") == "active":
            suffix_parts.append(f"p{target_player_index}:active")
        if action.get("bonus_damage"):
            suffix_parts.append(f"bonus:{action['bonus_damage']}")
        discard_ids = action.get("discard_attached_energy_ids")
        if discard_ids:
            suffix_parts.append(",".join(discard_ids))
        return ":".join(suffix_parts)
    return action["type"]


def _resolve_submitted_action(
    state: GameState,
    player_index: int,
    submitted_action: dict[str, Any],
    legal_actions: list[dict[str, Any]],
) -> dict[str, Any] | None:
    submitted_base_action = _base_action_payload(submitted_action)
    for legal_action in legal_actions:
        if submitted_base_action != legal_action:
            continue
        return _resolve_dynamic_action_fields(state, player_index, legal_action, submitted_action)
    return None


def _base_action_payload(action: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in action.items()
        if key not in _DYNAMIC_ACTION_FIELDS
    }


def _resolve_dynamic_action_fields(
    state: GameState,
    player_index: int,
    legal_action: dict[str, Any],
    submitted_action: dict[str, Any],
) -> dict[str, Any]:
    if legal_action["type"] not in {"play_supporter", "play_item"}:
        return dict(legal_action)

    hand_card_id = legal_action.get("hand_card_id")
    if not isinstance(hand_card_id, str):
        return dict(legal_action)

    player = state.players[player_index]
    card = card_definition(state, hand_card_id)
    resolved_action = dict(legal_action)
    auto_select = player_index != 0

    for effect_spec in card.effect_specs:
        if effect_spec.effect_type == "discard_from_hand":
            resolved_action["discard_from_hand_ids"] = _resolve_discard_from_hand_selection(
                player,
                hand_card_id,
                effect_spec,
                submitted_action,
                auto_select=auto_select,
            )
            continue

        if effect_spec.effect_type == "search_deck":
            resolved_action["search_result_ids"] = _resolve_search_deck_selection(
                state,
                player,
                effect_spec,
                submitted_action,
                auto_select=auto_select,
            )

    return resolved_action


def _resolve_discard_from_hand_selection(
    player: PlayerState,
    source_card_id: str,
    effect_spec: EffectSpec,
    submitted_action: dict[str, Any],
    *,
    auto_select: bool,
) -> list[str]:
    required_count = _required_choice_count(effect_spec)
    candidate_ids = _available_hand_choice_ids(player, source_card_id, effect_spec)
    if len(candidate_ids) < required_count:
        raise ValueError("Not enough cards are available to discard from hand.")

    selected_ids = _coerce_action_card_ids(submitted_action.get("discard_from_hand_ids"))
    if not selected_ids:
        if not auto_select:
            raise ValueError("This action requires cards to discard from hand.")
        return candidate_ids[:required_count]

    if len(selected_ids) != required_count or len(set(selected_ids)) != len(selected_ids):
        raise ValueError("Discard selections must match the required card count.")
    if any(instance_id not in candidate_ids for instance_id in selected_ids):
        raise ValueError("Discard selections must come from the acting player's hand.")
    return selected_ids


def _resolve_search_deck_selection(
    state: GameState,
    player: PlayerState,
    effect_spec: EffectSpec,
    submitted_action: dict[str, Any],
    *,
    auto_select: bool,
) -> list[str]:
    required_count = _required_choice_count(effect_spec)
    candidate_ids = _available_deck_choice_ids(state, player, effect_spec)
    if len(candidate_ids) < required_count:
        raise ValueError("Not enough cards are available in deck for this search.")

    selected_ids = _coerce_action_card_ids(submitted_action.get("search_result_ids"))
    if not selected_ids:
        if not auto_select:
            raise ValueError("This action requires cards to be chosen from deck.")
        return candidate_ids[:required_count]

    if len(selected_ids) != required_count or len(set(selected_ids)) != len(selected_ids):
        raise ValueError("Deck search selections must match the required card count.")
    if any(instance_id not in candidate_ids for instance_id in selected_ids):
        raise ValueError("Deck search selections must match the card filter.")
    return selected_ids


def _required_choice_count(effect_spec: EffectSpec) -> int:
    count = effect_spec.choose_count if effect_spec.choose_count is not None else effect_spec.count
    if not isinstance(count, int) or count <= 0:
        raise ValueError(f"Unsupported choice count for effect: {effect_spec.effect_type}")
    return count


def _coerce_action_card_ids(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError("Action card selections must be provided as a list of instance ids.")
    return list(value)


def card_definition(state: GameState, instance_id: str) -> CardDefinition:
    return state.card_definitions[state.cards[instance_id].card_id]


def get_top_card_definition(state: GameState, pokemon: PokemonInPlay | None) -> CardDefinition | None:
    if pokemon is None or not pokemon.stack:
        return None
    return card_definition(state, pokemon.stack[-1])


def _build_player_state(
    name: str,
    deck_definition: DeckDefinition,
    owner: int,
    rng: random.Random,
) -> tuple[PlayerState, dict[str, CardInstance], dict[str, CardDefinition]]:
    counters: dict[str, int] = defaultdict(int)
    cards: dict[str, CardInstance] = {}
    card_definitions: dict[str, CardDefinition] = {}
    deck_ids: list[str] = []

    for entry in load_deck_cards(deck_definition.deck_id):
        card_definitions[entry.card_id] = _to_card_definition(entry)
        for _ in range(entry.quantity):
            counters[entry.card_id] += 1
            instance_id = f"p{owner}_{entry.card_id}_{counters[entry.card_id]}"
            cards[instance_id] = CardInstance(instance_id=instance_id, card_id=entry.card_id, owner=owner)
            deck_ids.append(instance_id)

    player = PlayerState(
        name=name,
        deck_name=deck_definition.name,
        element=deck_definition.element,
        deck=deck_ids,
    )
    _ensure_deck_has_basic(player, cards, card_definitions)
    rng.shuffle(player.deck)
    _draw_opening_hand(player)
    return player, cards, card_definitions


def _to_card_definition(entry: DeckCardDefinition) -> CardDefinition:
    return CardDefinition(
        card_id=entry.card_id,
        name=entry.name,
        kind=entry.kind,
        element=entry.element,
        stage=entry.stage,
        is_basic=entry.is_basic,
        evolves_from=entry.evolves_from,
        hp=entry.hp,
        attacks=entry.attacks,
        image_url=entry.image_url,
        card_tags=entry.card_tags,
        rules_text=entry.rules_text,
        effect_specs=entry.effect_specs,
    )


def _draw_opening_hand(player: PlayerState) -> None:
    player.hand = player.deck[:OPENING_HAND_SIZE]
    del player.deck[:OPENING_HAND_SIZE]


def _redraw_opening_hand(player: PlayerState, rng: random.Random) -> None:
    player.deck.extend(player.hand)
    player.hand.clear()
    rng.shuffle(player.deck)
    _draw_opening_hand(player)
    player.mulligans_taken += 1


def _ensure_deck_has_basic(
    player: PlayerState,
    cards: dict[str, CardInstance],
    card_definitions: dict[str, CardDefinition],
) -> None:
    if any(card_definitions[cards[instance_id].card_id].is_basic for instance_id in player.deck):
        return
    raise ValueError(f"Standard deck '{player.deck_name}' has no Basic Pokemon in catalog data.")


def _hand_has_basic(
    player: PlayerState,
    cards: dict[str, CardInstance],
    card_definitions: dict[str, CardDefinition],
) -> bool:
    return any(card_definitions[cards[instance_id].card_id].is_basic for instance_id in player.hand)


def _remove_from_hand(player: PlayerState, instance_id: str) -> str:
    player.hand.remove(instance_id)
    return instance_id


def _draw_cards(player: PlayerState, count: int) -> int:
    drawn_cards = player.deck[:count]
    del player.deck[: len(drawn_cards)]
    player.hand.extend(drawn_cards)
    return len(drawn_cards)


def _start_turn(state: GameState, player_index: int) -> None:
    state.current_player = player_index
    player = state.players[player_index]
    player.turns_taken += 1
    player.supporter_played_this_turn = False
    player.energy_attached_this_turn = False


def _draw_turn_card(state: GameState, player_index: int) -> None:
    player = state.players[player_index]
    actor_name = "You" if player_index == 0 else player.name
    if not player.deck:
        state.log.append(f"{actor_name} had no card to draw at the start of the turn.")
        return

    drawn_card_id = player.deck.pop(0)
    player.hand.append(drawn_card_id)
    state.log.append(f"{actor_name} drew {card_definition(state, drawn_card_id).name}.")


def _flip_coin(state: GameState) -> bool:
    return bool(state.rng.choice([True, False]))


def _resolve_effect_specs(
    state: GameState,
    player_index: int,
    effect_specs: tuple[EffectSpec, ...],
    action: dict[str, Any] | None = None,
) -> None:
    player = state.players[player_index]
    actor_name = "You" if player_index == 0 else player.name

    for effect_spec in effect_specs:
        if effect_spec.effect_type == "draw":
            draw_count = effect_spec.count or 0
            drawn = _draw_cards(player, draw_count)
            state.log.append(f"{actor_name} drew {drawn} card{'' if drawn == 1 else 's'}.")
            continue

        if effect_spec.effect_type == "shuffle_zone_into_deck":
            if effect_spec.source_zone != "hand" or effect_spec.destination_zone != "deck":
                raise ValueError(f"Unsupported shuffle effect configuration: {effect_spec}")
            cards_to_shuffle = list(player.hand)
            player.hand.clear()
            player.deck.extend(cards_to_shuffle)
            if effect_spec.shuffle_destination:
                state.rng.shuffle(player.deck)
            state.log.append(
                f"{actor_name} shuffled {len(cards_to_shuffle)} card"
                f"{'' if len(cards_to_shuffle) == 1 else 's'} into the deck."
            )
            continue

        if effect_spec.effect_type == "discard_from_hand":
            discarded = _discard_cards_from_hand_for_effect(player, action, effect_spec)
            state.log.append(
                f"{actor_name} discarded {len(discarded)} card"
                f"{'' if len(discarded) == 1 else 's'} from hand."
            )
            continue

        if effect_spec.effect_type == "search_deck":
            moved_cards = _move_cards_from_deck_for_effect(state, player, action, effect_spec)
            moved_names = [card_definition(state, instance_id).name for instance_id in moved_cards]
            if moved_names:
                if len(moved_names) == 1:
                    state.log.append(f"{actor_name} searched the deck for {moved_names[0]} and put it into hand.")
                else:
                    card_list = ", ".join(moved_names)
                    state.log.append(f"{actor_name} searched the deck for {card_list} and put them into hand.")
            if effect_spec.shuffle_destination:
                state.rng.shuffle(player.deck)
                state.log.append(f"{actor_name} shuffled the deck.")
            continue

        if effect_spec.effect_type == "heal_damage":
            heal_amount = effect_spec.count or 0
            target = _resolve_board_target(player, action)
            healed = min(heal_amount, target.damage)
            target.damage = max(0, target.damage - heal_amount)
            target_name = _board_target_name(state, target)
            state.log.append(
                f"{actor_name} healed {healed} damage from {target_name}."
            )
            continue

        if effect_spec.effect_type == "switch_active_with_bench":
            if player.active is None:
                raise ValueError("Cannot switch without an Active Pokemon.")
            if action is None or action.get("target_zone") != "bench":
                raise ValueError("Switch requires a benched target.")
            bench_index = action.get("target_bench_index")
            if not isinstance(bench_index, int):
                raise ValueError("Switch target is missing a bench index.")
            try:
                bench_pokemon = player.bench.pop(bench_index)
            except IndexError as exc:
                raise ValueError("Switch target is out of range.") from exc
            previous_active = player.active
            player.active = bench_pokemon
            player.bench.append(previous_active)
            state.log.append(f"{actor_name} switched to {_board_target_name(state, player.active)}.")
            continue

        raise ValueError(f"Unsupported Standard effect type: {effect_spec.effect_type}")


def _discard_cards_from_hand_for_effect(
    player: PlayerState,
    action: dict[str, Any] | None,
    effect_spec: EffectSpec,
) -> list[str]:
    if action is None:
        raise ValueError("Discarding from hand requires an action payload.")

    selected_ids = _coerce_action_card_ids(action.get("discard_from_hand_ids"))
    required_count = _required_choice_count(effect_spec)
    if len(selected_ids) != required_count:
        raise ValueError("Discard effect is missing the required number of selected cards.")

    discarded: list[str] = []
    for instance_id in selected_ids:
        player.discard.append(_remove_from_hand(player, instance_id))
        discarded.append(instance_id)
    return discarded


def _move_cards_from_deck_for_effect(
    state: GameState,
    player: PlayerState,
    action: dict[str, Any] | None,
    effect_spec: EffectSpec,
) -> list[str]:
    if action is None:
        raise ValueError("Searching the deck requires an action payload.")

    selected_ids = _coerce_action_card_ids(action.get("search_result_ids"))
    required_count = _required_choice_count(effect_spec)
    if len(selected_ids) != required_count:
        raise ValueError("Search effect is missing the required number of selected cards.")
    if effect_spec.source_zone != "deck" or effect_spec.destination_zone != "hand":
        raise ValueError(f"Unsupported search effect configuration: {effect_spec}")

    moved_cards: list[str] = []
    for instance_id in selected_ids:
        player.deck.remove(instance_id)
        player.hand.append(instance_id)
        moved_cards.append(instance_id)
    return moved_cards


def _list_bench_basic_actions(
    state: GameState,
    player_index: int,
) -> list[dict[str, Any]]:
    player = state.players[player_index]
    if len(player.bench) >= BENCH_LIMIT:
        return []

    return [
        {
            "type": "bench_basic",
            "hand_card_id": instance_id,
            "label": f"Bench {card_definition(state, instance_id).name}",
        }
        for instance_id in player.hand
        if card_definition(state, instance_id).is_basic
    ]


def _list_energy_attachment_actions(
    state: GameState,
    player_index: int,
) -> list[dict[str, Any]]:
    player = state.players[player_index]
    if player.energy_attached_this_turn:
        return []
    if player.active is None:
        return []

    energy_cards = [
        instance_id
        for instance_id in player.hand
        if card_definition(state, instance_id).kind == "energy"
    ]
    if not energy_cards:
        return []

    targets: list[tuple[str, int | None, str]] = [("active", None, "Active Pokemon")]
    targets.extend(
        ("bench", bench_index, get_top_card_definition(state, pokemon).name)
        for bench_index, pokemon in enumerate(player.bench)
        if get_top_card_definition(state, pokemon) is not None
    )

    actions: list[dict[str, Any]] = []
    for instance_id in energy_cards:
        energy_card = card_definition(state, instance_id)
        for target_zone, target_bench_index, target_name in targets:
            actions.append(
                {
                    "type": "play_energy",
                    "hand_card_id": instance_id,
                    "target_zone": target_zone,
                    "target_bench_index": target_bench_index,
                    "label": f"Attach {energy_card.name} to {target_name}",
                }
            )
    return actions


def _list_evolution_actions(
    state: GameState,
    player_index: int,
) -> list[dict[str, Any]]:
    player = state.players[player_index]
    actions: list[dict[str, Any]] = []

    for instance_id in player.hand:
        card = card_definition(state, instance_id)
        if card.kind != "pokemon" or card.stage not in {"stage1", "stage2"} or not card.evolves_from:
            continue

        if _can_evolve(state, player, player.active, card):
            actions.append(
                {
                    "type": "evolve",
                    "hand_card_id": instance_id,
                    "target_zone": "active",
                    "target_bench_index": None,
                    "label": f"Evolve {_board_target_name(state, player.active)} into {card.name}",
                }
            )

        for bench_index, pokemon in enumerate(player.bench):
            if not _can_evolve(state, player, pokemon, card):
                continue
            actions.append(
                {
                    "type": "evolve",
                    "hand_card_id": instance_id,
                    "target_zone": "bench",
                    "target_bench_index": bench_index,
                    "label": f"Evolve {_board_target_name(state, pokemon)} into {card.name}",
                }
            )

    return actions


def _list_trainer_actions(
    state: GameState,
    player_index: int,
) -> list[dict[str, Any]]:
    player = state.players[player_index]
    actions: list[dict[str, Any]] = []
    for instance_id in player.hand:
        card = card_definition(state, instance_id)
        if card.kind != "trainer" or not card.effect_specs:
            continue

        is_supporter = "supporter" in card.card_tags
        is_item = "item" in card.card_tags
        if is_supporter and (
            player.supporter_played_this_turn or _supporters_blocked_for_turn(state, player_index)
        ):
            continue

        targeted_actions = _list_targeted_trainer_actions(state, player_index, instance_id, card)
        if targeted_actions:
            actions.extend(targeted_actions)
            continue

        if any(
            effect_spec.options
            or effect_spec.selection_count is not None
            or effect_spec.destination_position is not None
            for effect_spec in card.effect_specs
        ):
            continue
        if not _can_resolve_non_targeted_trainer_effects(state, player_index, instance_id, card.effect_specs):
            continue

        if is_supporter:
            action_type = "play_supporter"
        elif is_item:
            action_type = "play_item"
        else:
            continue

        actions.append(
            {
                "type": action_type,
                "hand_card_id": instance_id,
                "label": f"Play {card.name}",
            }
        )
    return actions


def _supporters_blocked_for_turn(state: GameState, player_index: int) -> bool:
    return state.turn_number == 1 and player_index == state.starting_player


def _list_targeted_trainer_actions(
    state: GameState,
    player_index: int,
    instance_id: str,
    card: CardDefinition,
) -> list[dict[str, Any]]:
    player = state.players[player_index]
    action_type = "play_supporter" if "supporter" in card.card_tags else "play_item"
    actions: list[dict[str, Any]] = []

    for effect_spec in card.effect_specs:
        if effect_spec.effect_type == "heal_damage":
            heal_amount = effect_spec.count or 0
            if player.active is not None and player.active.damage > 0:
                actions.append(
                    {
                        "type": action_type,
                        "hand_card_id": instance_id,
                        "target_zone": "active",
                        "target_bench_index": None,
                        "label": f"Play {card.name} on {_board_target_name(state, player.active)}",
                    }
                )
            for bench_index, pokemon in enumerate(player.bench):
                if pokemon.damage <= 0:
                    continue
                actions.append(
                    {
                        "type": action_type,
                        "hand_card_id": instance_id,
                        "target_zone": "bench",
                        "target_bench_index": bench_index,
                        "label": f"Play {card.name} on {_board_target_name(state, pokemon)}",
                    }
                )
            return actions

        if effect_spec.effect_type == "switch_active_with_bench":
            if player.active is None or not player.bench:
                return []
            for bench_index, pokemon in enumerate(player.bench):
                actions.append(
                    {
                        "type": action_type,
                        "hand_card_id": instance_id,
                        "target_zone": "bench",
                        "target_bench_index": bench_index,
                        "label": f"Play {card.name} and switch to {_board_target_name(state, pokemon)}",
                    }
                )
            return actions

    return []


def _can_resolve_non_targeted_trainer_effects(
    state: GameState,
    player_index: int,
    source_card_id: str,
    effect_specs: tuple[EffectSpec, ...],
) -> bool:
    player = state.players[player_index]
    for effect_spec in effect_specs:
        if effect_spec.effect_type in {"draw", "shuffle_zone_into_deck"}:
            continue
        if effect_spec.effect_type == "discard_from_hand":
            if len(_available_hand_choice_ids(player, source_card_id, effect_spec)) < _required_choice_count(effect_spec):
                return False
            continue
        if effect_spec.effect_type == "search_deck":
            if len(_available_deck_choice_ids(state, player, effect_spec)) < _required_choice_count(effect_spec):
                return False
            continue
        return False
    return True


def _available_hand_choice_ids(
    player: PlayerState,
    source_card_id: str,
    effect_spec: EffectSpec,
) -> list[str]:
    return [
        instance_id
        for instance_id in player.hand
        if not (effect_spec.exclude_source_card and instance_id == source_card_id)
    ]


def _available_deck_choice_ids(
    state: GameState,
    player: PlayerState,
    effect_spec: EffectSpec,
) -> list[str]:
    return [
        instance_id
        for instance_id in player.deck
        if _card_matches_effect_filters(card_definition(state, instance_id), effect_spec.search_filters)
    ]


def _card_matches_effect_filters(card: CardDefinition, search_filters: tuple[str, ...]) -> bool:
    if not search_filters:
        return True

    for search_filter in search_filters:
        if search_filter == "pokemon" and card.kind != "pokemon":
            return False
        if search_filter == "basic_pokemon" and (card.kind != "pokemon" or not card.is_basic):
            return False
    return True


def _list_attack_actions(
    state: GameState,
    player_index: int,
) -> list[dict[str, Any]]:
    player = state.players[player_index]
    opponent = state.players[1 - player_index]
    if player.active is None or opponent.active is None:
        return []
    if _attacks_blocked_for_turn(state, player_index):
        return []

    active_card = get_top_card_definition(state, player.active)
    if active_card is None:
        return []

    attached_energy = len(player.active.attached_energy)
    actions: list[dict[str, Any]] = []
    for attack_index, attack in enumerate(active_card.attacks):
        if attached_energy < attack.cost:
            continue
        attack_actions = _build_attack_actions_for_definition(
            state,
            player_index,
            player.active,
            attack_index,
            attack,
        )
        actions.extend(attack_actions)
    return actions


def _attacks_blocked_for_turn(state: GameState, player_index: int) -> bool:
    return state.turn_number == 1 and player_index == state.starting_player


def _build_attack_actions_for_definition(
    state: GameState,
    player_index: int,
    pokemon: PokemonInPlay,
    attack_index: int,
    attack: AttackDefinition,
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = [
        {
            "type": "attack",
            "attack_index": attack_index,
            "label": f"Use {attack.name}",
        }
    ]

    for effect_spec in attack.effect_specs:
        if effect_spec.effect_type == "discard_attached_energy":
            discard_ids = _pick_attached_energy_ids_for_effect(state, pokemon, effect_spec)
            if not discard_ids:
                return []
            actions = [
                {
                    **action,
                    "discard_attached_energy_ids": discard_ids,
                }
                for action in actions
            ]
            continue

        if effect_spec.effect_type == "optional_discard_attached_energy_for_bonus_damage":
            discard_ids = _pick_attached_energy_ids_for_effect(state, pokemon, effect_spec)
            if discard_ids:
                actions = [
                    {
                        **action,
                        "discard_attached_energy_ids": discard_ids,
                        "bonus_damage": effect_spec.bonus_damage or 0,
                    }
                    for action in actions
                ]
            continue

        if effect_spec.effect_type == "damage_target":
            actions = _expand_attack_actions_for_targeted_damage(
                state,
                player_index,
                actions,
                effect_spec,
            )
            continue

    return actions


def _expand_attack_actions_for_targeted_damage(
    state: GameState,
    player_index: int,
    actions: list[dict[str, Any]],
    effect_spec: Any,
) -> list[dict[str, Any]]:
    if effect_spec.target_player != "opponent":
        return actions
    if effect_spec.selection_count != 1:
        return actions

    targets = _list_attack_targets_for_effect(state, player_index, effect_spec)
    if not targets:
        return actions

    expanded_actions: list[dict[str, Any]] = []
    for action in actions:
        for target in targets:
            expanded_actions.append({**action, **target})
    return expanded_actions


def _list_attack_targets_for_effect(
    state: GameState,
    player_index: int,
    effect_spec: Any,
) -> list[dict[str, Any]]:
    opponent_index = 1 - player_index
    opponent = state.players[opponent_index]
    targets: list[dict[str, Any]] = []
    if effect_spec.target_zone in {"active", "any"} and opponent.active is not None:
        targets.append(
            {
                "target_player_index": opponent_index,
                "target_zone": "active",
                "target_bench_index": None,
            }
        )
    if effect_spec.target_zone in {"bench", "any"}:
        targets.extend(
            {
                "target_player_index": opponent_index,
                "target_zone": "bench",
                "target_bench_index": bench_index,
            }
            for bench_index in range(len(opponent.bench))
        )
    return targets


def _pick_attached_energy_ids_for_effect(
    state: GameState,
    pokemon: PokemonInPlay,
    effect_spec: Any,
) -> list[str]:
    amount = int(effect_spec.amount or 0)
    if amount <= 0:
        return []
    matching_ids = [
        instance_id
        for instance_id in pokemon.attached_energy
        if _attached_energy_matches(state, instance_id, effect_spec.energy_type)
    ]
    if len(matching_ids) < amount:
        return []
    return matching_ids[-amount:]


def _attached_energy_matches(
    state: GameState,
    instance_id: str,
    energy_type: str | None,
) -> bool:
    if energy_type is None:
        return True
    card = card_definition(state, instance_id)
    return card.element == energy_type


def _resolve_energy_target(player: PlayerState, action: dict[str, Any]) -> PokemonInPlay:
    target_zone = action.get("target_zone")
    if target_zone == "active":
        if player.active is None:
            raise ValueError("Cannot attach Energy without an Active Pokemon.")
        return player.active

    if target_zone == "bench":
        bench_index = action.get("target_bench_index")
        if not isinstance(bench_index, int):
            raise ValueError("Bench Energy attachment is missing a target index.")
        try:
            return player.bench[bench_index]
        except IndexError as exc:
            raise ValueError("Bench Energy attachment target is out of range.") from exc

    raise ValueError(f"Unsupported Energy attachment target zone: {target_zone}")


def _energy_target_name(
    state: GameState,
    player: PlayerState,
    action: dict[str, Any],
) -> str:
    pokemon = _resolve_energy_target(player, action)
    top_card = get_top_card_definition(state, pokemon)
    if top_card is None:
        return "Pokemon"
    return top_card.name


def _resolve_board_target(player: PlayerState, action: dict[str, Any] | None) -> PokemonInPlay:
    if action is None:
        raise ValueError("This effect requires a target.")

    target_zone = action.get("target_zone")
    if target_zone == "active":
        if player.active is None:
            raise ValueError("No Active Pokemon is available.")
        return player.active
    if target_zone == "bench":
        bench_index = action.get("target_bench_index")
        if not isinstance(bench_index, int):
            raise ValueError("Bench target is missing an index.")
        try:
            return player.bench[bench_index]
        except IndexError as exc:
            raise ValueError("Bench target is out of range.") from exc
    raise ValueError(f"Unsupported board target zone: {target_zone}")


def _board_target_name(state: GameState, pokemon: PokemonInPlay | None) -> str:
    if pokemon is None:
        return "Pokemon"
    top_card = get_top_card_definition(state, pokemon)
    return top_card.name if top_card is not None else "Pokemon"


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
    if pokemon.entered_play_turn >= player.turns_taken:
        return False

    top_card = get_top_card_definition(state, pokemon)
    return top_card is not None and top_card.name == evolution_card.evolves_from


def _attack_damage_value(damage_text: str) -> int:
    matches = _DAMAGE_PATTERN.findall(str(damage_text))
    if not matches:
        return 0
    return sum(int(match) for match in matches)


def _resolve_attack(
    state: GameState,
    player_index: int,
    attack: AttackDefinition,
    action: dict[str, Any],
) -> None:
    player = state.players[player_index]

    for effect_spec in attack.effect_specs:
        if effect_spec.effect_type in {
            "discard_attached_energy",
            "optional_discard_attached_energy_for_bonus_damage",
        }:
            _resolve_attack_effect_spec(state, player_index, attack, effect_spec, action)

    damage = _resolve_attack_damage_amount(state, player_index, attack, action)
    if damage > 0:
        _apply_attack_damage(
            state,
            attacker_index=player_index,
            target_player_index=1 - player_index,
            target_zone="active",
            target_bench_index=None,
            damage=damage,
        )

    for effect_spec in attack.effect_specs:
        if effect_spec.effect_type not in {
            "discard_attached_energy",
            "optional_discard_attached_energy_for_bonus_damage",
        }:
            _resolve_attack_effect_spec(state, player_index, attack, effect_spec, action)


def _resolve_attack_effect_spec(
    state: GameState,
    player_index: int,
    attack: AttackDefinition,
    effect_spec: Any,
    action: dict[str, Any],
) -> None:
    player = state.players[player_index]
    actor_name = "You" if player_index == 0 else player.name

    if effect_spec.effect_type == "draw_cards":
        drawn = _draw_cards(player, int(effect_spec.amount or 0))
        state.log.append(f"{actor_name} drew {drawn} card{'' if drawn == 1 else 's'}.")
        return

    if effect_spec.effect_type == "discard_attached_energy":
        discarded = _discard_attached_energy_from_attack_action(state, player_index, action)
        if discarded:
            state.log.append(
                f"{actor_name} discarded {len(discarded)} attached Energy"
                f"{'' if len(discarded) == 1 else ' cards'}."
            )
        return

    if effect_spec.effect_type == "optional_discard_attached_energy_for_bonus_damage":
        discarded = _discard_attached_energy_from_attack_action(state, player_index, action)
        if discarded:
            state.log.append(
                f"{actor_name} discarded {len(discarded)} attached Energy"
                f"{'' if len(discarded) == 1 else ' cards'} for extra damage."
            )
        return

    if effect_spec.effect_type == "damage_target":
        target = _resolve_attack_effect_target(state, player_index, effect_spec, action)
        if target is None:
            return
        target_player_index, target_zone, target_bench_index = target
        _apply_attack_damage(
            state,
            attacker_index=player_index,
            target_player_index=target_player_index,
            target_zone=target_zone,
            target_bench_index=target_bench_index,
            damage=int(effect_spec.amount or 0),
        )
        return

    if effect_spec.effect_type == "apply_protection":
        if player.active is None:
            return
        if effect_spec.condition == "basic_pokemon_attack_damage":
            player.active.lingering_effects.append(
                LingeringEffect(
                    effect_type="prevent_damage_from_basic_pokemon_attacks",
                    source_player=player_index,
                    expires_end_of_player_turn=1 - player_index,
                    condition=effect_spec.condition,
                )
            )
            state.log.append(f"{_board_target_name(state, player.active)} is shielded from Basic Pokemon next turn.")
        return


def _resolve_attack_damage_amount(
    state: GameState,
    player_index: int,
    attack: AttackDefinition,
    action: dict[str, Any],
) -> int:
    damage = _attack_damage_value(attack.damage) + int(action.get("bonus_damage", 0) or 0)
    actor_name = "You" if player_index == 0 else state.players[player_index].name

    for effect_spec in attack.effect_specs:
        if effect_spec.effect_type == "damage_per_opponent_prizes_taken":
            opponent = state.players[1 - player_index]
            prizes_taken = max(0, 6 - opponent.prize_cards_remaining)
            damage = int(effect_spec.amount or 0) * prizes_taken
            continue

        if effect_spec.effect_type == "coin_flip_damage_on_heads_only":
            heads = _flip_coin(state)
            state.log.append(f"{actor_name} flipped {'heads' if heads else 'tails'}.")
            if not heads:
                state.log.append(f"{attack.name} did no damage.")
                return 0

    return damage


def _discard_attached_energy_from_attack_action(
    state: GameState,
    player_index: int,
    action: dict[str, Any],
) -> list[str]:
    player = state.players[player_index]
    if player.active is None:
        return []
    discard_ids = list(action.get("discard_attached_energy_ids") or [])
    if not discard_ids:
        return []

    discarded: list[str] = []
    for instance_id in discard_ids:
        if instance_id not in player.active.attached_energy:
            continue
        player.active.attached_energy.remove(instance_id)
        player.discard.append(instance_id)
        discarded.append(instance_id)
    return discarded


def _resolve_attack_effect_target(
    state: GameState,
    player_index: int,
    effect_spec: Any,
    action: dict[str, Any],
) -> tuple[int, str, int | None] | None:
    if effect_spec.target_player == "self":
        return player_index, effect_spec.target_zone or "active", action.get("target_bench_index")

    target_player_index = int(action.get("target_player_index", 1 - player_index))
    target_zone = action.get("target_zone")
    if target_zone is None:
        if effect_spec.target_zone == "bench":
            if not state.players[target_player_index].bench:
                return None
            target_zone = "bench"
        else:
            target_zone = effect_spec.target_zone or "active"
    return target_player_index, target_zone, action.get("target_bench_index")


def _apply_attack_damage(
    state: GameState,
    *,
    attacker_index: int,
    target_player_index: int,
    target_zone: str,
    target_bench_index: int | None,
    damage: int,
) -> None:
    if damage <= 0:
        return

    target_pokemon = _resolve_attack_damage_target(
        state,
        target_player_index=target_player_index,
        target_zone=target_zone,
        target_bench_index=target_bench_index,
    )
    if target_pokemon is None:
        return

    attacker_card = get_top_card_definition(state, state.players[attacker_index].active)
    if _is_attack_damage_prevented(attacker_index, target_player_index, target_pokemon, attacker_card):
        state.log.append(f"Damage to {_board_target_name(state, target_pokemon)} was prevented.")
        return

    target_pokemon.damage += damage
    state.log.append(f"{_board_target_name(state, target_pokemon)} took {damage} damage.")


def _resolve_attack_damage_target(
    state: GameState,
    *,
    target_player_index: int,
    target_zone: str,
    target_bench_index: int | None,
) -> PokemonInPlay | None:
    player = state.players[target_player_index]
    if target_zone == "active":
        return player.active
    if target_zone == "bench":
        if not isinstance(target_bench_index, int):
            return None
        if not 0 <= target_bench_index < len(player.bench):
            return None
        return player.bench[target_bench_index]
    return None


def _is_attack_damage_prevented(
    attacker_index: int,
    target_player_index: int,
    target_pokemon: PokemonInPlay,
    attacker_card: CardDefinition | None,
) -> bool:
    if attacker_index == target_player_index or attacker_card is None:
        return False
    if not attacker_card.is_basic:
        return False
    return any(
        effect.effect_type == "prevent_damage_from_basic_pokemon_attacks"
        for effect in target_pokemon.lingering_effects
    )


def _resolve_knockouts_after_attack(state: GameState, attacker_index: int) -> None:
    for knocked_player_index in (1 - attacker_index, attacker_index):
        _resolve_player_knockouts(state, knocked_player_index)
        if state.winner is not None:
            return


def _resolve_player_knockouts(state: GameState, knocked_player_index: int) -> None:
    player = state.players[knocked_player_index]
    opponent_index = 1 - knocked_player_index

    for bench_index in range(len(player.bench) - 1, -1, -1):
        pokemon = player.bench[bench_index]
        if not _pokemon_is_knocked_out(state, pokemon):
            continue
        knocked_out = player.bench.pop(bench_index)
        knocked_out_name = _board_target_name(state, knocked_out)
        player.discard.extend(knocked_out.stack)
        player.discard.extend(knocked_out.attached_energy)
        state.log.append(f"{knocked_out_name} was Knocked Out.")
        _award_prize_for_knockout(state, winner_index=opponent_index)
        if state.winner is not None:
            return

    if player.active is None or not _pokemon_is_knocked_out(state, player.active):
        return

    knocked_out_name = _board_target_name(state, player.active)
    player.discard.extend(player.active.stack)
    player.discard.extend(player.active.attached_energy)
    player.active = None
    state.log.append(f"{knocked_out_name} was Knocked Out.")
    _award_prize_for_knockout(state, winner_index=opponent_index)
    if state.winner is not None:
        return

    if player.bench:
        promoted = player.bench.pop(0)
        player.active = promoted
        state.log.append(f"{player.name} promoted {_board_target_name(state, promoted)} to the Active Spot.")
        return

    state.winner = opponent_index
    state.log.append(f"{'You' if opponent_index == 0 else state.players[opponent_index].name} won the game.")


def _pokemon_is_knocked_out(state: GameState, pokemon: PokemonInPlay | None) -> bool:
    if pokemon is None:
        return False
    card = get_top_card_definition(state, pokemon)
    if card is None or card.hp is None:
        return False
    return pokemon.damage >= card.hp


def _award_prize_for_knockout(state: GameState, winner_index: int) -> None:
    player = state.players[winner_index]
    if player.prize_cards_remaining > 0:
        player.prize_cards_remaining -= 1
        actor_name = "You" if winner_index == 0 else player.name
        state.log.append(
            f"{actor_name} took a Prize card. {player.prize_cards_remaining} Prize"
            f"{'' if player.prize_cards_remaining == 1 else 's'} remaining."
        )
    if player.prize_cards_remaining == 0:
        state.winner = winner_index
        state.log.append(f"{'You' if winner_index == 0 else player.name} won the game.")


def _expire_lingering_effects(state: GameState, ending_player_index: int) -> None:
    for player in state.players:
        if player.active is not None:
            player.active.lingering_effects = [
                effect
                for effect in player.active.lingering_effects
                if effect.expires_end_of_player_turn != ending_player_index
            ]
        for pokemon in player.bench:
            pokemon.lingering_effects = [
                effect
                for effect in pokemon.lingering_effects
                if effect.expires_end_of_player_turn != ending_player_index
            ]


def _advance_turn_after_attack(state: GameState, player_index: int) -> None:
    next_player_index = 1 - player_index
    _expire_lingering_effects(state, ending_player_index=player_index)
    _start_turn(state, next_player_index)
    if player_index == 0:
        state.log.append(f"Turn {state.turn_number}: {state.players[1].name}'s turn.")
    else:
        state.turn_number += 1
        state.log.append(f"Turn {state.turn_number}: Your turn.")
    _draw_turn_card(state, next_player_index)
