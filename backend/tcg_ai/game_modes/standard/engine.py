from __future__ import annotations

from collections import defaultdict
import random
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
    PlayerState,
    PokemonInPlay,
)

OPENING_HAND_SIZE = 7
BENCH_LIMIT = 5


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
    actions.extend(_list_supporter_actions(state, player_index))
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
    if action not in legal_actions:
        raise ValueError("Illegal action.")

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
        player.active = PokemonInPlay(stack=[card_id])
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
        player.bench.append(PokemonInPlay(stack=[card_id]))
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

    if action["type"] == "play_supporter":
        card_id = _remove_from_hand(player, action["hand_card_id"])
        card = card_definition(state, card_id)
        state.log.append(f"{actor_name} played {card.name}.")
        _resolve_effect_specs(state, player_index, card.effect_specs)
        player.discard.append(card_id)
        player.supporter_played_this_turn = True
        return state

    if action["type"] == "end_setup":
        state.setup_phase = None
        state.players[state.current_player].supporter_played_this_turn = False
        state.players[state.current_player].energy_attached_this_turn = False
        state.log.append("Opening setup is complete. Turn 1 begins. You go first.")
        return state

    if action["type"] == "end_turn":
        state.current_player = 1 - player_index
        state.players[state.current_player].supporter_played_this_turn = False
        state.players[state.current_player].energy_attached_this_turn = False
        if player_index == 0:
            state.log.append("You ended the turn.")
            state.log.append(f"Turn {state.turn_number}: {state.players[1].name}'s turn.")
        else:
            state.turn_number += 1
            state.log.append(f"{player.name} ended the turn.")
            state.log.append(f"Turn {state.turn_number}: Your turn.")
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
    if action["type"] == "play_supporter":
        return f"play_supporter:{action['hand_card_id']}"
    return action["type"]


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


def _resolve_effect_specs(
    state: GameState,
    player_index: int,
    effect_specs: tuple[EffectSpec, ...],
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

        raise ValueError(f"Unsupported Standard effect type: {effect_spec.effect_type}")


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


def _list_supporter_actions(
    state: GameState,
    player_index: int,
) -> list[dict[str, Any]]:
    player = state.players[player_index]
    if player.supporter_played_this_turn:
        return []

    actions: list[dict[str, Any]] = []
    for instance_id in player.hand:
        card = card_definition(state, instance_id)
        if "supporter" not in card.card_tags or not card.effect_specs:
            continue
        if any(
            effect_spec.options
            or effect_spec.choose_count is not None
            or effect_spec.selection_count is not None
            or effect_spec.destination_position is not None
            or effect_spec.search_filters
            for effect_spec in card.effect_specs
        ):
            continue
        actions.append(
            {
                "type": "play_supporter",
                "hand_card_id": instance_id,
                "label": f"Play {card.name}",
            }
        )
    return actions


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
