from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Literal

CardKind = Literal["pokemon", "trainer", "energy"]
SetupPhase = Literal["choose_active", "awaiting_end_setup"]
EffectCountMode = Literal["fixed", "all"]


@dataclass(frozen=True)
class AttackEffectSpec:
    effect_type: str
    amount: int | None = None
    target_player: str = "self"
    target_zone: str | None = None
    selection_count: int | None = None
    energy_type: str | None = None
    optional: bool = False
    bonus_damage: int | None = None
    condition: str | None = None
    duration: str | None = None


@dataclass(frozen=True)
class AttackDefinition:
    name: str
    cost: int
    damage: str
    effect: str = "none"
    text: str = ""
    effect_specs: tuple[AttackEffectSpec, ...] = ()


@dataclass(frozen=True)
class EffectOption:
    option_id: str
    label: str
    effect_specs: tuple["EffectSpec", ...] = ()


@dataclass(frozen=True)
class EffectSpec:
    effect_type: str
    count: int | None = None
    count_mode: EffectCountMode = "fixed"
    source_zone: str | None = None
    destination_zone: str | None = None
    destination_position: str | None = None
    target_player: str = "self"
    selection_count: int | None = None
    choose_count: int | None = None
    search_filters: tuple[str, ...] = ()
    options: tuple[EffectOption, ...] = ()
    shuffle_destination: bool = False
    exclude_source_card: bool = False
    revealed_to: str | None = None
    changes_hidden_information: bool = False


@dataclass(frozen=True)
class CardDefinition:
    card_id: str
    name: str
    kind: CardKind
    element: str | None = None
    stage: str | None = None
    is_basic: bool = False
    evolves_from: str | None = None
    hp: int | None = None
    attacks: tuple[AttackDefinition, ...] = ()
    image_url: str | None = None
    card_tags: tuple[str, ...] = ()
    rules_text: tuple[str, ...] = ()
    effect_specs: tuple[EffectSpec, ...] = ()


@dataclass
class CardInstance:
    instance_id: str
    card_id: str
    owner: int


@dataclass(frozen=True)
class LingeringEffect:
    effect_type: str
    source_player: int
    expires_end_of_player_turn: int | None = None
    condition: str | None = None


@dataclass
class PokemonInPlay:
    stack: list[str] = field(default_factory=list)
    damage: int = 0
    attached_energy: list[str] = field(default_factory=list)
    entered_play_turn: int = 0
    lingering_effects: list[LingeringEffect] = field(default_factory=list)


@dataclass
class PlayerState:
    name: str
    deck_name: str
    element: str
    deck: list[str] = field(default_factory=list)
    hand: list[str] = field(default_factory=list)
    discard: list[str] = field(default_factory=list)
    active: PokemonInPlay | None = None
    bench: list[PokemonInPlay] = field(default_factory=list)
    prize_cards_remaining: int = 6
    mulligans_taken: int = 0
    supporter_played_this_turn: bool = False
    energy_attached_this_turn: bool = False
    turns_taken: int = 0


@dataclass
class GameState:
    cards: dict[str, CardInstance]
    card_definitions: dict[str, CardDefinition]
    players: list[PlayerState]
    current_player: int
    rng: random.Random = field(repr=False)
    starting_player: int = 0
    turn_number: int = 1
    winner: int | None = None
    log: list[str] = field(default_factory=list)
    seed: int = 0
    setup_phase: SetupPhase | None = "choose_active"
