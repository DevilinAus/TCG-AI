from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Literal

CardKind = Literal["pokemon", "trainer", "energy"]
Stage = Literal["basic", "stage1"]
AttackEffect = Literal["none", "coin_flip_bonus_30", "coin_flip_fail"]
TrainerEffect = Literal["heal_30", "switch_active"]


@dataclass(frozen=True)
class AttackDefinition:
    name: str
    cost: int
    damage: int
    effect: AttackEffect = "none"


@dataclass(frozen=True)
class CardDefinition:
    card_id: str
    name: str
    kind: CardKind
    element: str | None = None
    hp: int | None = None
    stage: Stage | None = None
    evolves_from: str | None = None
    attacks: tuple[AttackDefinition, ...] = ()
    trainer_effect: TrainerEffect | None = None


@dataclass
class CardInstance:
    instance_id: str
    card_id: str
    owner: int


@dataclass
class PokemonInPlay:
    stack: list[str] = field(default_factory=list)
    damage: int = 0
    entered_play_turn: int = 0


@dataclass
class PlayerState:
    name: str
    deck_name: str
    element: str
    deck: list[str] = field(default_factory=list)
    hand: list[str] = field(default_factory=list)
    discard: list[str] = field(default_factory=list)
    energy_zone: list[str] = field(default_factory=list)
    active: PokemonInPlay | None = None
    bench: list[PokemonInPlay] = field(default_factory=list)
    prize_tokens_remaining: int = 3
    energy_played_this_turn: bool = False
    turns_taken: int = 0


@dataclass
class GameState:
    cards: dict[str, CardInstance]
    players: list[PlayerState]
    current_player: int
    rng: random.Random = field(repr=False)
    turn_number: int = 0
    winner: int | None = None
    log: list[str] = field(default_factory=list)
    seed: int = 0
    pending_promotion_for: int | None = None
    turn_starts_after_promotion: bool = False
