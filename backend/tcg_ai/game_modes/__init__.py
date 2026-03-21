from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .my_first_battle import bot as my_first_battle_bot
from .my_first_battle import cards as my_first_battle_cards
from .my_first_battle import engine as my_first_battle_engine
from .my_first_battle import learning as my_first_battle_learning
from .my_first_battle import presentation as my_first_battle_presentation

DEFAULT_GAME_MODE = "my_first_battle"


@dataclass(frozen=True)
class GameModeDefinition:
    game_mode: str
    name: str
    description: str
    available: bool
    create_game: Callable[..., Any] | None = None
    apply_action: Callable[..., Any] | None = None
    serialize_state: Callable[..., dict[str, Any]] | None = None
    choose_action: Callable[..., dict[str, Any] | None] | None = None
    summarize_state: Callable[..., Any] | None = None
    calculate_reward: Callable[..., float] | None = None
    extract_action_features: Callable[..., tuple[str, ...]] | None = None
    default_human_deck_id: str | None = None
    deck_definitions: dict[str, Any] | None = None
    available_deck_snapshots: Callable[[str | None], list[dict[str, object]]] | None = None
    paired_deck_id_for: Callable[[str], str] | None = None

    def snapshot(self, selected: bool = False) -> dict[str, object]:
        return {
            "id": self.game_mode,
            "name": self.name,
            "description": self.description,
            "available": self.available,
            "selected": selected,
        }

    @property
    def supports_decks(self) -> bool:
        return (
            self.default_human_deck_id is not None
            and self.deck_definitions is not None
            and self.available_deck_snapshots is not None
        )


GAME_MODES: dict[str, GameModeDefinition] = {
    "my_first_battle": GameModeDefinition(
        game_mode="my_first_battle",
        name="My First Battle",
        description="Tutorial-style beginner decks with simplified rules.",
        available=True,
        create_game=my_first_battle_engine.create_game,
        apply_action=my_first_battle_engine.apply_action,
        serialize_state=my_first_battle_presentation.serialize_state,
        choose_action=my_first_battle_bot.choose_action,
        summarize_state=my_first_battle_learning.summarize_state,
        calculate_reward=my_first_battle_learning.calculate_reward,
        extract_action_features=my_first_battle_learning.extract_action_features,
        default_human_deck_id=my_first_battle_cards.DEFAULT_HUMAN_DECK_ID,
        deck_definitions=my_first_battle_cards.DECK_DEFINITIONS,
        available_deck_snapshots=my_first_battle_cards.available_deck_snapshots,
        paired_deck_id_for=my_first_battle_cards.paired_deck_id_for,
    ),
    "standard": GameModeDefinition(
        game_mode="standard",
        name="Standard",
        description="Full-size decks with Standard rules. Not available yet.",
        available=False,
    ),
}


def get_game_mode(game_mode: str) -> GameModeDefinition | None:
    return GAME_MODES.get(game_mode)


def available_game_mode_snapshots(selected_id: str | None = None) -> list[dict[str, object]]:
    return [
        mode.snapshot(selected=mode.game_mode == selected_id) for mode in GAME_MODES.values()
    ]
