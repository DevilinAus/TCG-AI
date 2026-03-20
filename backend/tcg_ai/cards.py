from __future__ import annotations

from dataclasses import dataclass

from .models import AttackDefinition, CardDefinition


def attack(name: str, cost: int, damage: int, effect: str = "none") -> AttackDefinition:
    return AttackDefinition(name=name, cost=cost, damage=damage, effect=effect)


def pokemon(
    card_id: str,
    name: str,
    element: str,
    hp: int,
    stage: str,
    attacks: tuple[AttackDefinition, ...],
    evolves_from: str | None = None,
) -> CardDefinition:
    return CardDefinition(
        card_id=card_id,
        name=name,
        kind="pokemon",
        element=element,
        hp=hp,
        stage=stage,
        evolves_from=evolves_from,
        attacks=attacks,
    )


def trainer(card_id: str, name: str, effect: str) -> CardDefinition:
    return CardDefinition(card_id=card_id, name=name, kind="trainer", trainer_effect=effect)


def energy(card_id: str, name: str, element: str) -> CardDefinition:
    return CardDefinition(card_id=card_id, name=name, kind="energy", element=element)


CARD_DEFINITIONS: dict[str, CardDefinition] = {
    "charmander": pokemon(
        "charmander",
        "Charmander",
        "fire",
        70,
        "basic",
        (attack("Scratch", 1, 10), attack("Ember", 2, 30)),
    ),
    "charmeleon": pokemon(
        "charmeleon",
        "Charmeleon",
        "fire",
        100,
        "stage1",
        (attack("Ember", 1, 30), attack("Flamethrower", 3, 60)),
        evolves_from="Charmander",
    ),
    "vulpix": pokemon(
        "vulpix",
        "Vulpix",
        "fire",
        60,
        "basic",
        (attack("Will-O-Wisp", 1, 20),),
    ),
    "ninetales": pokemon(
        "ninetales",
        "Ninetales",
        "fire",
        100,
        "stage1",
        (attack("Flame Tail", 2, 70),),
        evolves_from="Vulpix",
    ),
    "growlithe": pokemon(
        "growlithe",
        "Growlithe",
        "fire",
        80,
        "basic",
        (attack("Gnaw", 2, 30), attack("Take Down", 3, 50)),
    ),
    "arcanine": pokemon(
        "arcanine",
        "Arcanine",
        "fire",
        140,
        "stage1",
        (attack("Flare Blitz", 3, 90, "coin_flip_bonus_30"),),
        evolves_from="Growlithe",
    ),
    "magmar": pokemon(
        "magmar",
        "Magmar",
        "fire",
        90,
        "basic",
        (attack("Flare", 1, 20), attack("Magma Punch", 2, 50)),
    ),
    "squirtle": pokemon(
        "squirtle",
        "Squirtle",
        "water",
        70,
        "basic",
        (attack("Tackle", 1, 10), attack("Water Gun", 2, 30)),
    ),
    "wartortle": pokemon(
        "wartortle",
        "Wartortle",
        "water",
        100,
        "stage1",
        (attack("Water Gun", 1, 30), attack("Skull Bash", 3, 60)),
        evolves_from="Squirtle",
    ),
    "poliwag": pokemon(
        "poliwag",
        "Poliwag",
        "water",
        60,
        "basic",
        (attack("Pound", 1, 20),),
    ),
    "poliwhirl": pokemon(
        "poliwhirl",
        "Poliwhirl",
        "water",
        90,
        "stage1",
        (attack("Wave Splash", 1, 50),),
        evolves_from="Poliwag",
    ),
    "magikarp": pokemon(
        "magikarp",
        "Magikarp",
        "water",
        30,
        "basic",
        (attack("Splash", 1, 10),),
    ),
    "gyarados": pokemon(
        "gyarados",
        "Gyarados",
        "water",
        160,
        "stage1",
        (attack("Tail Smash", 3, 90, "coin_flip_fail"), attack("Hyper Beam", 4, 120)),
        evolves_from="Magikarp",
    ),
    "lapras": pokemon(
        "lapras",
        "Lapras",
        "water",
        110,
        "basic",
        (attack("Surf", 2, 50),),
    ),
    "potion": trainer("potion", "Potion", "heal_30"),
    "switch": trainer("switch", "Switch", "switch_active"),
    "fire_energy": energy("fire_energy", "Fire Energy", "fire"),
    "water_energy": energy("water_energy", "Water Energy", "water"),
}


@dataclass(frozen=True)
class DeckDefinition:
    deck_id: str
    name: str
    element: str
    starter_pokemon: str
    starting_energy: str
    entries: tuple[tuple[str, int], ...]


DECK_DEFINITIONS: dict[str, DeckDefinition] = {
    "charmander": DeckDefinition(
        deck_id="charmander",
        name="Charmander Deck",
        element="fire",
        starter_pokemon="charmander",
        starting_energy="fire_energy",
        entries=(
            ("charmander", 2),
            ("charmeleon", 2),
            ("vulpix", 2),
            ("ninetales", 1),
            ("growlithe", 2),
            ("arcanine", 1),
            ("magmar", 1),
            ("potion", 1),
            ("switch", 1),
            ("fire_energy", 4),
        ),
    ),
    "squirtle": DeckDefinition(
        deck_id="squirtle",
        name="Squirtle Deck",
        element="water",
        starter_pokemon="squirtle",
        starting_energy="water_energy",
        entries=(
            ("squirtle", 2),
            ("wartortle", 2),
            ("poliwag", 2),
            ("poliwhirl", 1),
            ("magikarp", 2),
            ("gyarados", 1),
            ("lapras", 1),
            ("potion", 1),
            ("switch", 1),
            ("water_energy", 4),
        ),
    ),
}
