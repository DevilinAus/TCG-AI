from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path

from .models import AttackDefinition, EffectSpec

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATA_ROOT = PROJECT_ROOT / "backend" / "tcg_ai" / "game_modes" / "standard" / "data"
MANIFEST_PATH = DATA_ROOT / "manifest.json"
CARD_CATALOG_PATH = DATA_ROOT / "catalog" / "cards.json"
EX_BATTLE_DECKS_DIR = DATA_ROOT / "decks" / "ex-battle"

TRAINER_CARD_CATEGORIES = {"Item", "Pokémon Tool", "Supporter", "Stadium"}
ENERGY_CATEGORY = "Energy"
ELEMENT_BY_CATEGORY = {
    "Colorless": "colorless",
    "Darkness": "darkness",
    "Dragon": "dragon",
    "Fighting": "fighting",
    "Fire": "fire",
    "Grass": "grass",
    "Lightning": "lightning",
    "Metal": "metal",
    "Psychic": "psychic",
    "Water": "water",
}


@dataclass(frozen=True)
class DeckDefinition:
    deck_id: str
    name: str
    element: str
    release_date: str
    product_line: str
    product_line_name: str
    card_count: int
    preview_card_image_url: str | None


@dataclass(frozen=True)
class DeckCardDefinition:
    card_id: str
    name: str
    quantity: int
    kind: str
    element: str | None
    stage: str | None
    is_basic: bool
    hp: int | None
    attacks: tuple[AttackDefinition, ...]
    image_url: str | None
    card_tags: tuple[str, ...]
    rules_text: tuple[str, ...]
    effect_specs: tuple[EffectSpec, ...]


TRAINER_TAGS_BY_SUBTYPE = {
    "Supporter": "supporter",
    "Item": "item",
    "Stadium": "stadium",
    "Pokémon Tool": "pokemon_tool",
}

TRAINER_EFFECT_SPECS: dict[str, tuple[EffectSpec, ...]] = {
    "sv1-180": (
        EffectSpec(
            effect_type="draw",
            count=3,
            source_zone="deck",
            destination_zone="hand",
            changes_hidden_information=True,
        ),
    ),
    "sv1-198": (
        EffectSpec(
            effect_type="shuffle_zone_into_deck",
            count_mode="all",
            source_zone="hand",
            destination_zone="deck",
            shuffle_destination=True,
            exclude_source_card=True,
            changes_hidden_information=True,
        ),
        EffectSpec(
            effect_type="draw",
            count=5,
            source_zone="deck",
            destination_zone="hand",
            changes_hidden_information=True,
        ),
    ),
}


@lru_cache(maxsize=1)
def _load_manifest() -> dict[str, object]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _load_card_catalog() -> dict[str, dict[str, object]]:
    payload = json.loads(CARD_CATALOG_PATH.read_text(encoding="utf-8"))
    cards = payload.get("cards", {})
    if not isinstance(cards, dict):
        raise ValueError("Standard card catalog is malformed.")
    return {
        card_id: card_payload
        for card_id, card_payload in cards.items()
        if isinstance(card_payload, dict)
    }


def _normalize_element(category: str | None) -> str:
    if not category:
        return "colorless"
    return ELEMENT_BY_CATEGORY.get(category, "colorless")


def _infer_deck_element(cards: list[dict[str, object]]) -> str:
    for card in cards:
        category = card.get("category")
        if category in TRAINER_CARD_CATEGORIES or category == ENERGY_CATEGORY:
            continue
        return _normalize_element(category if isinstance(category, str) else None)

    for card in cards:
        if card.get("category") != ENERGY_CATEGORY:
            continue
        detail = card.get("category_detail")
        if isinstance(detail, str) and detail:
            return _normalize_element(detail)

    return "colorless"


def _pick_preview_card_image(cards: list[dict[str, object]]) -> str | None:
    for card in cards:
        category = card.get("category")
        if category in TRAINER_CARD_CATEGORIES or category == ENERGY_CATEGORY:
            continue
        image_url = card.get("local_image_url")
        if isinstance(image_url, str) and image_url:
            return image_url

    if not cards:
        return None
    image_url = cards[0].get("local_image_url")
    return image_url if isinstance(image_url, str) and image_url else None


def _card_kind(category: str, category_detail: str | None) -> str:
    del category_detail
    if category == ENERGY_CATEGORY:
        return "energy"
    if category in TRAINER_CARD_CATEGORIES:
        return "trainer"
    return "pokemon"


def _card_element(category: str, category_detail: str | None) -> str | None:
    if category == ENERGY_CATEGORY:
        if category_detail is None:
            return None
        return _normalize_element(category_detail)
    if category in TRAINER_CARD_CATEGORIES:
        return None
    return _normalize_element(category)


def _catalog_stage(card_id: str, kind: str) -> tuple[str | None, bool]:
    if kind != "pokemon":
        return None, False

    card = _load_card_catalog().get(card_id)
    if card is None:
        raise ValueError(f"Missing Standard catalog data for card '{card_id}'.")

    subtypes = card.get("subtypes", [])
    if not isinstance(subtypes, list):
        raise ValueError(f"Malformed subtype data for Standard card '{card_id}'.")

    normalized_subtypes = {
        subtype for subtype in subtypes if isinstance(subtype, str) and subtype
    }
    if "Basic" in normalized_subtypes:
        return "basic", True
    if "Stage 1" in normalized_subtypes:
        return "stage1", False
    if "Stage 2" in normalized_subtypes:
        return "stage2", False
    raise ValueError(f"Unknown Pokemon stage data for Standard card '{card_id}'.")


def _catalog_hp(card_id: str, kind: str) -> int | None:
    if kind != "pokemon":
        return None

    card = _load_card_catalog().get(card_id)
    if card is None:
        raise ValueError(f"Missing Standard catalog data for card '{card_id}'.")

    hp = card.get("hp")
    if hp is None:
        return None
    return int(hp)


def _catalog_attacks(card_id: str, kind: str) -> tuple[AttackDefinition, ...]:
    if kind != "pokemon":
        return ()

    card = _load_card_catalog().get(card_id)
    if card is None:
        raise ValueError(f"Missing Standard catalog data for card '{card_id}'.")

    attacks = card.get("attacks", [])
    if not isinstance(attacks, list):
        raise ValueError(f"Malformed attack data for Standard card '{card_id}'.")

    definitions: list[AttackDefinition] = []
    for attack in attacks:
        if not isinstance(attack, dict):
            continue
        name = attack.get("name")
        converted_cost = attack.get("convertedEnergyCost")
        damage = attack.get("damage", "")
        if not isinstance(name, str) or not isinstance(converted_cost, int):
            continue
        definitions.append(
            AttackDefinition(
                name=name,
                cost=converted_cost,
                damage=str(damage or ""),
            )
        )
    return tuple(definitions)


def _catalog_card_tags(card_id: str) -> tuple[str, ...]:
    card = _load_card_catalog().get(card_id)
    if card is None:
        raise ValueError(f"Missing Standard catalog data for card '{card_id}'.")

    tags: list[str] = []
    subtypes = card.get("subtypes", [])
    if isinstance(subtypes, list):
        for subtype in subtypes:
            if not isinstance(subtype, str):
                continue
            tag = TRAINER_TAGS_BY_SUBTYPE.get(subtype)
            if tag and tag not in tags:
                tags.append(tag)

    abilities = card.get("abilities")
    if isinstance(abilities, list) and abilities and "ability" not in tags:
        tags.append("ability")

    return tuple(tags)


def _catalog_rules_text(card_id: str) -> tuple[str, ...]:
    card = _load_card_catalog().get(card_id)
    if card is None:
        raise ValueError(f"Missing Standard catalog data for card '{card_id}'.")

    rules = card.get("rules")
    if not isinstance(rules, list):
        return ()
    return tuple(rule for rule in rules if isinstance(rule, str) and rule)


def _effect_specs_for_card(card_id: str) -> tuple[EffectSpec, ...]:
    return TRAINER_EFFECT_SPECS.get(card_id, ())


@lru_cache(maxsize=1)
def _load_planned_ex_battle_decks() -> tuple[DeckDefinition, ...]:
    manifest = _load_manifest()
    product_line = manifest["product_lines"]["ex-battle"]
    deck_entries = product_line["decks"]
    definitions: list[DeckDefinition] = []
    for manifest_entry in deck_entries:
        if not manifest_entry.get("planned_for_initial_implementation"):
            continue
        deck_id = manifest_entry["deck_id"]
        payload_path = EX_BATTLE_DECKS_DIR / f"{deck_id}.json"
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        cards = payload.get("cards", [])
        definitions.append(
            DeckDefinition(
                deck_id=deck_id,
                name=payload["name"],
                element=_infer_deck_element(cards),
                release_date=payload["release_date"],
                product_line=payload["product_line"],
                product_line_name=payload["product_line_name"],
                card_count=int(payload["card_count"]),
                preview_card_image_url=_pick_preview_card_image(cards),
            )
        )
    return tuple(definitions)


@lru_cache(maxsize=None)
def load_deck_cards(deck_id: str) -> tuple[DeckCardDefinition, ...]:
    payload_path = EX_BATTLE_DECKS_DIR / f"{deck_id}.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    cards: list[DeckCardDefinition] = []
    for raw_card in payload.get("cards", []):
        category = raw_card.get("category")
        if not isinstance(category, str) or not category:
            continue
        category_detail = raw_card.get("category_detail")
        detail = category_detail if isinstance(category_detail, str) and category_detail else None
        kind = _card_kind(category, detail)
        name = raw_card.get("name")
        image_url = raw_card.get("local_image_url")
        card_id = raw_card.get("card_id")
        quantity = raw_card.get("quantity")
        if not isinstance(name, str) or not isinstance(card_id, str):
            continue
        stage, is_basic = _catalog_stage(card_id, kind)
        cards.append(
            DeckCardDefinition(
                card_id=card_id,
                name=name,
                quantity=int(quantity or 0),
                kind=kind,
                element=_card_element(category, detail),
                stage=stage,
                is_basic=is_basic,
                hp=_catalog_hp(card_id, kind),
                attacks=_catalog_attacks(card_id, kind),
                image_url=image_url if isinstance(image_url, str) and image_url else None,
                card_tags=_catalog_card_tags(card_id),
                rules_text=_catalog_rules_text(card_id),
                effect_specs=_effect_specs_for_card(card_id),
            )
        )
    return tuple(cards)


def _build_pairings(deck_ids_in_order: list[str]) -> dict[str, str]:
    pairings: dict[str, str] = {}
    ordered_definitions = [DECK_DEFINITIONS[deck_id] for deck_id in deck_ids_in_order]
    release_groups: dict[str, list[str]] = {}
    for definition in ordered_definitions:
        release_groups.setdefault(definition.release_date, []).append(definition.deck_id)

    for release_date in sorted(release_groups):
        grouped_ids = release_groups[release_date]
        if len(grouped_ids) == 1:
            pairings[grouped_ids[0]] = grouped_ids[0]
            continue
        for index in range(0, len(grouped_ids), 2):
            left = grouped_ids[index]
            right = grouped_ids[index + 1] if index + 1 < len(grouped_ids) else grouped_ids[index]
            pairings[left] = right
            pairings[right] = left
    return pairings


_DECKS_IN_ORDER = _load_planned_ex_battle_decks()
DECK_DEFINITIONS: dict[str, DeckDefinition] = {
    definition.deck_id: definition for definition in _DECKS_IN_ORDER
}
DEFAULT_HUMAN_DECK_ID = _DECKS_IN_ORDER[0].deck_id
PAIRED_DECK_IDS = _build_pairings([definition.deck_id for definition in _DECKS_IN_ORDER])


def paired_deck_id_for(deck_id: str) -> str:
    return PAIRED_DECK_IDS[deck_id]


def available_deck_snapshots(selected_id: str | None = None) -> list[dict[str, object]]:
    snapshots = []
    for definition in _DECKS_IN_ORDER:
        paired_id = paired_deck_id_for(definition.deck_id)
        paired_definition = DECK_DEFINITIONS[paired_id]
        snapshots.append(
            {
                "id": definition.deck_id,
                "name": definition.name,
                "element": definition.element,
                "paired_deck_id": paired_id,
                "paired_deck_name": paired_definition.name,
                "product_line": definition.product_line,
                "product_line_name": definition.product_line_name,
                "release_date": definition.release_date,
                "card_count": definition.card_count,
                "preview_card_image_url": definition.preview_card_image_url,
                "selected": definition.deck_id == selected_id,
            }
        )
    return snapshots
