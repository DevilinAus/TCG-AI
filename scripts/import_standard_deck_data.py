#!/usr/bin/env python3

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen

ROOT_DIR = Path(__file__).resolve().parents[1]
STANDARD_DATA_DIR = ROOT_DIR / "backend" / "tcg_ai" / "game_modes" / "standard" / "data"
DECKS_DIR = STANDARD_DATA_DIR / "decks"
CATALOG_DIR = STANDARD_DATA_DIR / "catalog"
ASSET_DIR = ROOT_DIR / "frontend" / "assets" / "cards" / "standard" / "shared"

BULBAPEDIA_RAW_URL = "https://bulbapedia.bulbagarden.net/w/index.php?title={title}&action=raw"
BULBAPEDIA_PAGE_URL = "https://bulbapedia.bulbagarden.net/wiki/{title}"
TCG_DATA_SETS_URL = "https://raw.githubusercontent.com/PokemonTCG/pokemon-tcg-data/master/sets/en.json"
TCG_DATA_SET_URL = "https://raw.githubusercontent.com/PokemonTCG/pokemon-tcg-data/master/cards/en/{set_id}.json"
USER_AGENT = "TCG-AI deck import/1.0"

SET_NAME_ALIASES = {
    "SVP Promo": "Scarlet & Violet Black Star Promos",
    "SVE Energy": "Scarlet & Violet Energies",
}

PRODUCT_LINES = {
    "ex-battle": {
        "name": "ex Battle Deck",
        "play_level": 1,
        "planned_for_initial_implementation": True,
    },
    "ex-deluxe-battle": {
        "name": "ex Deluxe Battle Deck",
        "play_level": 2,
        "planned_for_initial_implementation": False,
    },
}

DECK_SPECS = (
    {
        "deck_id": "ampharos-ex-battle-deck",
        "name": "Ampharos ex Battle Deck",
        "page_title": "Ampharos_ex_Battle_Deck_(TCG)",
        "product_line": "ex-battle",
    },
    {
        "deck_id": "lucario-ex-battle-deck",
        "name": "Lucario ex Battle Deck",
        "page_title": "Lucario_ex_Battle_Deck_(TCG)",
        "product_line": "ex-battle",
    },
    {
        "deck_id": "chien-pao-ex-battle-deck",
        "name": "Chien-Pao ex Battle Deck",
        "page_title": "Chien-Pao_ex_Battle_Deck_(TCG)",
        "product_line": "ex-battle",
    },
    {
        "deck_id": "tinkaton-ex-battle-deck",
        "name": "Tinkaton ex Battle Deck",
        "page_title": "Tinkaton_ex_Battle_Deck_(TCG)",
        "product_line": "ex-battle",
    },
    {
        "deck_id": "meowscarada-ex-deluxe-battle-deck",
        "name": "Meowscarada ex Deluxe Battle Deck",
        "page_title": "Meowscarada_ex_Deluxe_Battle_Deck_(TCG)",
        "product_line": "ex-deluxe-battle",
    },
    {
        "deck_id": "quaquaval-ex-deluxe-battle-deck",
        "name": "Quaquaval ex Deluxe Battle Deck",
        "page_title": "Quaquaval_ex_Deluxe_Battle_Deck_(TCG)",
        "product_line": "ex-deluxe-battle",
    },
    {
        "deck_id": "greninja-ex-battle-deck",
        "name": "Greninja ex Battle Deck",
        "page_title": "Greninja_ex_Battle_Deck_(TCG)",
        "product_line": "ex-battle",
    },
    {
        "deck_id": "kangaskhan-ex-battle-deck",
        "name": "Kangaskhan ex Battle Deck",
        "page_title": "Kangaskhan_ex_Battle_deck_(TCG)",
        "product_line": "ex-battle",
    },
    {
        "deck_id": "ninetales-ex-deluxe-battle-deck",
        "name": "Ninetales ex Deluxe Battle Deck",
        "page_title": "Ninetales_ex_Deluxe_Battle_Deck_(TCG)",
        "product_line": "ex-deluxe-battle",
    },
    {
        "deck_id": "zapdos-ex-deluxe-battle-deck",
        "name": "Zapdos ex Deluxe Battle Deck",
        "page_title": "Zapdos_ex_Deluxe_Battle_Deck_(TCG)",
        "product_line": "ex-deluxe-battle",
    },
    {
        "deck_id": "houndoom-ex-battle-deck",
        "name": "Houndoom ex Battle Deck",
        "page_title": "Houndoom_ex_Battle_Deck_(TCG)",
        "product_line": "ex-battle",
    },
    {
        "deck_id": "melmetal-ex-battle-deck",
        "name": "Melmetal ex Battle Deck",
        "page_title": "Melmetal_ex_Battle_Deck_(TCG)",
        "product_line": "ex-battle",
    },
    {
        "deck_id": "miraidon-ex-battle-deck",
        "name": "Miraidon ex Battle Deck",
        "page_title": "Ex_Battle_Decks—Victini_&_Miraidon_(TCG)",
        "product_line": "ex-battle",
    },
    {
        "deck_id": "victini-ex-battle-deck",
        "name": "Victini ex Battle Deck",
        "page_title": "Ex_Battle_Decks—Victini_&_Miraidon_(TCG)",
        "product_line": "ex-battle",
    },
    {
        "deck_id": "miraidon-ex-deluxe-battle-deck",
        "name": "Miraidon ex Deluxe Battle Deck",
        "page_title": "Deluxe_Battle_Decks—Koraidon_ex_&_Miraidon_ex_(TCG)",
        "product_line": "ex-deluxe-battle",
    },
    {
        "deck_id": "koraidon-ex-deluxe-battle-deck",
        "name": "Koraidon ex Deluxe Battle Deck",
        "page_title": "Deluxe_Battle_Decks—Koraidon_ex_&_Miraidon_ex_(TCG)",
        "product_line": "ex-deluxe-battle",
    },
    {
        "deck_id": "tapu-koko-ex-battle-deck",
        "name": "Tapu Koko ex Battle Deck",
        "page_title": "Ex_Battle_Decks—Tapu_Koko_&_Iron_Leaves_(TCG)",
        "product_line": "ex-battle",
    },
    {
        "deck_id": "iron-leaves-ex-battle-deck",
        "name": "Iron Leaves ex Battle Deck",
        "page_title": "Ex_Battle_Decks—Tapu_Koko_&_Iron_Leaves_(TCG)",
        "product_line": "ex-battle",
    },
)


def fetch_json(url: str) -> Any:
    with urlopen(make_request(url), timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_text(url: str) -> str:
    with urlopen(make_request(url), timeout=30) as response:
        return response.read().decode("utf-8")


def make_request(url: str) -> Request:
    return Request(url, headers={"User-Agent": USER_AGENT})


def split_top_level(text: str, separator: str = "|") -> list[str]:
    fields: list[str] = []
    current: list[str] = []
    template_depth = 0
    link_depth = 0
    index = 0
    while index < len(text):
        token = text[index : index + 2]
        if token == "{{":
            template_depth += 1
            current.append(token)
            index += 2
            continue
        if token == "}}" and template_depth > 0:
            template_depth -= 1
            current.append(token)
            index += 2
            continue
        if token == "[[":
            link_depth += 1
            current.append(token)
            index += 2
            continue
        if token == "]]" and link_depth > 0:
            link_depth -= 1
            current.append(token)
            index += 2
            continue
        if text[index] == separator and template_depth == 0 and link_depth == 0:
            fields.append("".join(current))
            current = []
            index += 1
            continue
        current.append(text[index])
        index += 1
    fields.append("".join(current))
    return fields


def decode_redirect(raw_text: str) -> str | None:
    match = re.match(r"#REDIRECT \[\[(.+?)\]\]", raw_text.strip())
    return match.group(1) if match else None


def fetch_bulbapedia_raw(title: str) -> tuple[str, str]:
    visited: set[str] = set()
    current = title
    while True:
        if current in visited:
            raise RuntimeError(f"Redirect loop while fetching Bulbapedia page {title!r}.")
        visited.add(current)
        raw_url = BULBAPEDIA_RAW_URL.format(title=quote(current, safe="()_"))
        raw_text = fetch_text(raw_url)
        redirect_target = decode_redirect(raw_text)
        if redirect_target is None:
            return current, raw_text
        current = redirect_target.replace(" ", "_")


def parse_release_date(raw_text: str) -> str:
    match = re.search(r"^\|release=(.+)$", raw_text, flags=re.MULTILINE)
    if not match:
        raise RuntimeError("Could not find release date in deck page.")
    return datetime.strptime(match.group(1).strip(), "%B %d, %Y").date().isoformat()


def parse_header_args(line: str) -> dict[str, str]:
    content = line.strip()[2:-2]
    fields = split_top_level(content)
    values: dict[str, str] = {}
    for field in fields[1:]:
        key, _, value = field.partition("=")
        values[key.strip()] = value.strip()
    return values


def parse_card_reference(field: str) -> dict[str, str]:
    field = field.strip()
    if "{{TCG ID|" in field:
        start = field.index("{{TCG ID|") + 2
        end = field.index("}}", start)
        template_fields = split_top_level(field[start:end])
        if len(template_fields) < 4:
            raise RuntimeError(f"Unsupported TCG ID reference: {field}")
        _, set_name, card_name, number = template_fields[:4]
        return {
            "set_name": canonical_set_name(set_name),
            "card_name": strip_html(card_name),
            "number": normalize_card_number(number),
        }

    link_match = re.search(r"\[\[([^|\]]+)(?:\|([^\]]+))?\]\]", field)
    if not link_match:
        raise RuntimeError(f"Unsupported card reference: {field}")
    link_target = link_match.group(1)
    label = link_match.group(2) or link_target
    if "{{ex}}" in field and not label.endswith(" ex"):
        label = f"{label} ex"

    target_match = re.match(r"^(.*?) \((.+) (\d+)\)$", link_target)
    if not target_match:
        raise RuntimeError(f"Could not parse linked card reference: {field}")

    set_alias = target_match.group(2)
    return {
        "set_name": canonical_set_name(set_alias),
        "card_name": strip_html(label),
        "number": normalize_card_number(target_match.group(3)),
    }


def canonical_set_name(set_name: str) -> str:
    return SET_NAME_ALIASES.get(set_name, set_name)


def normalize_card_number(number: str) -> str:
    normalized = number.strip()
    if normalized.isdigit():
        return str(int(normalized))
    return normalized


def strip_html(value: str) -> str:
    value = re.sub(r"<[^>]+>", "", value)
    return value.replace("'''", "").replace("''", "").strip()


def normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def parse_deck_entries(raw_text: str, deck_name: str) -> tuple[str, list[dict[str, Any]]]:
    lines = raw_text.splitlines()
    collecting = False
    deck_type = ""
    entries: list[dict[str, Any]] = []
    expected_title = normalize_text(deck_name)
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("{{halfdecklist/header|"):
            header = parse_header_args(stripped)
            collecting = normalize_text(header.get("title", "")) == expected_title
            if collecting:
                deck_type = header.get("type", "")
                entries = []
            continue
        if collecting and stripped.startswith("{{halfdecklist/footer"):
            return deck_type, entries
        if collecting and stripped.startswith("{{halfdecklist/entry|"):
            content = stripped[2:-2]
            fields = split_top_level(content)
            if len(fields) < 7:
                raise RuntimeError(f"Unexpected deck entry format for {deck_name}: {stripped}")
            reference = parse_card_reference(fields[3])
            entries.append(
                {
                    "order": len(entries) + 1,
                    "raw_number": fields[1].strip(),
                    "mark": fields[2].strip(),
                    "set_name": reference["set_name"],
                    "card_name": reference["card_name"],
                    "number": reference["number"],
                    "category": fields[4].strip(),
                    "category_detail": fields[5].strip(),
                    "quantity": int(fields[6].strip()),
                }
            )
    raise RuntimeError(f"Could not find deck list for {deck_name}.")


def load_set_catalog() -> dict[str, dict[str, Any]]:
    return {entry["name"]: entry for entry in fetch_json(TCG_DATA_SETS_URL)}


def load_cards_for_set(set_id: str, cache: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    if set_id not in cache:
        cache[set_id] = fetch_json(TCG_DATA_SET_URL.format(set_id=set_id))
    return cache[set_id]


def resolve_card_record(
    set_entry: dict[str, Any],
    number: str,
    card_name: str,
    set_cache: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    cards = load_cards_for_set(set_entry["id"], set_cache)
    normalized_number = normalize_card_number(number)
    exact_matches = [
        card
        for card in cards
        if normalize_card_number(str(card.get("number", ""))) == normalized_number
    ]
    if len(exact_matches) == 1:
        return exact_matches[0]
    if exact_matches:
        normalized_name = normalize_text(card_name)
        named_matches = [card for card in exact_matches if normalize_text(card["name"]) == normalized_name]
        if len(named_matches) == 1:
            return named_matches[0]
    raise RuntimeError(
        f"Could not resolve card {card_name!r} #{number} in set {set_entry['name']!r} ({set_entry['id']})."
    )


def export_card(
    card: dict[str, Any],
    set_entry: dict[str, Any],
    local_image_url: str,
    local_image_path: str,
    deck_memberships: list[str],
) -> dict[str, Any]:
    exported = {
        "card_id": card["id"],
        "name": card["name"],
        "number": card["number"],
        "set_id": set_entry["id"],
        "set_name": set_entry["name"],
        "supertype": card.get("supertype"),
        "subtypes": card.get("subtypes", []),
        "hp": card.get("hp"),
        "types": card.get("types", []),
        "evolves_from": card.get("evolvesFrom"),
        "evolves_to": card.get("evolvesTo", []),
        "abilities": card.get("abilities", []),
        "attacks": card.get("attacks", []),
        "rules": card.get("rules", []),
        "weaknesses": card.get("weaknesses", []),
        "resistances": card.get("resistances", []),
        "retreat_cost": card.get("retreatCost", []),
        "converted_retreat_cost": card.get("convertedRetreatCost"),
        "rarity": card.get("rarity"),
        "regulation_mark": card.get("regulationMark"),
        "legalities": card.get("legalities", {}),
        "images": {
            "remote_small": card.get("images", {}).get("small"),
            "remote_large": card.get("images", {}).get("large"),
            "local_small": local_image_url,
            "local_path": local_image_path,
        },
        "used_in_decks": sorted(deck_memberships),
        "source": {
            "card_data_repo": "https://github.com/PokemonTCG/pokemon-tcg-data",
            "set_file_url": TCG_DATA_SET_URL.format(set_id=set_entry["id"]),
        },
    }
    if "rules" in card and any("Rule Box" in rule for rule in card["rules"]):
        exported["has_rule_box"] = True
    return exported


def download_image(url: str, destination: Path) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(make_request(url), timeout=30) as response:
        destination.write_bytes(response.read())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    set_catalog = load_set_catalog()
    set_cache: dict[str, list[dict[str, Any]]] = {}
    used_sets: dict[str, dict[str, Any]] = {}
    cards_by_id: dict[str, dict[str, Any]] = {}
    card_memberships: dict[str, set[str]] = defaultdict(set)
    deck_manifests: dict[str, list[dict[str, Any]]] = defaultdict(list)
    asset_manifest: dict[str, str] = {}

    for deck_spec in DECK_SPECS:
        resolved_title, raw_text = fetch_bulbapedia_raw(deck_spec["page_title"])
        release_date = parse_release_date(raw_text)
        deck_type, entries = parse_deck_entries(raw_text, deck_spec["name"])
        product_line = PRODUCT_LINES[deck_spec["product_line"]]
        deck_cards: list[dict[str, Any]] = []

        for entry in entries:
            set_entry = set_catalog.get(entry["set_name"])
            if set_entry is None:
                raise RuntimeError(f"Unknown set name {entry['set_name']!r} while importing {deck_spec['name']}.")

            card = resolve_card_record(
                set_entry=set_entry,
                number=entry["number"],
                card_name=entry["card_name"],
                set_cache=set_cache,
            )
            used_sets[set_entry["id"]] = {
                "id": set_entry["id"],
                "name": set_entry["name"],
                "series": set_entry["series"],
                "release_date": set_entry["releaseDate"],
                "updated_at": set_entry["updatedAt"],
                "images": set_entry["images"],
            }
            card_memberships[card["id"]].add(deck_spec["deck_id"])
            asset_filename = f"{card['id']}.png"
            asset_relative_path = Path("frontend/assets/cards/standard/shared") / asset_filename
            asset_manifest[card["id"]] = asset_filename
            local_image_url = f"/assets/cards/standard/shared/{asset_filename}"
            local_image_path = str(asset_relative_path)

            deck_cards.append(
                {
                    "order": entry["order"],
                    "card_id": card["id"],
                    "name": card["name"],
                    "set_id": set_entry["id"],
                    "set_name": set_entry["name"],
                    "number": card["number"],
                    "quantity": entry["quantity"],
                    "category": entry["category"],
                    "category_detail": entry["category_detail"] or None,
                    "local_image_url": local_image_url,
                }
            )

            cards_by_id[card["id"]] = export_card(
                card=card,
                set_entry=set_entry,
                local_image_url=local_image_url,
                local_image_path=local_image_path,
                deck_memberships=sorted(card_memberships[card["id"]]),
            )

            image_url = card.get("images", {}).get("small")
            if image_url:
                download_image(image_url, ASSET_DIR / asset_filename)

        deck_payload = {
            "deck_id": deck_spec["deck_id"],
            "name": deck_spec["name"],
            "product_line": deck_spec["product_line"],
            "product_line_name": product_line["name"],
            "play_level": product_line["play_level"],
            "planned_for_initial_implementation": product_line["planned_for_initial_implementation"],
            "release_date": release_date,
            "deck_type": deck_type,
            "card_count": sum(card["quantity"] for card in deck_cards),
            "cards": deck_cards,
            "source": {
                "bulbapedia_page_title": resolved_title,
                "bulbapedia_page_url": BULBAPEDIA_PAGE_URL.format(
                    title=quote(resolved_title, safe="()_")
                ),
                "bulbapedia_raw_url": BULBAPEDIA_RAW_URL.format(
                    title=quote(resolved_title, safe="()_")
                ),
            },
        }
        deck_manifests[deck_spec["product_line"]].append(
            {
                "deck_id": deck_payload["deck_id"],
                "name": deck_payload["name"],
                "release_date": deck_payload["release_date"],
                "planned_for_initial_implementation": deck_payload[
                    "planned_for_initial_implementation"
                ],
                "card_count": deck_payload["card_count"],
            }
        )
        write_json(
            DECKS_DIR / deck_spec["product_line"] / f"{deck_spec['deck_id']}.json",
            deck_payload,
        )

    for card_id, memberships in card_memberships.items():
        cards_by_id[card_id]["used_in_decks"] = sorted(memberships)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_summary": {
            "deck_lists": "Bulbapedia raw wiki pages",
            "card_details": "PokemonTCG/pokemon-tcg-data",
            "artwork_files": "Downloaded from the images referenced by PokemonTCG/pokemon-tcg-data",
        },
        "deck_scope": {
            "requested_decks_only": True,
            "deck_count": len(DECK_SPECS),
        },
        "product_lines": {
            product_line: {
                "name": PRODUCT_LINES[product_line]["name"],
                "planned_for_initial_implementation": PRODUCT_LINES[product_line][
                    "planned_for_initial_implementation"
                ],
                "decks": sorted(deck_manifests[product_line], key=lambda deck: deck["name"]),
            }
            for product_line in sorted(PRODUCT_LINES)
        },
        "card_catalog_path": "backend/tcg_ai/game_modes/standard/data/catalog/cards.json",
        "set_catalog_path": "backend/tcg_ai/game_modes/standard/data/catalog/sets.json",
        "local_asset_manifest_path": "frontend/assets/cards/standard/shared/manifest.json",
    }

    write_json(CATALOG_DIR / "cards.json", {"cards": cards_by_id})
    write_json(CATALOG_DIR / "sets.json", {"sets": used_sets})
    write_json(STANDARD_DATA_DIR / "manifest.json", manifest)
    write_json(
        ASSET_DIR / "manifest.json",
        {
            "cards": asset_manifest,
        },
    )

    print(
        f"Imported {len(DECK_SPECS)} decks, {len(cards_by_id)} unique cards, and {len(used_sets)} sets."
    )


if __name__ == "__main__":
    main()
