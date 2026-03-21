from __future__ import annotations

import json
from pathlib import Path
import unittest


ROOT_DIR = Path(__file__).resolve().parents[1]
STANDARD_DATA_DIR = ROOT_DIR / "backend" / "tcg_ai" / "game_modes" / "standard" / "data"
DECKS_DIR = STANDARD_DATA_DIR / "decks"
CATALOG_DIR = STANDARD_DATA_DIR / "catalog"
ASSET_MANIFEST_PATH = ROOT_DIR / "frontend" / "assets" / "cards" / "standard" / "shared" / "manifest.json"

EXPECTED_EX_BATTLE_DECK_IDS = {
    "ampharos-ex-battle-deck",
    "lucario-ex-battle-deck",
    "chien-pao-ex-battle-deck",
    "tinkaton-ex-battle-deck",
    "greninja-ex-battle-deck",
    "kangaskhan-ex-battle-deck",
    "houndoom-ex-battle-deck",
    "melmetal-ex-battle-deck",
    "miraidon-ex-battle-deck",
    "victini-ex-battle-deck",
    "tapu-koko-ex-battle-deck",
    "iron-leaves-ex-battle-deck",
}

EXPECTED_EX_DELUXE_DECK_IDS = {
    "meowscarada-ex-deluxe-battle-deck",
    "quaquaval-ex-deluxe-battle-deck",
    "ninetales-ex-deluxe-battle-deck",
    "zapdos-ex-deluxe-battle-deck",
    "miraidon-ex-deluxe-battle-deck",
    "koraidon-ex-deluxe-battle-deck",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class StandardDataTests(unittest.TestCase):
    def test_manifest_is_scoped_to_requested_decks_only(self) -> None:
        manifest = read_json(STANDARD_DATA_DIR / "manifest.json")

        self.assertTrue(manifest["deck_scope"]["requested_decks_only"])
        self.assertEqual(manifest["deck_scope"]["deck_count"], 18)

        ex_battle = manifest["product_lines"]["ex-battle"]
        ex_deluxe = manifest["product_lines"]["ex-deluxe-battle"]

        self.assertTrue(ex_battle["planned_for_initial_implementation"])
        self.assertFalse(ex_deluxe["planned_for_initial_implementation"])
        self.assertEqual(
            {deck["deck_id"] for deck in ex_battle["decks"]},
            EXPECTED_EX_BATTLE_DECK_IDS,
        )
        self.assertEqual(
            {deck["deck_id"] for deck in ex_deluxe["decks"]},
            EXPECTED_EX_DELUXE_DECK_IDS,
        )

    def test_deck_directories_match_expected_product_line_split(self) -> None:
        ex_battle_paths = sorted((DECKS_DIR / "ex-battle").glob("*.json"))
        ex_deluxe_paths = sorted((DECKS_DIR / "ex-deluxe-battle").glob("*.json"))

        self.assertEqual({path.stem for path in ex_battle_paths}, EXPECTED_EX_BATTLE_DECK_IDS)
        self.assertEqual({path.stem for path in ex_deluxe_paths}, EXPECTED_EX_DELUXE_DECK_IDS)

    def test_each_imported_deck_is_a_full_sixty_card_list_with_expected_flags(self) -> None:
        for product_line, expected_ids, planned in (
            ("ex-battle", EXPECTED_EX_BATTLE_DECK_IDS, True),
            ("ex-deluxe-battle", EXPECTED_EX_DELUXE_DECK_IDS, False),
        ):
            for deck_id in expected_ids:
                deck = read_json(DECKS_DIR / product_line / f"{deck_id}.json")
                self.assertEqual(deck["deck_id"], deck_id)
                self.assertEqual(deck["product_line"], product_line)
                self.assertEqual(deck["card_count"], 60)
                self.assertEqual(sum(card["quantity"] for card in deck["cards"]), 60)
                self.assertEqual(deck["planned_for_initial_implementation"], planned)

    def test_card_catalog_and_asset_manifest_cover_all_referenced_cards(self) -> None:
        cards_payload = read_json(CATALOG_DIR / "cards.json")
        asset_manifest = read_json(ASSET_MANIFEST_PATH)

        requested_deck_ids = EXPECTED_EX_BATTLE_DECK_IDS | EXPECTED_EX_DELUXE_DECK_IDS
        referenced_card_ids: set[str] = set()

        for product_line, deck_ids in (
            ("ex-battle", EXPECTED_EX_BATTLE_DECK_IDS),
            ("ex-deluxe-battle", EXPECTED_EX_DELUXE_DECK_IDS),
        ):
            for deck_id in deck_ids:
                deck = read_json(DECKS_DIR / product_line / f"{deck_id}.json")
                referenced_card_ids.update(card["card_id"] for card in deck["cards"])

        self.assertEqual(referenced_card_ids, set(cards_payload["cards"]))
        self.assertEqual(referenced_card_ids, set(asset_manifest["cards"]))

        for card_id, card in cards_payload["cards"].items():
            self.assertTrue(set(card["used_in_decks"]).issubset(requested_deck_ids))
            self.assertEqual(
                card["images"]["local_small"],
                f"/assets/cards/standard/shared/{asset_manifest['cards'][card_id]}",
            )


if __name__ == "__main__":
    unittest.main()
