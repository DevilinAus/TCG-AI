from __future__ import annotations

import unittest

from backend.tcg_ai.game_modes.standard.decision_payload import build_decision_request
from backend.tcg_ai.game_modes.standard.engine import apply_action, card_definition, create_game, list_legal_actions
from backend.tcg_ai.game_modes.standard.ml.knowledge_state import serialize_knowledge_actions
from backend.tcg_ai.game_modes.standard.models import PokemonInPlay


class StandardActionMetadataTests(unittest.TestCase):
    def test_curated_states_cover_expected_standard_action_types(self) -> None:
        examples = self._collect_action_examples()

        self.assertTrue(
            {
                "play_basic_to_active",
                "bench_basic",
                "end_setup",
                "play_energy",
                "retreat",
                "evolve",
                "play_supporter",
                "play_item",
                "attack",
                "end_turn",
                "promote",
            }.issubset(examples.keys())
        )

    def test_knowledge_actions_include_rule_metadata_for_retreat_and_search(self) -> None:
        examples = self._collect_action_examples()

        retreat_state, retreat_player_index, retreat_action = examples["retreat"]
        retreat_payload = serialize_knowledge_actions(
            retreat_state,
            acting_player_index=retreat_player_index,
            legal_actions=[retreat_action],
        )[0]
        self.assertEqual(retreat_payload["card_instance_id"], retreat_state.players[0].active.stack[-1])
        self.assertIn("retreat", retreat_payload["effect_tags"])
        self.assertIn("switch_active", retreat_payload["effect_tags"])
        self.assertEqual(retreat_payload["resource_costs"]["discard_attached_energy_count"], 1)
        self.assertEqual(retreat_payload["resource_costs"]["retreat_turn_cost"], 1)
        self.assertTrue(retreat_payload["expected_state_delta"]["active_changes"])
        self.assertTrue(retreat_payload["expected_state_delta"]["retreat_flag_set"])
        self.assertFalse(retreat_payload["expected_state_delta"]["turn_ends"])

        item_state, item_player_index, item_action = examples["play_item"]
        item_payload = serialize_knowledge_actions(
            item_state,
            acting_player_index=item_player_index,
            legal_actions=[item_action],
        )[0]
        self.assertIn("search_deck", item_payload["effect_tags"])
        self.assertIn("discard_from_hand", item_payload["effect_tags"])
        self.assertEqual(item_payload["resource_costs"]["discard_from_hand_count"], 2)
        self.assertEqual(item_payload["resource_costs"]["search_selection_count"], 1)
        self.assertEqual(item_payload["expected_state_delta"]["discard_count_delta_known"], 3)
        self.assertEqual(item_payload["expected_state_delta"]["hand_count_delta_known"], -2)

    def test_decision_payload_actions_include_metadata_fields(self) -> None:
        examples = self._collect_action_examples()
        state, player_index, action = examples["attack"]

        request = build_decision_request(
            state,
            session_id="session-metadata",
            decision_id="decision-metadata",
            decision_type="turn",
            acting_player_index=player_index,
            ai_trainer_id="brock",
            ai_deck_id="lucario-ex-battle-deck",
            legal_actions=[action],
        )

        payload = request["legal_actions"][0]
        self.assertIn("card_instance_id", payload)
        self.assertIn("effect_tags", payload)
        self.assertIn("resource_costs", payload)
        self.assertIn("expected_state_delta", payload)
        self.assertIn("attack", payload["effect_tags"])
        self.assertIn("damage", payload["effect_tags"])
        self.assertEqual(payload["resource_costs"]["attack_energy_cost"], 1)
        self.assertTrue(payload["expected_state_delta"]["turn_ends"])

    def _collect_action_examples(self) -> dict[str, tuple[object, int, dict[str, object]]]:
        examples: dict[str, tuple[object, int, dict[str, object]]] = {}

        opening_state = create_game(seed=1, human_deck_id="ampharos-ex-battle-deck")
        opening_actions = list_legal_actions(opening_state)
        examples["play_basic_to_active"] = (
            opening_state,
            0,
            next(action for action in opening_actions if action["type"] == "play_basic_to_active"),
        )

        setup_state = create_game(seed=1, human_deck_id="ampharos-ex-battle-deck")
        active_action = next(action for action in list_legal_actions(setup_state) if action["type"] == "play_basic_to_active")
        apply_action(setup_state, active_action)
        setup_actions = list_legal_actions(setup_state)
        examples["bench_basic"] = (setup_state, 0, next(action for action in setup_actions if action["type"] == "bench_basic"))
        examples["end_setup"] = (setup_state, 0, next(action for action in setup_actions if action["type"] == "end_setup"))

        midgame_state = create_game(seed=1, human_deck_id="ampharos-ex-battle-deck")
        midgame_state.setup_phase = None
        midgame_state.current_player = 0
        midgame_state.turn_number = 2
        midgame_state.players[0].turns_taken = 2
        midgame_state.players[1].turns_taken = 2
        self._set_named_active_pokemon(midgame_state, 0, "Mareep")
        self._set_named_bench_pokemon(midgame_state, 0, "Wattrel")
        self._set_named_active_pokemon(midgame_state, 1, "Mankey")
        retreat_energy_id = self._take_named_card(midgame_state, 0, "Basic Lightning Energy")
        bench_energy_id = self._take_named_card(midgame_state, 0, "Basic Lightning Energy")
        midgame_state.players[0].active.attached_energy = [retreat_energy_id]
        midgame_state.players[0].bench[0].attached_energy = [bench_energy_id]

        extra_basic_id = self._move_named_card_to_hand(midgame_state, 0, "Mareep")
        energy_id = self._move_named_card_to_hand(midgame_state, 0, "Basic Lightning Energy")
        flaaffy_id = self._move_named_card_to_hand(midgame_state, 0, "Flaaffy")
        nemona_id = self._move_named_card_to_hand(midgame_state, 0, "Nemona")
        ultra_ball_id = self._move_named_card_to_hand(midgame_state, 0, "Ultra Ball")
        nest_ball_id = self._move_named_card_to_hand(midgame_state, 0, "Nest Ball")
        potion_id = self._move_named_card_to_hand(midgame_state, 0, "Potion")
        switch_id = self._move_named_card_to_hand(midgame_state, 0, "Switch")
        self._set_exact_hand(
            midgame_state,
            0,
            [extra_basic_id, energy_id, flaaffy_id, nemona_id, ultra_ball_id, nest_ball_id, potion_id, switch_id],
        )

        midgame_actions = list_legal_actions(midgame_state)
        examples["play_energy"] = (midgame_state, 0, next(action for action in midgame_actions if action["type"] == "play_energy"))
        examples["retreat"] = (midgame_state, 0, next(action for action in midgame_actions if action["type"] == "retreat"))
        examples["evolve"] = (midgame_state, 0, next(action for action in midgame_actions if action["type"] == "evolve"))
        examples["play_supporter"] = (
            midgame_state,
            0,
            next(
                action
                for action in midgame_actions
                if action["type"] == "play_supporter"
                and card_definition(midgame_state, action["hand_card_id"]).name == "Nemona"
            ),
        )
        examples["play_item"] = (
            midgame_state,
            0,
            next(
                action
                for action in midgame_actions
                if action["type"] == "play_item"
                and card_definition(midgame_state, action["hand_card_id"]).name == "Ultra Ball"
            ),
        )
        examples["attack"] = (midgame_state, 0, next(action for action in midgame_actions if action["type"] == "attack"))
        examples["end_turn"] = (midgame_state, 0, next(action for action in midgame_actions if action["type"] == "end_turn"))

        promote_state = create_game(seed=1, human_deck_id="ampharos-ex-battle-deck")
        promote_state.setup_phase = None
        promote_state.current_player = 0
        promote_state.turn_number = 2
        promote_state.players[0].turns_taken = 2
        promote_state.players[1].turns_taken = 2
        self._set_named_bench_pokemon(promote_state, 0, "Wattrel")
        self._set_named_active_pokemon(promote_state, 1, "Mankey")
        promote_state.players[0].active = None
        promote_state.pending_promotion_for = 0
        promote_actions = list_legal_actions(promote_state)
        examples["promote"] = (promote_state, 0, next(action for action in promote_actions if action["type"] == "promote"))

        return examples

    def _move_named_card_to_hand(self, state, player_index: int, card_name: str) -> str:
        player = state.players[player_index]
        for zone_name in ("hand", "deck", "discard", "prizes"):
            zone = getattr(player, zone_name)
            for instance_id in list(zone):
                if card_definition(state, instance_id).name != card_name:
                    continue
                if zone_name != "hand":
                    zone.remove(instance_id)
                    player.hand.append(instance_id)
                return instance_id
        self.fail(f"Could not find {card_name} for player {player_index}")

    def _take_named_card(self, state, player_index: int, card_name: str) -> str:
        player = state.players[player_index]
        for zone_name in ("hand", "deck", "discard", "prizes"):
            zone = getattr(player, zone_name)
            for instance_id in list(zone):
                if card_definition(state, instance_id).name != card_name:
                    continue
                zone.remove(instance_id)
                return instance_id
        self.fail(f"Could not take {card_name} for player {player_index}")

    def _set_exact_hand(self, state, player_index: int, ordered_instance_ids: list[str]) -> None:
        player = state.players[player_index]
        kept = set(ordered_instance_ids)
        extras = [instance_id for instance_id in player.hand if instance_id not in kept]
        player.hand = list(ordered_instance_ids)
        player.deck.extend(extras)

    def _set_named_active_pokemon(self, state, player_index: int, card_name: str) -> None:
        instance_id = self._find_instance_id(state, player_index, card_name)
        player = state.players[player_index]
        for zone_name in ("hand", "deck", "discard", "prizes"):
            zone = getattr(player, zone_name)
            if instance_id in zone:
                zone.remove(instance_id)
                break
        player.active = PokemonInPlay(stack=[instance_id])

    def _set_named_bench_pokemon(self, state, player_index: int, card_name: str) -> None:
        instance_id = self._find_instance_id(state, player_index, card_name)
        player = state.players[player_index]
        for zone_name in ("hand", "deck", "discard", "prizes"):
            zone = getattr(player, zone_name)
            if instance_id in zone:
                zone.remove(instance_id)
                break
        player.bench.append(PokemonInPlay(stack=[instance_id]))

    def _find_instance_id(self, state, player_index: int, card_name: str) -> str:
        player = state.players[player_index]
        for zone_name in ("hand", "deck", "discard", "prizes"):
            zone = getattr(player, zone_name)
            for instance_id in zone:
                if card_definition(state, instance_id).name == card_name:
                    return instance_id
        self.fail(f"Could not find {card_name} for player {player_index}")
