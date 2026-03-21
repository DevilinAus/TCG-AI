from __future__ import annotations

import unittest
from unittest.mock import patch

from backend.tcg_ai.game_modes.standard.engine import (
    apply_action,
    card_definition,
    choose_action,
    create_game,
    list_legal_actions,
)
from backend.tcg_ai.game_modes.standard.models import PokemonInPlay


class StandardEngineTests(unittest.TestCase):
    def test_only_implemented_direct_draw_supporters_are_playable(self) -> None:
        state = create_game(seed=1, human_deck_id="ampharos-ex-battle-deck")
        self._finish_opening_setup(state)
        self._move_named_card_to_hand(state, 0, "Nemona")
        self._move_named_card_to_hand(state, 0, "Youngster")
        self._move_named_card_to_hand(state, 0, "Jacq")

        supporter_actions = [action for action in list_legal_actions(state) if action["type"] == "play_supporter"]

        self.assertEqual(
            sorted(card_definition(state, action["hand_card_id"]).name for action in supporter_actions),
            ["Nemona", "Youngster"],
        )

    def test_nemona_draws_three_cards_and_is_discarded(self) -> None:
        state = create_game(seed=1, human_deck_id="ampharos-ex-battle-deck")
        self._finish_opening_setup(state)
        nemona_id = self._move_named_card_to_hand(state, 0, "Nemona")
        self._set_exact_hand(state, 0, [nemona_id])
        expected_draw = list(state.players[0].deck[:3])

        action = self._find_action_for_card_name(state, "play_supporter", "Nemona")
        apply_action(state, action)

        self.assertEqual(state.players[0].hand, expected_draw)
        self.assertEqual(card_definition(state, state.players[0].discard[-1]).name, "Nemona")
        self.assertTrue(state.players[0].supporter_played_this_turn)
        self.assertEqual([action["type"] for action in list_legal_actions(state)].count("play_supporter"), 0)

    def test_youngster_shuffles_other_hand_cards_into_deck_then_draws_five(self) -> None:
        state = create_game(seed=1, human_deck_id="ampharos-ex-battle-deck")
        self._finish_opening_setup(state)
        youngster_id = self._move_named_card_to_hand(state, 0, "Youngster")
        mareep_id = self._move_named_card_to_hand(state, 0, "Mareep")
        wattrel_id = self._move_named_card_to_hand(state, 0, "Wattrel")
        self._set_exact_hand(state, 0, [youngster_id, mareep_id, wattrel_id])
        expected_draw = list(state.players[0].deck[:5])

        action = self._find_action_for_card_name(state, "play_supporter", "Youngster")
        with patch.object(state.rng, "shuffle", lambda cards: None):
            apply_action(state, action)

        self.assertEqual(state.players[0].hand, expected_draw)
        self.assertEqual(card_definition(state, state.players[0].discard[-1]).name, "Youngster")
        self.assertIn(mareep_id, state.players[0].deck)
        self.assertIn(wattrel_id, state.players[0].deck)
        self.assertTrue(state.players[0].supporter_played_this_turn)

    def test_play_energy_attaches_to_active_and_uses_the_turn_attachment(self) -> None:
        state = create_game(seed=1, human_deck_id="ampharos-ex-battle-deck")
        self._finish_opening_setup(state)
        energy_id = self._move_named_card_to_hand(state, 0, "Basic Lightning Energy")
        self._set_exact_hand(state, 0, [energy_id])

        action = next(action for action in list_legal_actions(state) if action["type"] == "play_energy")
        apply_action(state, action)

        self.assertEqual(state.players[0].hand, [])
        self.assertEqual(state.players[0].active.attached_energy, [energy_id])
        self.assertTrue(state.players[0].energy_attached_this_turn)
        self.assertEqual([action["type"] for action in list_legal_actions(state)].count("play_energy"), 0)

    def test_play_energy_targets_active_and_bench_pokemon(self) -> None:
        state = create_game(seed=1, human_deck_id="ampharos-ex-battle-deck")
        self._finish_opening_setup(state)
        mareep_id = self._move_named_card_to_hand(state, 0, "Mareep")
        self._set_exact_hand(state, 0, [mareep_id])
        bench_action = next(action for action in list_legal_actions(state) if action["type"] == "bench_basic")
        apply_action(state, bench_action)

        energy_id = self._move_named_card_to_hand(state, 0, "Basic Lightning Energy")
        self._set_exact_hand(state, 0, [energy_id])

        energy_actions = [action for action in list_legal_actions(state) if action["type"] == "play_energy"]

        self.assertEqual(len(energy_actions), 2)
        self.assertEqual(
            {(action["target_zone"], action["target_bench_index"]) for action in energy_actions},
            {("active", None), ("bench", 0)},
        )

    def test_ai_can_choose_energy_attachment_over_ending_the_turn(self) -> None:
        state = create_game(seed=1, human_deck_id="ampharos-ex-battle-deck")
        self._finish_opening_setup(state)
        state.current_player = 1
        self._set_first_basic_as_active(state, 1)
        energy_id = self._move_named_card_to_hand(state, 1, "Basic Fighting Energy")
        self._set_exact_hand(state, 1, [energy_id])

        action = choose_action(state, 1)

        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(action["type"], "play_energy")
        self.assertEqual(action["hand_card_id"], energy_id)

    def _finish_opening_setup(self, state) -> None:
        if state.setup_phase != "choose_active":
            return

        active_action = next(
            action for action in list_legal_actions(state) if action["type"] == "play_basic_to_active"
        )
        apply_action(state, active_action)
        end_setup = next(action for action in list_legal_actions(state) if action["type"] == "end_setup")
        apply_action(state, end_setup)

    def _find_action_for_card_name(self, state, action_type: str, card_name: str):
        for action in list_legal_actions(state):
            if action["type"] != action_type:
                continue
            if card_definition(state, action["hand_card_id"]).name == card_name:
                return action
        self.fail(f"Could not find {action_type} action for {card_name}")

    def _move_named_card_to_hand(self, state, player_index: int, card_name: str) -> str:
        player = state.players[player_index]
        for zone_name in ("hand", "deck", "discard"):
            zone = getattr(player, zone_name)
            for instance_id in list(zone):
                if card_definition(state, instance_id).name != card_name:
                    continue
                if zone_name != "hand":
                    zone.remove(instance_id)
                    player.hand.append(instance_id)
                return instance_id
        self.fail(f"Could not find {card_name} for player {player_index}")

    def _set_exact_hand(self, state, player_index: int, ordered_instance_ids: list[str]) -> None:
        player = state.players[player_index]
        kept = set(ordered_instance_ids)
        extras = [instance_id for instance_id in player.hand if instance_id not in kept]
        player.hand = list(ordered_instance_ids)
        player.deck.extend(extras)

    def _set_first_basic_as_active(self, state, player_index: int) -> None:
        player = state.players[player_index]
        if player.active is not None:
            return

        for zone_name in ("hand", "deck", "discard"):
            zone = getattr(player, zone_name)
            for instance_id in list(zone):
                if not card_definition(state, instance_id).is_basic:
                    continue
                zone.remove(instance_id)
                player.active = PokemonInPlay(stack=[instance_id])
                return
        self.fail(f"Could not find a Basic Pokemon to make active for player {player_index}")


if __name__ == "__main__":
    unittest.main()
