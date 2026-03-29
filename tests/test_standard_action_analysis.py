from __future__ import annotations

import unittest

from backend.tcg_ai.game_modes.standard.action_analysis import analyze_legal_actions
from backend.tcg_ai.game_modes.standard.engine import action_id_for, card_definition, list_legal_actions
from backend.tcg_ai.game_modes.standard.ml.tactical_suite import (
    _build_attack_for_immediate_win_state,
    _build_attack_over_redundant_retreat_state,
    _build_energy_retrieval_non_empty_state,
    _build_pokegear_empty_hand_thinning_state,
)


class StandardActionAnalysisTests(unittest.TestCase):
    def test_immediate_winning_attack_gets_finish_game_reason(self) -> None:
        state, acting_player_index = _build_attack_for_immediate_win_state()
        legal_actions = list_legal_actions(state, player_index=acting_player_index)

        analysis = analyze_legal_actions(
            state,
            acting_player_index=acting_player_index,
            legal_actions=legal_actions,
        )
        attack = next(action for action in legal_actions if action.get("type") == "attack")
        attack_analysis = analysis[action_id_for(attack)]

        self.assertTrue(attack_analysis["tactical_outcomes"]["wins_game_now"])
        self.assertIn("finish_game", attack_analysis["intent_tags"])
        self.assertEqual(attack_analysis["reason_summary"], "acted to close out the game.")

    def test_redundant_retreat_is_marked_as_missing_immediate_win(self) -> None:
        state, acting_player_index = _build_attack_over_redundant_retreat_state()
        legal_actions = list_legal_actions(state, player_index=acting_player_index)

        analysis = analyze_legal_actions(
            state,
            acting_player_index=acting_player_index,
            legal_actions=legal_actions,
        )
        retreat = next(action for action in legal_actions if action.get("type") == "retreat")
        retreat_analysis = analysis[action_id_for(retreat)]

        self.assertIn("misses_immediate_win", retreat_analysis["quality_flags"])
        self.assertIn("pivot_ready_attacker", retreat_analysis["intent_tags"])
        self.assertEqual(
            retreat_analysis["reason_summary"],
            "set up a same-turn prize line.",
        )

    def test_empty_energy_retrieval_is_dominated_when_recovery_exists(self) -> None:
        state, acting_player_index = _build_energy_retrieval_non_empty_state()
        legal_actions = list_legal_actions(state, player_index=acting_player_index)

        analysis = analyze_legal_actions(
            state,
            acting_player_index=acting_player_index,
            legal_actions=legal_actions,
        )
        empty_retrieval = next(
            action
            for action in legal_actions
            if action.get("type") == "play_item"
            and card_definition(state, action["hand_card_id"]).name == "Energy Retrieval"
            and action.get("recover_from_discard_ids", []) == []
        )
        empty_analysis = analysis[action_id_for(empty_retrieval)]

        self.assertTrue(empty_analysis["resolution_facts"]["optional_choice_empty"])
        self.assertTrue(empty_analysis["resolution_facts"]["productive_variant_exists"])
        self.assertIn("dominated_optional_play", empty_analysis["quality_flags"])
        self.assertNotIn("hand_thinning", empty_analysis["intent_tags"])

    def test_empty_pokegear_can_be_valid_hand_thinning(self) -> None:
        state, acting_player_index = _build_pokegear_empty_hand_thinning_state()
        legal_actions = list_legal_actions(state, player_index=acting_player_index)

        analysis = analyze_legal_actions(
            state,
            acting_player_index=acting_player_index,
            legal_actions=legal_actions,
        )
        pokegear = next(
            action
            for action in legal_actions
            if action.get("type") == "play_item"
            and card_definition(state, action["hand_card_id"]).name == "Pokégear 3.0"
        )
        pokegear_analysis = analysis[action_id_for(pokegear)]

        self.assertTrue(pokegear_analysis["resolution_facts"]["optional_choice_empty"])
        self.assertFalse(pokegear_analysis["resolution_facts"]["productive_variant_exists"])
        self.assertIn("hand_thinning", pokegear_analysis["intent_tags"])
        self.assertEqual(pokegear_analysis["quality_flags"], [])
        self.assertEqual(
            pokegear_analysis["reason_summary"],
            "thinned a low-value card out of hand.",
        )


if __name__ == "__main__":
    unittest.main()
