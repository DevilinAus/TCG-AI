from __future__ import annotations

from copy import deepcopy
import unittest

from backend.tcg_ai.game_modes.standard.ml.neural_policy import (
    ACTION_VECTOR_SIZE,
    STATE_VECTOR_SIZE,
    encode_action_vector,
    encode_state_vector,
    infer_checkpoint_model_dimensions,
)


def _sample_belief_state() -> dict[str, object]:
    return {
        "turn_number": 4,
        "current_player": 0,
        "starting_player": 0,
        "winner": None,
        "setup_phase": None,
        "perspective_player_index": 0,
        "players": [
            {
                "deck_count": 42,
                "hand_count": 5,
                "discard_count": 6,
                "prize_count": 6,
                "known_prize_cards_unordered": [],
                "deck_inspected_this_game": False,
                "supporter_played_this_turn": False,
                "energy_attached_this_turn": False,
                "retreated_this_turn": False,
                "active": {
                    "hp": 120,
                    "remaining_hp": 120,
                    "retreat_cost": 2,
                    "attached_energy_count": 1,
                    "turns_until_ready": 1,
                    "attacks": [
                        {"cost": 2, "remaining_cost": 1, "damage": "90"},
                    ],
                    "card": {
                        "prize_card_value": 2,
                        "is_basic": True,
                        "stage": "basic",
                    },
                },
                "bench": [
                    {
                        "hp": 140,
                        "remaining_hp": 140,
                        "retreat_cost": 1,
                        "attached_energy_count": 0,
                        "turns_until_ready": 2,
                        "attacks": [
                            {"cost": 3, "remaining_cost": 3, "damage": "120"},
                        ],
                        "card": {
                            "prize_card_value": 2,
                            "is_basic": True,
                            "stage": "basic",
                        },
                    },
                    {
                        "hp": 110,
                        "remaining_hp": 90,
                        "retreat_cost": 1,
                        "attached_energy_count": 2,
                        "turns_until_ready": 0,
                        "attacks": [
                            {"cost": 2, "remaining_cost": 0, "damage": "50"},
                        ],
                        "card": {
                            "prize_card_value": 1,
                            "is_basic": False,
                            "stage": "stage1",
                        },
                    },
                ],
            },
            {
                "deck_count": 40,
                "hand_count": 4,
                "discard_count": 3,
                "prize_count": 6,
                "supporter_played_this_turn": False,
                "energy_attached_this_turn": False,
                "retreated_this_turn": False,
                "active": {
                    "hp": 110,
                    "remaining_hp": 110,
                    "retreat_cost": 1,
                    "attached_energy_count": 1,
                    "turns_until_ready": 0,
                    "attacks": [
                        {"cost": 1, "remaining_cost": 0, "damage": "50"},
                    ],
                    "card": {
                        "prize_card_value": 1,
                        "is_basic": True,
                        "stage": "basic",
                    },
                },
                "bench": [],
            },
        ],
        "derived_features": {
            "player_active_likely_knockout_next_turn": False,
            "opponent_active_likely_knockout_next_turn": True,
            "player_energy_at_risk_on_active": 1,
            "opponent_energy_at_risk_on_active": 1,
            "player_board_investment": 5,
            "opponent_board_investment": 2,
        },
    }


class StandardNeuralPolicyEncodingTests(unittest.TestCase):
    def test_state_vector_is_fixed_width_and_slot_sensitive(self) -> None:
        belief_state = _sample_belief_state()
        swapped = deepcopy(belief_state)
        swapped["players"][0]["bench"] = list(reversed(swapped["players"][0]["bench"]))

        encoded = encode_state_vector(belief_state)
        swapped_encoded = encode_state_vector(swapped)

        self.assertEqual(len(encoded), STATE_VECTOR_SIZE)
        self.assertEqual(len(swapped_encoded), STATE_VECTOR_SIZE)
        self.assertNotEqual(encoded, swapped_encoded)

    def test_state_vector_changes_when_retreat_facts_change(self) -> None:
        belief_state = _sample_belief_state()
        changed = deepcopy(belief_state)
        changed["players"][0]["retreated_this_turn"] = True
        changed["players"][0]["active"]["retreat_cost"] = 3

        self.assertNotEqual(encode_state_vector(belief_state), encode_state_vector(changed))

    def test_action_vector_uses_board_context_for_attach_targets(self) -> None:
        belief_state = _sample_belief_state()
        active_attach = {
            "type": "play_energy",
            "source_card": {
                "kind": "energy",
                "is_basic": False,
                "is_basic_energy": True,
                "prize_card_value": 0,
                "stage": None,
            },
            "target": {"player_index": 0, "zone": "active", "bench_index": None},
            "resource_costs": {
                "hand_card_count": 1,
                "discard_from_hand_count": 0,
                "discard_attached_energy_count": 0,
                "recover_from_discard_count": 0,
                "search_selection_count": 0,
                "attack_energy_cost": 0,
                "retreat_energy_cost": 0,
            },
            "expected_state_delta": {
                "cards_drawn_known": 0,
                "hand_count_delta_known": -1,
                "bench_count_delta": 0,
                "discard_count_delta_known": 0,
                "active_changes": False,
                "turn_ends": False,
            },
            "effect_tags": ["attach_energy"],
            "consumes_supporter_for_turn": False,
            "consumes_attachment_for_turn": True,
            "consumes_retreat_for_turn": False,
            "search_selection": [],
        }
        bench_attach = deepcopy(active_attach)
        bench_attach["target"] = {"player_index": 0, "zone": "bench", "bench_index": 0}

        active_vector = encode_action_vector(active_attach, belief_state=belief_state)
        bench_vector = encode_action_vector(bench_attach, belief_state=belief_state)

        self.assertEqual(len(active_vector), ACTION_VECTOR_SIZE)
        self.assertEqual(len(bench_vector), ACTION_VECTOR_SIZE)
        self.assertNotEqual(active_vector, bench_vector)

    def test_action_vector_includes_intent_and_quality_analysis_features(self) -> None:
        belief_state = _sample_belief_state()
        base_action = {
            "type": "play_item",
            "source_card": {
                "kind": "trainer",
                "is_basic": False,
                "is_basic_energy": False,
                "prize_card_value": 0,
                "stage": None,
            },
            "target": {"player_index": 0, "zone": None, "bench_index": None},
            "resource_costs": {
                "hand_card_count": 1,
                "discard_from_hand_count": 0,
                "discard_attached_energy_count": 0,
                "recover_from_discard_count": 0,
                "search_selection_count": 0,
                "attack_energy_cost": 0,
                "retreat_energy_cost": 0,
            },
            "expected_state_delta": {
                "cards_drawn_known": 0,
                "hand_count_delta_known": -1,
                "bench_count_delta": 0,
                "discard_count_delta_known": 1,
                "active_changes": False,
                "turn_ends": False,
            },
            "effect_tags": ["recover_from_discard"],
            "consumes_supporter_for_turn": False,
            "consumes_attachment_for_turn": False,
            "consumes_retreat_for_turn": False,
            "tactical_outcomes": {
                "wins_game_now": False,
                "takes_prize_now": False,
                "prizes_taken_now": 0,
                "creates_same_turn_prize_line": False,
                "creates_live_attack_this_turn": False,
                "changes_active": False,
                "saves_board_investment": False,
                "reduces_active_ko_risk": False,
            },
            "resolution_facts": {
                "optional_choice_empty": True,
                "productive_variant_exists": False,
                "net_known_hand_delta": -1,
                "net_known_bench_delta": 0,
                "net_known_discard_delta": 1,
            },
            "intent_tags": [],
            "quality_flags": [],
        }
        analyzed_action = deepcopy(base_action)
        analyzed_action["tactical_outcomes"] = {
            **base_action["tactical_outcomes"],
            "takes_prize_now": True,
            "prizes_taken_now": 1,
        }
        analyzed_action["resolution_facts"] = {
            **base_action["resolution_facts"],
            "productive_variant_exists": True,
        }
        analyzed_action["intent_tags"] = ["take_prize", "recover_resource"]
        analyzed_action["quality_flags"] = ["dominated_optional_play"]

        base_vector = encode_action_vector(base_action, belief_state=belief_state)
        analyzed_vector = encode_action_vector(analyzed_action, belief_state=belief_state)

        self.assertEqual(len(base_vector), ACTION_VECTOR_SIZE)
        self.assertEqual(len(analyzed_vector), ACTION_VECTOR_SIZE)
        self.assertNotEqual(base_vector, analyzed_vector)

    def test_checkpoint_dimension_inference_prefers_explicit_model_config(self) -> None:
        checkpoint = {
            "model_config": {"state_dim": 99, "action_dim": 44},
            "state_dict": {},
        }

        self.assertEqual(infer_checkpoint_model_dimensions(checkpoint), (99, 44))

    def test_checkpoint_dimension_inference_falls_back_to_state_dict_shapes(self) -> None:
        class FakeWeight:
            def __init__(self, rows: int, cols: int) -> None:
                self.shape = (rows, cols)

        checkpoint = {
            "state_dict": {
                "state_encoder.0.weight": FakeWeight(128, 61),
                "action_encoder.0.weight": FakeWeight(128, 37),
            }
        }

        self.assertEqual(infer_checkpoint_model_dimensions(checkpoint), (61, 37))


if __name__ == "__main__":
    unittest.main()
