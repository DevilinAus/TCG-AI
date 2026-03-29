from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from backend.tcg_ai.server import (
    AI_ACTION_DELAY_MAX_MS,
    AI_ACTION_DELAY_MIN_MS,
    ApiError,
    TcgApplication,
)
from backend.tcg_ai.game_modes.standard.cards import load_deck_cards
from backend.tcg_ai.game_modes.standard.policy import StandardPolicyConfig, StandardRemoteDecisionError
from backend.tcg_ai.learning import RewardLearner
from backend.tcg_ai.models import PokemonInPlay


class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.state_path = Path(self.temp_dir.name) / "trainer_progress.txt"
        self.policy_state_path = Path(self.temp_dir.name) / "standard_policy_progress.json"
        self.app = TcgApplication(
            trainer_state_path=self.state_path,
            standard_policy_state_path=self.policy_state_path,
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_new_game_returns_a_session_id_and_display_ready_state(self) -> None:
        state = self.app.new_game({"human_first": True})

        self.assertIsInstance(state["session_id"], str)
        self.assertEqual(state["game_mode"], "my_first_battle")
        self.assertEqual(state["shared_assets"]["face_down_card_image_url"], "/assets/cards/shared/card-back.png")
        self.assertEqual(state["current_player"], 0)
        self.assertEqual(state["players"][0]["active"]["card_id"], "charmander")
        self.assertEqual(state["players"][0]["deck_pile"]["count"], state["players"][0]["deck_count"])
        self.assertEqual(state["players"][0]["prize_pile"]["count"], 3)
        self.assertEqual(
            state["players"][0]["prize_pile"]["image_url"],
            "/assets/coins/my-first-battle/charmander-prize-coin.png",
        )
        self.assertEqual(
            state["players"][1]["prize_pile"]["image_url"],
            "/assets/coins/my-first-battle/squirtle-prize-coin.png",
        )
        self.assertEqual(state["players"][0]["energy_count"], 0)
        self.assertTrue(state["players"][0]["hand"][0]["image_url"].startswith("/assets/cards/"))
        self.assertEqual(state["ai_learning"]["games_played"], 0)
        self.assertEqual(state["ai_trainer"]["id"], "brock")
        self.assertEqual(state["ai_trainer"]["level"], 1)
        self.assertEqual(len(state["available_trainers"]), 8)

    def test_play_energy_targets_the_shared_energy_space(self) -> None:
        state = self.app.new_game({"human_first": True, "seed": 1})

        play_energy_action = self._find_action(state, "play_energy")

        self.assertEqual(play_energy_action["label"], "Play Fire Energy")
        self.assertEqual(play_energy_action["target"]["name"], "Shared Energy")

    def test_lobby_exposes_available_trainers_and_decks_before_a_game_starts(self) -> None:
        lobby = self.app.lobby()

        self.assertEqual(lobby["game_mode"], "my_first_battle")
        self.assertEqual(lobby["standard_ai_mode"], "local")
        self.assertEqual(lobby["human_deck_id"], "charmander")
        self.assertEqual(lobby["ai_deck_id"], "squirtle")
        self.assertEqual(lobby["ai_trainer"]["id"], "brock")
        self.assertEqual(
            [mode["id"] for mode in lobby["available_game_modes"]],
            ["my_first_battle", "standard"],
        )
        standard_mode = next(mode for mode in lobby["available_game_modes"] if mode["id"] == "standard")
        self.assertTrue(standard_mode["available"])
        self.assertEqual(
            [deck["id"] for deck in lobby["available_decks"]],
            ["bulbasaur", "charmander", "squirtle", "pikachu"],
        )
        selected_deck = next(deck for deck in lobby["available_decks"] if deck["selected"])
        self.assertEqual(selected_deck["id"], "charmander")
        self.assertEqual(
            selected_deck["prize_coin_image_url"],
            "/assets/coins/my-first-battle/charmander-prize-coin.png",
        )

    def test_lobby_can_preview_standard_mode_decks_while_standard_is_unavailable(self) -> None:
        lobby = self.app.lobby("standard")

        self.assertEqual(lobby["game_mode"], "standard")
        self.assertEqual(lobby["standard_ai_mode"], "local")
        self.assertEqual(lobby["human_deck_id"], "ampharos-ex-battle-deck")
        self.assertEqual(lobby["ai_deck_id"], "lucario-ex-battle-deck")
        self.assertEqual(len(lobby["available_decks"]), 12)
        self.assertEqual(
            [mode["id"] for mode in lobby["available_game_modes"]],
            ["my_first_battle", "standard"],
        )
        standard_mode = next(mode for mode in lobby["available_game_modes"] if mode["id"] == "standard")
        self.assertTrue(standard_mode["available"])
        self.assertTrue(standard_mode["selected"])
        selected_deck = next(deck for deck in lobby["available_decks"] if deck["selected"])
        self.assertEqual(selected_deck["id"], "ampharos-ex-battle-deck")
        self.assertEqual(selected_deck["paired_deck_id"], "lucario-ex-battle-deck")
        self.assertEqual(selected_deck["element"], "lightning")
        self.assertEqual(selected_deck["product_line"], "ex-battle")

    def test_new_game_can_start_standard_and_draw_opening_hands(self) -> None:
        state = self.app.new_game(
            {
                "game_mode": "standard",
                "human_first": True,
                "human_deck_id": "ampharos-ex-battle-deck",
                "seed": 1,
            }
        )

        self.assertEqual(state["game_mode"], "standard")
        self.assertEqual(state["standard_ai_mode"], "local")
        self.assertEqual(state["current_player"], 0)
        self.assertEqual(state["setup_phase"], "choose_active")
        self.assertEqual(state["players"][0]["deck_name"], "Ampharos ex Battle Deck")
        self.assertEqual(state["players"][1]["deck_name"], "Lucario ex Battle Deck")
        self.assertEqual(len(state["players"][0]["hand"]), 7)
        self.assertEqual(state["players"][0]["hand_count"], 7)
        self.assertEqual(state["players"][1]["hand_count"], 6)
        self.assertEqual(state["players"][0]["deck_count"], 47)
        self.assertEqual(state["players"][1]["deck_count"], 47)
        self.assertIsNone(state["players"][0]["active"])
        self.assertTrue(state["players"][1]["active"]["face_down"])
        self.assertEqual(
            state["players"][1]["active"]["image_url"],
            "/assets/cards/shared/card-back.png",
        )
        self.assertEqual(state["players"][0]["bench"], [])
        self.assertEqual(
            [action["type"] for action in state["legal_actions"]],
            ["play_basic_to_active", "play_basic_to_active", "play_basic_to_active"],
        )
        self.assertEqual(state["legal_actions"][0]["target"]["zone"], "active")
        self.assertEqual(state["legal_actions"][0]["source"]["zone"], "hand")
        self.assertTrue(
            all(
                card["image_url"].startswith("/assets/cards/standard/shared/")
                for card in state["players"][0]["hand"]
            )
        )
        self.assertEqual(state["ai_learning"], {})
        self.assertEqual(state["ai_decision_debug"]["provider_type"], "fallback")
        self.assertEqual(len(state["ai_decision_debug"]["pending_traces"]), 1)
        self.assertIn("opening hand", state["log"][1]["text"].lower())

    def test_standard_new_game_records_an_ai_opening_trace_without_persisting_it(self) -> None:
        state = self.app.new_game(
            {
                "game_mode": "standard",
                "human_first": True,
                "human_deck_id": "ampharos-ex-battle-deck",
                "seed": 1,
            }
        )

        self.assertEqual(state["players"][1]["active"]["name"], "Face-down Active Pokemon")
        self.assertTrue(state["players"][1]["active"]["face_down"])
        self.assertEqual(state["standard_ai_mode"], "local")
        self.assertEqual(state["ai_decision_debug"]["last_decision"]["decision_type"], "opening_active")
        self.assertEqual(state["ai_decision_debug"]["last_decision"]["source"], "local")
        self.assertEqual(state["ai_decision_debug"]["pending_traces"][0]["chosen_card_id"], "sv1-124")
        self.assertFalse(self.policy_state_path.exists())

    def test_standard_supporters_expose_machine_readable_effect_specs_in_hand_and_actions(self) -> None:
        state = self.app.new_game(
            {
                "game_mode": "standard",
                "human_first": True,
                "human_deck_id": "ampharos-ex-battle-deck",
                "seed": 1,
            }
        )

        session = self.app.sessions.get(state["session_id"])
        from backend.tcg_ai.game_modes.standard.engine import apply_action, card_definition, list_legal_actions

        active_action = next(
            action for action in list_legal_actions(session.state) if action["type"] == "play_basic_to_active"
        )
        apply_action(session.state, active_action)
        end_setup = next(action for action in list_legal_actions(session.state) if action["type"] == "end_setup")
        apply_action(session.state, end_setup)
        session.state.turn_number = 2

        player = session.state.players[0]
        nemona_id = self._move_standard_named_card_to_hand(session.state, 0, "Nemona")
        player.hand = [nemona_id]

        snapshot = self.app.get_game(state["session_id"])
        nemona = snapshot["players"][0]["hand"][0]
        supporter_action = next(
            action for action in snapshot["legal_actions"] if action["type"] == "play_supporter"
        )

        self.assertEqual(card_definition(session.state, nemona_id).name, "Nemona")
        self.assertEqual(nemona["card_tags"], ["supporter"])
        self.assertEqual(nemona["effect_specs"][0]["effect_type"], "draw")
        self.assertEqual(nemona["effect_specs"][0]["count"], 3)
        self.assertEqual(nemona["effect_specs"][0]["destination_zone"], "hand")
        self.assertTrue(nemona["effect_specs"][0]["changes_hidden_information"])
        self.assertEqual(supporter_action["source"]["card_id"], "sv1-180")
        self.assertEqual(supporter_action["card_tags"], ["supporter"])
        self.assertEqual(supporter_action["effect_specs"][0]["effect_type"], "draw")
        self.assertTrue(supporter_action["changes_hidden_information"])

    def test_standard_jacq_exposes_evolution_search_metadata_and_adds_selected_cards_to_hand(self) -> None:
        state = self.app.new_game(
            {
                "game_mode": "standard",
                "human_first": True,
                "human_deck_id": "ampharos-ex-battle-deck",
                "seed": 1,
            }
        )

        session = self.app.sessions.get(state["session_id"])
        from backend.tcg_ai.game_modes.standard.engine import apply_action, list_legal_actions

        active_action = next(
            action for action in list_legal_actions(session.state) if action["type"] == "play_basic_to_active"
        )
        apply_action(session.state, active_action)
        end_setup = next(action for action in list_legal_actions(session.state) if action["type"] == "end_setup")
        apply_action(session.state, end_setup)
        session.state.turn_number = 2

        player = session.state.players[0]
        jacq_id = self._move_standard_named_card_to_hand(session.state, 0, "Jacq")
        player.hand = [jacq_id]

        ampharos_id = self._find_standard_instance_id(session.state, 0, "Ampharos ex")
        staraptor_id = self._find_standard_instance_id(session.state, 0, "Staraptor")
        mareep_id = self._find_standard_instance_id(session.state, 0, "Mareep")
        self._reorder_standard_deck(session.state, 0, [mareep_id, ampharos_id, staraptor_id])

        snapshot = self.app.get_game(state["session_id"])
        jacq = snapshot["players"][0]["hand"][0]
        jacq_actions = [
            action
            for action in snapshot["legal_actions"]
            if action["type"] == "play_supporter" and action["source"]["instance_id"] == jacq_id
        ]

        self.assertEqual(jacq["card_tags"], ["supporter"])
        self.assertEqual(jacq["effect_specs"][0]["effect_type"], "search_deck")
        self.assertEqual(jacq["effect_specs"][0]["destination_zone"], "hand")
        self.assertEqual(jacq["effect_specs"][0]["search_filters"], ["evolution_pokemon"])
        self.assertEqual(jacq["effect_specs"][0]["choose_count"], 2)
        self.assertTrue(jacq["effect_specs"][0]["optional"])
        self.assertIn([], [action["action"]["search_deck_ids"] for action in jacq_actions])
        self.assertIn([ampharos_id], [action["action"]["search_deck_ids"] for action in jacq_actions])
        self.assertIn([staraptor_id], [action["action"]["search_deck_ids"] for action in jacq_actions])
        self.assertIn(
            [ampharos_id, staraptor_id],
            [action["action"]["search_deck_ids"] for action in jacq_actions],
        )
        self.assertNotIn([mareep_id], [action["action"]["search_deck_ids"] for action in jacq_actions])

        chosen_action = next(
            action
            for action in jacq_actions
            if action["action"]["search_deck_ids"] == [ampharos_id, staraptor_id]
        )
        updated = self.app.human_action(
            {
                "session_id": state["session_id"],
                "action": chosen_action["action"],
            }
        )

        self.assertIn("Ampharos ex", [card["name"] for card in updated["players"][0]["hand"]])
        self.assertIn("Staraptor", [card["name"] for card in updated["players"][0]["hand"]])
        self.assertEqual(updated["players"][0]["discard_top"]["name"], "Jacq")

    def test_standard_ml_status_reports_not_configured_when_remote_disabled(self) -> None:
        status = self.app.standard_ml_status()

        self.assertFalse(status["configured"])
        self.assertFalse(status["ready"])
        self.assertFalse(status["model_loaded"])
        self.assertIn("not configured", status["error"])

    def test_standard_ml_status_checks_remote_worker_readiness(self) -> None:
        app = TcgApplication(
            trainer_state_path=self.state_path,
            standard_policy_state_path=self.policy_state_path,
            standard_policy_config=StandardPolicyConfig(
                remote_enabled=True,
                remote_url="http://127.0.0.1:8100/api/standard-ml/decision",
                remote_timeout_ms=1_500,
                remote_api_token="secret-token",
            ),
        )
        request_details: dict[str, object] = {}

        def fake_urlopen(request, timeout):
            request_details["url"] = request.full_url
            request_details["timeout"] = timeout
            request_details["token"] = request.get_header("X-standard-ml-token")
            return _FakeResponse(
                {
                    "ready": True,
                    "backend": "torch",
                    "model_loaded": True,
                    "checkpoint_path": "/models/champion.pt",
                }
            )

        with patch("backend.tcg_ai.server.urllib_request.urlopen", side_effect=fake_urlopen):
            status = app.standard_ml_status()

        self.assertTrue(status["configured"])
        self.assertTrue(status["ready"])
        self.assertTrue(status["model_loaded"])
        self.assertEqual(status["backend"], "torch")
        self.assertEqual(status["checkpoint_path"], "/models/champion.pt")
        self.assertEqual(status["ready_url"], "http://127.0.0.1:8100/readyz")
        self.assertEqual(request_details["url"], "http://127.0.0.1:8100/readyz")
        self.assertEqual(request_details["timeout"], 1.5)
        self.assertEqual(request_details["token"], "secret-token")

    def test_standard_remote_game_requires_loaded_remote_model(self) -> None:
        app = TcgApplication(
            trainer_state_path=self.state_path,
            standard_policy_state_path=self.policy_state_path,
            standard_policy_config=StandardPolicyConfig(
                remote_enabled=True,
                remote_url="http://127.0.0.1:8100/api/standard-ml/decision",
            ),
        )

        with patch(
            "backend.tcg_ai.server.urllib_request.urlopen",
            return_value=_FakeResponse(
                {
                    "ready": True,
                    "backend": "heuristic",
                    "model_loaded": False,
                }
            ),
        ):
            with self.assertRaises(ApiError) as exc_info:
                app.new_game(
                    {
                        "game_mode": "standard",
                        "human_first": True,
                        "human_deck_id": "ampharos-ex-battle-deck",
                        "standard_ai_mode": "remote",
                    }
                )

        self.assertEqual(exc_info.exception.code, "standard_ml_unavailable")
        self.assertIn("no model checkpoint is loaded", exc_info.exception.message.lower())

    def test_standard_remote_game_uses_remote_mode_when_worker_is_ready(self) -> None:
        app = TcgApplication(
            trainer_state_path=self.state_path,
            standard_policy_state_path=self.policy_state_path,
            standard_policy_config=StandardPolicyConfig(
                remote_enabled=True,
                remote_url="http://127.0.0.1:8100/api/standard-ml/decision",
                remote_batch_eval_url="http://127.0.0.1:8100/api/standard-ml/batch-eval",
            ),
        )

        with patch(
            "backend.tcg_ai.server.urllib_request.urlopen",
            return_value=_FakeResponse(
                {
                    "ready": True,
                    "backend": "torch",
                    "model_loaded": True,
                    "checkpoint_path": "/models/champion.pt",
                }
            ),
        ):
            state = app.new_game(
                {
                    "game_mode": "standard",
                    "human_first": True,
                    "human_deck_id": "ampharos-ex-battle-deck",
                    "seed": 1,
                    "standard_ai_mode": "remote",
                }
            )

        session = app.sessions.get(state["session_id"])
        self.assertEqual(state["standard_ai_mode"], "remote")
        self.assertEqual(session.standard_ai_mode, "remote")
        self.assertTrue(session.standard_policy_config.remote_enabled)
        self.assertEqual(
            session.standard_policy_config.remote_batch_eval_url,
            "http://127.0.0.1:8100/api/standard-ml/batch-eval",
        )

    def test_standard_local_game_disables_remote_config_for_the_session(self) -> None:
        app = TcgApplication(
            trainer_state_path=self.state_path,
            standard_policy_state_path=self.policy_state_path,
            standard_policy_config=StandardPolicyConfig(
                remote_enabled=True,
                remote_url="http://127.0.0.1:8100/api/standard-ml/decision",
            ),
        )

        state = app.new_game(
            {
                "game_mode": "standard",
                "human_first": True,
                "human_deck_id": "ampharos-ex-battle-deck",
                "seed": 1,
                "standard_ai_mode": "local",
            }
        )

        session = app.sessions.get(state["session_id"])
        self.assertEqual(state["standard_ai_mode"], "local")
        self.assertEqual(session.standard_ai_mode, "local")
        self.assertFalse(session.standard_policy_config.remote_enabled)
        self.assertIsNone(session.standard_policy_config.remote_url)

    def test_standard_call_for_family_exposes_bench_search_metadata_and_benches_two_basics(self) -> None:
        state = self.app.new_game(
            {
                "game_mode": "standard",
                "human_first": True,
                "human_deck_id": "lucario-ex-battle-deck",
                "seed": 1,
            }
        )

        session = self.app.sessions.get(state["session_id"])
        session.state.setup_phase = None
        session.state.current_player = 0
        session.state.turn_number = 2
        session.state.players[0].turns_taken = 2
        session.state.players[1].turns_taken = 2
        self._set_standard_named_active_pokemon(session.state, 0, "Squawkabilly")
        self._set_standard_named_active_pokemon(session.state, 1, "Mareep")
        session.state.players[0].active.attached_energy = [
            self._find_standard_instance_id(session.state, 0, "Basic Fighting Energy"),
        ]

        riolu_id = self._find_standard_instance_id(session.state, 0, "Riolu")
        lechonk_id = self._find_standard_instance_id(session.state, 0, "Lechonk")
        oinkologne_id = self._find_standard_instance_id(session.state, 0, "Oinkologne")
        self._reorder_standard_deck(session.state, 0, [riolu_id, oinkologne_id, lechonk_id])

        snapshot = self.app.get_game(state["session_id"])
        call_for_family = snapshot["players"][0]["active"]["attacks"][0]
        call_for_family_actions = [
            action
            for action in snapshot["legal_actions"]
            if action["type"] == "attack"
            and action["source"]["instance_id"] == snapshot["players"][0]["active"]["instance_id"]
            and action["action"]["attack_index"] == 0
            and isinstance(action["action"].get("search_deck_ids"), list)
        ]

        self.assertEqual(call_for_family["name"], "Call for Family")
        self.assertEqual(call_for_family["effect_specs"][0]["effect_type"], "search_deck")
        self.assertEqual(call_for_family["effect_specs"][0]["destination_zone"], "bench")
        self.assertEqual(call_for_family["effect_specs"][0]["search_filters"], ["basic_pokemon"])
        self.assertEqual(call_for_family["effect_specs"][0]["choose_count"], 2)
        self.assertTrue(call_for_family["effect_specs"][0]["optional"])
        self.assertIn([], [action["action"]["search_deck_ids"] for action in call_for_family_actions])
        self.assertIn([riolu_id], [action["action"]["search_deck_ids"] for action in call_for_family_actions])
        self.assertIn([lechonk_id], [action["action"]["search_deck_ids"] for action in call_for_family_actions])
        self.assertIn(
            [riolu_id, lechonk_id],
            [action["action"]["search_deck_ids"] for action in call_for_family_actions],
        )
        self.assertNotIn([oinkologne_id], [action["action"]["search_deck_ids"] for action in call_for_family_actions])

        selected_action = next(
            action
            for action in call_for_family_actions
            if action["action"]["search_deck_ids"] == [riolu_id, lechonk_id]
        )

        updated = self.app.human_action(
            {
                "session_id": state["session_id"],
                "action": selected_action["action"],
            }
        )

        bench_names = [card["name"] for card in updated["players"][0]["bench"]]
        bench_ids = [card["instance_id"] for card in updated["players"][0]["bench"]]
        self.assertIn("Riolu", bench_names)
        self.assertIn("Lechonk", bench_names)
        self.assertNotIn("Oinkologne", bench_names)
        self.assertIn(riolu_id, bench_ids)
        self.assertIn(lechonk_id, bench_ids)
        self.assertTrue(
            any("put 2 cards onto the bench" in entry["text"].lower() for entry in updated["log"]),
        )

    def test_standard_energy_retrieval_exposes_discard_cards_and_recovery_choices(self) -> None:
        state = self.app.new_game(
            {
                "game_mode": "standard",
                "human_first": True,
                "human_deck_id": "ampharos-ex-battle-deck",
                "seed": 1,
            }
        )
        session = self.app.sessions.get(state["session_id"])
        from backend.tcg_ai.game_modes.standard.engine import apply_action, list_legal_actions

        active_action = next(
            action for action in list_legal_actions(session.state) if action["type"] == "play_basic_to_active"
        )
        apply_action(session.state, active_action)
        end_setup = next(action for action in list_legal_actions(session.state) if action["type"] == "end_setup")
        apply_action(session.state, end_setup)

        energy_retrieval_id = self._move_standard_named_card_to_hand(session.state, 0, "Energy Retrieval")
        lightning_a_id = self._take_standard_named_card(session.state, 0, "Basic Lightning Energy")
        lightning_b_id = self._take_standard_named_card(session.state, 0, "Basic Lightning Energy")
        potion_id = self._take_standard_named_card(session.state, 0, "Potion")
        session.state.players[0].discard.extend([lightning_a_id, lightning_b_id, potion_id])
        self._set_standard_exact_hand(session.state, 0, [energy_retrieval_id])

        snapshot = self.app.get_game(state["session_id"])
        energy_retrieval = next(
            card for card in snapshot["players"][0]["hand"] if card["instance_id"] == energy_retrieval_id
        )
        energy_retrieval_actions = [
            action
            for action in snapshot["legal_actions"]
            if action["type"] == "play_item" and action["source"]["instance_id"] == energy_retrieval_id
        ]
        recover_choices = [action["action"]["recover_from_discard_ids"] for action in energy_retrieval_actions]

        self.assertEqual(energy_retrieval["effect_specs"][0]["effect_type"], "recover_from_discard")
        self.assertEqual(energy_retrieval["effect_specs"][0]["source_zone"], "discard")
        self.assertEqual(energy_retrieval["effect_specs"][0]["destination_zone"], "hand")
        self.assertEqual(energy_retrieval["effect_specs"][0]["search_filters"], ["basic_energy"])
        self.assertEqual(energy_retrieval["effect_specs"][0]["choose_count"], 2)
        self.assertTrue(energy_retrieval["effect_specs"][0]["optional"])
        self.assertCountEqual(
            [card["name"] for card in snapshot["players"][0]["discard_cards"]],
            ["Basic Lightning Energy", "Basic Lightning Energy", "Potion"],
        )
        self.assertIn([], recover_choices)
        self.assertIn([lightning_a_id], recover_choices)
        self.assertIn([lightning_b_id], recover_choices)
        self.assertIn([lightning_a_id, lightning_b_id], recover_choices)
        self.assertNotIn([potion_id], recover_choices)

    def test_standard_acu_punch_ture_exposes_attack_choice_metadata(self) -> None:
        state = self.app.new_game(
            {
                "game_mode": "standard",
                "human_first": True,
                "human_deck_id": "lucario-ex-battle-deck",
                "seed": 1,
            }
        )
        session = self.app.sessions.get(state["session_id"])
        session.state.setup_phase = None
        session.state.current_player = 0
        session.state.turn_number = 2
        session.state.players[0].turns_taken = 2
        session.state.players[1].turns_taken = 2
        self._set_standard_named_active_pokemon(session.state, 0, "Medicham")
        self._set_standard_named_active_pokemon(session.state, 1, "Ampharos ex")
        session.state.players[0].active.attached_energy = [
            self._take_standard_named_card(session.state, 0, "Basic Fighting Energy"),
        ]

        snapshot = self.app.get_game(state["session_id"])
        medicham = snapshot["players"][0]["active"]
        acu_punch_ture = medicham["attacks"][0]
        acu_punch_ture_actions = [
            action
            for action in snapshot["legal_actions"]
            if action["type"] == "attack"
            and action["source"]["instance_id"] == medicham["instance_id"]
            and action["action"]["attack_index"] == 0
            and isinstance(action["action"].get("blocked_attack_index"), int)
        ]

        self.assertEqual(acu_punch_ture["name"], "Acu-Punch-Ture")
        self.assertEqual(acu_punch_ture["effect_specs"][0]["effect_type"], "block_selected_opponent_attack")
        self.assertEqual(
            {action["action"]["blocked_attack_index"] for action in acu_punch_ture_actions},
            {0, 1},
        )
        self.assertEqual(
            {action["label"] for action in acu_punch_ture_actions},
            {
                "Use Acu-Punch-Ture and block Electro Ball",
                "Use Acu-Punch-Ture and block Thunderstrike Tail",
            },
        )

    def test_standard_ultra_ball_exposes_deck_search_metadata_and_viewer_deck_cards(self) -> None:
        state = self.app.new_game(
            {
                "game_mode": "standard",
                "human_first": True,
                "human_deck_id": "ampharos-ex-battle-deck",
                "seed": 1,
            }
        )

        session = self.app.sessions.get(state["session_id"])
        ultra_ball_id = self._move_standard_named_card_to_hand(session.state, 0, "Ultra Ball")
        snapshot = self.app.get_game(state["session_id"])

        ultra_ball = next(card for card in snapshot["players"][0]["hand"] if card["instance_id"] == ultra_ball_id)
        self.assertEqual(ultra_ball["effect_specs"][1]["effect_type"], "search_deck")
        self.assertEqual(ultra_ball["effect_specs"][1]["destination_zone"], "hand")
        self.assertEqual(ultra_ball["effect_specs"][1]["search_filters"], ["pokemon"])
        self.assertEqual(ultra_ball["effect_specs"][1]["choose_count"], 1)
        self.assertGreater(len(snapshot["players"][0]["deck_cards"]), 0)
        self.assertEqual(snapshot["players"][1]["deck_cards"], [])

    def test_standard_ultra_ball_search_selection_confirm_adds_chosen_pokemon_to_hand(self) -> None:
        state = self.app.new_game(
            {
                "game_mode": "standard",
                "human_first": True,
                "human_deck_id": "ampharos-ex-battle-deck",
                "seed": 1,
            }
        )
        session = self.app.sessions.get(state["session_id"])
        from backend.tcg_ai.game_modes.standard.engine import apply_action, card_definition, list_legal_actions

        active_action = next(
            action for action in list_legal_actions(session.state) if action["type"] == "play_basic_to_active"
        )
        apply_action(session.state, active_action)
        end_setup = next(action for action in list_legal_actions(session.state) if action["type"] == "end_setup")
        apply_action(session.state, end_setup)

        ultra_ball_id = self._move_standard_named_card_to_hand(session.state, 0, "Ultra Ball")
        discard_a_id = self._move_standard_named_card_to_hand(session.state, 0, "Potion")
        discard_b_id = self._move_standard_named_card_to_hand(session.state, 0, "Switch")
        self._set_standard_exact_hand(session.state, 0, [ultra_ball_id, discard_a_id, discard_b_id])

        chosen_pokemon_id = next(
            instance_id
            for instance_id in session.state.players[0].deck
            if card_definition(session.state, instance_id).kind == "pokemon"
        )
        chosen_pokemon_name = card_definition(session.state, chosen_pokemon_id).name

        snapshot = self.app.get_game(state["session_id"])
        ultra_ball_actions = [
            action
            for action in snapshot["legal_actions"]
            if action["type"] == "play_item"
            and action["source"]["instance_id"] == ultra_ball_id
            and isinstance(action["action"].get("search_deck_ids"), list)
            and isinstance(action["action"].get("discard_from_hand_ids"), list)
        ]
        self.assertGreater(len(ultra_ball_actions), 0)

        selected_action = next(
            action
            for action in ultra_ball_actions
            if action["action"]["search_deck_ids"] == [chosen_pokemon_id]
            and set(action["action"]["discard_from_hand_ids"]) == {discard_a_id, discard_b_id}
        )

        updated = self.app.human_action(
            {
                "session_id": state["session_id"],
                "action": selected_action["action"],
            }
        )

        self.assertIn(
            chosen_pokemon_name,
            [card["name"] for card in updated["players"][0]["hand"]],
        )
        self.assertEqual(updated["players"][0]["discard_top"]["name"], "Ultra Ball")

    def test_standard_nest_ball_exposes_bench_search_metadata_and_benches_the_selected_basic(self) -> None:
        state = self.app.new_game(
            {
                "game_mode": "standard",
                "human_first": True,
                "human_deck_id": "ampharos-ex-battle-deck",
                "seed": 1,
            }
        )
        session = self.app.sessions.get(state["session_id"])
        from backend.tcg_ai.game_modes.standard.engine import apply_action, card_definition, list_legal_actions

        active_action = next(
            action for action in list_legal_actions(session.state) if action["type"] == "play_basic_to_active"
        )
        apply_action(session.state, active_action)
        end_setup = next(action for action in list_legal_actions(session.state) if action["type"] == "end_setup")
        apply_action(session.state, end_setup)

        nest_ball_id = self._move_standard_named_card_to_hand(session.state, 0, "Nest Ball")
        self._set_standard_exact_hand(session.state, 0, [nest_ball_id])

        chosen_basic_id = next(
            instance_id
            for instance_id in session.state.players[0].deck
            if card_definition(session.state, instance_id).is_basic
        )
        chosen_basic_name = card_definition(session.state, chosen_basic_id).name

        snapshot = self.app.get_game(state["session_id"])
        nest_ball = next(card for card in snapshot["players"][0]["hand"] if card["instance_id"] == nest_ball_id)
        self.assertEqual(nest_ball["effect_specs"][0]["effect_type"], "search_deck")
        self.assertEqual(nest_ball["effect_specs"][0]["destination_zone"], "bench")
        self.assertEqual(nest_ball["effect_specs"][0]["search_filters"], ["basic_pokemon"])
        self.assertEqual(nest_ball["effect_specs"][0]["choose_count"], 1)

        nest_ball_actions = [
            action
            for action in snapshot["legal_actions"]
            if action["type"] == "play_item"
            and action["source"]["instance_id"] == nest_ball_id
            and isinstance(action["action"].get("search_deck_ids"), list)
        ]
        self.assertGreater(len(nest_ball_actions), 0)

        selected_action = next(
            action
            for action in nest_ball_actions
            if action["action"]["search_deck_ids"] == [chosen_basic_id]
        )

        updated = self.app.human_action(
            {
                "session_id": state["session_id"],
                "action": selected_action["action"],
            }
        )

        self.assertIn(
            chosen_basic_name,
            [card["name"] for card in updated["players"][0]["bench"]],
        )
        self.assertNotIn(
            chosen_basic_name,
            [card["name"] for card in updated["players"][0]["hand"]],
        )
        self.assertEqual(updated["players"][0]["discard_top"]["name"], "Nest Ball")

    def test_standard_nest_ball_is_not_legal_when_the_bench_is_full(self) -> None:
        state = self.app.new_game(
            {
                "game_mode": "standard",
                "human_first": True,
                "human_deck_id": "ampharos-ex-battle-deck",
                "seed": 1,
            }
        )
        session = self.app.sessions.get(state["session_id"])
        from backend.tcg_ai.game_modes.standard.engine import apply_action, list_legal_actions

        active_action = next(
            action for action in list_legal_actions(session.state) if action["type"] == "play_basic_to_active"
        )
        apply_action(session.state, active_action)
        end_setup = next(action for action in list_legal_actions(session.state) if action["type"] == "end_setup")
        apply_action(session.state, end_setup)

        bench_ids = [
            self._move_standard_named_card_to_hand(session.state, 0, card_name)
            for card_name in ("Mareep", "Mareep", "Wattrel", "Starly", "Rotom")
        ]
        self._set_standard_exact_hand(session.state, 0, bench_ids)
        for _ in range(5):
            bench_action = next(
                action for action in list_legal_actions(session.state) if action["type"] == "bench_basic"
            )
            apply_action(session.state, bench_action)

        nest_ball_id = self._move_standard_named_card_to_hand(session.state, 0, "Nest Ball")
        self._set_standard_exact_hand(session.state, 0, [nest_ball_id])

        snapshot = self.app.get_game(state["session_id"])
        nest_ball_actions = [
            action
            for action in snapshot["legal_actions"]
            if action["type"] == "play_item" and action["source"]["instance_id"] == nest_ball_id
        ]

        self.assertEqual(len(snapshot["players"][0]["bench"]), 5)
        self.assertEqual(nest_ball_actions, [])

    def test_standard_pokegear_exposes_top_seven_supporter_search_metadata_and_adds_the_selected_supporter(self) -> None:
        state = self.app.new_game(
            {
                "game_mode": "standard",
                "human_first": True,
                "human_deck_id": "ampharos-ex-battle-deck",
                "seed": 1,
            }
        )
        session = self.app.sessions.get(state["session_id"])
        from backend.tcg_ai.game_modes.standard.engine import apply_action, card_definition, list_legal_actions

        active_action = next(
            action for action in list_legal_actions(session.state) if action["type"] == "play_basic_to_active"
        )
        apply_action(session.state, active_action)
        end_setup = next(action for action in list_legal_actions(session.state) if action["type"] == "end_setup")
        apply_action(session.state, end_setup)

        pokegear_id = self._move_standard_named_card_to_hand(session.state, 0, "Pok\u00e9gear 3.0")
        self._set_standard_exact_hand(session.state, 0, [pokegear_id])

        player = session.state.players[0]
        top_supporter_id = self._find_standard_instance_id(session.state, 0, "Nemona")
        outside_supporter_id = self._find_standard_instance_id(session.state, 0, "Youngster")
        filler_ids = [
            self._find_standard_instance_id(session.state, 0, card_name)
            for card_name in ("Mareep", "Wattrel", "Rotom", "Starly", "Switch", "Potion")
        ]
        ordered_top_cards = [
            filler_ids[0],
            filler_ids[1],
            top_supporter_id,
            filler_ids[2],
            filler_ids[3],
            filler_ids[4],
            filler_ids[5],
        ]
        prioritized_ids = set(ordered_top_cards + [outside_supporter_id])
        player.deck = ordered_top_cards + [
            instance_id
            for instance_id in player.deck
            if instance_id not in prioritized_ids
        ] + [outside_supporter_id]

        snapshot = self.app.get_game(state["session_id"])
        pokegear = next(card for card in snapshot["players"][0]["hand"] if card["instance_id"] == pokegear_id)
        self.assertEqual(pokegear["effect_specs"][0]["effect_type"], "search_deck")
        self.assertEqual(pokegear["effect_specs"][0]["count"], 7)
        self.assertEqual(pokegear["effect_specs"][0]["destination_zone"], "hand")
        self.assertEqual(pokegear["effect_specs"][0]["search_filters"], ["supporter"])
        self.assertEqual(pokegear["effect_specs"][0]["choose_count"], 1)

        pokegear_actions = [
            action
            for action in snapshot["legal_actions"]
            if action["type"] == "play_item"
            and action["source"]["instance_id"] == pokegear_id
            and isinstance(action["action"].get("search_deck_ids"), list)
        ]
        self.assertIn(
            [],
            [action["action"]["search_deck_ids"] for action in pokegear_actions],
        )
        self.assertIn(
            [top_supporter_id],
            [action["action"]["search_deck_ids"] for action in pokegear_actions],
        )
        self.assertNotIn(
            [outside_supporter_id],
            [action["action"]["search_deck_ids"] for action in pokegear_actions],
        )

        selected_action = next(
            action
            for action in pokegear_actions
            if action["action"]["search_deck_ids"] == [top_supporter_id]
        )
        selected_supporter_name = card_definition(session.state, top_supporter_id).name

        updated = self.app.human_action(
            {
                "session_id": state["session_id"],
                "action": selected_action["action"],
            }
        )

        self.assertIn(
            selected_supporter_name,
            [card["name"] for card in updated["players"][0]["hand"]],
        )
        self.assertEqual(updated["players"][0]["discard_top"]["name"], "Pok\u00e9gear 3.0")

    def test_standard_pokegear_is_still_playable_when_the_top_seven_has_no_supporter(self) -> None:
        state = self.app.new_game(
            {
                "game_mode": "standard",
                "human_first": True,
                "human_deck_id": "ampharos-ex-battle-deck",
                "seed": 1,
            }
        )
        session = self.app.sessions.get(state["session_id"])
        from backend.tcg_ai.game_modes.standard.engine import apply_action, list_legal_actions

        active_action = next(
            action for action in list_legal_actions(session.state) if action["type"] == "play_basic_to_active"
        )
        apply_action(session.state, active_action)
        end_setup = next(action for action in list_legal_actions(session.state) if action["type"] == "end_setup")
        apply_action(session.state, end_setup)

        pokegear_id = self._move_standard_named_card_to_hand(session.state, 0, "Pok\u00e9gear 3.0")
        self._set_standard_exact_hand(session.state, 0, [pokegear_id])

        top_cards = [
            self._find_standard_instance_id(session.state, 0, card_name)
            for card_name in ("Mareep", "Wattrel", "Rotom", "Starly", "Switch", "Potion", "Nest Ball")
        ]
        trailing_supporter_id = self._find_standard_instance_id(session.state, 0, "Nemona")
        self._reorder_standard_deck(session.state, 0, top_cards, [trailing_supporter_id])

        snapshot = self.app.get_game(state["session_id"])
        pokegear_actions = [
            action
            for action in snapshot["legal_actions"]
            if action["type"] == "play_item"
            and action["source"]["instance_id"] == pokegear_id
            and isinstance(action["action"].get("search_deck_ids"), list)
        ]
        self.assertEqual(
            [action["action"]["search_deck_ids"] for action in pokegear_actions],
            [[]],
        )

        updated = self.app.human_action(
            {
                "session_id": state["session_id"],
                "action": pokegear_actions[0]["action"],
            }
        )

        self.assertEqual(updated["players"][0]["hand"], [])
        self.assertEqual(updated["players"][0]["discard_top"]["name"], "Pok\u00e9gear 3.0")
        self.assertIn("did not add a card to hand", updated["log"][-1]["text"].lower())

    def test_standard_potion_exposes_targeted_item_actions_and_heals_the_selected_pokemon(self) -> None:
        state = self.app.new_game(
            {
                "game_mode": "standard",
                "human_first": True,
                "human_deck_id": "ampharos-ex-battle-deck",
                "seed": 1,
            }
        )
        session = self.app.sessions.get(state["session_id"])
        from backend.tcg_ai.game_modes.standard.engine import apply_action, list_legal_actions

        active_action = next(
            action for action in list_legal_actions(session.state) if action["type"] == "play_basic_to_active"
        )
        apply_action(session.state, active_action)
        bench_action = next(
            action for action in list_legal_actions(session.state) if action["type"] == "bench_basic"
        )
        apply_action(session.state, bench_action)
        end_setup = next(action for action in list_legal_actions(session.state) if action["type"] == "end_setup")
        apply_action(session.state, end_setup)

        potion_id = self._move_standard_named_card_to_hand(session.state, 0, "Potion")
        self._set_standard_exact_hand(session.state, 0, [potion_id])
        session.state.players[0].active.damage = 40
        session.state.players[0].bench[0].damage = 20

        snapshot = self.app.get_game(state["session_id"])
        potion = snapshot["players"][0]["hand"][0]
        potion_actions = [action for action in snapshot["legal_actions"] if action["type"] == "play_item"]

        self.assertEqual(potion["card_id"], "sv1-188")
        self.assertEqual(potion["effect_specs"][0]["effect_type"], "heal_damage")
        self.assertEqual(potion["effect_specs"][0]["count"], 30)
        self.assertEqual(
            {(action["target"]["zone"], action["target"].get("bench_index")) for action in potion_actions},
            {("active", None), ("bench", 0)},
        )

        active_target_action = next(action for action in potion_actions if action["target"]["zone"] == "active")
        updated = self.app.human_action(
            {
                "session_id": state["session_id"],
                "action": active_target_action["action"],
            }
        )

        self.assertEqual(updated["players"][0]["active"]["damage"], 10)
        self.assertEqual(updated["players"][0]["bench"][0]["damage"], 20)
        self.assertEqual(updated["players"][0]["discard_top"]["name"], "Potion")

    def test_standard_evolution_exposes_targeted_actions_and_evolves_the_selected_pokemon(self) -> None:
        state = self.app.new_game(
            {
                "game_mode": "standard",
                "human_first": True,
                "human_deck_id": "ampharos-ex-battle-deck",
                "seed": 1,
            }
        )
        session = self.app.sessions.get(state["session_id"])
        from backend.tcg_ai.game_modes.standard.engine import apply_action, list_legal_actions

        active_action = next(
            action for action in list_legal_actions(session.state) if action["type"] == "play_basic_to_active"
        )
        apply_action(session.state, active_action)
        end_setup = next(action for action in list_legal_actions(session.state) if action["type"] == "end_setup")
        apply_action(session.state, end_setup)

        session.state.players[0].turns_taken = 2
        kilowattrel_id = self._move_standard_named_card_to_hand(session.state, 0, "Kilowattrel")
        self._set_standard_exact_hand(session.state, 0, [kilowattrel_id])

        snapshot = self.app.get_game(state["session_id"])
        evolve_actions = [action for action in snapshot["legal_actions"] if action["type"] == "evolve"]

        self.assertEqual(len(evolve_actions), 1)
        self.assertEqual(evolve_actions[0]["source"]["zone"], "hand")
        self.assertEqual(evolve_actions[0]["source"]["card_id"], "sv1-79")
        self.assertEqual(evolve_actions[0]["target"]["zone"], "active")

        updated = self.app.human_action(
            {
                "session_id": state["session_id"],
                "action": evolve_actions[0]["action"],
            }
        )

        self.assertEqual(updated["players"][0]["active"]["name"], "Kilowattrel")
        self.assertEqual(updated["players"][0]["active"]["stage"], "stage1")
        self.assertEqual(updated["players"][0]["hand_count"], 0)

    def test_standard_switch_exposes_bench_targets_and_swaps_active_pokemon(self) -> None:
        state = self.app.new_game(
            {
                "game_mode": "standard",
                "human_first": True,
                "human_deck_id": "ampharos-ex-battle-deck",
                "seed": 1,
            }
        )
        session = self.app.sessions.get(state["session_id"])
        from backend.tcg_ai.game_modes.standard.engine import apply_action, list_legal_actions

        active_action = next(
            action for action in list_legal_actions(session.state) if action["type"] == "play_basic_to_active"
        )
        apply_action(session.state, active_action)
        bench_action = next(
            action for action in list_legal_actions(session.state) if action["type"] == "bench_basic"
        )
        apply_action(session.state, bench_action)
        end_setup = next(action for action in list_legal_actions(session.state) if action["type"] == "end_setup")
        apply_action(session.state, end_setup)

        switch_id = self._move_standard_named_card_to_hand(session.state, 0, "Switch")
        self._set_standard_exact_hand(session.state, 0, [switch_id])
        expected_bench_name = session.state.players[0].bench[0].stack[-1]
        expected_active_name = session.state.players[0].active.stack[-1]

        snapshot = self.app.get_game(state["session_id"])
        switch_card = snapshot["players"][0]["hand"][0]
        switch_actions = [action for action in snapshot["legal_actions"] if action["type"] == "play_item"]

        self.assertEqual(switch_card["card_id"], "sv1-194")
        self.assertEqual(switch_card["effect_specs"][0]["effect_type"], "switch_active_with_bench")
        self.assertEqual(len(switch_actions), 1)
        self.assertEqual(switch_actions[0]["target"]["zone"], "bench")
        self.assertEqual(switch_actions[0]["target"]["bench_index"], 0)

        updated = self.app.human_action(
            {
                "session_id": state["session_id"],
                "action": switch_actions[0]["action"],
            }
        )

        self.assertEqual(updated["players"][0]["active"]["instance_id"], expected_bench_name)
        self.assertEqual(updated["players"][0]["bench"][0]["instance_id"], expected_active_name)
        self.assertEqual(updated["players"][0]["discard_top"]["name"], "Switch")

    def test_standard_knock_out_prompts_the_human_to_choose_a_new_active_pokemon(self) -> None:
        state = self.app.new_game(
            {
                "game_mode": "standard",
                "human_first": True,
                "human_deck_id": "ampharos-ex-battle-deck",
                "seed": 1,
            }
        )
        session = self.app.sessions.get(state["session_id"])
        from backend.tcg_ai.game_modes.standard.engine import apply_action, list_legal_actions

        session.state.setup_phase = None
        session.state.current_player = 1
        session.state.turn_number = 2
        session.state.players[0].turns_taken = 2
        session.state.players[1].turns_taken = 2
        self._set_standard_named_active_pokemon(session.state, 0, "Mareep")
        self._set_standard_named_bench_pokemon(session.state, 0, "Wattrel")
        self._set_standard_named_active_pokemon(session.state, 1, "Mankey")
        session.state.players[0].active.damage = 30
        session.state.players[1].active.attached_energy = [
            self._find_standard_instance_id(session.state, 1, "Basic Fighting Energy"),
        ]

        attack_action = next(action for action in list_legal_actions(session.state) if action["type"] == "attack")
        apply_action(session.state, attack_action)

        snapshot = self.app.get_game(state["session_id"])
        promote_action = self._find_action(snapshot, "promote")

        self.assertEqual(snapshot["current_player"], 0)
        self.assertEqual(snapshot["turn_number"], 2)
        self.assertEqual(snapshot["pending_promotion_for"], 0)
        self.assertIsNone(snapshot["players"][0]["active"])
        self.assertEqual(promote_action["source"]["zone"], "bench")
        self.assertEqual(promote_action["source"]["name"], "Wattrel")
        self.assertEqual(promote_action["target"]["zone"], "active")
        self.assertIsNone(promote_action["target"]["instance_id"])

        updated = self.app.human_action(
            {
                "session_id": state["session_id"],
                "action": promote_action["action"],
            }
        )

        self.assertEqual(updated["current_player"], 0)
        self.assertEqual(updated["turn_number"], 3)
        self.assertIsNone(updated["pending_promotion_for"])
        self.assertEqual(updated["players"][0]["active"]["name"], "Wattrel")

    def test_standard_energy_attachment_actions_expose_targets_and_update_state(self) -> None:
        state = self.app.new_game(
            {
                "game_mode": "standard",
                "human_first": True,
                "human_deck_id": "ampharos-ex-battle-deck",
                "seed": 1,
            }
        )
        session = self.app.sessions.get(state["session_id"])
        from backend.tcg_ai.game_modes.standard.engine import apply_action, list_legal_actions

        active_action = next(
            action for action in list_legal_actions(session.state) if action["type"] == "play_basic_to_active"
        )
        apply_action(session.state, active_action)
        bench_action = next(
            action for action in list_legal_actions(session.state) if action["type"] == "bench_basic"
        )
        apply_action(session.state, bench_action)
        end_setup = next(action for action in list_legal_actions(session.state) if action["type"] == "end_setup")
        apply_action(session.state, end_setup)

        energy_id = self._move_standard_named_card_to_hand(session.state, 0, "Basic Lightning Energy")
        self._set_standard_exact_hand(session.state, 0, [energy_id])

        snapshot = self.app.get_game(state["session_id"])
        energy_actions = [action for action in snapshot["legal_actions"] if action["type"] == "play_energy"]

        self.assertEqual(len(energy_actions), 2)
        active_target = next(action for action in energy_actions if action["target"]["zone"] == "active")
        bench_target = next(action for action in energy_actions if action["target"]["zone"] == "bench")
        self.assertEqual(
            active_target["target"]["instance_id"],
            snapshot["players"][0]["active"]["ref"]["instance_id"],
        )
        self.assertEqual(bench_target["target"]["bench_index"], 0)
        self.assertEqual(
            bench_target["target"]["instance_id"],
            snapshot["players"][0]["bench"][0]["ref"]["instance_id"],
        )

        updated = self.app.human_action(
            {
                "session_id": state["session_id"],
                "action": active_target["action"],
            }
        )

        self.assertEqual(updated["players"][0]["hand_count"], 0)
        self.assertEqual(updated["players"][0]["energy_count"], 1)
        self.assertFalse(updated["players"][0]["energy_attachment_available"])
        self.assertEqual(updated["players"][0]["active"]["attached_energy_count"], 1)
        self.assertEqual(updated["players"][0]["active"]["attached_energy"][0]["name"], "Basic Lightning Energy")
        self.assertFalse(updated["players"][0]["active"]["can_attack"])
        self.assertEqual(
            [action["type"] for action in updated["legal_actions"]],
            ["retreat", "end_turn"],
        )

    def test_standard_retreat_actions_expose_active_to_bench_and_update_state(self) -> None:
        state = self.app.new_game(
            {
                "game_mode": "standard",
                "human_first": True,
                "human_deck_id": "ampharos-ex-battle-deck",
                "seed": 1,
            }
        )
        session = self.app.sessions.get(state["session_id"])

        session.state.setup_phase = None
        session.state.current_player = 0
        session.state.turn_number = 2
        session.state.players[0].turns_taken = 2
        session.state.players[1].turns_taken = 2
        self._set_standard_named_active_pokemon(session.state, 0, "Mareep")
        self._set_standard_named_bench_pokemon(session.state, 0, "Wattrel")
        self._set_standard_named_active_pokemon(session.state, 1, "Mankey")
        retreat_energy_id = self._take_standard_named_card(session.state, 0, "Basic Lightning Energy")
        bench_energy_id = self._take_standard_named_card(session.state, 0, "Basic Lightning Energy")
        session.state.players[0].active.attached_energy = [retreat_energy_id]
        session.state.players[0].bench[0].attached_energy = [bench_energy_id]

        snapshot = self.app.get_game(state["session_id"])
        retreat_action = self._find_action(snapshot, "retreat")

        self.assertEqual(retreat_action["source"]["zone"], "active")
        self.assertEqual(retreat_action["source"]["instance_id"], snapshot["players"][0]["active"]["ref"]["instance_id"])
        self.assertEqual(retreat_action["target"]["zone"], "bench")
        self.assertEqual(retreat_action["target"]["bench_index"], 0)

        updated = self.app.human_action(
            {
                "session_id": state["session_id"],
                "action": retreat_action["action"],
            }
        )

        self.assertEqual(updated["players"][0]["active"]["name"], "Wattrel")
        self.assertEqual(updated["players"][0]["bench"][0]["name"], "Mareep")
        self.assertEqual(updated["players"][0]["discard_top"]["name"], "Basic Lightning Energy")
        self.assertNotIn("retreat", [action["type"] for action in updated["legal_actions"]])
        self.assertIn("attack", [action["type"] for action in updated["legal_actions"]])

    def test_standard_collect_attack_draws_a_card_and_passes_the_turn_on_a_later_turn(self) -> None:
        state = self.app.new_game(
            {
                "game_mode": "standard",
                "human_first": True,
                "human_deck_id": "ampharos-ex-battle-deck",
                "seed": 1,
            }
        )
        session = self.app.sessions.get(state["session_id"])
        from backend.tcg_ai.game_modes.standard.engine import apply_action, list_legal_actions

        active_action = next(
            action for action in list_legal_actions(session.state) if action["type"] == "play_basic_to_active"
        )
        apply_action(session.state, active_action)
        end_setup = next(action for action in list_legal_actions(session.state) if action["type"] == "end_setup")
        apply_action(session.state, end_setup)
        session.state.turn_number = 2
        session.state.players[0].turns_taken = 2

        energy_id = self._move_standard_named_card_to_hand(session.state, 0, "Basic Lightning Energy")
        self._set_standard_exact_hand(session.state, 0, [energy_id])
        attach_action = next(action for action in list_legal_actions(session.state) if action["type"] == "play_energy")
        apply_action(session.state, attach_action)

        updated = self.app.human_action(
            {
                "session_id": state["session_id"],
                "action": next(
                    action["action"]
                    for action in self.app.get_game(state["session_id"])["legal_actions"]
                    if action["type"] == "attack"
                ),
            }
        )

        self.assertEqual(updated["current_player"], 1)
        self.assertEqual(updated["players"][1]["active"]["damage"], 0)
        self.assertEqual(updated["players"][0]["hand_count"], 1)
        self.assertTrue(any("drew 1 card" in entry["text"].lower() for entry in updated["log"]))

    def test_standard_opening_hand_shuffle_is_seeded(self) -> None:
        first = self.app.new_game(
            {
                "game_mode": "standard",
                "human_first": True,
                "human_deck_id": "ampharos-ex-battle-deck",
                "seed": 77,
            }
        )
        second = self.app.new_game(
            {
                "game_mode": "standard",
                "human_first": True,
                "human_deck_id": "ampharos-ex-battle-deck",
                "seed": 77,
            }
        )

        self.assertEqual(
            [card["card_id"] for card in first["players"][0]["hand"]],
            [card["card_id"] for card in second["players"][0]["hand"]],
        )
        self.assertEqual(
            [action["label"] for action in first["legal_actions"]],
            [action["label"] for action in second["legal_actions"]],
        )

    def test_standard_cards_use_catalog_basic_and_stage_data(self) -> None:
        deck_cards = {card.card_id: card for card in load_deck_cards("ampharos-ex-battle-deck")}

        self.assertTrue(deck_cards["sv1-66"].is_basic)
        self.assertEqual(deck_cards["sv1-66"].stage, "basic")
        self.assertFalse(deck_cards["svp-15"].is_basic)
        self.assertEqual(deck_cards["svp-15"].stage, "stage1")
        self.assertFalse(deck_cards["svp-16"].is_basic)
        self.assertEqual(deck_cards["svp-16"].stage, "stage2")

    def test_standard_opening_hand_mulligan_redraws_when_no_basic_is_present(self) -> None:
        state = self.app.new_game(
            {
                "game_mode": "standard",
                "human_first": True,
                "human_deck_id": "ampharos-ex-battle-deck",
                "seed": 4,
            }
        )

        self.assertEqual(
            [card["name"] for card in state["players"][0]["hand"]],
            [
                "Youngster",
                "Basic Lightning Energy",
                "Basic Lightning Energy",
                "Basic Lightning Energy",
                "Nemona",
                "Ultra Ball",
                "Basic Lightning Energy",
            ],
        )
        self.assertEqual([action["type"] for action in state["legal_actions"]], ["mulligan"])

        updated = self.app.human_action(
            {
                "session_id": state["session_id"],
                "action": state["legal_actions"][0]["action"],
            }
        )

        self.assertEqual(updated["players"][0]["hand_count"], 7)
        self.assertEqual(updated["players"][0]["deck_count"], 47)
        self.assertEqual(len(updated["players"][0]["hand"]), 7)
        self.assertTrue(any(card["is_basic"] for card in updated["players"][0]["hand"]))
        self.assertTrue(updated["legal_actions"])
        self.assertTrue(
            all(action["type"] == "play_basic_to_active" for action in updated["legal_actions"])
        )

    def _move_standard_named_card_to_hand(self, state, player_index: int, card_name: str) -> str:
        player = state.players[player_index]
        from backend.tcg_ai.game_modes.standard.engine import card_definition

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
        self.assertIn("7 cards were redrawn", updated["log"][-2]["text"])

    def _take_standard_named_card(self, state, player_index: int, card_name: str) -> str:
        player = state.players[player_index]
        from backend.tcg_ai.game_modes.standard.engine import card_definition

        for zone_name in ("hand", "deck", "discard", "prizes"):
            zone = getattr(player, zone_name)
            for instance_id in list(zone):
                if card_definition(state, instance_id).name != card_name:
                    continue
                zone.remove(instance_id)
                return instance_id
        self.fail(f"Could not take {card_name} for player {player_index}")

    def _set_standard_exact_hand(self, state, player_index: int, ordered_instance_ids: list[str]) -> None:
        player = state.players[player_index]
        kept = set(ordered_instance_ids)
        extras = [instance_id for instance_id in player.hand if instance_id not in kept]
        player.hand = list(ordered_instance_ids)
        player.deck.extend(extras)

    def _set_standard_named_active_pokemon(self, state, player_index: int, card_name: str) -> None:
        from backend.tcg_ai.game_modes.standard.models import PokemonInPlay

        instance_id = self._find_standard_instance_id(state, player_index, card_name)
        player = state.players[player_index]
        for zone_name in ("hand", "deck", "discard", "prizes"):
            zone = getattr(player, zone_name)
            if instance_id in zone:
                zone.remove(instance_id)
                break
        player.active = PokemonInPlay(stack=[instance_id])

    def _set_standard_named_bench_pokemon(self, state, player_index: int, card_name: str) -> None:
        from backend.tcg_ai.game_modes.standard.models import PokemonInPlay

        instance_id = self._find_standard_instance_id(state, player_index, card_name)
        player = state.players[player_index]
        for zone_name in ("hand", "deck", "discard", "prizes"):
            zone = getattr(player, zone_name)
            if instance_id in zone:
                zone.remove(instance_id)
                break
        player.bench.append(PokemonInPlay(stack=[instance_id]))

    def _find_standard_instance_id(self, state, player_index: int, card_name: str) -> str:
        from backend.tcg_ai.game_modes.standard.engine import card_definition

        player = state.players[player_index]
        for zone_name in ("hand", "deck", "discard", "prizes"):
            zone = getattr(player, zone_name)
            for instance_id in zone:
                if card_definition(state, instance_id).name == card_name:
                    return instance_id
        self.fail(f"Could not find {card_name} for player {player_index}")

    def _reorder_standard_deck(
        self,
        state,
        player_index: int,
        leading_instance_ids: list[str],
        trailing_instance_ids: list[str] | None = None,
    ) -> None:
        player = state.players[player_index]
        trailing_instance_ids = trailing_instance_ids or []
        prioritized_ids = set(leading_instance_ids + trailing_instance_ids)
        player.deck = list(leading_instance_ids) + [
            instance_id
            for instance_id in player.deck
            if instance_id not in prioritized_ids
        ] + list(trailing_instance_ids)

    def test_standard_can_play_a_basic_from_hand_into_the_active_spot(self) -> None:
        state = self.app.new_game(
            {
                "game_mode": "standard",
                "human_first": True,
                "human_deck_id": "ampharos-ex-battle-deck",
                "seed": 1,
            }
        )

        action = state["legal_actions"][0]
        self.assertEqual(action["type"], "play_basic_to_active")

        updated = self.app.human_action(
            {
                "session_id": state["session_id"],
                "action": action["action"],
            }
        )

        self.assertEqual(updated["players"][0]["active"]["name"], "Wattrel")
        self.assertEqual(updated["players"][0]["active"]["stage"], "basic")
        self.assertEqual(updated["players"][0]["active"]["hp"], 50)
        self.assertEqual(updated["players"][0]["active"]["ref"]["zone"], "active")
        self.assertEqual(updated["players"][0]["hand_count"], 6)
        self.assertEqual(updated["players"][0]["deck_count"], 47)
        self.assertTrue(updated["players"][1]["active"]["face_down"])
        self.assertEqual(updated["setup_phase"], "awaiting_end_setup")
        self.assertEqual(
            [action["type"] for action in updated["legal_actions"]],
            ["bench_basic", "bench_basic", "end_setup"],
        )

    def test_standard_can_bench_a_basic_during_setup_after_choosing_active(self) -> None:
        state = self.app.new_game(
            {
                "game_mode": "standard",
                "human_first": True,
                "human_deck_id": "ampharos-ex-battle-deck",
                "seed": 1,
            }
        )
        after_active = self.app.human_action(
            {
                "session_id": state["session_id"],
                "action": state["legal_actions"][0]["action"],
            }
        )
        bench_action = next(
            action for action in after_active["legal_actions"] if action["type"] == "bench_basic"
        )

        updated = self.app.human_action(
            {
                "session_id": state["session_id"],
                "action": bench_action["action"],
            }
        )

        self.assertEqual(len(updated["players"][0]["bench"]), 1)
        self.assertEqual(updated["players"][0]["bench"][0]["name"], "Mareep")
        self.assertEqual(updated["players"][0]["hand_count"], 5)
        self.assertEqual(updated["setup_phase"], "awaiting_end_setup")
        self.assertEqual(
            [action["type"] for action in updated["legal_actions"]],
            ["bench_basic", "end_setup"],
        )
        self.assertIn("during setup", updated["log"][-1]["text"])

    def test_standard_end_setup_begins_turn_one_and_enables_end_turn(self) -> None:
        state = self.app.new_game(
            {
                "game_mode": "standard",
                "human_first": True,
                "human_deck_id": "ampharos-ex-battle-deck",
                "seed": 1,
            }
        )
        choose_active = state["legal_actions"][0]
        after_active = self.app.human_action(
            {
                "session_id": state["session_id"],
                "action": choose_active["action"],
            }
        )
        self.assertEqual(
            [action["type"] for action in after_active["legal_actions"]],
            ["bench_basic", "bench_basic", "end_setup"],
        )
        end_setup_action = next(
            action for action in after_active["legal_actions"] if action["type"] == "end_setup"
        )

        after_setup = self.app.human_action(
            {
                "session_id": state["session_id"],
                "action": end_setup_action["action"],
            }
        )

        self.assertIsNone(after_setup["setup_phase"])
        self.assertEqual(after_setup["current_player"], 0)
        self.assertEqual(after_setup["turn_number"], 1)
        self.assertEqual(after_setup["players"][0]["hand_count"], 7)
        self.assertEqual(after_setup["players"][0]["deck_count"], 46)
        self.assertEqual(after_setup["players"][1]["active"]["name"], "Koraidon")
        self.assertFalse(after_setup["players"][1]["active"]["face_down"])
        legal_action_types = [action["type"] for action in after_setup["legal_actions"]]
        self.assertIn("end_turn", legal_action_types)
        self.assertIn("play_energy", legal_action_types)
        self.assertGreaterEqual(legal_action_types.count("bench_basic"), 2)
        self.assertIn("Turn 1 begins", after_setup["log"][-2]["text"])
        self.assertIn("You drew", after_setup["log"][-1]["text"])

    def test_standard_can_bench_a_basic_during_the_turn(self) -> None:
        state = self.app.new_game(
            {
                "game_mode": "standard",
                "human_first": True,
                "human_deck_id": "ampharos-ex-battle-deck",
                "seed": 1,
            }
        )
        after_active = self.app.human_action(
            {
                "session_id": state["session_id"],
                "action": state["legal_actions"][0]["action"],
            }
        )
        end_setup_action = next(
            action for action in after_active["legal_actions"] if action["type"] == "end_setup"
        )
        after_setup = self.app.human_action(
            {
                "session_id": state["session_id"],
                "action": end_setup_action["action"],
            }
        )
        bench_action = next(
            action for action in after_setup["legal_actions"] if action["type"] == "bench_basic"
        )

        updated = self.app.human_action(
            {
                "session_id": state["session_id"],
                "action": bench_action["action"],
            }
        )

        self.assertEqual(len(updated["players"][0]["bench"]), 1)
        self.assertEqual(updated["players"][0]["bench"][0]["name"], "Mareep")
        self.assertIsNone(updated["setup_phase"])
        legal_action_types = [action["type"] for action in updated["legal_actions"]]
        self.assertIn("end_turn", legal_action_types)
        self.assertGreaterEqual(legal_action_types.count("play_energy"), 2)
        self.assertGreaterEqual(legal_action_types.count("bench_basic"), 1)

    def test_standard_ai_turn_can_finish_and_pass_the_turn_back(self) -> None:
        state = self.app.new_game(
            {
                "game_mode": "standard",
                "human_first": True,
                "human_deck_id": "ampharos-ex-battle-deck",
                "seed": 1,
            }
        )
        after_active = self.app.human_action(
            {
                "session_id": state["session_id"],
                "action": state["legal_actions"][0]["action"],
            }
        )
        end_setup_action = next(
            action for action in after_active["legal_actions"] if action["type"] == "end_setup"
        )
        after_setup = self.app.human_action(
            {
                "session_id": state["session_id"],
                "action": end_setup_action["action"],
            }
        )
        end_turn_action = next(
            action for action in after_setup["legal_actions"] if action["type"] == "end_turn"
        )
        after_end_turn = self.app.human_action(
            {
                "session_id": state["session_id"],
                "action": end_turn_action["action"],
            }
        )

        self.assertEqual(after_end_turn["current_player"], 1)
        actions_taken: list[str] = []
        ai_state = after_end_turn
        for _ in range(10):
            ai_state = self.app.ai_step({"session_id": state["session_id"]})
            actions_taken.append(ai_state["ai_step"]["action"]["type"])
            if ai_state["current_player"] == 0:
                break

        self.assertEqual(ai_state["current_player"], 0)
        self.assertEqual(ai_state["turn_number"], 2)
        self.assertEqual(actions_taken[-1], "end_turn")
        legal_action_types = [action["type"] for action in ai_state["legal_actions"]]
        self.assertIn("end_turn", legal_action_types)
        self.assertGreaterEqual(legal_action_types.count("bench_basic"), 2)
        self.assertGreaterEqual(legal_action_types.count("play_supporter"), 2)

    def test_standard_ai_step_can_play_a_supporter_and_draw_cards(self) -> None:
        state = self.app.new_game(
            {
                "game_mode": "standard",
                "human_first": True,
                "human_deck_id": "ampharos-ex-battle-deck",
                "seed": 1,
            }
        )
        session = self.app.sessions.get(state["session_id"])

        after_active = self.app.human_action(
            {
                "session_id": state["session_id"],
                "action": state["legal_actions"][0]["action"],
            }
        )
        end_setup_action = next(
            action for action in after_active["legal_actions"] if action["type"] == "end_setup"
        )
        after_setup = self.app.human_action(
            {
                "session_id": state["session_id"],
                "action": end_setup_action["action"],
            }
        )
        end_turn_action = next(
            action for action in after_setup["legal_actions"] if action["type"] == "end_turn"
        )
        after_end_turn = self.app.human_action(
            {
                "session_id": state["session_id"],
                "action": end_turn_action["action"],
            }
        )

        nemona_id = self._move_standard_named_card_to_hand(session.state, 1, "Nemona")
        self._set_standard_exact_hand(session.state, 1, [nemona_id])
        expected_ai_draw = list(session.state.players[1].deck[:3])

        ai_step = self.app.ai_step({"session_id": state["session_id"]})

        self.assertEqual(after_end_turn["current_player"], 1)
        self.assertEqual(ai_step["ai_step"]["action"]["type"], "play_supporter")
        self.assertEqual(ai_step["current_player"], 1)
        self.assertEqual(ai_step["players"][1]["hand_count"], 3)
        self.assertEqual(
            session.state.players[1].hand,
            expected_ai_draw,
        )
        from backend.tcg_ai.game_modes.standard.engine import card_definition

        self.assertEqual(ai_step["players"][1]["discard_count"], 1)
        self.assertEqual(card_definition(session.state, session.state.players[1].discard[-1]).name, "Nemona")
        self.assertIn("played Nemona", ai_step["log"][-2]["text"])
        self.assertIn("drew 3 cards", ai_step["log"][-1]["text"])

    def test_standard_ai_step_can_attach_energy_to_its_active_pokemon(self) -> None:
        state = self.app.new_game(
            {
                "game_mode": "standard",
                "human_first": True,
                "human_deck_id": "ampharos-ex-battle-deck",
                "seed": 1,
            }
        )
        session = self.app.sessions.get(state["session_id"])

        after_active = self.app.human_action(
            {
                "session_id": state["session_id"],
                "action": state["legal_actions"][0]["action"],
            }
        )
        end_setup_action = next(
            action for action in after_active["legal_actions"] if action["type"] == "end_setup"
        )
        after_setup = self.app.human_action(
            {
                "session_id": state["session_id"],
                "action": end_setup_action["action"],
            }
        )
        end_turn_action = next(
            action for action in after_setup["legal_actions"] if action["type"] == "end_turn"
        )
        after_end_turn = self.app.human_action(
            {
                "session_id": state["session_id"],
                "action": end_turn_action["action"],
            }
        )

        energy_id = self._move_standard_named_card_to_hand(session.state, 1, "Basic Fighting Energy")
        self._set_standard_exact_hand(session.state, 1, [energy_id])

        ai_step = self.app.ai_step({"session_id": state["session_id"]})

        self.assertEqual(after_end_turn["current_player"], 1)
        self.assertEqual(ai_step["ai_step"]["action"]["type"], "play_energy")
        self.assertEqual(ai_step["current_player"], 1)
        self.assertEqual(ai_step["players"][1]["hand_count"], 0)
        self.assertEqual(ai_step["players"][1]["energy_count"], 1)
        self.assertEqual(ai_step["players"][1]["active"]["attached_energy_count"], 1)
        self.assertEqual(ai_step["players"][1]["active"]["attached_energy"][0]["name"], "Basic Fighting Energy")
        self.assertIn("attached Basic Fighting Energy", ai_step["log"][-1]["text"])

    def test_standard_player_view_redacts_opponent_named_turn_draw(self) -> None:
        state = self.app.new_game(
            {
                "game_mode": "standard",
                "human_first": True,
                "human_deck_id": "ampharos-ex-battle-deck",
                "seed": 1,
            }
        )
        session = self.app.sessions.get(state["session_id"])

        after_active = self.app.human_action(
            {
                "session_id": state["session_id"],
                "action": state["legal_actions"][0]["action"],
            }
        )
        end_setup_action = next(
            action for action in after_active["legal_actions"] if action["type"] == "end_setup"
        )
        after_setup = self.app.human_action(
            {
                "session_id": state["session_id"],
                "action": end_setup_action["action"],
            }
        )
        end_turn_action = next(
            action for action in after_setup["legal_actions"] if action["type"] == "end_turn"
        )
        after_end_turn = self.app.human_action(
            {
                "session_id": state["session_id"],
                "action": end_turn_action["action"],
            }
        )

        opponent_name = session.state.players[1].name
        self.assertRegex(session.state.log[-1], rf"^{opponent_name} drew .+\.$")
        self.assertNotEqual(after_end_turn["log"][-1]["text"], session.state.log[-1])
        self.assertEqual(after_end_turn["log"][-1]["text"], f"{opponent_name} drew a card.")

    def test_new_game_can_select_a_human_deck_and_exposes_available_decks(self) -> None:
        state = self.app.new_game({"human_first": True, "human_deck_id": "bulbasaur"})

        self.assertEqual(state["human_deck_id"], "bulbasaur")
        self.assertEqual(state["ai_deck_id"], "pikachu")
        self.assertEqual(state["players"][0]["active"]["card_id"], "bulbasaur")
        self.assertEqual(state["players"][1]["active"]["card_id"], "pikachu")
        self.assertEqual(
            [deck["id"] for deck in state["available_decks"]],
            ["bulbasaur", "charmander", "squirtle", "pikachu"],
        )
        selected_deck = next(deck for deck in state["available_decks"] if deck["id"] == "bulbasaur")
        self.assertTrue(selected_deck["selected"])
        self.assertEqual(selected_deck["paired_deck_id"], "pikachu")
        self.assertEqual(
            selected_deck["prize_coin_image_url"],
            "/assets/coins/my-first-battle/bulbasaur-prize-coin.png",
        )

    def test_new_game_rejects_an_unknown_human_deck(self) -> None:
        with self.assertRaises(ApiError) as context:
            self.app.new_game({"human_first": True, "human_deck_id": "missingno"})

        self.assertEqual(context.exception.code, "human_deck_not_found")

    def test_new_game_can_target_a_specific_gym_leader(self) -> None:
        state = self.app.new_game({"human_first": True, "trainer_id": "misty"})

        self.assertEqual(state["ai_trainer"]["id"], "misty")
        self.assertEqual(state["ai_trainer"]["name"], "Misty")
        self.assertEqual(state["players"][1]["name"], "Misty")
        self.assertEqual(
            [trainer["id"] for trainer in state["available_trainers"]],
            ["brock", "misty", "lt_surge", "erika", "koga", "sabrina", "blaine", "giovanni"],
        )
        selected_trainer = next(
            trainer for trainer in state["available_trainers"] if trainer["id"] == "misty"
        )
        self.assertTrue(selected_trainer["selected"])

    def test_new_game_rolls_for_the_starting_player_when_turn_order_is_not_provided(self) -> None:
        with patch("backend.tcg_ai.server.roll_starting_player_die", return_value=5):
            state = self.app.new_game({"trainer_id": "misty"})

        self.assertEqual(state["current_player"], 1)
        self.assertIn("Opening die roll: 5. Misty goes first.", [entry["text"] for entry in state["log"]])

    def test_sessions_are_isolated_from_each_other(self) -> None:
        first_state = self.app.new_game({"human_first": True})
        second_state = self.app.new_game({"human_first": True})

        attack_action = self._find_action(first_state, "end_turn")
        updated_first = self.app.human_action(
            {"session_id": first_state["session_id"], "action": attack_action["action"]}
        )
        unchanged_second = self.app.get_game(second_state["session_id"])

        self.assertEqual(updated_first["current_player"], 1)
        self.assertEqual(updated_first["game_mode"], "my_first_battle")
        self.assertEqual(unchanged_second["current_player"], 0)
        self.assertEqual(unchanged_second["game_mode"], "my_first_battle")
        self.assertEqual(unchanged_second["players"][1]["active"]["damage"], 0)

    def test_unknown_session_id_raises_a_structured_error(self) -> None:
        with self.assertRaises(ApiError) as context:
            self.app.get_game("missing-session")

        self.assertEqual(context.exception.code, "session_not_found")

    def test_resubmitting_an_expired_action_is_rejected(self) -> None:
        state = self.app.new_game({"human_first": True})
        attack_action = self._find_action(state, "end_turn")

        self.app.human_action({"session_id": state["session_id"], "action": attack_action["action"]})

        with self.assertRaises(ApiError) as context:
            self.app.human_action({"session_id": state["session_id"], "action": attack_action["action"]})

        self.assertEqual(context.exception.code, "wrong_turn")

    def test_illegal_action_payload_is_reported(self) -> None:
        state = self.app.new_game({"human_first": True})
        illegal_action = {"type": "attack", "attack_index": 9, "label": "Impossible attack"}

        with self.assertRaises(ApiError) as context:
            self.app.human_action({"session_id": state["session_id"], "action": illegal_action})

        self.assertEqual(context.exception.code, "illegal_action")

    def test_ai_turn_records_learning_progress_when_a_game_finishes(self) -> None:
        self.app.learner.exploration_rate = 0.0
        state = self.app.new_game({"human_first": False})
        session = self.app.sessions.get(state["session_id"])
        self._move_card_to_energy_zone(session.state, 1, "water_energy")
        session.state.players[0].active.damage = 60
        session.state.players[0].bench.clear()

        updated = self.app.ai_turn({"session_id": state["session_id"]})

        self.assertEqual(updated["winner"], 1)
        self.assertEqual(updated["ai_learning"]["games_played"], 1)
        self.assertEqual(updated["ai_learning"]["wins"], 1)
        self.assertGreater(updated["ai_learning"]["recent_episode_rewards"][0], 0)
        self.assertEqual(len(updated["ai_turn_replay"]["steps"]), 1)
        self.assertEqual(updated["ai_turn_replay"]["steps"][0]["action"]["type"], "attack")
        self.assertDelayInRange(updated["ai_turn_replay"]["steps"][0]["delay_ms"])
        self.assertEqual(updated["ai_trainer"]["experience"], 26)
        self.assertEqual(updated["ai_trainer"]["level"], 1)
        attack_bias = self._find_action_bias(updated["ai_learning"]["action_biases"], "attack")
        self.assertEqual(attack_bias["samples"], 1)

    def test_trainer_progress_is_saved_and_loaded_from_text_file(self) -> None:
        self.app.learner.exploration_rate = 0.0
        state = self.app.new_game({"human_first": False, "trainer_id": "brock"})
        session = self.app.sessions.get(state["session_id"])
        self._move_card_to_energy_zone(session.state, 1, "water_energy")
        session.state.players[0].active.damage = 60
        session.state.players[0].bench.clear()

        self.app.ai_turn({"session_id": state["session_id"]})

        self.assertTrue(self.state_path.exists())
        saved_text = self.state_path.read_text(encoding="utf-8")
        self.assertIn('"version": 2', saved_text)
        self.assertIn('"id": "brock"', saved_text)
        self.assertIn('"my_first_battle"', saved_text)
        self.assertIn('"experience": 26', saved_text)
        self.assertIn('"feature_weights"', saved_text)
        self.assertNotIn('"standard"', saved_text)

        reloaded_app = TcgApplication(trainer_state_path=self.state_path)
        reloaded_state = reloaded_app.new_game({"human_first": True, "trainer_id": "brock"})

        self.assertEqual(reloaded_state["ai_trainer"]["experience"], 26)
        self.assertEqual(reloaded_state["ai_learning"]["games_played"], 1)
        self.assertEqual(reloaded_state["ai_learning"]["wins"], 1)

    def test_legacy_trainer_progress_migrates_into_my_first_battle_only(self) -> None:
        learner = RewardLearner()
        learner.record_step_reward(("action:attack", "card:wartortle"), "attack", 4.0)
        legacy_payload = {
            "version": 1,
            "trainers": [
                {
                    "id": "brock",
                    "name": "Brock",
                    "specialty": "Rock",
                    "experience": 125,
                    "learner": learner.export_state(),
                }
            ],
        }
        self.state_path.write_text(json.dumps(legacy_payload), encoding="utf-8")

        migrated_app = TcgApplication(trainer_state_path=self.state_path)
        state = migrated_app.new_game({"human_first": True, "trainer_id": "brock"})
        exported = migrated_app.trainers.export_state()
        brock_state = next(trainer for trainer in exported["trainers"] if trainer["id"] == "brock")

        self.assertEqual(state["game_mode"], "my_first_battle")
        self.assertEqual(state["ai_trainer"]["experience"], 125)
        self.assertEqual(state["ai_trainer"]["level"], 2)
        self.assertEqual(exported["version"], 2)
        self.assertIn("my_first_battle", brock_state["modes"])
        self.assertNotIn("standard", brock_state["modes"])
        self.assertEqual(brock_state["modes"]["my_first_battle"]["experience"], 125)
        self.assertEqual(state["ai_learning"]["tracked_feature_count"], 2)

    def test_ai_turn_returns_replay_snapshots_for_each_action_in_the_turn(self) -> None:
        self.app.learner.exploration_rate = 0.0
        state = self.app.new_game({"human_first": False})
        session = self.app.sessions.get(state["session_id"])
        player = session.state.players[1]
        water_energy_id = self._find_instance_id(session.state, 1, "water_energy")
        player.hand = [water_energy_id]
        player.energy_zone.clear()

        updated = self.app.ai_turn({"session_id": state["session_id"]})

        replay_steps = updated["ai_turn_replay"]["steps"]
        self.assertEqual([step["action"]["type"] for step in replay_steps], ["play_energy", "attack"])
        self.assertDelayInRange(replay_steps[0]["delay_ms"])
        self.assertDelayInRange(replay_steps[1]["delay_ms"])
        self.assertEqual(replay_steps[0]["state"]["players"][1]["energy_count"], 1)
        self.assertEqual(replay_steps[0]["state"]["current_player"], 1)
        self.assertEqual(replay_steps[1]["state"]["current_player"], 0)
        self.assertEqual(replay_steps[-1]["state"]["turn_number"], updated["turn_number"])
        self.assertEqual(replay_steps[-1]["state"]["players"][0]["active"]["damage"], 10)
        self.assertEqual(updated["players"][0]["active"]["damage"], 10)

    def test_ai_step_advances_the_ai_turn_one_action_at_a_time(self) -> None:
        self.app.learner.exploration_rate = 0.0
        state = self.app.new_game({"human_first": False})
        session = self.app.sessions.get(state["session_id"])
        player = session.state.players[1]
        water_energy_id = self._find_instance_id(session.state, 1, "water_energy")
        player.hand = [water_energy_id]
        player.energy_zone.clear()

        first_step = self.app.ai_step({"session_id": state["session_id"]})
        second_step = self.app.ai_step({"session_id": state["session_id"]})

        self.assertEqual(first_step["ai_step"]["action"]["type"], "play_energy")
        self.assertDelayInRange(first_step["ai_step"]["delay_ms"])
        self.assertEqual(first_step["players"][1]["energy_count"], 1)
        self.assertEqual(first_step["current_player"], 1)
        self.assertEqual(second_step["ai_step"]["action"]["type"], "attack")
        self.assertDelayInRange(second_step["ai_step"]["delay_ms"])
        self.assertEqual(second_step["players"][0]["active"]["damage"], 10)
        self.assertEqual(second_step["current_player"], 0)

    def test_standard_remote_ai_step_skips_fake_replay_delay(self) -> None:
        app = TcgApplication(
            trainer_state_path=self.state_path,
            standard_policy_state_path=self.policy_state_path,
            standard_policy_config=StandardPolicyConfig(
                remote_enabled=True,
                remote_url="http://127.0.0.1:8100/api/standard-ml/decision",
                remote_batch_eval_url="http://127.0.0.1:8100/api/standard-ml/batch-eval",
            ),
        )
        app.learner.exploration_rate = 0.0

        with patch(
            "backend.tcg_ai.server.urllib_request.urlopen",
            return_value=_FakeResponse(
                {
                    "ready": True,
                    "backend": "torch:cuda",
                    "model_loaded": True,
                    "checkpoint_path": "/models/champion.pt",
                }
            ),
        ):
            state = app.new_game(
                {
                    "game_mode": "standard",
                    "human_first": True,
                    "human_deck_id": "ampharos-ex-battle-deck",
                    "seed": 1,
                    "standard_ai_mode": "remote",
                }
            )

        session = app.sessions.get(state["session_id"])
        after_active = app.human_action(
            {
                "session_id": state["session_id"],
                "action": state["legal_actions"][0]["action"],
            }
        )
        end_setup_action = next(
            action for action in after_active["legal_actions"] if action["type"] == "end_setup"
        )
        after_setup = app.human_action(
            {
                "session_id": state["session_id"],
                "action": end_setup_action["action"],
            }
        )
        end_turn_action = next(
            action for action in after_setup["legal_actions"] if action["type"] == "end_turn"
        )
        app.human_action(
            {
                "session_id": state["session_id"],
                "action": end_turn_action["action"],
            }
        )

        energy_id = self._move_standard_named_card_to_hand(session.state, 1, "Basic Fighting Energy")
        self._set_standard_exact_hand(session.state, 1, [energy_id])

        with patch(
            "backend.tcg_ai.game_modes.standard.policy.RemoteStandardDecisionProvider.choose_action",
            side_effect=StandardRemoteDecisionError("worker unavailable"),
        ):
            ai_step = app.ai_step({"session_id": state["session_id"]})

        self.assertEqual(session.standard_ai_mode, "remote")
        self.assertEqual(session.ai_replay_delay_ms, 0)
        self.assertEqual(ai_step["standard_ai_mode"], "remote")
        self.assertEqual(ai_step["ai_step"]["action"]["type"], "play_energy")
        self.assertEqual(ai_step["ai_step"]["delay_ms"], 0)

    def test_standard_remote_ai_step_uses_full_state_decision_endpoint_without_batch_eval(self) -> None:
        from backend.tcg_ai.game_modes.standard.engine import action_id_for, list_legal_actions

        app = TcgApplication(
            trainer_state_path=self.state_path,
            standard_policy_state_path=self.policy_state_path,
            standard_policy_config=StandardPolicyConfig(
                remote_enabled=True,
                remote_url="http://127.0.0.1:8100/api/standard-ml/decision",
                remote_batch_eval_url="http://127.0.0.1:8100/api/standard-ml/batch-eval",
            ),
        )

        state, session = self._start_standard_remote_ai_turn(app)
        energy_id = self._move_standard_named_card_to_hand(session.state, 1, "Basic Fighting Energy")
        self._set_standard_exact_hand(session.state, 1, [energy_id])
        request_payloads: list[dict[str, object]] = []
        request_urls: list[str] = []

        def fake_remote_decision(request, timeout):
            del timeout
            if request.full_url.endswith("/batch-eval"):
                raise AssertionError("live remote turns should not call /batch-eval")
            request_urls.append(request.full_url)
            payload = json.loads((request.data or b"{}").decode("utf-8"))
            request_payloads.append(payload)
            self.assertEqual(payload["decision_type"], "turn_action")
            self.assertIn("state", payload)
            self.assertIn("search_config", payload)
            self.assertNotIn("legal_actions", payload)
            legal_actions = list_legal_actions(session.state, player_index=1)
            chosen_action = next(action for action in legal_actions if action["type"] == "play_energy")
            return _FakeResponse(
                {
                    "decision_id": payload["decision_id"],
                    "chosen_action_id": action_id_for(chosen_action),
                    "diagnostics": {"planner": "remote"},
                }
            )

        with patch(
            "backend.tcg_ai.game_modes.standard.policy.urllib_request.urlopen",
            side_effect=fake_remote_decision,
        ):
            ai_step = app.ai_step({"session_id": state["session_id"]})

        self.assertEqual(ai_step["ai_step"]["action"]["type"], "play_energy")
        self.assertEqual(request_urls, ["http://127.0.0.1:8100/api/standard-ml/decision"])
        self.assertIn("cards", request_payloads[0]["state"])

    def test_standard_remote_ai_turn_uses_full_state_decision_endpoint_without_batch_eval(self) -> None:
        from backend.tcg_ai.game_modes.standard.engine import action_id_for, list_legal_actions

        app = TcgApplication(
            trainer_state_path=self.state_path,
            standard_policy_state_path=self.policy_state_path,
            standard_policy_config=StandardPolicyConfig(
                remote_enabled=True,
                remote_url="http://127.0.0.1:8100/api/standard-ml/decision",
                remote_batch_eval_url="http://127.0.0.1:8100/api/standard-ml/batch-eval",
            ),
        )

        state, session = self._start_standard_remote_ai_turn(app)
        energy_id = self._move_standard_named_card_to_hand(session.state, 1, "Basic Fighting Energy")
        self._set_standard_exact_hand(session.state, 1, [energy_id])
        request_urls: list[str] = []

        def fake_remote_decision(request, timeout):
            del timeout
            if request.full_url.endswith("/batch-eval"):
                raise AssertionError("live remote turns should not call /batch-eval")
            request_urls.append(request.full_url)
            payload = json.loads((request.data or b"{}").decode("utf-8"))
            legal_actions = list_legal_actions(session.state, player_index=1)
            if any(action["type"] == "play_energy" for action in legal_actions):
                chosen_action = next(action for action in legal_actions if action["type"] == "play_energy")
            elif any(action["type"] == "attack" for action in legal_actions):
                chosen_action = next(action for action in legal_actions if action["type"] == "attack")
            else:
                chosen_action = legal_actions[0]
            return _FakeResponse(
                {
                    "decision_id": payload["decision_id"],
                    "chosen_action_id": action_id_for(chosen_action),
                    "diagnostics": {"planner": "remote"},
                }
            )

        with patch(
            "backend.tcg_ai.game_modes.standard.policy.urllib_request.urlopen",
            side_effect=fake_remote_decision,
        ):
            replay = app.ai_turn({"session_id": state["session_id"]})

        step_types = [step["action"]["type"] for step in replay["ai_turn_replay"]["steps"]]
        self.assertEqual(step_types, ["play_energy", "end_turn"])
        self.assertTrue(request_urls)
        self.assertTrue(all(url == "http://127.0.0.1:8100/api/standard-ml/decision" for url in request_urls))

    def test_standard_remote_ai_turn_replay_skips_fake_replay_delay(self) -> None:
        app = TcgApplication(
            trainer_state_path=self.state_path,
            standard_policy_state_path=self.policy_state_path,
            standard_policy_config=StandardPolicyConfig(
                remote_enabled=True,
                remote_url="http://127.0.0.1:8100/api/standard-ml/decision",
                remote_batch_eval_url="http://127.0.0.1:8100/api/standard-ml/batch-eval",
            ),
        )
        app.learner.exploration_rate = 0.0

        with patch(
            "backend.tcg_ai.server.urllib_request.urlopen",
            return_value=_FakeResponse(
                {
                    "ready": True,
                    "backend": "torch:cuda",
                    "model_loaded": True,
                    "checkpoint_path": "/models/champion.pt",
                }
            ),
        ):
            state = app.new_game(
                {
                    "game_mode": "standard",
                    "human_first": True,
                    "human_deck_id": "ampharos-ex-battle-deck",
                    "seed": 1,
                    "standard_ai_mode": "remote",
                }
            )

        session = app.sessions.get(state["session_id"])
        after_active = app.human_action(
            {
                "session_id": state["session_id"],
                "action": state["legal_actions"][0]["action"],
            }
        )
        end_setup_action = next(
            action for action in after_active["legal_actions"] if action["type"] == "end_setup"
        )
        after_setup = app.human_action(
            {
                "session_id": state["session_id"],
                "action": end_setup_action["action"],
            }
        )
        end_turn_action = next(
            action for action in after_setup["legal_actions"] if action["type"] == "end_turn"
        )
        app.human_action(
            {
                "session_id": state["session_id"],
                "action": end_turn_action["action"],
            }
        )

        energy_id = self._move_standard_named_card_to_hand(session.state, 1, "Basic Fighting Energy")
        self._set_standard_exact_hand(session.state, 1, [energy_id])

        with patch(
            "backend.tcg_ai.game_modes.standard.policy.RemoteStandardDecisionProvider.choose_action",
            side_effect=StandardRemoteDecisionError("worker unavailable"),
        ):
            replay = app.ai_turn({"session_id": state["session_id"]})

        self.assertEqual(session.standard_ai_mode, "remote")
        self.assertEqual(replay["standard_ai_mode"], "remote")
        self.assertEqual(replay["ai_turn_replay"]["step_delay_ms"], 0)
        self.assertTrue(replay["ai_turn_replay"]["steps"])
        self.assertTrue(all(step["delay_ms"] == 0 for step in replay["ai_turn_replay"]["steps"]))

    def test_promotion_actions_reference_a_benched_source_and_empty_active_target(self) -> None:
        state = self.app.new_game({"human_first": True})
        session = self.app.sessions.get(state["session_id"])
        self._move_card_to_bench(session.state, 0, "growlithe")
        session.state.players[0].active = None
        session.state.current_player = 0
        session.state.pending_promotion_for = 0

        snapshot = self.app.get_game(state["session_id"])
        promote_action = self._find_action(snapshot, "promote")

        self.assertEqual(promote_action["source"]["zone"], "bench")
        self.assertEqual(promote_action["source"]["name"], "Growlithe")
        self.assertEqual(promote_action["target"]["zone"], "active")
        self.assertIsNone(promote_action["target"]["instance_id"])

    def _find_action(self, state: dict, action_type: str) -> dict:
        for action in state["legal_actions"]:
            if action["type"] == action_type:
                return action
        self.fail(f"Could not find action of type {action_type}")

    def _find_action_bias(self, action_biases: list[dict], action_type: str) -> dict:
        for action_bias in action_biases:
            if action_bias["action_type"] == action_type:
                return action_bias
        self.fail(f"Could not find action bias for {action_type}")

    def _move_card_to_energy_zone(self, state, player_index: int, card_id: str) -> None:
        player = state.players[player_index]
        instance_id = self._find_instance_id(state, player_index, card_id)
        for zone_name in ("hand", "deck", "discard"):
            zone = getattr(player, zone_name)
            if instance_id in zone:
                zone.remove(instance_id)
        player.energy_zone.append(instance_id)

    def _move_card_to_bench(self, state, player_index: int, card_id: str) -> None:
        player = state.players[player_index]
        instance_id = self._find_instance_id(state, player_index, card_id)
        for zone_name in ("hand", "deck", "discard", "energy_zone"):
            zone = getattr(player, zone_name)
            if instance_id in zone:
                zone.remove(instance_id)
        player.bench.append(PokemonInPlay(stack=[instance_id]))

    def _find_instance_id(self, state, player_index: int, card_id: str) -> str:
        player = state.players[player_index]
        for zone_name in ("hand", "deck", "discard", "energy_zone"):
            zone = getattr(player, zone_name)
            for instance_id in zone:
                if state.cards[instance_id].card_id == card_id:
                    return instance_id
        self.fail(f"Could not find instance of {card_id} for player {player_index}")

    def _start_standard_remote_ai_turn(self, app: TcgApplication):
        with patch(
            "backend.tcg_ai.server.urllib_request.urlopen",
            return_value=_FakeResponse(
                {
                    "ready": True,
                    "backend": "torch:cuda",
                    "model_loaded": True,
                    "checkpoint_path": "/models/champion.pt",
                }
            ),
        ):
            state = app.new_game(
                {
                    "game_mode": "standard",
                    "human_first": True,
                    "human_deck_id": "ampharos-ex-battle-deck",
                    "seed": 1,
                    "standard_ai_mode": "remote",
                }
            )
        session = app.sessions.get(state["session_id"])
        after_active = app.human_action(
            {
                "session_id": state["session_id"],
                "action": state["legal_actions"][0]["action"],
            }
        )
        end_setup_action = next(
            action for action in after_active["legal_actions"] if action["type"] == "end_setup"
        )
        after_setup = app.human_action(
            {
                "session_id": state["session_id"],
                "action": end_setup_action["action"],
            }
        )
        end_turn_action = next(
            action for action in after_setup["legal_actions"] if action["type"] == "end_turn"
        )
        app.human_action(
            {
                "session_id": state["session_id"],
                "action": end_turn_action["action"],
            }
        )
        return state, session

    def assertDelayInRange(self, delay_ms: int) -> None:
        self.assertGreaterEqual(delay_ms, AI_ACTION_DELAY_MIN_MS)
        self.assertLessEqual(delay_ms, AI_ACTION_DELAY_MAX_MS)


class _FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


if __name__ == "__main__":
    unittest.main()
