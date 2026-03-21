from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from backend.tcg_ai.game_modes.standard.decision_payload import build_decision_request
from backend.tcg_ai.game_modes.standard.engine import action_id_for, create_game, list_legal_actions
from backend.tcg_ai.game_modes.standard.ml.canonical_state import deserialize_state, serialize_state
from backend.tcg_ai.game_modes.standard.ml.experience import StandardExperienceStore
from backend.tcg_ai.game_modes.standard.ml.planner import PlannerConfig, StandardTurnPlanner
from backend.tcg_ai.game_modes.standard.ml.service import StandardMlService


class StandardMlTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.experience_store = StandardExperienceStore(base_dir=Path(self.temp_dir.name))
        self.service = StandardMlService(experience_store=self.experience_store)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_canonical_round_trip_preserves_legal_actions(self) -> None:
        state = create_game(seed=1, human_deck_id="ampharos-ex-battle-deck", ai_name="Brock")

        rebuilt = deserialize_state(serialize_state(state))

        self.assertEqual(
            [action_id_for(action) for action in list_legal_actions(state, player_index=0)],
            [action_id_for(action) for action in list_legal_actions(rebuilt, player_index=0)],
        )
        self.assertEqual(rebuilt.seed, state.seed)
        self.assertEqual(rebuilt.setup_phase, state.setup_phase)

    def test_planner_returns_legal_action_for_full_state_payload(self) -> None:
        state = create_game(seed=1, human_deck_id="ampharos-ex-battle-deck", ai_name="Brock")
        planner = StandardTurnPlanner(config=PlannerConfig(max_depth=2, beam_width=3))

        decision = planner.plan(state, acting_player_index=0)

        legal_action_ids = {action_id_for(action) for action in list_legal_actions(state, player_index=0)}
        self.assertIn(decision["chosen_action_id"], legal_action_ids)
        self.assertIn("top_candidates", decision["diagnostics"])

    def test_service_handles_full_state_payload(self) -> None:
        state = create_game(seed=1, human_deck_id="ampharos-ex-battle-deck", ai_name="Brock")
        payload = {
            "schema_version": 2,
            "decision_id": "session-a:turn_action",
            "decision_type": "turn_action",
            "acting_player_index": 0,
            "search_config": {
                "max_depth": 2,
                "beam_width": 3,
                "opponent_branch_width": 2,
            },
            "state": serialize_state(state),
        }

        response = self.service.choose_action(payload)

        legal_action_ids = {action_id_for(action) for action in list_legal_actions(state, player_index=0)}
        self.assertIn(response["chosen_action_id"], legal_action_ids)
        self.assertEqual(response["decision_id"], "session-a:turn_action")
        self.assertTrue((Path(self.temp_dir.name) / "decisions.jsonl").exists())

    def test_service_handles_legacy_payload(self) -> None:
        state = create_game(seed=1, human_deck_id="ampharos-ex-battle-deck", ai_name="Brock")
        actions = list_legal_actions(state, player_index=1)
        payload = build_decision_request(
            state,
            session_id="session-a",
            decision_id="session-a:opening_active",
            decision_type="opening_active",
            acting_player_index=1,
            ai_trainer_id="brock",
            ai_deck_id="lucario-ex-battle-deck",
            legal_actions=actions,
        )

        response = self.service.choose_action(payload)

        self.assertIn(
            response["chosen_action_id"],
            {action["action_id"] for action in payload["legal_actions"]},
        )
        self.assertEqual(response["decision_id"], "session-a:opening_active")
        self.assertIn("selection_mode", response["diagnostics"])

    def test_service_records_outcome_payload(self) -> None:
        response = self.service.record_outcome(
            {
                "session_id": "session-a",
                "winner": 1,
                "terminal_reward": 1.0,
            }
        )

        self.assertTrue(response["ok"])
        self.assertTrue((Path(self.temp_dir.name) / "outcomes.jsonl").exists())
