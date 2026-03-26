from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from backend.tcg_ai.game_modes.standard.decision_payload import build_decision_request
from backend.tcg_ai.game_modes.standard.engine import action_id_for, apply_action, create_game, list_legal_actions
from backend.tcg_ai.game_modes.standard.ml.canonical_state import deserialize_state, serialize_state
from backend.tcg_ai.game_modes.standard.ml.experience import StandardExperienceStore
from backend.tcg_ai.game_modes.standard.ml.knowledge_state import (
    serialize_knowledge_actions,
    serialize_knowledge_state,
)
from backend.tcg_ai.game_modes.standard.ml.neural_policy import PolicyValueBackend
from backend.tcg_ai.game_modes.standard.ml.oracle import (
    BackendPolicyValueOracle,
    PolicyValueRequest,
)
from backend.tcg_ai.game_modes.standard.ml.planner import PlannerConfig, StandardTurnPlanner
from backend.tcg_ai.game_modes.standard.ml.self_play import (
    SelfPlayConfig,
    _build_player_planners,
    _should_record_decision,
    play_self_play_game,
)
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

    def test_knowledge_state_hides_opponent_hand_and_prize_positions(self) -> None:
        state = create_game(seed=1, human_deck_id="ampharos-ex-battle-deck", ai_name="Brock")

        knowledge_state = serialize_knowledge_state(state, perspective_player_index=0)

        self.assertEqual(knowledge_state["players"][0]["hand_count"], 7)
        self.assertNotIn("hand", knowledge_state["players"][1])
        self.assertEqual(knowledge_state["players"][0]["known_prize_cards_unordered"], [])
        self.assertNotIn("prizes", knowledge_state["players"][0])
        self.assertEqual(knowledge_state["players"][1]["prize_count"], 6)

    def test_knowledge_state_reveals_unordered_known_prizes_after_full_deck_search(self) -> None:
        state = create_game(seed=1, human_deck_id="ampharos-ex-battle-deck", ai_name="Brock")
        self._finish_opening_setup(state)
        state.turn_number = 2
        jacq_id = self._move_named_card_to_hand(state, 0, "Jacq")
        state.players[0].hand = [jacq_id]

        action = next(
            action
            for action in list_legal_actions(state)
            if action["type"] == "play_supporter"
        )
        apply_action(state, action)

        knowledge_state = serialize_knowledge_state(state, perspective_player_index=0)

        self.assertTrue(state.players[0].deck_inspected_this_game)
        self.assertEqual(len(knowledge_state["players"][0]["known_prize_cards_unordered"]), 6)
        self.assertNotIn("position", knowledge_state["players"][0]["known_prize_cards_unordered"][0])

    def test_service_evaluates_belief_batch_payloads(self) -> None:
        state = create_game(seed=1, human_deck_id="ampharos-ex-battle-deck", ai_name="Brock")
        legal_actions = list_legal_actions(state, player_index=0)
        payload = {
            "schema_version": 1,
            "evaluations": [
                {
                    "acting_player_index": 0,
                    "root_player_index": 0,
                    "belief_state": serialize_knowledge_state(state, perspective_player_index=0),
                    "legal_actions": serialize_knowledge_actions(
                        state,
                        acting_player_index=0,
                        legal_actions=legal_actions,
                    ),
                }
            ],
        }

        response = self.service.evaluate_batch(payload)

        self.assertEqual(len(response["evaluations"]), 1)
        self.assertIn("value", response["evaluations"][0])
        self.assertEqual(
            set(response["evaluations"][0]["action_priors"]),
            {action_id_for(action) for action in legal_actions},
        )

    def test_backend_policy_value_oracle_evaluates_in_process_requests(self) -> None:
        state = create_game(seed=1, human_deck_id="ampharos-ex-battle-deck", ai_name="Brock")
        legal_actions = list_legal_actions(state, player_index=0)
        backend = PolicyValueBackend(checkpoint_path=Path(self.temp_dir.name) / "missing.pt")
        oracle = BackendPolicyValueOracle(backend=backend)

        results = oracle.evaluate_batch(
            [
                PolicyValueRequest(
                    state=state,
                    acting_player_index=0,
                    root_player_index=0,
                    legal_actions=legal_actions,
                )
            ]
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(set(results[0].action_priors), {action_id_for(action) for action in legal_actions})
        self.assertEqual(results[0].diagnostics["backend"], "heuristic")

    def test_self_play_game_produces_completed_training_records(self) -> None:
        summary, records = play_self_play_game(
            game_id="self-play-test",
            seed=5,
            config=SelfPlayConfig(
                player0_deck_id="ampharos-ex-battle-deck",
                planner_config=PlannerConfig(max_depth=1, beam_width=2, opponent_branch_width=1),
                max_actions_per_game=120,
                include_setup_decisions=True,
                record_forced_actions=True,
            ),
        )

        self.assertFalse(summary.truncated)
        self.assertIn(summary.winner, {0, 1})
        self.assertEqual(summary.decision_samples, len(records))
        self.assertGreater(len(records), 0)
        self.assertTrue(all(record["winner"] == summary.winner for record in records))
        self.assertTrue(all(record["value_target"] in {-100.0, 100.0} for record in records))
        self.assertTrue(
            all(
                record["chosen_action_id"] in {action["action_id"] for action in record["legal_actions"]}
                for record in records
            )
        )

    def test_self_play_builds_player_specific_planners_and_honors_collect_flag(self) -> None:
        class TaggedOracle:
            def __init__(self, tag: str) -> None:
                self.tag = tag

        default_oracle = TaggedOracle("default")
        player0_oracle = TaggedOracle("player0")
        player1_oracle = TaggedOracle("player1")

        planners = _build_player_planners(
            planner_config=PlannerConfig(max_depth=1, beam_width=1, opponent_branch_width=1),
            default_oracle=default_oracle,
            oracle_by_player={
                0: player0_oracle,
                1: player1_oracle,
            },
        )

        self.assertIs(planners[0].oracle, player0_oracle)
        self.assertIs(planners[1].oracle, player1_oracle)
        self.assertFalse(
            _should_record_decision(
                legal_actions=[{"type": "end_turn"}],
                setup_phase="opening",
                config=SelfPlayConfig(
                    collect_training_records=False,
                    include_setup_decisions=True,
                    record_forced_actions=True,
                ),
            )
        )

    def _finish_opening_setup(self, state) -> None:
        active_action = next(
            action for action in list_legal_actions(state) if action["type"] == "play_basic_to_active"
        )
        apply_action(state, active_action)
        end_setup_action = next(
            action for action in list_legal_actions(state) if action["type"] == "end_setup"
        )
        apply_action(state, end_setup_action)

    def _move_named_card_to_hand(self, state, player_index: int, card_name: str) -> str:
        player = state.players[player_index]
        for zone_name in ("hand", "deck", "discard", "prizes"):
            zone = getattr(player, zone_name)
            for instance_id in list(zone):
                from backend.tcg_ai.game_modes.standard.engine import card_definition

                if card_definition(state, instance_id).name != card_name:
                    continue
                if zone_name != "hand":
                    zone.remove(instance_id)
                    player.hand.append(instance_id)
                return instance_id
        self.fail(f"Could not find {card_name} for player {player_index}")
