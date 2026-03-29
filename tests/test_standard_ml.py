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
from backend.tcg_ai.game_modes.standard.ml.neural_policy import (
    ActionConditionedPolicyValueNet,
    PolicyValueBackend,
    PolicyValueBackendStatus,
    torch,
)
from backend.tcg_ai.game_modes.standard.ml.oracle import (
    BackendPolicyValueOracle,
    PolicyValueRequest,
)
from backend.tcg_ai.game_modes.standard.ml.planner import PlannerConfig, StandardTurnPlanner
from backend.tcg_ai.game_modes.standard.ml.self_play import (
    SelfPlayConfig,
    _discounted_prize_progress_target,
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
        self.assertEqual(
            {entry["action_id"] for entry in decision["diagnostics"]["policy_target_scores"]},
            legal_action_ids,
        )

    def test_planner_batches_oracle_requests_for_sibling_states(self) -> None:
        class RecordingOracle:
            def __init__(self, checkpoint_path: Path) -> None:
                self.batch_sizes: list[int] = []
                self.delegate = BackendPolicyValueOracle(
                    backend=PolicyValueBackend(checkpoint_path=checkpoint_path)
                )

            def evaluate_batch(self, requests):
                self.batch_sizes.append(len(requests))
                return self.delegate.evaluate_batch(requests)

        state = create_game(seed=1, human_deck_id="ampharos-ex-battle-deck", ai_name="Brock")
        oracle = RecordingOracle(Path(self.temp_dir.name) / "missing.pt")
        planner = StandardTurnPlanner(config=PlannerConfig(max_depth=1, beam_width=2), oracle=oracle)

        planner.plan(state, acting_player_index=0)

        self.assertGreater(len(list_legal_actions(state, player_index=0)), 1)
        self.assertGreater(max(oracle.batch_sizes), 1)
        self.assertGreater(sum(oracle.batch_sizes), len(oracle.batch_sizes))

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

    def test_service_full_state_path_uses_backend_oracle(self) -> None:
        class RecordingBackend:
            def __init__(self) -> None:
                self.calls = 0
                self.status = PolicyValueBackendStatus(
                    backend="recording",
                    model_loaded=True,
                    checkpoint_path="/tmp/fake.pt",
                )

            def evaluate_batch(self, evaluations):
                self.calls += 1
                responses = []
                for evaluation in evaluations:
                    legal_actions = list(evaluation.get("legal_actions", []))
                    action_ids = [str(action.get("action_id", "")) for action in legal_actions]
                    uniform = round(1.0 / len(action_ids), 6) if action_ids else 0.0
                    responses.append(
                        {
                            "value": 0.0,
                            "action_priors": {action_id: uniform for action_id in action_ids},
                            "diagnostics": {"backend": "recording", "model_loaded": True},
                        }
                    )
                return responses

        state = create_game(seed=1, human_deck_id="ampharos-ex-battle-deck", ai_name="Brock")
        self.service.policy_backend = RecordingBackend()
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
        self.assertGreater(self.service.policy_backend.calls, 0)
        self.assertIn(response["chosen_action_id"], legal_action_ids)

    def test_service_full_state_decision_matches_local_backend_planner(self) -> None:
        class DeterministicBackend:
            status = PolicyValueBackendStatus(
                backend="deterministic",
                model_loaded=True,
                checkpoint_path="/tmp/fake.pt",
            )

            @staticmethod
            def evaluate_batch(evaluations):
                responses = []
                for evaluation in evaluations:
                    legal_actions = list(evaluation.get("legal_actions", []))
                    action_ids = [str(action.get("action_id", "")) for action in legal_actions]
                    priors = {action_id: 0.0 for action_id in action_ids}
                    if action_ids:
                        priors[min(action_ids)] = 1.0
                    responses.append(
                        {
                            "value": 0.0,
                            "action_priors": priors,
                            "diagnostics": {"backend": "deterministic", "model_loaded": True},
                        }
                    )
                return responses

        backend = DeterministicBackend()
        self.service.policy_backend = backend
        state = create_game(seed=1, human_deck_id="ampharos-ex-battle-deck", ai_name="Brock")
        config = PlannerConfig(max_depth=2, beam_width=3, opponent_branch_width=2)
        local_planner = StandardTurnPlanner(
            config=config,
            oracle=BackendPolicyValueOracle(backend=backend),
        )

        local_decision = local_planner.plan(state, acting_player_index=0)
        response = self.service.choose_action(
            {
                "schema_version": 2,
                "decision_id": "session-a:turn_action",
                "decision_type": "turn_action",
                "acting_player_index": 0,
                "search_config": {
                    "max_depth": config.max_depth,
                    "beam_width": config.beam_width,
                    "opponent_branch_width": config.opponent_branch_width,
                    "include_opponent_turn": config.include_opponent_turn,
                },
                "state": serialize_state(state),
            }
        )

        self.assertEqual(response["chosen_action_id"], local_decision["chosen_action_id"])

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

    @unittest.skipIf(torch is None, "PyTorch is not available.")
    def test_model_backed_batch_eval_matches_single_item_path(self) -> None:
        backend = PolicyValueBackend(checkpoint_path=Path(self.temp_dir.name) / "missing.pt")
        torch.manual_seed(7)
        backend._model = ActionConditionedPolicyValueNet(
            state_dim=backend._state_dim,
            action_dim=backend._action_dim,
        )
        backend._model.eval()
        backend._status = PolicyValueBackendStatus(
            backend="torch:cpu",
            model_loaded=True,
            checkpoint_path=None,
        )

        state = create_game(seed=1, human_deck_id="ampharos-ex-battle-deck", ai_name="Brock")
        legal_actions = list_legal_actions(state, player_index=0)
        evaluations = [
            {
                "acting_player_index": 0,
                "root_player_index": 0,
                "belief_state": serialize_knowledge_state(state, perspective_player_index=0),
                "legal_actions": serialize_knowledge_actions(
                    state,
                    acting_player_index=0,
                    legal_actions=legal_actions,
                ),
            },
            {
                "acting_player_index": 0,
                "root_player_index": 0,
                "belief_state": serialize_knowledge_state(state, perspective_player_index=0),
                "legal_actions": serialize_knowledge_actions(
                    state,
                    acting_player_index=0,
                    legal_actions=legal_actions[:2],
                ),
            },
            {
                "acting_player_index": 0,
                "root_player_index": 0,
                "belief_state": serialize_knowledge_state(state, perspective_player_index=0),
                "legal_actions": [],
            },
        ]

        expected = [backend._model_evaluation(evaluation) for evaluation in evaluations]
        actual = backend.evaluate_batch(evaluations)

        self.assertEqual(len(actual), len(expected))
        for expected_row, actual_row in zip(expected, actual):
            self.assertAlmostEqual(actual_row["value"], expected_row["value"], places=6)
            self.assertEqual(set(actual_row["action_priors"]), set(expected_row["action_priors"]))
            for action_id, expected_prior in expected_row["action_priors"].items():
                self.assertAlmostEqual(actual_row["action_priors"][action_id], expected_prior, places=6)

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
        self.assertTrue(
            all(
                isinstance(record["value_target"], float) and -100.0 <= record["value_target"] <= 100.0
                for record in records
            )
        )
        self.assertTrue(all(record["terminal_outcome_target"] in {-100.0, 100.0} for record in records))
        self.assertTrue(
            all(
                isinstance(record["discounted_prize_progress_target"], float)
                for record in records
            )
        )
        self.assertTrue(all("transition_summary" in record for record in records))
        self.assertTrue(
            all(
                record["chosen_action_id"] in {action["action_id"] for action in record["legal_actions"]}
                for record in records
            )
        )
        self.assertTrue(
            all(
                abs(sum(record["policy_target_probs"].values()) - 1.0) < 1e-5
                for record in records
            )
        )
        self.assertTrue(
            all(
                record["chosen_action_id"] in record["policy_target_probs"]
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

    def test_discounted_prize_progress_target_is_from_perspective_player(self) -> None:
        samples = [
            {
                "acting_player_index": 0,
                "transition_summary": {
                    "prizes_taken_by_actor": 1,
                    "prizes_lost_by_actor": 0,
                },
            },
            {
                "acting_player_index": 1,
                "transition_summary": {
                    "prizes_taken_by_actor": 2,
                    "prizes_lost_by_actor": 0,
                },
            },
        ]

        player0_target = _discounted_prize_progress_target(
            samples,
            start_index=0,
            perspective_player_index=0,
        )
        player1_target = _discounted_prize_progress_target(
            samples,
            start_index=0,
            perspective_player_index=1,
        )

        self.assertLess(player0_target, 0.0)
        self.assertGreater(player1_target, 0.0)
        self.assertAlmostEqual(player0_target, -player1_target, places=6)

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
