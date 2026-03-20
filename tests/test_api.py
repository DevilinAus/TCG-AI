from __future__ import annotations

import unittest
from unittest.mock import patch

from backend.tcg_ai.server import (
    AI_ACTION_DELAY_MAX_MS,
    AI_ACTION_DELAY_MIN_MS,
    ApiError,
    TcgApplication,
)
from backend.tcg_ai.models import PokemonInPlay


class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = TcgApplication()

    def test_new_game_returns_a_session_id_and_display_ready_state(self) -> None:
        state = self.app.new_game({"human_first": True})

        self.assertIsInstance(state["session_id"], str)
        self.assertEqual(state["current_player"], 0)
        self.assertEqual(state["players"][0]["active"]["card_id"], "charmander")
        self.assertEqual(state["players"][0]["energy_count"], 0)
        self.assertTrue(state["players"][0]["hand"][0]["image_url"].startswith("/assets/cards/"))
        self.assertEqual(state["ai_learning"]["games_played"], 0)
        self.assertEqual(state["ai_trainer"]["id"], "brock")
        self.assertEqual(state["ai_trainer"]["level"], 1)
        self.assertEqual(len(state["available_trainers"]), 8)

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
        self.assertEqual(unchanged_second["current_player"], 0)
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

    def assertDelayInRange(self, delay_ms: int) -> None:
        self.assertGreaterEqual(delay_ms, AI_ACTION_DELAY_MIN_MS)
        self.assertLessEqual(delay_ms, AI_ACTION_DELAY_MAX_MS)


if __name__ == "__main__":
    unittest.main()
