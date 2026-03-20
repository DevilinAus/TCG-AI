from __future__ import annotations

import unittest

from backend.tcg_ai.bot import choose_action
from backend.tcg_ai.engine import create_game
from backend.tcg_ai.learning import RewardLearner


class BotTests(unittest.TestCase):
    def test_ai_can_choose_setup_action_over_an_available_attack(self) -> None:
        state = create_game(seed=43, human_first=False)
        self._move_card_to_hand(state, 1, "wartortle")
        self._move_card_to_energy_zone(state, 1, "water_energy")

        learner = RewardLearner(exploration_rate=0.0, min_exploration_rate=0.0)
        action = choose_action(state, 1, learner=learner)

        self.assertIsNotNone(action)
        self.assertEqual(action["type"], "evolve")
        self.assertEqual(action["target"], "active")

    def _move_card_to_hand(self, state, player_index: int, card_id: str) -> str:
        player = state.players[player_index]
        for zone_name in ("hand", "deck", "discard"):
            zone = getattr(player, zone_name)
            for instance_id in list(zone):
                if state.cards[instance_id].card_id == card_id:
                    if zone_name != "hand":
                        zone.remove(instance_id)
                        player.hand.append(instance_id)
                    return instance_id
        self.fail(f"Could not find {card_id} for player {player_index}")

    def _move_card_to_energy_zone(self, state, player_index: int, card_id: str) -> None:
        player = state.players[player_index]
        instance_id = self._find_instance_id(state, player_index, card_id)
        for zone_name in ("hand", "deck", "discard"):
            zone = getattr(player, zone_name)
            if instance_id in zone:
                zone.remove(instance_id)
        player.energy_zone.append(instance_id)

    def _find_instance_id(self, state, player_index: int, card_id: str) -> str:
        player = state.players[player_index]
        for zone_name in ("hand", "deck", "discard", "energy_zone"):
            zone = getattr(player, zone_name)
            for instance_id in zone:
                if state.cards[instance_id].card_id == card_id:
                    return instance_id
        self.fail(f"Could not find instance of {card_id} for player {player_index}")


if __name__ == "__main__":
    unittest.main()
