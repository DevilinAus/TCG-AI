from __future__ import annotations

from unittest.mock import patch
import unittest

from backend.tcg_ai.engine import apply_action, card_definition, create_game, list_legal_actions
from backend.tcg_ai.models import PokemonInPlay


class EngineTests(unittest.TestCase):
    def test_new_game_sets_up_the_expected_opening_state(self) -> None:
        state = create_game(seed=7, human_first=True)

        self.assertEqual(card_definition(state, state.players[0].active.stack[-1]).name, "Charmander")
        self.assertEqual(card_definition(state, state.players[1].active.stack[-1]).name, "Squirtle")
        self.assertEqual(len(state.players[0].energy_zone), 0)
        self.assertEqual(len(state.players[1].energy_zone), 0)
        self.assertEqual(len(state.players[0].hand), 4)
        self.assertEqual(len(state.players[1].hand), 3)

    def test_evolution_is_allowed_on_the_first_turn(self) -> None:
        state = create_game(seed=11, human_first=True)
        self._move_card_to_hand(state, 0, "charmeleon")

        action = self._find_action(state, "evolve", target="active")
        apply_action(state, action)

        self.assertEqual(card_definition(state, state.players[0].active.stack[-1]).name, "Charmeleon")

    def test_potion_heals_thirty_damage(self) -> None:
        state = create_game(seed=13, human_first=True)
        self._move_card_to_hand(state, 0, "potion")
        state.players[0].active.damage = 40

        action = self._find_action(state, "play_potion", target="active")
        apply_action(state, action)

        self.assertEqual(state.players[0].active.damage, 10)
        self.assertEqual(card_definition(state, state.players[0].discard[-1]).name, "Potion")

    def test_knock_out_takes_a_prize_and_ends_the_game_if_no_bench_exists(self) -> None:
        state = create_game(seed=17, human_first=True)
        self._move_card_to_energy_zone(state, 0, "fire_energy")
        state.players[1].active.damage = 60
        state.players[1].bench.clear()

        action = self._find_action(state, "attack", attack_index=0)
        apply_action(state, action)

        self.assertEqual(state.players[0].prize_tokens_remaining, 2)
        self.assertEqual(state.winner, 0)

    def test_switch_swaps_the_active_with_the_selected_bench_pokemon(self) -> None:
        state = create_game(seed=19, human_first=True)
        self._move_card_to_hand(state, 0, "switch")
        self._bench_basic_via_action(state, 0, "magmar")

        action = self._find_action(state, "play_switch", bench_index=0)
        apply_action(state, action)

        self.assertEqual(card_definition(state, state.players[0].active.stack[-1]).name, "Magmar")
        self.assertEqual(card_definition(state, state.players[0].bench[0].stack[-1]).name, "Charmander")
        self.assertEqual(card_definition(state, state.players[0].discard[-1]).name, "Switch")

    def test_knock_out_requires_a_promotion_and_then_starts_the_defending_turn(self) -> None:
        state = create_game(seed=23, human_first=True)
        self._move_card_to_energy_zone(state, 0, "fire_energy")
        self._bench_basic_directly(state, 1, "lapras")
        state.players[1].active.damage = 60

        action = self._find_action(state, "attack", attack_index=0)
        apply_action(state, action)

        promote_actions = self._find_actions(state, "promote")
        self.assertEqual(state.current_player, 1)
        self.assertEqual(state.pending_promotion_for, 1)
        self.assertEqual(len(promote_actions), 1)

        apply_action(state, promote_actions[0])

        self.assertIsNone(state.pending_promotion_for)
        self.assertEqual(state.current_player, 1)
        self.assertEqual(state.turn_number, 2)
        self.assertEqual(card_definition(state, state.players[1].active.stack[-1]).name, "Lapras")

    def test_coin_flip_fail_attack_can_do_zero_damage(self) -> None:
        state = create_game(seed=29, human_first=False)
        self._set_active_pokemon(state, 1, ["gyarados"])
        self._move_card_to_energy_zone(state, 1, "water_energy")
        self._move_card_to_energy_zone(state, 1, "water_energy")
        self._move_card_to_energy_zone(state, 1, "water_energy")

        action = self._find_action(state, "attack", attack_index=0)
        with patch.object(state.rng, "choice", return_value="tails"):
            apply_action(state, action)

        self.assertEqual(state.players[0].active.damage, 0)
        self.assertEqual(state.current_player, 0)

    def test_bench_limit_blocks_benching_a_fourth_basic(self) -> None:
        state = create_game(seed=31, human_first=True)
        for card_id in ("magmar", "vulpix", "growlithe"):
            self._bench_basic_via_action(state, 0, card_id)

        self._move_card_to_hand(state, 0, "charmander")
        bench_actions = self._find_actions(state, "bench_basic")

        self.assertEqual(len(state.players[0].bench), 3)
        self.assertEqual(bench_actions, [])

    def test_potion_can_target_multiple_damaged_pokemon(self) -> None:
        state = create_game(seed=37, human_first=True)
        self._move_card_to_hand(state, 0, "potion")
        self._bench_basic_via_action(state, 0, "magmar")
        state.players[0].active.damage = 20
        state.players[0].bench[0].damage = 10

        potion_actions = self._find_actions(state, "play_potion")

        self.assertEqual(len(potion_actions), 2)
        self.assertEqual({action["target"] for action in potion_actions}, {"active", "bench:0"})

    def test_evolution_can_target_active_or_matching_benched_pokemon(self) -> None:
        state = create_game(seed=41, human_first=True)
        self._move_card_to_hand(state, 0, "charmander")
        self._move_card_to_hand(state, 0, "charmeleon")

        bench_action = self._find_action_by_card(state, "bench_basic", "charmander")
        apply_action(state, bench_action)

        evolve_actions = self._find_actions(state, "evolve")

        self.assertEqual(len(evolve_actions), 2)
        self.assertEqual({action["target"] for action in evolve_actions}, {"active", "bench:0"})

    def _find_action(self, state, action_type: str, **criteria):
        for action in list_legal_actions(state):
            if action["type"] != action_type:
                continue
            if all(action.get(key) == value for key, value in criteria.items()):
                return action
        self.fail(f"Could not find action {action_type} with criteria {criteria}")

    def _find_actions(self, state, action_type: str):
        return [action for action in list_legal_actions(state) if action["type"] == action_type]

    def _find_action_by_card(self, state, action_type: str, card_id: str):
        for action in list_legal_actions(state):
            if action["type"] != action_type:
                continue
            hand_card_id = action.get("hand_card_id")
            if hand_card_id and state.cards[hand_card_id].card_id == card_id:
                return action
        self.fail(f"Could not find action {action_type} for card {card_id}")

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

    def _find_instance_id(self, state, player_index: int, card_id: str) -> str:
        player = state.players[player_index]
        for zone_name in ("hand", "deck", "discard", "energy_zone"):
            zone = getattr(player, zone_name)
            for instance_id in zone:
                if state.cards[instance_id].card_id == card_id:
                    return instance_id
        self.fail(f"Could not find instance of {card_id} for player {player_index}")

    def _bench_basic_via_action(self, state, player_index: int, card_id: str) -> None:
        self._move_card_to_hand(state, player_index, card_id)
        action = self._find_action_by_card(state, "bench_basic", card_id)
        apply_action(state, action)

    def _bench_basic_directly(self, state, player_index: int, card_id: str) -> None:
        player = state.players[player_index]
        instance_id = self._find_instance_id(state, player_index, card_id)
        for zone_name in ("hand", "deck", "discard"):
            zone = getattr(player, zone_name)
            if instance_id in zone:
                zone.remove(instance_id)
        player.bench.append(PokemonInPlay(stack=[instance_id]))

    def _move_card_to_energy_zone(self, state, player_index: int, card_id: str) -> None:
        player = state.players[player_index]
        instance_id = self._find_instance_id(state, player_index, card_id)
        for zone_name in ("hand", "deck", "discard"):
            zone = getattr(player, zone_name)
            if instance_id in zone:
                zone.remove(instance_id)
        player.energy_zone.append(instance_id)

    def _set_active_pokemon(self, state, player_index: int, card_ids: list[str]) -> None:
        player = state.players[player_index]
        player.active = PokemonInPlay(stack=[])
        for card_id in card_ids:
            instance_id = self._find_instance_id(state, player_index, card_id)
            for zone_name in ("hand", "deck", "discard", "energy_zone"):
                zone = getattr(player, zone_name)
                if instance_id in zone:
                    zone.remove(instance_id)
            player.active.stack.append(instance_id)


if __name__ == "__main__":
    unittest.main()
