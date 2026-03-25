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
        state.turn_number = 2
        self._move_named_card_to_hand(state, 0, "Nemona")
        self._move_named_card_to_hand(state, 0, "Youngster")
        self._move_named_card_to_hand(state, 0, "Jacq")

        supporter_actions = [action for action in list_legal_actions(state) if action["type"] == "play_supporter"]

        self.assertEqual(
            sorted(card_definition(state, action["hand_card_id"]).name for action in supporter_actions),
            ["Nemona", "Youngster"],
        )

    def test_starting_player_cannot_play_supporters_on_turn_one(self) -> None:
        state = create_game(seed=1, human_deck_id="ampharos-ex-battle-deck")
        self._finish_opening_setup(state)

        supporter_actions = [action for action in list_legal_actions(state) if action["type"] == "play_supporter"]

        self.assertEqual(supporter_actions, [])

    def test_second_player_can_play_supporters_on_their_first_turn(self) -> None:
        state = create_game(seed=1, human_deck_id="ampharos-ex-battle-deck")
        self._finish_opening_setup(state)
        end_turn = next(action for action in list_legal_actions(state) if action["type"] == "end_turn")
        apply_action(state, end_turn)
        self._set_first_basic_as_active(state, 1)
        nemona_id = self._move_named_card_to_hand(state, 1, "Nemona")
        self._set_exact_hand(state, 1, [nemona_id])

        supporter_actions = [action for action in list_legal_actions(state, player_index=1) if action["type"] == "play_supporter"]

        self.assertEqual(len(supporter_actions), 1)
        self.assertEqual(card_definition(state, supporter_actions[0]["hand_card_id"]).name, "Nemona")

    def test_nemona_draws_three_cards_and_is_discarded(self) -> None:
        state = create_game(seed=1, human_deck_id="ampharos-ex-battle-deck")
        self._finish_opening_setup(state)
        state.turn_number = 2
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
        state.turn_number = 2
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

    def test_evolution_is_not_available_on_the_first_turn_in_standard(self) -> None:
        state = create_game(seed=1, human_deck_id="ampharos-ex-battle-deck")
        self._finish_opening_setup(state)
        kilowattrel_id = self._move_named_card_to_hand(state, 0, "Kilowattrel")
        self._set_exact_hand(state, 0, [kilowattrel_id])

        evolve_actions = [action for action in list_legal_actions(state) if action["type"] == "evolve"]

        self.assertEqual(evolve_actions, [])

    def test_end_setup_draws_a_card_to_begin_the_first_turn(self) -> None:
        state = create_game(seed=1, human_deck_id="ampharos-ex-battle-deck")
        active_action = next(
            action for action in list_legal_actions(state) if action["type"] == "play_basic_to_active"
        )
        apply_action(state, active_action)
        expected_draw = state.players[0].deck[0]
        end_setup = next(action for action in list_legal_actions(state) if action["type"] == "end_setup")

        apply_action(state, end_setup)

        self.assertEqual(state.players[0].hand[-1], expected_draw)
        self.assertEqual(len(state.players[0].hand), 7)
        self.assertIn("You drew", state.log[-1])

    def test_evolution_is_available_on_a_later_turn_for_a_matching_active_pokemon(self) -> None:
        state = create_game(seed=1, human_deck_id="ampharos-ex-battle-deck")
        self._finish_opening_setup(state)
        state.players[0].turns_taken = 2
        kilowattrel_id = self._move_named_card_to_hand(state, 0, "Kilowattrel")
        self._set_exact_hand(state, 0, [kilowattrel_id])

        evolve_actions = [action for action in list_legal_actions(state) if action["type"] == "evolve"]

        self.assertEqual(len(evolve_actions), 1)
        self.assertEqual(evolve_actions[0]["target_zone"], "active")

        apply_action(state, evolve_actions[0])

        self.assertEqual(card_definition(state, state.players[0].active.stack[-1]).name, "Kilowattrel")
        self.assertEqual(state.players[0].active.entered_play_turn, 2)

    def test_evolution_cannot_target_a_basic_played_this_turn_in_standard(self) -> None:
        state = create_game(seed=1, human_deck_id="ampharos-ex-battle-deck")
        self._finish_opening_setup(state)
        state.players[0].turns_taken = 2

        mareep_id = self._move_named_card_to_hand(state, 0, "Mareep")
        self._set_exact_hand(state, 0, [mareep_id])
        bench_action = next(action for action in list_legal_actions(state) if action["type"] == "bench_basic")
        apply_action(state, bench_action)

        flaaffy_id = self._move_named_card_to_hand(state, 0, "Flaaffy")
        self._set_exact_hand(state, 0, [flaaffy_id])

        evolve_actions = [action for action in list_legal_actions(state) if action["type"] == "evolve"]

        self.assertEqual(evolve_actions, [])

    def test_stage_two_evolution_cannot_happen_on_the_same_turn_as_stage_one(self) -> None:
        state = create_game(seed=1, human_deck_id="ampharos-ex-battle-deck")

        active_action = next(
            action for action in list_legal_actions(state) if action["type"] == "play_basic_to_active"
        )
        apply_action(state, active_action)
        setup_bench_action = next(
            action for action in list_legal_actions(state) if action["type"] == "bench_basic"
        )
        apply_action(state, setup_bench_action)
        end_setup = next(action for action in list_legal_actions(state) if action["type"] == "end_setup")
        apply_action(state, end_setup)

        state.players[0].turns_taken = 2
        flaaffy_id = self._move_named_card_to_hand(state, 0, "Flaaffy")
        ampharos_id = self._move_named_card_to_hand(state, 0, "Ampharos ex")
        self._set_exact_hand(state, 0, [flaaffy_id, ampharos_id])

        evolve_actions = [action for action in list_legal_actions(state) if action["type"] == "evolve"]
        flaaffy_action = next(
            action for action in evolve_actions if card_definition(state, action["hand_card_id"]).name == "Flaaffy"
        )
        self.assertEqual(flaaffy_action["target_zone"], "bench")
        self.assertEqual(flaaffy_action["target_bench_index"], 0)

        apply_action(state, flaaffy_action)

        same_turn_stage_two_actions = [
            action for action in list_legal_actions(state) if action["type"] == "evolve"
        ]
        self.assertEqual(same_turn_stage_two_actions, [])

        state.players[0].turns_taken = 3
        next_turn_stage_two_actions = [
            action for action in list_legal_actions(state) if action["type"] == "evolve"
        ]

        self.assertEqual(len(next_turn_stage_two_actions), 1)
        self.assertEqual(card_definition(state, next_turn_stage_two_actions[0]["hand_card_id"]).name, "Ampharos ex")
        self.assertEqual(next_turn_stage_two_actions[0]["target_zone"], "bench")

    def test_attack_becomes_legal_when_active_has_enough_energy(self) -> None:
        state = create_game(seed=1, human_deck_id="ampharos-ex-battle-deck")
        state.setup_phase = None
        state.turn_number = 2
        state.players[0].turns_taken = 2
        self._set_named_active_pokemon(state, 0, "Mareep")
        self._set_named_active_pokemon(state, 1, "Mankey")
        energy_id = self._move_named_card_to_hand(state, 0, "Basic Lightning Energy")
        self._set_exact_hand(state, 0, [energy_id])

        attach_action = next(action for action in list_legal_actions(state) if action["type"] == "play_energy")
        apply_action(state, attach_action)
        attack_actions = [action for action in list_legal_actions(state) if action["type"] == "attack"]

        self.assertEqual(len(attack_actions), 1)
        self.assertEqual(attack_actions[0]["attack_index"], 0)
        self.assertEqual(attack_actions[0]["label"], "Use Static Shock")

    def test_collect_draws_a_card_and_ends_the_turn(self) -> None:
        state = create_game(seed=1, human_deck_id="ampharos-ex-battle-deck")
        state.setup_phase = None
        state.current_player = 0
        state.turn_number = 2
        state.players[0].turns_taken = 2
        self._set_named_active_pokemon(state, 0, "Wattrel")
        self._set_named_active_pokemon(state, 1, "Mankey")
        energy_id = self._move_named_card_to_hand(state, 0, "Basic Lightning Energy")
        self._set_exact_hand(state, 0, [energy_id])
        expected_draw = state.players[0].deck[0]

        attach_action = next(action for action in list_legal_actions(state) if action["type"] == "play_energy")
        apply_action(state, attach_action)
        collect_action = next(
            action
            for action in list_legal_actions(state)
            if action["type"] == "attack" and action["attack_index"] == 0
        )
        apply_action(state, collect_action)

        self.assertEqual(state.players[0].hand, [expected_draw])
        self.assertEqual(state.current_player, 1)
        self.assertEqual(state.players[1].active.damage, 0)
        self.assertIn("drew 1 card", " ".join(state.log).lower())

    def test_damage_attack_applies_damage_and_passes_the_turn(self) -> None:
        state = create_game(seed=1, human_deck_id="ampharos-ex-battle-deck")
        state.setup_phase = None
        state.current_player = 0
        state.turn_number = 2
        state.players[0].turns_taken = 2
        self._set_named_active_pokemon(state, 0, "Mareep")
        self._set_named_active_pokemon(state, 1, "Mankey")
        state.players[0].active.attached_energy = [
            self._find_instance_id(state, 0, "Basic Lightning Energy"),
        ]

        attack_action = next(action for action in list_legal_actions(state) if action["type"] == "attack")
        apply_action(state, attack_action)

        self.assertEqual(state.players[1].active.damage, 10)
        self.assertEqual(state.current_player, 1)

    def test_linear_attack_can_target_the_opposing_bench(self) -> None:
        state = create_game(seed=1, human_deck_id="ampharos-ex-battle-deck")
        state.setup_phase = None
        state.current_player = 0
        state.turn_number = 2
        state.players[0].turns_taken = 2
        self._set_named_active_pokemon(state, 0, "Rotom")
        self._set_named_active_pokemon(state, 1, "Mankey")
        self._set_named_bench_pokemon(state, 1, "Lechonk")
        state.players[0].active.attached_energy = self._find_instance_ids(
            state,
            0,
            "Basic Lightning Energy",
            1,
        )

        attack_actions = [action for action in list_legal_actions(state) if action["type"] == "attack"]

        self.assertEqual(
            {(action["target_zone"], action["target_bench_index"]) for action in attack_actions},
            {("active", None), ("bench", 0)},
        )

        bench_action = next(action for action in attack_actions if action["target_zone"] == "bench")
        apply_action(state, bench_action)

        self.assertEqual(state.players[1].active.damage, 0)
        self.assertEqual(state.players[1].bench[0].damage, 20)
        self.assertEqual(state.current_player, 1)

    def test_lightning_laser_hits_active_and_selected_bench_target(self) -> None:
        state = create_game(seed=1, human_deck_id="ampharos-ex-battle-deck")
        state.setup_phase = None
        state.current_player = 0
        state.turn_number = 2
        state.players[0].turns_taken = 2
        self._set_named_active_pokemon(state, 0, "Miraidon")
        self._set_named_active_pokemon(state, 1, "Cyclizar")
        self._set_named_bench_pokemon(state, 1, "Lechonk")
        attack = card_definition(state, state.players[0].active.stack[-1]).attacks[1]
        state.players[0].active.attached_energy = self._find_instance_ids(
            state,
            0,
            "Basic Lightning Energy",
            attack.cost,
        )

        lightning_laser = next(
            action
            for action in list_legal_actions(state)
            if action["type"] == "attack"
            and action["attack_index"] == 1
            and action["target_zone"] == "bench"
        )
        apply_action(state, lightning_laser)

        self.assertEqual(state.players[1].active.damage, 90)
        self.assertEqual(state.players[1].bench[0].damage, 30)

    def test_nosedive_deals_damage_to_the_opponent_and_to_itself(self) -> None:
        state = create_game(seed=1, human_deck_id="ampharos-ex-battle-deck")
        state.setup_phase = None
        state.current_player = 0
        state.turn_number = 2
        state.players[0].turns_taken = 2
        self._set_named_active_pokemon(state, 0, "Flamigo")
        self._set_named_active_pokemon(state, 1, "Lucario ex")
        attack = card_definition(state, state.players[0].active.stack[-1]).attacks[1]
        state.players[0].active.attached_energy = self._find_instance_ids(
            state,
            0,
            "Basic Lightning Energy",
            attack.cost,
        )

        nosedive = next(
            action
            for action in list_legal_actions(state)
            if action["type"] == "attack" and action["attack_index"] == 1
        )
        apply_action(state, nosedive)

        self.assertEqual(state.players[1].active.damage, 110)
        self.assertEqual(state.players[0].active.damage, 20)

    def test_thunder_blast_discards_one_attached_lightning_energy(self) -> None:
        state = create_game(seed=1, human_deck_id="ampharos-ex-battle-deck")
        state.setup_phase = None
        state.current_player = 0
        state.turn_number = 2
        state.players[0].turns_taken = 2
        self._set_named_active_pokemon(state, 0, "Kilowattrel")
        self._set_named_active_pokemon(state, 1, "Lucario ex")
        attack = card_definition(state, state.players[0].active.stack[-1]).attacks[1]
        attached_energy = self._find_instance_ids(
            state,
            0,
            "Basic Lightning Energy",
            attack.cost,
        )
        state.players[0].active.attached_energy = list(attached_energy)

        thunder_blast = next(
            action
            for action in list_legal_actions(state)
            if action["type"] == "attack" and action["attack_index"] == 1
        )
        apply_action(state, thunder_blast)

        self.assertEqual(state.players[1].active.damage, 140)
        self.assertEqual(len(state.players[0].active.attached_energy), attack.cost - 1)
        self.assertEqual(card_definition(state, state.players[0].discard[-1]).name, "Basic Lightning Energy")

    def test_primeape_raging_punch_damages_itself(self) -> None:
        state = create_game(seed=1, human_deck_id="ampharos-ex-battle-deck")
        state.setup_phase = None
        state.current_player = 1
        state.turn_number = 2
        state.players[0].turns_taken = 2
        state.players[1].turns_taken = 2
        self._set_named_active_pokemon(state, 0, "Kilowattrel")
        self._set_named_active_pokemon(state, 1, "Primeape")
        attack = card_definition(state, state.players[1].active.stack[-1]).attacks[0]
        state.players[1].active.attached_energy = self._find_instance_ids(
            state,
            1,
            "Basic Fighting Energy",
            attack.cost,
        )

        raging_punch = next(action for action in list_legal_actions(state) if action["type"] == "attack")
        apply_action(state, raging_punch)

        self.assertEqual(state.players[0].active.damage, 70)
        self.assertEqual(state.players[1].active.damage, 20)

    def test_touring_draws_two_cards(self) -> None:
        state = create_game(seed=1, human_deck_id="ampharos-ex-battle-deck")
        state.setup_phase = None
        state.current_player = 1
        state.turn_number = 2
        state.players[0].turns_taken = 2
        state.players[1].turns_taken = 2
        self._set_named_active_pokemon(state, 0, "Kilowattrel")
        self._set_named_active_pokemon(state, 1, "Cyclizar")
        attack = card_definition(state, state.players[1].active.stack[-1]).attacks[0]
        state.players[1].active.attached_energy = self._find_instance_ids(
            state,
            1,
            "Basic Fighting Energy",
            attack.cost,
        )
        hand_size_before = len(state.players[1].hand)
        expected_draw = list(state.players[1].deck[:2])

        touring = next(
            action
            for action in list_legal_actions(state)
            if action["type"] == "attack" and action["attack_index"] == 0
        )
        apply_action(state, touring)

        self.assertEqual(len(state.players[1].hand), hand_size_before + 2)
        self.assertEqual(state.players[1].hand[-2:], expected_draw)

    def test_rage_fist_scales_with_prize_cards_taken(self) -> None:
        state = create_game(seed=1, human_deck_id="ampharos-ex-battle-deck")
        state.setup_phase = None
        state.current_player = 1
        state.turn_number = 2
        state.players[0].turns_taken = 2
        state.players[1].turns_taken = 2
        state.players[0].prize_cards_remaining = 4
        self._set_named_active_pokemon(state, 0, "Ampharos ex")
        self._set_named_active_pokemon(state, 1, "Annihilape")
        attack = card_definition(state, state.players[1].active.stack[-1]).attacks[0]
        state.players[1].active.attached_energy = self._find_instance_ids(
            state,
            1,
            "Basic Fighting Energy",
            attack.cost,
        )

        rage_fist = next(
            action
            for action in list_legal_actions(state)
            if action["type"] == "attack" and action["attack_index"] == 0
        )
        apply_action(state, rage_fist)

        self.assertEqual(state.players[0].active.damage, 140)

    def test_kick_shot_does_nothing_on_tails(self) -> None:
        state = create_game(seed=1, human_deck_id="ampharos-ex-battle-deck")
        state.setup_phase = None
        state.current_player = 1
        state.turn_number = 2
        state.players[0].turns_taken = 2
        state.players[1].turns_taken = 2
        self._set_named_active_pokemon(state, 0, "Kilowattrel")
        self._set_named_active_pokemon(state, 1, "Medicham")
        attack = card_definition(state, state.players[1].active.stack[-1]).attacks[1]
        state.players[1].active.attached_energy = self._find_instance_ids(
            state,
            1,
            "Basic Fighting Energy",
            attack.cost,
        )

        kick_shot = next(
            action
            for action in list_legal_actions(state)
            if action["type"] == "attack" and action["attack_index"] == 1
        )
        with patch.object(state.rng, "choice", return_value=False):
            apply_action(state, kick_shot)

        self.assertEqual(state.players[0].active.damage, 0)
        self.assertIn("flipped tails", " ".join(state.log).lower())

    def test_aura_sphere_hits_the_active_and_selected_bench_pokemon(self) -> None:
        state = create_game(seed=1, human_deck_id="ampharos-ex-battle-deck")
        state.setup_phase = None
        state.current_player = 1
        state.turn_number = 2
        state.players[0].turns_taken = 2
        state.players[1].turns_taken = 2
        self._set_named_active_pokemon(state, 0, "Ampharos ex")
        self._set_named_bench_pokemon(state, 0, "Mareep")
        self._set_named_active_pokemon(state, 1, "Lucario ex")
        attack = card_definition(state, state.players[1].active.stack[-1]).attacks[1]
        state.players[1].active.attached_energy = self._find_instance_ids(
            state,
            1,
            "Basic Fighting Energy",
            attack.cost,
        )

        aura_sphere = next(
            action
            for action in list_legal_actions(state)
            if action["type"] == "attack"
            and action["attack_index"] == 1
            and action["target_zone"] == "bench"
        )
        apply_action(state, aura_sphere)

        self.assertEqual(state.players[0].active.damage, 160)
        self.assertEqual(state.players[0].bench[0].damage, 50)

    def test_rampaging_fang_discards_three_energy(self) -> None:
        state = create_game(seed=1, human_deck_id="ampharos-ex-battle-deck")
        state.setup_phase = None
        state.current_player = 1
        state.turn_number = 2
        state.players[0].turns_taken = 2
        state.players[1].turns_taken = 2
        self._set_named_active_pokemon(state, 0, "Ampharos ex")
        self._set_named_active_pokemon(state, 1, "Koraidon")
        attack = card_definition(state, state.players[1].active.stack[-1]).attacks[1]
        state.players[1].active.attached_energy = self._find_instance_ids(
            state,
            1,
            "Basic Fighting Energy",
            attack.cost,
        )

        rampaging_fang = next(
            action
            for action in list_legal_actions(state)
            if action["type"] == "attack" and action["attack_index"] == 1
        )
        apply_action(state, rampaging_fang)

        self.assertEqual(len(state.players[1].active.attached_energy), 1)
        self.assertEqual(
            sum(1 for instance_id in state.players[1].discard if card_definition(state, instance_id).name == "Basic Fighting Energy"),
            3,
        )

    def test_tailspin_away_prevents_damage_from_basic_pokemon_during_the_next_turn(self) -> None:
        state = create_game(seed=1, human_deck_id="ampharos-ex-battle-deck")
        state.setup_phase = None
        state.current_player = 0
        state.turn_number = 2
        state.players[0].turns_taken = 2
        state.players[1].turns_taken = 2
        self._set_named_active_pokemon(state, 0, "Staraptor")
        self._set_named_active_pokemon(state, 1, "Cyclizar")
        staraptor_attack = card_definition(state, state.players[0].active.stack[-1]).attacks[0]
        mankey_attack = card_definition(state, state.players[1].active.stack[-1]).attacks[0]
        state.players[0].active.attached_energy = self._find_instance_ids(
            state,
            0,
            "Basic Lightning Energy",
            staraptor_attack.cost,
        )
        state.players[1].active.attached_energy = self._find_instance_ids(
            state,
            1,
            "Basic Fighting Energy",
            mankey_attack.cost,
        )

        tailspin_away = next(
            action
            for action in list_legal_actions(state)
            if action["type"] == "attack" and action["attack_index"] == 0
        )
        apply_action(state, tailspin_away)

        self.assertEqual(state.current_player, 1)
        self.assertEqual(state.players[0].active.damage, 0)

        basic_attack = next(action for action in list_legal_actions(state) if action["type"] == "attack")
        apply_action(state, basic_attack)

        self.assertEqual(state.players[0].active.damage, 0)
        self.assertEqual(state.players[0].active.lingering_effects, [])

    def test_ai_prefers_an_immediate_winning_attack_over_benching_more_basics(self) -> None:
        state = create_game(seed=1, human_deck_id="ampharos-ex-battle-deck")
        state.setup_phase = None
        state.current_player = 1
        state.turn_number = 2
        state.starting_player = 0
        state.players[0].turns_taken = 2
        state.players[1].turns_taken = 2
        self._set_named_active_pokemon(state, 0, "Mareep")
        self._set_named_active_pokemon(state, 1, "Primeape")
        state.players[0].bench = []
        state.players[1].bench = []
        state.players[1].active.attached_energy = self._find_instance_ids(
            state,
            1,
            "Basic Fighting Energy",
            5,
        )

        action = choose_action(state, 1)

        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(action["type"], "attack")
        self.assertEqual(action["attack_index"], 0)

    def test_first_player_cannot_attack_on_turn_one(self) -> None:
        state = create_game(seed=1, human_deck_id="ampharos-ex-battle-deck")
        state.setup_phase = None
        state.current_player = 0
        state.turn_number = 1
        state.starting_player = 0
        state.players[0].turns_taken = 1
        self._set_named_active_pokemon(state, 0, "Mareep")
        self._set_named_active_pokemon(state, 1, "Mankey")
        state.players[0].active.attached_energy = [
            self._find_instance_id(state, 0, "Basic Lightning Energy"),
        ]

        self.assertNotIn("attack", [action["type"] for action in list_legal_actions(state)])

    def test_second_player_can_attack_on_turn_one(self) -> None:
        state = create_game(seed=1, human_deck_id="ampharos-ex-battle-deck")
        state.setup_phase = None
        state.current_player = 1
        state.turn_number = 1
        state.starting_player = 0
        state.players[1].turns_taken = 1
        self._set_named_active_pokemon(state, 0, "Mareep")
        self._set_named_active_pokemon(state, 1, "Mankey")
        state.players[1].active.attached_energy = [
            self._find_instance_id(state, 1, "Basic Fighting Energy"),
        ]

        attack_actions = [action for action in list_legal_actions(state) if action["type"] == "attack"]

        self.assertEqual(len(attack_actions), 1)
        self.assertEqual(attack_actions[0]["label"], "Use Monkey Beatdown")

    def test_potion_targets_damaged_pokemon_and_heals_thirty(self) -> None:
        state = create_game(seed=1, human_deck_id="ampharos-ex-battle-deck")
        self._finish_opening_setup(state)
        mareep_id = self._move_named_card_to_hand(state, 0, "Mareep")
        self._set_exact_hand(state, 0, [mareep_id])
        bench_action = next(action for action in list_legal_actions(state) if action["type"] == "bench_basic")
        apply_action(state, bench_action)

        potion_id = self._move_named_card_to_hand(state, 0, "Potion")
        self._set_exact_hand(state, 0, [potion_id])
        assert state.players[0].active is not None
        state.players[0].active.damage = 40
        state.players[0].bench[0].damage = 20

        potion_actions = [action for action in list_legal_actions(state) if action["type"] == "play_item"]

        self.assertEqual(len(potion_actions), 2)
        self.assertEqual(
            {(action["target_zone"], action["target_bench_index"]) for action in potion_actions},
            {("active", None), ("bench", 0)},
        )

        active_potion = next(action for action in potion_actions if action["target_zone"] == "active")
        apply_action(state, active_potion)

        self.assertEqual(state.players[0].active.damage, 10)
        self.assertEqual(state.players[0].bench[0].damage, 20)
        self.assertEqual(card_definition(state, state.players[0].discard[-1]).name, "Potion")

    def test_switch_targets_each_benched_pokemon_and_swaps_with_active(self) -> None:
        state = create_game(seed=1, human_deck_id="ampharos-ex-battle-deck")
        self._finish_opening_setup(state)
        mareep_id = self._move_named_card_to_hand(state, 0, "Mareep")
        self._set_exact_hand(state, 0, [mareep_id])
        bench_action = next(action for action in list_legal_actions(state) if action["type"] == "bench_basic")
        apply_action(state, bench_action)

        switch_id = self._move_named_card_to_hand(state, 0, "Switch")
        self._set_exact_hand(state, 0, [switch_id])
        previous_active_name = card_definition(state, state.players[0].active.stack[-1]).name
        benched_name = card_definition(state, state.players[0].bench[0].stack[-1]).name

        switch_actions = [action for action in list_legal_actions(state) if action["type"] == "play_item"]

        self.assertEqual(len(switch_actions), 1)
        self.assertEqual(switch_actions[0]["target_zone"], "bench")
        self.assertEqual(switch_actions[0]["target_bench_index"], 0)

        apply_action(state, switch_actions[0])

        self.assertEqual(card_definition(state, state.players[0].active.stack[-1]).name, benched_name)
        self.assertEqual(card_definition(state, state.players[0].bench[0].stack[-1]).name, previous_active_name)
        self.assertEqual(card_definition(state, state.players[0].discard[-1]).name, "Switch")

    def test_ultra_ball_discards_two_other_cards_and_puts_a_pokemon_into_hand(self) -> None:
        state = create_game(seed=1, human_deck_id="ampharos-ex-battle-deck")
        self._finish_opening_setup(state)

        ultra_ball_id = self._move_named_card_to_hand(state, 0, "Ultra Ball")
        discard_ids = [
            self._move_named_card_to_hand(state, 0, "Basic Lightning Energy"),
            self._move_named_card_to_hand(state, 0, "Potion"),
        ]
        target_id = next(
            instance_id
            for instance_id in state.players[0].deck
            if card_definition(state, instance_id).kind == "pokemon"
        )

        self._set_exact_hand(state, 0, [ultra_ball_id, *discard_ids])
        ultra_ball_action = self._find_action_for_card_name(state, "play_item", "Ultra Ball")

        apply_action(
            state,
            {
                **ultra_ball_action,
                "discard_from_hand_ids": discard_ids,
                "search_result_ids": [target_id],
            },
        )

        self.assertIn(target_id, state.players[0].hand)
        self.assertNotIn(target_id, state.players[0].deck)
        self.assertTrue(all(instance_id in state.players[0].discard for instance_id in discard_ids))
        self.assertEqual(card_definition(state, state.players[0].discard[-1]).name, "Ultra Ball")

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

    def _set_named_active_pokemon(self, state, player_index: int, card_name: str) -> None:
        instance_id = self._find_instance_id(state, player_index, card_name)
        player = state.players[player_index]
        for zone_name in ("hand", "deck", "discard"):
            zone = getattr(player, zone_name)
            if instance_id in zone:
                zone.remove(instance_id)
                break
        player.active = PokemonInPlay(stack=[instance_id])

    def _set_named_bench_pokemon(self, state, player_index: int, card_name: str) -> None:
        instance_id = self._find_instance_id(state, player_index, card_name)
        player = state.players[player_index]
        for zone_name in ("hand", "deck", "discard"):
            zone = getattr(player, zone_name)
            if instance_id in zone:
                zone.remove(instance_id)
                break
        player.bench.append(PokemonInPlay(stack=[instance_id]))

    def _find_instance_id(self, state, player_index: int, card_name: str) -> str:
        player = state.players[player_index]
        for zone_name in ("hand", "deck", "discard"):
            zone = getattr(player, zone_name)
            for instance_id in zone:
                if card_definition(state, instance_id).name == card_name:
                    return instance_id
        self.fail(f"Could not find {card_name} for player {player_index}")

    def _find_instance_ids(self, state, player_index: int, card_name: str, count: int) -> list[str]:
        player = state.players[player_index]
        matches: list[str] = []
        for zone_name in ("hand", "deck", "discard"):
            zone = getattr(player, zone_name)
            for instance_id in zone:
                if card_definition(state, instance_id).name != card_name:
                    continue
                matches.append(instance_id)
                if len(matches) == count:
                    return matches
        self.fail(f"Could not find {count} copies of {card_name} for player {player_index}")


if __name__ == "__main__":
    unittest.main()
