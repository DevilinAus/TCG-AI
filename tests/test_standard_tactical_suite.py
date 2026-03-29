from __future__ import annotations

import unittest

from backend.tcg_ai.game_modes.standard.ml.oracle import HeuristicPolicyValueOracle
from backend.tcg_ai.game_modes.standard.ml.planner import PlannerConfig
from backend.tcg_ai.game_modes.standard.ml.tactical_suite import (
    core_tactical_scenarios,
    default_tactical_scenarios,
    run_tactical_suite,
    strategic_tactical_scenarios,
)


class StandardTacticalSuiteTests(unittest.TestCase):
    def test_default_core_tactical_scenarios_are_objective_and_covered(self) -> None:
        scenarios = default_tactical_scenarios()

        self.assertEqual(
            [scenario.name for scenario in scenarios],
            [
                "attack_for_immediate_win",
                "attack_for_immediate_win_over_supporter",
                "attach_to_enable_immediate_win",
                "retreat_into_ready_attacker",
                "retreat_into_ready_attacker_over_supporter",
                "attack_over_redundant_retreat_when_active_already_wins",
            ],
        )

    def test_tactical_suite_can_return_all_scenarios(self) -> None:
        scenarios = default_tactical_scenarios("all")
        self.assertEqual(
            [scenario.name for scenario in scenarios],
            [
                *[scenario.name for scenario in core_tactical_scenarios()],
                *[scenario.name for scenario in strategic_tactical_scenarios()],
            ],
        )

    def test_heuristic_oracle_passes_core_tactical_suite(self) -> None:
        results = run_tactical_suite(
            oracle=HeuristicPolicyValueOracle(),
            planner_config=PlannerConfig(max_depth=2, beam_width=4, opponent_branch_width=2),
        )

        self.assertEqual(len(results), 6)
        self.assertTrue(all(result.passed for result in results))
        self.assertEqual(
            {result.chosen_action_type for result in results},
            {"attack", "play_energy", "retreat"},
        )
        self.assertEqual({result.tier for result in results}, {"core"})

    def test_heuristic_oracle_passes_strategic_tactical_suite(self) -> None:
        results = run_tactical_suite(
            oracle=HeuristicPolicyValueOracle(),
            planner_config=PlannerConfig(max_depth=2, beam_width=4, opponent_branch_width=2),
            suite="strategic",
        )

        self.assertEqual(len(results), 10)
        self.assertTrue(all(result.passed for result in results))
        self.assertEqual(
            [result.name for result in results],
            [
                "call_for_family_over_supporter_draw",
                "attach_to_bench_for_retreat_win",
                "attach_to_bench_from_doomed_active_for_retreat_win",
                "overattach_to_active_for_retreat_win",
                "nest_ball_unique_target_for_retreat_win",
                "ultra_ball_lucario_ex_for_immediate_conversion",
                "ultra_ball_ampharos_ex_for_immediate_conversion",
                "youngster_over_low_value_ultra_ball",
                "energy_retrieval_non_empty_over_empty_variant",
                "pokegear_empty_hand_thinning_when_no_supporter_found",
            ],
        )
        self.assertEqual(
            {result.chosen_action_type for result in results},
            {"attack", "play_energy", "play_item", "play_supporter", "retreat"},
        )
        self.assertEqual({result.tier for result in results}, {"strategic"})


if __name__ == "__main__":
    unittest.main()
