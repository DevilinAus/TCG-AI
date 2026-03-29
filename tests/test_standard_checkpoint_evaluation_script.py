from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import time
import unittest

from backend.tcg_ai.game_modes.standard.ml.self_play import SelfPlayGameSummary


ROOT_DIR = Path(__file__).resolve().parents[1]
EVAL_SCRIPT_PATH = ROOT_DIR / "scripts" / "evaluate_standard_checkpoints.py"


def _load_evaluation_module():
    spec = importlib.util.spec_from_file_location("evaluate_standard_checkpoints_script", EVAL_SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load evaluation script from {EVAL_SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StandardCheckpointEvaluationScriptTests(unittest.TestCase):
    def test_single_worker_progress_reporter_counts_completed_games_once(self) -> None:
        evaluation_module = _load_evaluation_module()
        reporter = evaluation_module._SingleWorkerProgressReporter(
            total_games=99,
            start_time=time.perf_counter(),
            log_interval_seconds=3600.0,
        )

        reporter.begin_game({"task_index": 0, "local_game": 1, "task_game_count": 10, "global_game": 1})
        reporter.record_completed_game(
            {
                "games": 1,
                "candidate_result": "win",
                "candidate_deck_id": "ampharos-ex-battle-deck",
                "baseline_deck_id": "lucario-ex-battle-deck",
                "truncated": False,
                "turns": 6,
                "actions": 9,
            }
        )
        reporter.begin_game({"task_index": 0, "local_game": 2, "task_game_count": 10, "global_game": 2})
        reporter.record_completed_game(
            {
                "games": 1,
                "candidate_result": "loss",
                "candidate_deck_id": "lucario-ex-battle-deck",
                "baseline_deck_id": "ampharos-ex-battle-deck",
                "truncated": True,
                "turns": 8,
                "actions": 12,
            }
        )

        aggregate = reporter.aggregate_snapshot()

        self.assertEqual(aggregate["games"], 2)
        self.assertEqual(aggregate["candidate_wins"], 1)
        self.assertEqual(aggregate["baseline_wins"], 1)
        self.assertEqual(aggregate["draws"], 0)
        self.assertEqual(aggregate["truncated"], 1)
        self.assertEqual(aggregate["turns"], 14)
        self.assertEqual(aggregate["actions"], 21)

    def test_run_evaluation_chunk_reports_game_level_callbacks(self) -> None:
        evaluation_module = _load_evaluation_module()
        original_build_oracle = evaluation_module._build_oracle
        original_play_self_play_game = evaluation_module.play_self_play_game
        started: list[dict[str, object]] = []
        completed: list[dict[str, object]] = []
        fake_results = [
            SelfPlayGameSummary(
                schema_version=3,
                game_id="g0",
                seed=11,
                player0_deck_id="ampharos-ex-battle-deck",
                player1_deck_id="lucario-ex-battle-deck",
                winner=0,
                truncated=False,
                turn_number=5,
                action_count=7,
                decision_samples=0,
            ),
            SelfPlayGameSummary(
                schema_version=3,
                game_id="g1",
                seed=12,
                player0_deck_id="lucario-ex-battle-deck",
                player1_deck_id="ampharos-ex-battle-deck",
                winner=0,
                truncated=True,
                turn_number=9,
                action_count=13,
                decision_samples=0,
            ),
        ]

        def fake_play_self_play_game(**_: object):
            return fake_results[len(completed)], []

        evaluation_module._build_oracle = lambda spec: object()
        evaluation_module.play_self_play_game = fake_play_self_play_game
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                output_dir = Path(temp_dir)
                (output_dir / "games").mkdir()
                summary = evaluation_module._run_evaluation_chunk(
                    {
                        "task_index": 0,
                        "run_id": "eval_test",
                        "start_index": 0,
                        "game_count": 2,
                        "base_seed": 11,
                        "output_dir": str(output_dir),
                        "planner_config": {
                            "max_depth": 2,
                            "beam_width": 4,
                            "opponent_branch_width": 2,
                            "include_opponent_turn": True,
                        },
                        "max_actions_per_game": 200,
                        "candidate_spec": {"kind": "heuristic", "label": "candidate"},
                        "baseline_spec": {"kind": "heuristic", "label": "baseline"},
                        "progress_log": None,
                    },
                    game_start_callback=started.append,
                    progress_callback=completed.append,
                )
        finally:
            evaluation_module._build_oracle = original_build_oracle
            evaluation_module.play_self_play_game = original_play_self_play_game

        self.assertEqual([payload["global_game"] for payload in started], [1, 2])
        self.assertEqual([payload["candidate_result"] for payload in completed], ["win", "loss"])
        self.assertEqual(summary["games"], 2)
        self.assertEqual(summary["candidate_wins"], 1)
        self.assertEqual(summary["baseline_wins"], 1)
        self.assertEqual(summary["draws"], 0)
        self.assertEqual(summary["truncated"], 1)


if __name__ == "__main__":
    unittest.main()
