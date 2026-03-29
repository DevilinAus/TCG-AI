from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock


ROOT_DIR = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT_PATH = ROOT_DIR / "scripts" / "train_standard_model.py"


def _load_training_module():
    spec = importlib.util.spec_from_file_location("train_standard_model_script", TRAIN_SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load training script from {TRAIN_SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StandardTrainingScriptTests(unittest.TestCase):
    def test_iter_collated_batches_groups_streaming_samples_without_indexing(self) -> None:
        training_module = _load_training_module()
        original_collate = training_module._collate_training_batch
        training_module._collate_training_batch = lambda samples: [sample["id"] for sample in samples]
        try:
            samples = [{"id": 1}, {"id": 2}, {"id": 3}]
            batches = list(training_module._iter_collated_batches(samples, batch_size=2))
        finally:
            training_module._collate_training_batch = original_collate

        self.assertEqual(batches, [[1, 2], [3]])

    def test_json_safe_converts_paths_recursively(self) -> None:
        training_module = _load_training_module()

        payload = {
            "output_dir": Path("/tmp/model"),
            "nested": [Path("/tmp/a"), {"checkpoint": Path("/tmp/b")}],
        }

        sanitized = training_module._json_safe(payload)

        self.assertEqual(
            sanitized,
            {
                "output_dir": "/tmp/model",
                "nested": ["/tmp/a", {"checkpoint": "/tmp/b"}],
            },
        )

    def test_build_policy_target_vector_uses_soft_targets_when_present(self) -> None:
        training_module = _load_training_module()

        payload = {
            "policy_target_probs": {
                "a": 0.2,
                "b": 0.3,
                "c": 0.5,
            }
        }

        vector = training_module._build_policy_target_vector(
            payload=payload,
            action_ids=["a", "b", "c"],
            chosen_action_id="b",
        )

        self.assertEqual(vector, [0.2, 0.3, 0.5])

    def test_build_policy_target_vector_falls_back_to_one_hot(self) -> None:
        training_module = _load_training_module()

        vector = training_module._build_policy_target_vector(
            payload={},
            action_ids=["a", "b", "c"],
            chosen_action_id="b",
        )

        self.assertEqual(vector, [0.0, 1.0, 0.0])

    def test_encode_training_record_builds_auxiliary_intent_and_quality_targets(self) -> None:
        training_module = _load_training_module()

        payload = {
            "belief_state": {"players": [{}, {}]},
            "legal_actions": [
                {
                    "action_id": "a",
                    "type": "attack",
                    "intent_tags": ["take_prize"],
                    "quality_flags": [],
                },
                {
                    "action_id": "b",
                    "type": "play_item",
                    "intent_tags": ["hand_thinning"],
                    "quality_flags": ["dominated_optional_play"],
                },
            ],
            "chosen_action_id": "a",
            "value_target": 12.5,
        }

        encoded = training_module._encode_training_record(
            payload,
            state_dim=training_module.STATE_VECTOR_SIZE,
            action_dim=training_module.ACTION_VECTOR_SIZE,
        )

        self.assertIsNotNone(encoded)
        self.assertEqual(len(encoded["intent_targets"]), 2)
        self.assertEqual(len(encoded["quality_targets"]), 2)
        self.assertEqual(
            encoded["intent_targets"][0][training_module.INTENT_TAGS.index("take_prize")],
            1.0,
        )
        self.assertEqual(
            encoded["intent_targets"][1][training_module.INTENT_TAGS.index("hand_thinning")],
            1.0,
        )
        self.assertEqual(
            encoded["quality_targets"][1][training_module.QUALITY_FLAGS.index("dominated_optional_play")],
            1.0,
        )

    def test_quality_label_weights_emphasize_dominated_and_missed_conversions(self) -> None:
        training_module = _load_training_module()

        weights = training_module._quality_label_weights()

        self.assertEqual(len(weights), len(training_module.QUALITY_FLAGS))
        self.assertGreater(
            weights[training_module.QUALITY_FLAGS.index("dominated_optional_play")],
            weights[training_module.QUALITY_FLAGS.index("wastes_item")],
        )
        self.assertGreater(
            weights[training_module.QUALITY_FLAGS.index("misses_immediate_win")],
            weights[training_module.QUALITY_FLAGS.index("low_value_retreat")],
        )

    def test_resolve_input_dir_can_point_at_latest_incomplete_run(self) -> None:
        training_module = _load_training_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "run_20260329T090000Z").mkdir()
            latest_run = root / "run_20260329T100000Z"
            latest_run.mkdir()
            (latest_run / "coordinator_state.json").write_text("{}", encoding="utf-8")

            original_root = training_module.DEFAULT_SELF_PLAY_ROOT
            training_module.DEFAULT_SELF_PLAY_ROOT = root
            try:
                resolved = training_module._resolve_input_dir(None)
            finally:
                training_module.DEFAULT_SELF_PLAY_ROOT = original_root

        self.assertEqual(resolved, latest_run.resolve())

    def test_resolve_training_decision_paths_prefers_committed_summary_backed_shards(self) -> None:
        training_module = _load_training_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir)
            decisions_dir = input_dir / "decisions"
            summaries_dir = input_dir / "summaries"
            decisions_dir.mkdir()
            summaries_dir.mkdir()

            committed = decisions_dir / "shard_000000.jsonl"
            committed.write_text('{"decision":1}\n', encoding="utf-8")
            orphaned = decisions_dir / "shard_000001.jsonl"
            orphaned.write_text('{"decision":2}\n', encoding="utf-8")
            (summaries_dir / "shard_000000.json").write_text('{"samples":1}', encoding="utf-8")

            decision_paths, report = training_module._resolve_training_decision_paths(input_dir)

        self.assertEqual([path.name for path in decision_paths], ["shard_000000.jsonl"])
        self.assertEqual(report["selection_mode"], "summary_backed")
        self.assertEqual(report["usable_shards"], 1)
        self.assertEqual(report["ignored_uncommitted_shards"], 1)
        self.assertEqual(report["skipped_invalid_shards"], 0)

    def test_resolve_training_decision_paths_skips_malformed_committed_shards(self) -> None:
        training_module = _load_training_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir)
            decisions_dir = input_dir / "decisions"
            summaries_dir = input_dir / "summaries"
            decisions_dir.mkdir()
            summaries_dir.mkdir()

            valid_path = decisions_dir / "shard_000000.jsonl"
            valid_path.write_text('{"decision":1}\n', encoding="utf-8")
            invalid_path = decisions_dir / "shard_000001.jsonl"
            invalid_path.write_text('{"decision":2}\n{"decision":"unterminated}\n', encoding="utf-8")
            (summaries_dir / "shard_000000.json").write_text('{"samples":1}', encoding="utf-8")
            (summaries_dir / "shard_000001.json").write_text('{"samples":2}', encoding="utf-8")

            decision_paths, report = training_module._resolve_training_decision_paths(input_dir)

        self.assertEqual([path.name for path in decision_paths], ["shard_000000.jsonl"])
        self.assertEqual(report["selection_mode"], "summary_backed")
        self.assertEqual(report["usable_shards"], 1)
        self.assertEqual(report["skipped_invalid_shards"], 1)


class StandardCheckpointLoadingTests(unittest.TestCase):
    def test_load_trusted_checkpoint_uses_non_weights_only_load(self) -> None:
        from backend.tcg_ai.game_modes.standard.ml import neural_policy

        original_torch = neural_policy.torch
        fake_torch = Mock()
        fake_torch.load.return_value = {"state_dict": {}}
        neural_policy.torch = fake_torch
        try:
            checkpoint = neural_policy.load_trusted_checkpoint(Path("/tmp/champion.pt"), map_location="cpu")
        finally:
            neural_policy.torch = original_torch

        self.assertEqual(checkpoint, {"state_dict": {}})
        fake_torch.load.assert_called_once_with(Path("/tmp/champion.pt"), map_location="cpu", weights_only=False)

    def test_load_trusted_checkpoint_falls_back_when_weights_only_kwarg_is_unsupported(self) -> None:
        from backend.tcg_ai.game_modes.standard.ml import neural_policy

        original_torch = neural_policy.torch
        fake_torch = Mock()
        fake_torch.load.side_effect = [TypeError("unexpected keyword"), {"state_dict": {}}]
        neural_policy.torch = fake_torch
        try:
            checkpoint = neural_policy.load_trusted_checkpoint(Path("/tmp/champion.pt"), map_location="cpu")
        finally:
            neural_policy.torch = original_torch

        self.assertEqual(checkpoint, {"state_dict": {}})
        self.assertEqual(fake_torch.load.call_count, 2)
        self.assertEqual(
            fake_torch.load.call_args_list[0].kwargs,
            {"map_location": "cpu", "weights_only": False},
        )
        self.assertEqual(
            fake_torch.load.call_args_list[1].kwargs,
            {"map_location": "cpu"},
        )


if __name__ == "__main__":
    unittest.main()
