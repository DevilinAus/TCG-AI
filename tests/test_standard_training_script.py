from __future__ import annotations

import importlib.util
from pathlib import Path
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
