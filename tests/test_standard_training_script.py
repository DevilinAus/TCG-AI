from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


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


if __name__ == "__main__":
    unittest.main()
