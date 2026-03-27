from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import patch

from backend.tcg_ai.game_modes.standard.ml.neural_policy import PolicyValueBackendStatus
from backend.tcg_ai.game_modes.standard.ml.oracle import BackendPolicyValueOracle, HeuristicPolicyValueOracle
from backend.tcg_ai.game_modes.standard.ml.self_play_jobs import _build_oracle, resolve_oracle_status


class StandardSelfPlayJobsTests(unittest.TestCase):
    def test_auto_oracle_falls_back_to_heuristic_when_no_model_is_available(self) -> None:
        status = resolve_oracle_status(oracle="auto", checkpoint=Path("/tmp/does-not-exist.pt"))

        self.assertEqual(status["requested_oracle"], "auto")
        self.assertEqual(status["resolved_oracle"], "heuristic")
        self.assertFalse(status["model_loaded"])

        oracle = _build_oracle("auto", "/tmp/does-not-exist.pt")
        self.assertIsInstance(oracle, HeuristicPolicyValueOracle)

    def test_auto_oracle_uses_backend_model_when_checkpoint_is_loaded(self) -> None:
        fake_backend = type(
            "FakeBackend",
            (),
            {
                "status": PolicyValueBackendStatus(
                    backend="torch:cpu",
                    model_loaded=True,
                    checkpoint_path="/tmp/champion.pt",
                )
            },
        )()

        with patch(
            "backend.tcg_ai.game_modes.standard.ml.self_play_jobs.PolicyValueBackend",
            return_value=fake_backend,
        ):
            status = resolve_oracle_status(oracle="auto", checkpoint=Path("/tmp/champion.pt"))
            oracle = _build_oracle("auto", "/tmp/champion.pt")

        self.assertEqual(status["resolved_oracle"], "local-model")
        self.assertTrue(status["model_loaded"])
        self.assertIsInstance(oracle, BackendPolicyValueOracle)


if __name__ == "__main__":
    unittest.main()
