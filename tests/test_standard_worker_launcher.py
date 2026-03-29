from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from backend.tcg_ai.game_modes.standard.ml.worker_launcher import (
    DEFAULT_COORDINATOR_URL,
    build_worker_command,
    build_worker_id,
    resolve_coordinator_url,
    resolve_worker_count,
    resolve_worker_prefix,
)


class StandardWorkerLauncherTests(unittest.TestCase):
    def test_resolve_coordinator_url_defaults_to_repo_server(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(resolve_coordinator_url(None), DEFAULT_COORDINATOR_URL)

    def test_resolve_coordinator_url_prefers_explicit_value(self) -> None:
        with patch.dict(
            "os.environ",
            {"TCG_AI_STANDARD_SELF_PLAY_COORDINATOR_URL": "http://192.168.0.200:8787"},
            clear=True,
        ):
            self.assertEqual(resolve_coordinator_url("http://192.168.0.175:8787/"), DEFAULT_COORDINATOR_URL)

    def test_resolve_worker_count_uses_detected_cpu_cores(self) -> None:
        with patch("backend.tcg_ai.game_modes.standard.ml.worker_launcher.os.cpu_count", return_value=12):
            self.assertEqual(resolve_worker_count(None), 12)

    def test_resolve_worker_count_prefers_positive_explicit_value(self) -> None:
        self.assertEqual(resolve_worker_count(4), 4)
        self.assertEqual(resolve_worker_count(0), 1)

    def test_resolve_worker_prefix_defaults_to_hostname(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with patch("backend.tcg_ai.game_modes.standard.ml.worker_launcher.socket.gethostname", return_value="Mac Book Pro"):
                self.assertEqual(resolve_worker_prefix(None), "Mac-Book-Pro")

    def test_build_worker_id_uses_suffixes_for_multi_worker_machines(self) -> None:
        self.assertEqual(build_worker_id("macbook", 0, 8), "macbook-01")
        self.assertEqual(build_worker_id("macbook", 7, 8), "macbook-08")
        self.assertEqual(build_worker_id("macbook", 0, 1), "macbook")

    def test_build_worker_command_includes_progress_log_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            progress_dir = project_root / "progress"

            command = build_worker_command(
                python_executable="python3",
                project_root=project_root,
                coordinator_url=DEFAULT_COORDINATOR_URL,
                worker_id="macbook-01",
                machine_name="macbook",
                poll_seconds=5.0,
                reconnect_seconds=300.0,
                request_timeout_seconds=30.0,
                heartbeat_interval_seconds=15.0,
                progress_log_dir=progress_dir,
            )

        self.assertEqual(command[0], "python3")
        self.assertIn("scripts/run_standard_self_play_worker.py", command[1])
        self.assertIn("--coordinator-url", command)
        self.assertIn(DEFAULT_COORDINATOR_URL, command)
        self.assertIn("--worker-id", command)
        self.assertIn("macbook-01", command)
        self.assertIn("--machine-name", command)
        self.assertIn("macbook", command)
        self.assertIn("--reconnect-seconds", command)
        self.assertIn("300.0", command)
        self.assertIn("--progress-log", command)
        self.assertIn(str(progress_dir / "macbook-01.log"), command)


if __name__ == "__main__":
    unittest.main()
