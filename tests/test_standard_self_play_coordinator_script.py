from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest


ROOT_DIR = Path(__file__).resolve().parents[1]
COORDINATOR_SCRIPT_PATH = ROOT_DIR / "scripts" / "run_standard_self_play_coordinator.py"


def _load_coordinator_module():
    spec = importlib.util.spec_from_file_location("run_standard_self_play_coordinator_script", COORDINATOR_SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load coordinator script from {COORDINATOR_SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class StandardSelfPlayCoordinatorScriptTests(unittest.TestCase):
    def test_resolve_coordinator_run_resumes_newest_incomplete_run(self) -> None:
        coordinator_module = _load_coordinator_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir)
            completed_run = output_root / "run_20260329T090000Z"
            resumable_run = output_root / "run_20260329T100000Z"
            completed_run.mkdir()
            resumable_run.mkdir()

            (completed_run / "coordinator_state.json").write_text(
                json.dumps({"run_id": completed_run.name, "tasks": [{"status": "completed"}]}),
                encoding="utf-8",
            )
            (completed_run / coordinator_module.RUN_COMPLETE_FLAG_NAME).write_text(
                json.dumps({"run_id": completed_run.name, "completed_at": "2026-03-29T09:05:00+00:00"}),
                encoding="utf-8",
            )
            (resumable_run / "coordinator_state.json").write_text(
                json.dumps({"run_id": resumable_run.name, "tasks": [{"status": "pending"}]}),
                encoding="utf-8",
            )

            os.utime(completed_run / "coordinator_state.json", (100.0, 100.0))
            os.utime(resumable_run / "coordinator_state.json", (200.0, 200.0))

            resolved = coordinator_module.resolve_coordinator_run(
                run_id=None,
                output_dir=None,
                output_root=output_root,
            )

        self.assertTrue(resolved.resumed)
        self.assertEqual(resolved.run_id, resumable_run.name)
        self.assertEqual(resolved.output_dir, resumable_run.resolve())

    def test_resolve_coordinator_run_creates_new_run_when_only_completed_runs_exist(self) -> None:
        coordinator_module = _load_coordinator_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir)
            completed_run = output_root / "run_20260329T090000Z"
            completed_run.mkdir()
            (completed_run / coordinator_module.RUN_COMPLETE_FLAG_NAME).write_text(
                json.dumps({"run_id": completed_run.name, "completed_at": "2026-03-29T09:05:00+00:00"}),
                encoding="utf-8",
            )

            resolved = coordinator_module.resolve_coordinator_run(
                run_id=None,
                output_dir=None,
                output_root=output_root,
            )

        self.assertFalse(resolved.resumed)
        self.assertTrue(resolved.run_id.startswith("run_"))
        self.assertEqual(resolved.output_dir.parent, output_root.resolve())
        self.assertNotEqual(resolved.output_dir, completed_run.resolve())

    def test_resolve_coordinator_run_infers_run_id_from_existing_output_dir(self) -> None:
        coordinator_module = _load_coordinator_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "run_20260329T100000Z"
            output_dir.mkdir()
            (output_dir / "coordinator_state.json").write_text(
                json.dumps({"run_id": "run_20260329T100000Z", "tasks": [{"status": "pending"}]}),
                encoding="utf-8",
            )

            resolved = coordinator_module.resolve_coordinator_run(
                run_id=None,
                output_dir=output_dir,
                output_root=None,
            )

        self.assertTrue(resolved.resumed)
        self.assertEqual(resolved.run_id, "run_20260329T100000Z")
        self.assertEqual(resolved.output_dir, output_dir.resolve())


if __name__ == "__main__":
    unittest.main()
