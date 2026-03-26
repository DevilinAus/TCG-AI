from __future__ import annotations

from pathlib import Path
import tempfile
import time
import unittest

from backend.tcg_ai.game_modes.standard.ml.distributed_self_play import (
    DistributedSelfPlayCoordinator,
    decode_uploaded_chunk_text,
    encode_chunk_text_gzip_b64,
)
from backend.tcg_ai.game_modes.standard.ml.planner import PlannerConfig
from backend.tcg_ai.game_modes.standard.ml.self_play_jobs import (
    SelfPlayRunConfig,
    build_self_play_tasks,
    self_play_task_from_payload,
    self_play_task_to_payload,
)
from scripts.run_standard_self_play_worker import LeaseHeartbeatLoop


class DistributedStandardMlTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name) / "self_play_run"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_task_payload_round_trip_preserves_chunk_fields(self) -> None:
        task = build_self_play_tasks(
            run_id="run-test",
            config=SelfPlayRunConfig(
                games=10,
                chunk_size=3,
                seed=7,
                planner_config=PlannerConfig(max_depth=2, beam_width=4, opponent_branch_width=2),
                max_actions_per_game=150,
                include_setup_decisions=True,
                record_forced_actions=True,
            ),
        )[1]

        rebuilt = self_play_task_from_payload(self_play_task_to_payload(task))

        self.assertEqual(rebuilt.run_id, task.run_id)
        self.assertEqual(rebuilt.task_index, task.task_index)
        self.assertEqual(rebuilt.start_index, task.start_index)
        self.assertEqual(rebuilt.game_count, task.game_count)
        self.assertEqual(rebuilt.planner_config.max_depth, task.planner_config.max_depth)
        self.assertTrue(rebuilt.include_setup_decisions)
        self.assertTrue(rebuilt.record_forced_actions)

    def test_encode_and_decode_uploaded_chunk_text_round_trip(self) -> None:
        text = '{"hello":"world"}\n{"game":2}\n'
        payload = {"decisions_gzip_b64": encode_chunk_text_gzip_b64(text)}

        decoded = decode_uploaded_chunk_text(payload, "decisions")

        self.assertEqual(decoded, text)

    def test_coordinator_leases_and_accepts_chunk_submission(self) -> None:
        coordinator = DistributedSelfPlayCoordinator(
            output_dir=self.output_dir,
            run_id="run-test",
            run_config=SelfPlayRunConfig(
                games=4,
                chunk_size=2,
                seed=11,
            ),
            lease_timeout_seconds=60,
        )

        lease = coordinator.lease_chunk(
            worker_id="worker-a",
            worker_meta={"hostname": "host-a", "platform": "linux"},
        )

        self.assertFalse(lease["run_complete"])
        self.assertIsNotNone(lease["task"])
        self.assertEqual(lease["task"]["task_index"], 0)
        status_after_lease = lease["status"]
        self.assertEqual(status_after_lease["leased_tasks"], 1)
        self.assertEqual(status_after_lease["completed_tasks"], 0)

        heartbeat = coordinator.heartbeat(
            worker_id="worker-a",
            task_index=0,
            worker_meta={"hostname": "host-a", "platform": "linux"},
        )
        self.assertTrue(heartbeat["ok"])

        progress = coordinator.record_progress(
            worker_id="worker-a",
            task_index=0,
            worker_meta={"hostname": "host-a", "platform": "linux", "poll_seconds": 2, "heartbeat_interval_seconds": 5},
            progress={
                "local_game_index": 1,
                "task_game_count": 2,
                "global_game_index": 1,
                "winner_deck_id": "ampharos-ex-battle-deck",
                "turns": 9,
                "actions": 41,
                "samples": 6,
                "truncated": False,
                "duration_seconds": 1.25,
            },
        )
        self.assertTrue(progress["ok"])
        live_worker = progress["status"]["workers"]["worker-a"]
        self.assertEqual(live_worker["completed_games"], 1)
        self.assertEqual(live_worker["current_task_completed_games"], 1)
        self.assertEqual(live_worker["status"], "busy")
        self.assertGreaterEqual(progress["status"]["throughput"]["games_per_minute_1m"], 1.0)
        self.assertEqual(progress["status"]["reported"]["deck_wins"]["ampharos-ex-battle-deck"], 1)

        submit = coordinator.submit_chunk(
            worker_id="worker-a",
            task_index=0,
            summary={
                "games": 2,
                "samples": 7,
                "truncated": 0,
                "deck_wins": {
                    "ampharos-ex-battle-deck": 1,
                    "lucario-ex-battle-deck": 1,
                },
                "turns": 18,
                "actions": 110,
            },
            decisions_jsonl='{"decision":1}\n',
            games_jsonl='{"game":1}\n{"game":2}\n',
            worker_meta={"hostname": "host-a", "platform": "linux"},
        )

        self.assertTrue(submit["ok"])
        self.assertFalse(submit["duplicate"])
        status_after_submit = submit["status"]
        self.assertEqual(status_after_submit["completed_tasks"], 1)
        self.assertEqual(status_after_submit["aggregate"]["games"], 2)
        self.assertEqual(status_after_submit["workers"]["worker-a"]["submitted_tasks"], 1)
        self.assertEqual(status_after_submit["workers"]["worker-a"]["status"], "idle")
        self.assertTrue((self.output_dir / "decisions" / "shard_000000.jsonl").exists())
        self.assertTrue((self.output_dir / "games" / "shard_000000.jsonl").exists())
        self.assertTrue((self.output_dir / "manifest.json").exists())
        self.assertTrue((self.output_dir / "summary.json").exists())

    def test_coordinator_reissues_expired_leases(self) -> None:
        coordinator = DistributedSelfPlayCoordinator(
            output_dir=self.output_dir,
            run_id="run-test-expire",
            run_config=SelfPlayRunConfig(
                games=4,
                chunk_size=2,
                seed=22,
            ),
            lease_timeout_seconds=1,
        )

        first_lease = coordinator.lease_chunk(worker_id="worker-a", worker_meta=None)
        self.assertEqual(first_lease["task"]["task_index"], 0)

        state = coordinator._state  # noqa: SLF001
        state["tasks"][0]["lease_expires_at"] = "2000-01-01T00:00:00+00:00"
        state["tasks"][0]["leased_by"] = "worker-a"
        state["tasks"][0]["status"] = "leased"
        coordinator._save_state()  # noqa: SLF001

        second_lease = coordinator.lease_chunk(worker_id="worker-b", worker_meta=None)
        self.assertEqual(second_lease["task"]["task_index"], 0)
        self.assertEqual(second_lease["status"]["workers"]["worker-b"]["leased_task_index"], 0)

    def test_lease_heartbeat_loop_posts_periodically(self) -> None:
        calls: list[tuple[str, dict[str, object], float]] = []

        def fake_post_json(url: str, *, payload: dict[str, object], timeout_seconds: float) -> dict[str, object]:
            calls.append((url, payload, timeout_seconds))
            return {"ok": True}

        heartbeat = LeaseHeartbeatLoop(
            coordinator_url="http://127.0.0.1:8787",
            worker_id="worker-a",
            worker_meta={"hostname": "host-a"},
            task_index=3,
            heartbeat_interval_seconds=1.0,
            request_timeout_seconds=2.0,
            post_json=fake_post_json,
        )

        heartbeat.start()
        time.sleep(1.15)
        heartbeat.stop()

        self.assertGreaterEqual(len(calls), 1)
        url, payload, timeout_seconds = calls[0]
        self.assertEqual(url, "http://127.0.0.1:8787/api/standard-self-play/heartbeat")
        self.assertEqual(payload["worker_id"], "worker-a")
        self.assertEqual(payload["task_index"], 3)
        self.assertEqual(timeout_seconds, 2.0)


if __name__ == "__main__":
    unittest.main()
