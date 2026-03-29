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
    SelfPlayChunkResult,
    SelfPlayRunConfig,
    build_self_play_tasks,
    self_play_chunk_artifact_paths,
    self_play_task_from_payload,
    self_play_task_to_payload,
    write_self_play_chunk_artifacts,
)
from scripts.run_standard_self_play_worker import LeaseHeartbeatLoop
from scripts.run_standard_self_play_worker import _call_coordinator_with_retries
from scripts.run_standard_self_play_worker import _handle_idle_lease_response
from scripts.run_standard_self_play_worker import CoordinatorUnavailableError


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

    def test_coordinator_recovers_completed_shards_from_disk(self) -> None:
        write_self_play_chunk_artifacts(
            output_dir=self.output_dir,
            result=SelfPlayChunkResult(
                task_index=0,
                summary={
                    "games": 2,
                    "samples": 3,
                    "truncated": 1,
                    "deck_wins": {
                        "ampharos-ex-battle-deck": 1,
                        "lucario-ex-battle-deck": 1,
                    },
                    "turns": 18,
                    "actions": 57,
                },
                decisions_jsonl='{"decision":1}\n{"decision":2}\n{"decision":3}\n',
                games_jsonl=(
                    '{"winner_deck_id":"ampharos-ex-battle-deck","turn_number":9,"action_count":21,"truncated":false}\n'
                    '{"winner_deck_id":"lucario-ex-battle-deck","turn_number":9,"action_count":36,"truncated":true}\n'
                ),
            ),
        )
        artifact_paths = self_play_chunk_artifact_paths(output_dir=self.output_dir, task_index=0)
        artifact_paths.summary.unlink()

        coordinator = DistributedSelfPlayCoordinator(
            output_dir=self.output_dir,
            run_id="run-recover",
            run_config=SelfPlayRunConfig(
                games=4,
                chunk_size=2,
                seed=33,
            ),
            lease_timeout_seconds=60,
        )

        status = coordinator.status()
        self.assertEqual(status["completed_tasks"], 1)
        self.assertEqual(status["pending_tasks"], 1)
        self.assertEqual(status["aggregate"]["games"], 2)
        self.assertEqual(status["aggregate"]["samples"], 3)
        self.assertEqual(status["recovery"]["upgraded_legacy_summaries"], 1)
        self.assertTrue(artifact_paths.summary.exists())

        lease = coordinator.lease_chunk(worker_id="worker-b", worker_meta=None)
        self.assertEqual(lease["task"]["task_index"], 1)

    def test_coordinator_rebuilds_from_corrupt_state_file(self) -> None:
        write_self_play_chunk_artifacts(
            output_dir=self.output_dir,
            result=SelfPlayChunkResult(
                task_index=0,
                summary={
                    "games": 2,
                    "samples": 1,
                    "truncated": 0,
                    "deck_wins": {
                        "ampharos-ex-battle-deck": 2,
                        "lucario-ex-battle-deck": 0,
                    },
                    "turns": 12,
                    "actions": 24,
                },
                decisions_jsonl='{"decision":1}\n',
                games_jsonl=(
                    '{"winner_deck_id":"ampharos-ex-battle-deck","turn_number":6,"action_count":12,"truncated":false}\n'
                    '{"winner_deck_id":"ampharos-ex-battle-deck","turn_number":6,"action_count":12,"truncated":false}\n'
                ),
            ),
        )
        (self.output_dir / "coordinator_state.json").write_text("{not-valid-json", encoding="utf-8")

        coordinator = DistributedSelfPlayCoordinator(
            output_dir=self.output_dir,
            run_id="run-corrupt",
            run_config=SelfPlayRunConfig(
                games=4,
                chunk_size=2,
                seed=44,
            ),
            lease_timeout_seconds=60,
        )

        status = coordinator.status()
        self.assertEqual(status["completed_tasks"], 1)
        self.assertGreaterEqual(status["recovery"]["integrity_issue_count"], 1)
        self.assertEqual(status["recovery"]["integrity_issues"][0]["reason"], "state_rebuild")

    def test_coordinator_keeps_incomplete_artifacts_pending(self) -> None:
        artifact_paths = self_play_chunk_artifact_paths(output_dir=self.output_dir, task_index=0)
        artifact_paths.decisions.parent.mkdir(parents=True, exist_ok=True)
        artifact_paths.decisions.write_text('{"decision":1}\n', encoding="utf-8")

        coordinator = DistributedSelfPlayCoordinator(
            output_dir=self.output_dir,
            run_id="run-incomplete",
            run_config=SelfPlayRunConfig(
                games=4,
                chunk_size=2,
                seed=55,
            ),
            lease_timeout_seconds=60,
        )

        status = coordinator.status()
        self.assertEqual(status["completed_tasks"], 0)
        self.assertEqual(status["pending_tasks"], 2)
        self.assertGreaterEqual(status["recovery"]["integrity_issue_count"], 1)
        self.assertEqual(status["recovery"]["integrity_issues"][0]["reason"], "incomplete_artifacts")

        lease = coordinator.lease_chunk(worker_id="worker-c", worker_meta=None)
        self.assertEqual(lease["task"]["task_index"], 0)

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

    def test_worker_retries_until_coordinator_returns(self) -> None:
        attempts = {"count": 0}
        sleep_calls: list[float] = []

        def flaky_request() -> dict[str, object]:
            attempts["count"] += 1
            if attempts["count"] < 3:
                raise CoordinatorUnavailableError("coordinator offline")
            return {"ok": True}

        response = _call_coordinator_with_retries(
            operation_label="lease request",
            retry_interval_seconds=300.0,
            request=flaky_request,
            sleep_fn=sleep_calls.append,
        )

        self.assertEqual(response, {"ok": True})
        self.assertEqual(attempts["count"], 3)
        self.assertEqual(sleep_calls, [300.0, 300.0])

    def test_worker_stays_alive_when_run_is_complete(self) -> None:
        sleep_calls: list[float] = []

        should_exit = _handle_idle_lease_response(
            lease_response={"task": None, "run_complete": True},
            poll_seconds=5.0,
            reconnect_seconds=300.0,
            exit_when_run_complete=False,
            sleep_fn=sleep_calls.append,
        )

        self.assertFalse(should_exit)
        self.assertEqual(sleep_calls, [300.0])

    def test_worker_can_exit_when_run_is_complete_if_requested(self) -> None:
        should_exit = _handle_idle_lease_response(
            lease_response={"task": None, "run_complete": True},
            poll_seconds=5.0,
            reconnect_seconds=300.0,
            exit_when_run_complete=True,
            sleep_fn=lambda _seconds: None,
        )

        self.assertTrue(should_exit)


if __name__ == "__main__":
    unittest.main()
