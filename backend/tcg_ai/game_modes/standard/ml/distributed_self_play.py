from __future__ import annotations

from datetime import UTC, datetime, timedelta
import json
import os
from pathlib import Path
import threading
import time
from typing import Any

from .self_play_jobs import (
    SELF_PLAY_JOB_SCHEMA_VERSION,
    SelfPlayChunkResult,
    SelfPlayRunConfig,
    build_self_play_manifest,
    build_self_play_tasks,
    derive_self_play_chunk_summary_from_artifacts,
    empty_self_play_summary,
    load_self_play_chunk_summary,
    merge_chunk_summary,
    resolve_oracle_status,
    self_play_chunk_artifact_paths,
    self_play_run_config_from_payload,
    self_play_run_config_to_payload,
    self_play_task_to_payload,
    write_self_play_chunk_artifacts,
)

DEFAULT_DECK_WINS = {
    "ampharos-ex-battle-deck": 0,
    "lucario-ex-battle-deck": 0,
}
PROGRESS_BUCKET_RETENTION_MINUTES = 180
THROUGHPUT_SERIES_MINUTES = 30


class DistributedSelfPlayCoordinator:
    def __init__(
        self,
        *,
        output_dir: Path,
        run_id: str,
        run_config: SelfPlayRunConfig,
        lease_timeout_seconds: int = 1800,
    ) -> None:
        if lease_timeout_seconds <= 0:
            raise ValueError("lease_timeout_seconds must be positive.")
        self.output_dir = output_dir.resolve()
        self.run_id = run_id
        self.run_config = run_config
        self.lease_timeout_seconds = lease_timeout_seconds
        self.state_path = self.output_dir / "coordinator_state.json"
        self.summary_path = self.output_dir / "summary.json"
        self.manifest_path = self.output_dir / "manifest.json"
        self.completion_flag_path = self.output_dir / "RUN_COMPLETE"
        self._lock = threading.RLock()
        self._last_progress_persist_monotonic = 0.0
        self._recovery_report: dict[str, Any] = {
            "reconciled_at": None,
            "completed_tasks_from_artifacts": 0,
            "upgraded_legacy_summaries": 0,
            "integrity_issue_count": 0,
            "integrity_issues": [],
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._state = self._load_or_initialize_state()

    def lease_chunk(
        self,
        *,
        worker_id: str,
        worker_meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            now = _utc_now()
            self._reclaim_expired_leases(now)
            self._prune_progress_buckets(now)
            self._mark_worker_seen(worker_id, worker_meta, now)
            next_task = next(
                (entry for entry in self._state["tasks"] if entry["status"] == "pending"),
                None,
            )
            if next_task is None:
                self._save_state()
                return {
                    "task": None,
                    "run_complete": self._all_tasks_completed(),
                    "status": self.status(),
                }

            next_task["status"] = "leased"
            next_task["leased_by"] = worker_id
            next_task["leased_at"] = now.isoformat()
            next_task["lease_expires_at"] = (now + timedelta(seconds=self.lease_timeout_seconds)).isoformat()
            worker_state = self._state["workers"][worker_id]
            worker_state["leased_task_index"] = int(next_task["task"]["task_index"])
            worker_state["current_task_started_at"] = now.isoformat()
            worker_state["current_task_game_count"] = int(next_task["task"]["game_count"])
            worker_state["current_task_completed_games"] = 0
            self._save_state()
            return {
                "task": next_task["task"],
                "run_complete": False,
                "status": self.status(),
            }

    def heartbeat(
        self,
        *,
        worker_id: str,
        task_index: int,
        worker_meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            now = _utc_now()
            self._reclaim_expired_leases(now)
            self._prune_progress_buckets(now)
            self._mark_worker_seen(worker_id, worker_meta, now)
            entry = self._task_entry(task_index)
            accepted = False
            if entry is not None and entry["status"] == "leased" and entry["leased_by"] == worker_id:
                entry["lease_expires_at"] = (now + timedelta(seconds=self.lease_timeout_seconds)).isoformat()
                accepted = True
            self._save_state()
            return {
                "ok": accepted,
                "run_complete": self._all_tasks_completed(),
                "status": self.status(),
            }

    def record_progress(
        self,
        *,
        worker_id: str,
        task_index: int,
        progress: dict[str, Any],
        worker_meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            now = _utc_now()
            self._reclaim_expired_leases(now)
            self._prune_progress_buckets(now)
            self._mark_worker_seen(worker_id, worker_meta, now)
            entry = self._task_entry(task_index)
            accepted = False
            if entry is not None and entry["status"] == "leased" and entry["leased_by"] == worker_id:
                accepted = True
                worker_state = self._state["workers"][worker_id]
                local_game_index = max(1, _coerce_int(progress.get("local_game_index"), default=1))
                task_game_count = max(
                    1,
                    _coerce_int(progress.get("task_game_count"), default=worker_state["current_task_game_count"] or 1),
                )
                worker_state["current_task_game_count"] = task_game_count
                worker_state["current_task_completed_games"] = max(
                    int(worker_state.get("current_task_completed_games", 0)),
                    local_game_index,
                )
                worker_state["last_progress_at"] = now.isoformat()
                worker_state["last_completed_global_game"] = _coerce_int(progress.get("global_game_index"), default=0)
                worker_state["last_duration_seconds"] = _coerce_float(progress.get("duration_seconds"))
                worker_state["completed_games"] += 1
                worker_state["completed_actions"] += _coerce_int(progress.get("actions"), default=0)
                worker_state["completed_turns"] += _coerce_int(progress.get("turns"), default=0)
                worker_state["completed_samples"] += _coerce_int(progress.get("samples"), default=0)
                worker_state["truncated_games"] += 1 if bool(progress.get("truncated")) else 0
                winner_deck_id = progress.get("winner_deck_id")
                if isinstance(winner_deck_id, str):
                    worker_state["deck_wins"].setdefault(winner_deck_id, 0)
                    worker_state["deck_wins"][winner_deck_id] += 1
                _update_progress_buckets(
                    self._state["progress_buckets"],
                    now,
                    games=1,
                    actions=_coerce_int(progress.get("actions"), default=0),
                    turns=_coerce_int(progress.get("turns"), default=0),
                    samples=_coerce_int(progress.get("samples"), default=0),
                    duration_seconds=_coerce_float(progress.get("duration_seconds")) or 0.0,
                )
                _update_progress_buckets(
                    worker_state["progress_buckets"],
                    now,
                    games=1,
                    actions=_coerce_int(progress.get("actions"), default=0),
                    turns=_coerce_int(progress.get("turns"), default=0),
                    samples=_coerce_int(progress.get("samples"), default=0),
                    duration_seconds=_coerce_float(progress.get("duration_seconds")) or 0.0,
                )
                self._save_state_throttled()
            return {
                "ok": accepted,
                "run_complete": self._all_tasks_completed(),
                "status": self.status(),
            }

    def submit_chunk(
        self,
        *,
        worker_id: str,
        task_index: int,
        summary: dict[str, Any],
        decisions_jsonl: str,
        games_jsonl: str,
        worker_meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            now = _utc_now()
            self._reclaim_expired_leases(now)
            self._prune_progress_buckets(now)
            self._mark_worker_seen(worker_id, worker_meta, now)
            entry = self._task_entry(task_index)
            if entry is None:
                raise ValueError(f"Unknown task index: {task_index}")
            if entry["status"] == "completed":
                return {
                    "ok": True,
                    "duplicate": True,
                    "run_complete": self._all_tasks_completed(),
                    "status": self.status(),
                }
            if entry["status"] != "leased" or entry["leased_by"] != worker_id:
                raise ValueError(f"Task {task_index} is not currently leased by worker '{worker_id}'.")

            result = SelfPlayChunkResult(
                task_index=task_index,
                summary=dict(summary),
                decisions_jsonl=decisions_jsonl,
                games_jsonl=games_jsonl,
            )
            write_self_play_chunk_artifacts(output_dir=self.output_dir, result=result)
            merge_chunk_summary(self._state["aggregate"], result.summary)

            entry["status"] = "completed"
            entry["leased_by"] = None
            entry["leased_at"] = None
            entry["lease_expires_at"] = None
            entry["submitted_at"] = now.isoformat()
            worker_state = self._state["workers"][worker_id]
            worker_state["leased_task_index"] = None
            worker_state["submitted_tasks"] = int(worker_state.get("submitted_tasks", 0)) + 1
            worker_state["last_chunk_submitted_at"] = now.isoformat()
            worker_state["last_completed_task_index"] = task_index
            worker_state["current_task_started_at"] = None
            worker_state["current_task_game_count"] = 0
            worker_state["current_task_completed_games"] = 0

            self._write_summary_file(completed_at=now if self._all_tasks_completed() else None)
            self._save_state()
            return {
                "ok": True,
                "duplicate": False,
                "run_complete": self._all_tasks_completed(),
                "status": self.status(),
            }

    def status(self) -> dict[str, Any]:
        with self._lock:
            now = _utc_now()
            self._reclaim_expired_leases(now)
            self._prune_progress_buckets(now)
            tasks = self._state["tasks"]
            workers = {
                worker_id: self._serialize_worker_status(worker_id, worker_state, now)
                for worker_id, worker_state in self._state["workers"].items()
            }
            reported = _build_reported_totals(workers)
            created_at_text = self._state.get("created_at")
            created_at = datetime.fromisoformat(created_at_text) if isinstance(created_at_text, str) else now
            elapsed_seconds = max((now - created_at).total_seconds(), 0.0)
            throughput = _build_throughput_metrics(
                buckets=self._state["progress_buckets"],
                now=now,
                elapsed_seconds=elapsed_seconds,
                total_games_target=int(self.run_config.games),
                reported_games=reported["games"],
            )
            return {
                "schema_version": SELF_PLAY_JOB_SCHEMA_VERSION,
                "run_id": self.run_id,
                "output_dir": str(self.output_dir),
                "created_at": created_at.isoformat(),
                "elapsed_seconds": round(elapsed_seconds, 3),
                "total_games_target": int(self.run_config.games),
                "total_tasks": len(tasks),
                "pending_tasks": sum(1 for entry in tasks if entry["status"] == "pending"),
                "leased_tasks": sum(1 for entry in tasks if entry["status"] == "leased"),
                "completed_tasks": sum(1 for entry in tasks if entry["status"] == "completed"),
                "run_complete": self._all_tasks_completed(),
                "aggregate": dict(self._state["aggregate"]),
                "reported": reported,
                "recovery": dict(self._recovery_report),
                "throughput": throughput,
                "throughput_series": _bucket_series(
                    self._state["progress_buckets"],
                    now=now,
                    window_minutes=THROUGHPUT_SERIES_MINUTES,
                ),
                "workers": workers,
            }

    def persist_state(self) -> None:
        with self._lock:
            now = _utc_now()
            self._reclaim_expired_leases(now)
            self._prune_progress_buckets(now)
            self._write_summary_file(completed_at=now if self._all_tasks_completed() else None)
            self._save_state()

    def _load_or_initialize_state(self) -> dict[str, Any]:
        bootstrap_issues: list[dict[str, Any]] = []
        if self.state_path.exists():
            try:
                payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                normalized = self._build_default_state()
                bootstrap_issues.append(
                    {
                        "task_index": None,
                        "reason": "state_rebuild",
                        "detail": f"Coordinator state could not be loaded cleanly and was rebuilt: {exc}",
                    }
                )
            else:
                if payload.get("run_id") != self.run_id:
                    raise ValueError("Existing coordinator state run_id does not match.")
                try:
                    self.run_config = self_play_run_config_from_payload(dict(payload["run_config"]))
                    normalized = self._normalize_state(payload)
                except (KeyError, TypeError, ValueError) as exc:
                    normalized = self._build_default_state()
                    bootstrap_issues.append(
                        {
                            "task_index": None,
                            "reason": "state_rebuild",
                            "detail": f"Coordinator state could not be normalized and was rebuilt: {exc}",
                        }
                    )
        else:
            normalized = self._build_default_state()

        self._write_manifest_if_missing()
        self._recovery_report = self._reconcile_state_with_artifacts(normalized)
        if bootstrap_issues:
            self._recovery_report["integrity_issues"] = (
                bootstrap_issues + list(self._recovery_report["integrity_issues"])
            )[:50]
            self._recovery_report["integrity_issue_count"] = int(self._recovery_report["integrity_issue_count"]) + len(
                bootstrap_issues
            )
        self._reclaim_expired_leases_in_state(normalized, _utc_now())
        self._write_summary_file_for_state(
            normalized,
            completed_at=_utc_now() if all(entry["status"] == "completed" for entry in normalized["tasks"]) else None,
        )
        _atomic_write_json(self.state_path, normalized)
        return normalized

    def _build_default_state(self) -> dict[str, Any]:
        return self._normalize_state(
            {
                "schema_version": SELF_PLAY_JOB_SCHEMA_VERSION,
                "run_id": self.run_id,
                "created_at": _utc_now().isoformat(),
                "lease_timeout_seconds": self.lease_timeout_seconds,
                "run_config": self_play_run_config_to_payload(self.run_config),
                "tasks": [
                    {
                        "task": self_play_task_to_payload(task),
                        "status": "pending",
                        "leased_by": None,
                        "leased_at": None,
                        "lease_expires_at": None,
                        "submitted_at": None,
                    }
                    for task in build_self_play_tasks(run_id=self.run_id, config=self.run_config)
                ],
                "aggregate": empty_self_play_summary(),
                "progress_buckets": {},
                "workers": {},
            }
        )

    def _normalize_state(self, payload: dict[str, Any]) -> dict[str, Any]:
        aggregate = payload.get("aggregate")
        normalized_aggregate = empty_self_play_summary()
        if isinstance(aggregate, dict):
            normalized_aggregate["games"] = _coerce_int(aggregate.get("games"), default=0)
            normalized_aggregate["samples"] = _coerce_int(aggregate.get("samples"), default=0)
            normalized_aggregate["truncated"] = _coerce_int(aggregate.get("truncated"), default=0)
            normalized_aggregate["turns"] = _coerce_int(aggregate.get("turns"), default=0)
            normalized_aggregate["actions"] = _coerce_int(aggregate.get("actions"), default=0)
            for deck_id, wins in (aggregate.get("deck_wins") or {}).items():
                if isinstance(deck_id, str):
                    normalized_aggregate["deck_wins"][deck_id] = _coerce_int(wins, default=0)
        payload["aggregate"] = normalized_aggregate
        payload["progress_buckets"] = _normalize_progress_buckets(payload.get("progress_buckets"))

        workers_payload = payload.get("workers")
        normalized_workers: dict[str, dict[str, Any]] = {}
        if isinstance(workers_payload, dict):
            for worker_id, worker_state in workers_payload.items():
                if isinstance(worker_id, str):
                    normalized_workers[worker_id] = _normalize_worker_state(worker_id, worker_state)
        payload["workers"] = normalized_workers
        return payload

    def _reconcile_state_with_artifacts(self, state: dict[str, Any]) -> dict[str, Any]:
        recovered_summaries: dict[int, dict[str, Any]] = {}
        integrity_issues: list[dict[str, Any]] = []
        upgraded_legacy_summaries = 0
        known_task_indices = {int(entry["task"]["task_index"]) for entry in state["tasks"]}

        for entry in state["tasks"]:
            task_index = int(entry["task"]["task_index"])
            expected_games = int(entry["task"]["game_count"])
            artifact_paths = self_play_chunk_artifact_paths(output_dir=self.output_dir, task_index=task_index)
            decisions_exists = artifact_paths.decisions.exists()
            games_exists = artifact_paths.games.exists()
            summary_exists = artifact_paths.summary.exists()

            summary: dict[str, Any] | None = None
            if summary_exists and decisions_exists and games_exists:
                try:
                    summary = load_self_play_chunk_summary(output_dir=self.output_dir, task_index=task_index)
                except ValueError as exc:
                    integrity_issues.append(
                        {
                            "task_index": task_index,
                            "reason": "invalid_summary",
                            "detail": str(exc),
                        }
                    )
            elif decisions_exists and games_exists:
                try:
                    summary = derive_self_play_chunk_summary_from_artifacts(
                        output_dir=self.output_dir,
                        task_index=task_index,
                    )
                    _atomic_write_json(artifact_paths.summary, summary)
                    upgraded_legacy_summaries += 1
                except ValueError as exc:
                    integrity_issues.append(
                        {
                            "task_index": task_index,
                            "reason": "invalid_artifacts",
                            "detail": str(exc),
                        }
                    )
            elif decisions_exists or games_exists or summary_exists:
                integrity_issues.append(
                    {
                        "task_index": task_index,
                        "reason": "incomplete_artifacts",
                        "detail": _artifact_presence_detail(
                            decisions_exists=decisions_exists,
                            games_exists=games_exists,
                            summary_exists=summary_exists,
                        ),
                    }
                )

            if summary is None:
                if entry["status"] == "completed":
                    integrity_issues.append(
                        {
                            "task_index": task_index,
                            "reason": "missing_persisted_chunk",
                            "detail": "Coordinator state marked this task completed, but no recoverable shard was found.",
                        }
                    )
                    self._clear_task_lease(entry)
                    entry["status"] = "pending"
                    entry["submitted_at"] = None
                continue

            if int(summary["games"]) != expected_games:
                integrity_issues.append(
                    {
                        "task_index": task_index,
                        "reason": "unexpected_game_count",
                        "detail": f"Expected {expected_games} games but recovered {summary['games']}.",
                    }
                )
                if entry["status"] == "completed":
                    self._clear_task_lease(entry)
                    entry["status"] = "pending"
                    entry["submitted_at"] = None
                continue

            recovered_summaries[task_index] = summary
            entry["status"] = "completed"
            self._clear_task_lease(entry)
            if not isinstance(entry.get("submitted_at"), str):
                entry["submitted_at"] = _artifact_submitted_at(artifact_paths)

        aggregate = empty_self_play_summary()
        for task_index in sorted(recovered_summaries):
            merge_chunk_summary(aggregate, recovered_summaries[task_index])
        state["aggregate"] = aggregate
        self._normalize_worker_leases(state)

        unexpected_indices = _scan_unexpected_artifact_indices(output_dir=self.output_dir, known_task_indices=known_task_indices)
        for task_index in unexpected_indices:
            integrity_issues.append(
                {
                    "task_index": task_index,
                    "reason": "unexpected_artifacts",
                    "detail": "Found shard artifacts that do not belong to this run configuration.",
                }
            )

        return {
            "reconciled_at": _utc_now().isoformat(),
            "completed_tasks_from_artifacts": len(recovered_summaries),
            "upgraded_legacy_summaries": upgraded_legacy_summaries,
            "integrity_issue_count": len(integrity_issues),
            "integrity_issues": integrity_issues[:50],
        }

    def _write_manifest_if_missing(self) -> None:
        if self.manifest_path.exists():
            return
        manifest = build_self_play_manifest(
            run_id=self.run_id,
            config=self.run_config,
            oracle_status=resolve_oracle_status(
                oracle=self.run_config.oracle,
                checkpoint=Path(self.run_config.checkpoint) if self.run_config.checkpoint else None,
            ),
        )
        manifest["execution_mode"] = "distributed_coordinator"
        manifest["lease_timeout_seconds"] = self.lease_timeout_seconds
        _atomic_write_json(self.manifest_path, manifest)

    def _write_summary_file_if_missing(self, state: dict[str, Any]) -> None:
        if self.summary_path.exists():
            return
        self._write_summary_file_for_state(state, completed_at=None)

    def _write_summary_file(self, *, completed_at: datetime | None) -> None:
        self._write_summary_file_for_state(self._state, completed_at=completed_at)

    def _write_summary_file_for_state(self, state: dict[str, Any], *, completed_at: datetime | None) -> None:
        payload = {
            "run_id": self.run_id,
            "completed_at": completed_at.isoformat() if completed_at is not None else None,
            **state["aggregate"],
        }
        _atomic_write_json(self.summary_path, payload)
        self._write_completion_flag(completed_at=completed_at)

    def _write_completion_flag(self, *, completed_at: datetime | None) -> None:
        if completed_at is None:
            try:
                self.completion_flag_path.unlink()
            except FileNotFoundError:
                return
            return
        _atomic_write_json(
            self.completion_flag_path,
            {
                "run_id": self.run_id,
                "completed_at": completed_at.isoformat(),
            },
        )

    def _task_entry(self, task_index: int) -> dict[str, Any] | None:
        for entry in self._state["tasks"]:
            if int(entry["task"]["task_index"]) == task_index:
                return entry
        return None

    def _clear_task_lease(self, entry: dict[str, Any]) -> None:
        entry["leased_by"] = None
        entry["leased_at"] = None
        entry["lease_expires_at"] = None

    def _normalize_worker_leases(self, state: dict[str, Any]) -> None:
        leased_task_indices = {
            int(entry["task"]["task_index"])
            for entry in state["tasks"]
            if entry["status"] == "leased"
        }
        for worker_state in state["workers"].values():
            leased_task_index = worker_state.get("leased_task_index")
            if leased_task_index in leased_task_indices:
                continue
            worker_state["leased_task_index"] = None
            worker_state["current_task_started_at"] = None
            worker_state["current_task_game_count"] = 0
            worker_state["current_task_completed_games"] = 0

    def _reclaim_expired_leases(self, now: datetime) -> None:
        self._reclaim_expired_leases_in_state(self._state, now)

    def _reclaim_expired_leases_in_state(self, state: dict[str, Any], now: datetime) -> None:
        for entry in state["tasks"]:
            if entry["status"] != "leased":
                continue
            expiry_text = entry.get("lease_expires_at")
            if not isinstance(expiry_text, str):
                continue
            expires_at = datetime.fromisoformat(expiry_text)
            if expires_at > now:
                continue
            worker_id = entry.get("leased_by")
            if isinstance(worker_id, str) and worker_id in state["workers"]:
                worker_state = state["workers"][worker_id]
                worker_state["leased_task_index"] = None
                worker_state["current_task_started_at"] = None
                worker_state["current_task_game_count"] = 0
                worker_state["current_task_completed_games"] = 0
            entry["status"] = "pending"
            entry["leased_by"] = None
            entry["leased_at"] = None
            entry["lease_expires_at"] = None

    def _prune_progress_buckets(self, now: datetime) -> None:
        _prune_progress_buckets(self._state["progress_buckets"], now)
        for worker_state in self._state["workers"].values():
            _prune_progress_buckets(worker_state["progress_buckets"], now)

    def _mark_worker_seen(
        self,
        worker_id: str,
        worker_meta: dict[str, Any] | None,
        now: datetime,
    ) -> None:
        worker_state = self._state["workers"].setdefault(worker_id, _default_worker_state(worker_id))
        worker_state["last_seen_at"] = now.isoformat()
        if not isinstance(worker_meta, dict):
            return
        for key in ("hostname", "platform", "python_version"):
            value = worker_meta.get(key)
            if isinstance(value, str) and value:
                worker_state[key] = value
        machine_name = worker_meta.get("machine_name")
        if isinstance(machine_name, str) and machine_name:
            worker_state["machine_name"] = machine_name
        for key in ("poll_seconds", "heartbeat_interval_seconds"):
            value = worker_meta.get(key)
            numeric_value = _coerce_float(value)
            if numeric_value is not None and numeric_value > 0:
                worker_state[key] = numeric_value

    def _serialize_worker_status(
        self,
        worker_id: str,
        worker_state: dict[str, Any],
        now: datetime,
    ) -> dict[str, Any]:
        seconds_since_seen = _seconds_since(worker_state.get("last_seen_at"), now)
        seconds_since_progress = _seconds_since(worker_state.get("last_progress_at"), now)
        contact_threshold_seconds = _contact_threshold_seconds(worker_state)
        leased_task_index = worker_state.get("leased_task_index")
        is_online = seconds_since_seen is not None and seconds_since_seen <= contact_threshold_seconds
        if is_online:
            status = "busy" if leased_task_index is not None else "idle"
        else:
            status = "stalled" if leased_task_index is not None else "offline"
        current_task_game_count = int(worker_state.get("current_task_game_count", 0) or 0)
        current_task_completed_games = int(worker_state.get("current_task_completed_games", 0) or 0)
        current_task_progress = (
            round(current_task_completed_games / current_task_game_count, 4)
            if current_task_game_count > 0
            else 0.0
        )
        completed_games = int(worker_state.get("completed_games", 0) or 0)
        completed_turns = int(worker_state.get("completed_turns", 0) or 0)
        completed_actions = int(worker_state.get("completed_actions", 0) or 0)
        completed_samples = int(worker_state.get("completed_samples", 0) or 0)
        current_task_elapsed_seconds = _seconds_since(worker_state.get("current_task_started_at"), now)
        recent_1m = _window_totals(worker_state["progress_buckets"], now=now, window_minutes=1)
        recent_5m = _window_totals(worker_state["progress_buckets"], now=now, window_minutes=5)
        return {
            "worker_id": worker_id,
            "machine_name": worker_state.get("machine_name"),
            "hostname": worker_state.get("hostname"),
            "platform": worker_state.get("platform"),
            "python_version": worker_state.get("python_version"),
            "status": status,
            "is_online": is_online,
            "last_seen_at": worker_state.get("last_seen_at"),
            "last_progress_at": worker_state.get("last_progress_at"),
            "seconds_since_seen": round(seconds_since_seen, 3) if seconds_since_seen is not None else None,
            "seconds_since_progress": round(seconds_since_progress, 3) if seconds_since_progress is not None else None,
            "contact_threshold_seconds": contact_threshold_seconds,
            "leased_task_index": leased_task_index,
            "submitted_tasks": int(worker_state.get("submitted_tasks", 0) or 0),
            "completed_games": completed_games,
            "completed_actions": completed_actions,
            "completed_turns": completed_turns,
            "completed_samples": completed_samples,
            "truncated_games": int(worker_state.get("truncated_games", 0) or 0),
            "average_turns_per_game": round(completed_turns / completed_games, 3) if completed_games else 0.0,
            "average_actions_per_game": round(completed_actions / completed_games, 3) if completed_games else 0.0,
            "average_samples_per_game": round(completed_samples / completed_games, 3) if completed_games else 0.0,
            "deck_wins": dict(worker_state.get("deck_wins") or DEFAULT_DECK_WINS),
            "current_task_started_at": worker_state.get("current_task_started_at"),
            "current_task_game_count": current_task_game_count,
            "current_task_completed_games": current_task_completed_games,
            "current_task_progress": current_task_progress,
            "current_task_elapsed_seconds": round(current_task_elapsed_seconds, 3) if current_task_elapsed_seconds is not None else None,
            "last_completed_global_game": int(worker_state.get("last_completed_global_game", 0) or 0),
            "last_duration_seconds": worker_state.get("last_duration_seconds"),
            "last_chunk_submitted_at": worker_state.get("last_chunk_submitted_at"),
            "last_completed_task_index": worker_state.get("last_completed_task_index"),
            "recent_games_per_minute_1m": round(recent_1m["games"] / 1.0, 3),
            "recent_games_per_minute_5m": round(recent_5m["games"] / 5.0, 3),
            "throughput_series": _bucket_series(worker_state["progress_buckets"], now=now, window_minutes=THROUGHPUT_SERIES_MINUTES),
        }

    def _all_tasks_completed(self) -> bool:
        return all(entry["status"] == "completed" for entry in self._state["tasks"])

    def _save_state(self) -> None:
        _atomic_write_json(self.state_path, self._state)
        self._last_progress_persist_monotonic = time.monotonic()

    def _save_state_throttled(self, *, minimum_interval_seconds: float = 2.0) -> None:
        now = time.monotonic()
        if now - self._last_progress_persist_monotonic < minimum_interval_seconds:
            return
        self._save_state()


def decode_uploaded_chunk_text(payload: dict[str, Any], key_prefix: str) -> str:
    text_key = f"{key_prefix}_jsonl"
    gzip_key = f"{key_prefix}_gzip_b64"
    if isinstance(payload.get(text_key), str):
        return str(payload[text_key])
    encoded = payload.get(gzip_key)
    if not isinstance(encoded, str) or not encoded:
        raise ValueError(f"Missing {text_key} or {gzip_key}.")
    import base64
    import gzip

    try:
        return gzip.decompress(base64.b64decode(encoded.encode("ascii"))).decode("utf-8")
    except Exception as exc:  # pragma: no cover - invalid payload path
        raise ValueError(f"Invalid {gzip_key} payload.") from exc


def encode_chunk_text_gzip_b64(text: str) -> str:
    import base64
    import gzip

    return base64.b64encode(gzip.compress(text.encode("utf-8"))).decode("ascii")


def _default_worker_state(worker_id: str) -> dict[str, Any]:
    return {
        "worker_id": worker_id,
        "machine_name": None,
        "hostname": None,
        "platform": None,
        "python_version": None,
        "poll_seconds": None,
        "heartbeat_interval_seconds": None,
        "last_seen_at": None,
        "last_progress_at": None,
        "leased_task_index": None,
        "submitted_tasks": 0,
        "completed_games": 0,
        "completed_actions": 0,
        "completed_turns": 0,
        "completed_samples": 0,
        "truncated_games": 0,
        "deck_wins": dict(DEFAULT_DECK_WINS),
        "current_task_started_at": None,
        "current_task_game_count": 0,
        "current_task_completed_games": 0,
        "last_completed_global_game": 0,
        "last_duration_seconds": None,
        "last_chunk_submitted_at": None,
        "last_completed_task_index": None,
        "progress_buckets": {},
    }


def _normalize_worker_state(worker_id: str, payload: Any) -> dict[str, Any]:
    worker_state = _default_worker_state(worker_id)
    if not isinstance(payload, dict):
        return worker_state
    for key in (
        "machine_name",
        "hostname",
        "platform",
        "python_version",
        "last_seen_at",
        "last_progress_at",
        "current_task_started_at",
        "last_chunk_submitted_at",
    ):
        value = payload.get(key)
        if isinstance(value, str) or value is None:
            worker_state[key] = value
    for key in (
        "leased_task_index",
        "submitted_tasks",
        "completed_games",
        "completed_actions",
        "completed_turns",
        "completed_samples",
        "truncated_games",
        "current_task_game_count",
        "current_task_completed_games",
        "last_completed_global_game",
        "last_completed_task_index",
    ):
        worker_state[key] = _coerce_int(payload.get(key), default=worker_state[key])
    for key in ("poll_seconds", "heartbeat_interval_seconds", "last_duration_seconds"):
        value = _coerce_float(payload.get(key))
        if value is not None:
            worker_state[key] = value
    deck_wins = payload.get("deck_wins")
    if isinstance(deck_wins, dict):
        worker_state["deck_wins"] = dict(DEFAULT_DECK_WINS)
        for deck_id, wins in deck_wins.items():
            if isinstance(deck_id, str):
                worker_state["deck_wins"][deck_id] = _coerce_int(wins, default=0)
    worker_state["progress_buckets"] = _normalize_progress_buckets(payload.get("progress_buckets"))
    return worker_state


def _normalize_progress_buckets(payload: Any) -> dict[str, dict[str, float]]:
    normalized: dict[str, dict[str, float]] = {}
    if not isinstance(payload, dict):
        return normalized
    for bucket_key, values in payload.items():
        if not isinstance(bucket_key, str) or not isinstance(values, dict):
            continue
        normalized[bucket_key] = {
            "games": float(_coerce_int(values.get("games"), default=0)),
            "actions": float(_coerce_int(values.get("actions"), default=0)),
            "turns": float(_coerce_int(values.get("turns"), default=0)),
            "samples": float(_coerce_int(values.get("samples"), default=0)),
            "duration_seconds": float(_coerce_float(values.get("duration_seconds")) or 0.0),
        }
    return normalized


def _build_reported_totals(workers: dict[str, dict[str, Any]]) -> dict[str, Any]:
    totals = {
        "games": 0,
        "actions": 0,
        "turns": 0,
        "samples": 0,
        "truncated": 0,
        "deck_wins": dict(DEFAULT_DECK_WINS),
    }
    for worker_state in workers.values():
        totals["games"] += int(worker_state.get("completed_games", 0) or 0)
        totals["actions"] += int(worker_state.get("completed_actions", 0) or 0)
        totals["turns"] += int(worker_state.get("completed_turns", 0) or 0)
        totals["samples"] += int(worker_state.get("completed_samples", 0) or 0)
        totals["truncated"] += int(worker_state.get("truncated_games", 0) or 0)
        for deck_id, wins in worker_state.get("deck_wins", {}).items():
            if isinstance(deck_id, str):
                totals["deck_wins"].setdefault(deck_id, 0)
                totals["deck_wins"][deck_id] += int(wins)
    return totals


def _build_throughput_metrics(
    *,
    buckets: dict[str, dict[str, float]],
    now: datetime,
    elapsed_seconds: float,
    total_games_target: int,
    reported_games: int,
) -> dict[str, Any]:
    recent_1m = _window_totals(buckets, now=now, window_minutes=1)
    recent_5m = _window_totals(buckets, now=now, window_minutes=5)
    recent_15m = _window_totals(buckets, now=now, window_minutes=15)
    games_per_minute_overall = (reported_games / max(elapsed_seconds, 1.0)) * 60.0
    games_per_minute_5m = recent_5m["games"] / 5.0
    eta_seconds = None
    remaining_games = max(total_games_target - reported_games, 0)
    if games_per_minute_5m > 0:
        eta_seconds = round((remaining_games / games_per_minute_5m) * 60.0, 3)
    return {
        "games_per_minute_overall": round(games_per_minute_overall, 3),
        "games_per_minute_1m": round(recent_1m["games"] / 1.0, 3),
        "games_per_minute_5m": round(games_per_minute_5m, 3),
        "games_per_minute_15m": round(recent_15m["games"] / 15.0, 3),
        "actions_per_minute_5m": round(recent_5m["actions"] / 5.0, 3),
        "samples_per_minute_5m": round(recent_5m["samples"] / 5.0, 3),
        "average_game_duration_seconds_5m": round(recent_5m["duration_seconds"] / max(recent_5m["games"], 1.0), 3),
        "eta_seconds": eta_seconds,
    }


def _window_totals(
    buckets: dict[str, dict[str, float]],
    *,
    now: datetime,
    window_minutes: int,
) -> dict[str, float]:
    start = _minute_floor(now - timedelta(minutes=max(window_minutes - 1, 0)))
    totals = {
        "games": 0.0,
        "actions": 0.0,
        "turns": 0.0,
        "samples": 0.0,
        "duration_seconds": 0.0,
    }
    for bucket_key, values in buckets.items():
        bucket_time = datetime.fromisoformat(bucket_key)
        if bucket_time < start:
            continue
        for metric in totals:
            totals[metric] += float(values.get(metric, 0.0))
    return totals


def _bucket_series(
    buckets: dict[str, dict[str, float]],
    *,
    now: datetime,
    window_minutes: int,
) -> list[dict[str, Any]]:
    start = _minute_floor(now - timedelta(minutes=max(window_minutes - 1, 0)))
    entries: list[dict[str, Any]] = []
    current = start
    for _ in range(window_minutes):
        bucket = buckets.get(current.isoformat(), {})
        entries.append(
            {
                "minute": current.isoformat(),
                "games": int(bucket.get("games", 0)),
                "actions": int(bucket.get("actions", 0)),
                "turns": int(bucket.get("turns", 0)),
                "samples": int(bucket.get("samples", 0)),
                "duration_seconds": round(float(bucket.get("duration_seconds", 0.0)), 3),
            }
        )
        current += timedelta(minutes=1)
    return entries


def _update_progress_buckets(
    buckets: dict[str, dict[str, float]],
    now: datetime,
    *,
    games: int,
    actions: int,
    turns: int,
    samples: int,
    duration_seconds: float,
) -> None:
    bucket_key = _minute_floor(now).isoformat()
    bucket = buckets.setdefault(
        bucket_key,
        {
            "games": 0.0,
            "actions": 0.0,
            "turns": 0.0,
            "samples": 0.0,
            "duration_seconds": 0.0,
        },
    )
    bucket["games"] += float(games)
    bucket["actions"] += float(actions)
    bucket["turns"] += float(turns)
    bucket["samples"] += float(samples)
    bucket["duration_seconds"] += float(duration_seconds)


def _prune_progress_buckets(buckets: dict[str, dict[str, float]], now: datetime) -> None:
    cutoff = _minute_floor(now - timedelta(minutes=PROGRESS_BUCKET_RETENTION_MINUTES))
    stale_keys = [bucket_key for bucket_key in buckets if datetime.fromisoformat(bucket_key) < cutoff]
    for bucket_key in stale_keys:
        buckets.pop(bucket_key, None)


def _contact_threshold_seconds(worker_state: dict[str, Any]) -> float:
    poll_seconds = _coerce_float(worker_state.get("poll_seconds")) or 5.0
    heartbeat_interval_seconds = _coerce_float(worker_state.get("heartbeat_interval_seconds")) or 30.0
    return round(max(15.0, (poll_seconds * 3.0) + 2.0, (heartbeat_interval_seconds * 2.0) + 5.0), 3)


def _seconds_since(value: Any, now: datetime) -> float | None:
    if not isinstance(value, str):
        return None
    try:
        then = datetime.fromisoformat(value)
    except ValueError:
        return None
    return max((now - then).total_seconds(), 0.0)


def _minute_floor(moment: datetime) -> datetime:
    return moment.replace(second=0, microsecond=0)


def _coerce_int(value: Any, *, default: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True))
        handle.flush()
        os.fsync(handle.fileno())
    temp_path.replace(path)
    _fsync_directory(path.parent)


def _artifact_presence_detail(*, decisions_exists: bool, games_exists: bool, summary_exists: bool) -> str:
    return (
        f"decisions={int(decisions_exists)} "
        f"games={int(games_exists)} "
        f"summary={int(summary_exists)}"
    )


def _artifact_submitted_at(artifact_paths: Any) -> str | None:
    candidate_paths = [artifact_paths.summary, artifact_paths.games, artifact_paths.decisions]
    timestamps = []
    for path in candidate_paths:
        try:
            timestamps.append(path.stat().st_mtime)
        except OSError:
            continue
    if not timestamps:
        return None
    return datetime.fromtimestamp(max(timestamps), UTC).isoformat()


def _scan_unexpected_artifact_indices(*, output_dir: Path, known_task_indices: set[int]) -> list[int]:
    unexpected_indices: set[int] = set()
    for directory, suffix in (
        (output_dir / "decisions", ".jsonl"),
        (output_dir / "games", ".jsonl"),
        (output_dir / "summaries", ".json"),
    ):
        if not directory.exists():
            continue
        for path in directory.glob(f"shard_*{suffix}"):
            suffix_length = len(suffix)
            stem = path.name[:-suffix_length]
            task_index = _parse_shard_task_index(stem)
            if task_index is None or task_index in known_task_indices:
                continue
            unexpected_indices.add(task_index)
    return sorted(unexpected_indices)


def _parse_shard_task_index(value: str) -> int | None:
    if not value.startswith("shard_"):
        return None
    index_text = value.removeprefix("shard_")
    try:
        return int(index_text)
    except ValueError:
        return None


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _utc_now() -> datetime:
    return datetime.now(UTC)
