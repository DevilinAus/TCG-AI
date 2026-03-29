#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import platform
import socket
import sys
import threading
import time
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.tcg_ai.game_modes.standard.ml.distributed_self_play import encode_chunk_text_gzip_b64
from backend.tcg_ai.game_modes.standard.ml.self_play_jobs import (
    run_self_play_chunk,
    self_play_task_from_payload,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a distributed Standard self-play worker.")
    parser.add_argument("--coordinator-url", required=True, help="Coordinator base URL, for example http://192.168.1.10:8787")
    parser.add_argument("--worker-id", default=None, help="Stable worker ID. Defaults to hostname-pid.")
    parser.add_argument("--machine-name", default=None, help="Human-friendly machine label used for dashboard grouping.")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument(
        "--reconnect-seconds",
        type=float,
        default=300.0,
        help="How long to wait before retrying after the coordinator is unreachable.",
    )
    parser.add_argument("--request-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--heartbeat-interval-seconds", type=float, default=30.0)
    parser.add_argument("--progress-log", type=Path, default=None, help="Optional local log file for worker progress.")
    parser.add_argument(
        "--exit-when-run-complete",
        action="store_true",
        help="Exit instead of idling when the coordinator reports that the current run is complete.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    worker_id = args.worker_id or f"{socket.gethostname()}-{os_getpid()}"
    machine_name = args.machine_name or socket.gethostname()
    coordinator_url = args.coordinator_url.rstrip("/")
    worker_meta = {
        "hostname": socket.gethostname(),
        "machine_name": machine_name,
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "poll_seconds": args.poll_seconds,
        "heartbeat_interval_seconds": args.heartbeat_interval_seconds,
    }
    print(f"[worker] worker_id={worker_id}")
    print(f"[worker] machine_name={machine_name}")
    print(f"[worker] coordinator={coordinator_url}")
    print(f"[worker] reconnect_seconds={max(1.0, args.reconnect_seconds)}")
    while True:
        lease_response = _call_coordinator_with_retries(
            operation_label="lease request",
            retry_interval_seconds=args.reconnect_seconds,
            request=lambda: _post_json(
                f"{coordinator_url}/api/standard-self-play/lease-chunk",
                payload={
                    "worker_id": worker_id,
                    "worker_meta": worker_meta,
                },
                timeout_seconds=args.request_timeout_seconds,
            ),
        )
        task_payload = lease_response.get("task")
        if task_payload is None:
            if _handle_idle_lease_response(
                lease_response=lease_response,
                poll_seconds=args.poll_seconds,
                reconnect_seconds=args.reconnect_seconds,
                exit_when_run_complete=args.exit_when_run_complete,
            ):
                return 0
            continue

        task = self_play_task_from_payload(task_payload)
        print(
            "[worker] "
            f"leased shard={task.task_index:06d} "
            f"games={task.game_count} start={task.start_index}"
        )
        heartbeat_loop = LeaseHeartbeatLoop(
            coordinator_url=coordinator_url,
            worker_id=worker_id,
            worker_meta=worker_meta,
            task_index=task.task_index,
            heartbeat_interval_seconds=args.heartbeat_interval_seconds,
            request_timeout_seconds=args.request_timeout_seconds,
        )

        def on_progress(line: str) -> None:
            print(line)
            if args.progress_log is not None:
                _append_progress_line(args.progress_log, line)

        def on_progress_event(event: dict[str, Any]) -> None:
            try:
                _post_json(
                    f"{coordinator_url}/api/standard-self-play/progress",
                    payload={
                        "worker_id": worker_id,
                        "task_index": task.task_index,
                        "worker_meta": worker_meta,
                        "progress": event,
                    },
                    timeout_seconds=args.request_timeout_seconds,
                )
            except (CoordinatorUnavailableError, CoordinatorRequestError) as exc:
                print(f"[worker] progress update failed for shard={task.task_index:06d}: {exc}", file=sys.stderr)

        heartbeat_loop.start()
        try:
            chunk_result = run_self_play_chunk(
                task,
                progress_callback=on_progress,
                progress_event_callback=on_progress_event,
            )
        finally:
            heartbeat_loop.stop()
        submit_response = _call_coordinator_with_retries(
            operation_label=f"submit shard={task.task_index:06d}",
            retry_interval_seconds=args.reconnect_seconds,
            request=lambda: _post_json(
                f"{coordinator_url}/api/standard-self-play/submit-chunk",
                payload={
                    "worker_id": worker_id,
                    "task_index": task.task_index,
                    "worker_meta": worker_meta,
                    "summary": chunk_result.summary,
                    "decisions_gzip_b64": encode_chunk_text_gzip_b64(chunk_result.decisions_jsonl),
                    "games_gzip_b64": encode_chunk_text_gzip_b64(chunk_result.games_jsonl),
                },
                timeout_seconds=max(args.request_timeout_seconds, 120.0),
            ),
        )
        print(
            "[worker] "
            f"submitted shard={task.task_index:06d} "
            f"duplicate={submit_response.get('duplicate', False)} "
            f"run_complete={submit_response.get('run_complete', False)}"
        )
        if submit_response.get("run_complete"):
            if args.exit_when_run_complete:
                print("[worker] coordinator reports run complete; exiting")
                return 0
            print(
                "[worker] coordinator reports run complete; "
                f"idling for {_format_duration(args.reconnect_seconds)} before checking again"
            )
            time.sleep(max(1.0, args.reconnect_seconds))


class CoordinatorUnavailableError(RuntimeError):
    pass


class CoordinatorRequestError(RuntimeError):
    pass


def _post_json(url: str, *, payload: dict[str, Any], timeout_seconds: float) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib_request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    try:
        with urllib_request.urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read().decode("utf-8")
    except urllib_error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        message = f"Request failed for {url}: {exc.code} {error_body}"
        if exc.code >= 500:
            raise CoordinatorUnavailableError(message) from exc
        raise CoordinatorRequestError(message) from exc
    except (urllib_error.URLError, TimeoutError, OSError) as exc:
        raise CoordinatorUnavailableError(f"Could not reach coordinator {url}: {exc}") from exc
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CoordinatorUnavailableError(f"Coordinator returned malformed JSON for {url}.") from exc
    if not isinstance(decoded, dict):
        raise CoordinatorUnavailableError(f"Coordinator returned malformed JSON object for {url}.")
    return decoded


def _call_coordinator_with_retries(
    *,
    operation_label: str,
    retry_interval_seconds: float,
    request: Any,
    sleep_fn: Any = None,
) -> dict[str, Any]:
    retry_interval_seconds = max(1.0, float(retry_interval_seconds))
    sleep_fn = sleep_fn or time.sleep
    while True:
        try:
            return request()
        except CoordinatorUnavailableError as exc:
            print(
                f"[worker] {operation_label} failed: {exc}; "
                f"retrying in {_format_duration(retry_interval_seconds)}",
                file=sys.stderr,
            )
            sleep_fn(retry_interval_seconds)
        except CoordinatorRequestError as exc:
            raise SystemExit(str(exc)) from exc


def _handle_idle_lease_response(
    *,
    lease_response: dict[str, Any],
    poll_seconds: float,
    reconnect_seconds: float,
    exit_when_run_complete: bool,
    sleep_fn: Any = None,
) -> bool:
    sleep_fn = sleep_fn or time.sleep
    if lease_response.get("run_complete"):
        if exit_when_run_complete:
            print("[worker] run complete; exiting")
            return True
        idle_seconds = max(1.0, reconnect_seconds)
        print(
            "[worker] run complete for now; "
            f"idling for {_format_duration(idle_seconds)} before checking again"
        )
        sleep_fn(idle_seconds)
        return False
    idle_seconds = max(0.1, poll_seconds)
    print("[worker] no chunk available yet; sleeping")
    sleep_fn(idle_seconds)
    return False


class LeaseHeartbeatLoop:
    def __init__(
        self,
        *,
        coordinator_url: str,
        worker_id: str,
        worker_meta: dict[str, Any],
        task_index: int,
        heartbeat_interval_seconds: float,
        request_timeout_seconds: float,
        post_json: Any = None,
    ) -> None:
        self.coordinator_url = coordinator_url.rstrip("/")
        self.worker_id = worker_id
        self.worker_meta = dict(worker_meta)
        self.task_index = task_index
        self.heartbeat_interval_seconds = max(1.0, heartbeat_interval_seconds)
        self.request_timeout_seconds = request_timeout_seconds
        self._post_json = post_json or _post_json
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"self-play-heartbeat-{task_index}", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=max(1.0, self.request_timeout_seconds + 1.0))

    def _run(self) -> None:
        while not self._stop_event.wait(self.heartbeat_interval_seconds):
            try:
                self._post_json(
                    f"{self.coordinator_url}/api/standard-self-play/heartbeat",
                    payload={
                        "worker_id": self.worker_id,
                        "task_index": self.task_index,
                        "worker_meta": self.worker_meta,
                    },
                    timeout_seconds=self.request_timeout_seconds,
                )
            except (CoordinatorUnavailableError, CoordinatorRequestError) as exc:
                print(f"[worker] heartbeat failed for shard={self.task_index:06d}: {exc}", file=sys.stderr)


def _append_progress_line(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.write("\n")


def os_getpid() -> int:
    import os

    return os.getpid()


def _format_duration(seconds: float) -> str:
    total_seconds = max(1, int(math.ceil(seconds)))
    minutes, remaining_seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{remaining_seconds:02d}s"
    return f"{remaining_seconds}s"


if __name__ == "__main__":
    raise SystemExit(main())
