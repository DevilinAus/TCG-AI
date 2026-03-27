from __future__ import annotations

import argparse
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import Sequence


DEFAULT_COORDINATOR_URL = "http://192.168.0.175:8787"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch one or more distributed Standard self-play workers on this machine."
    )
    parser.add_argument(
        "coordinator_url",
        nargs="?",
        default=None,
        help=f"Coordinator base URL. Defaults to {DEFAULT_COORDINATOR_URL}.",
    )
    parser.add_argument(
        "worker_prefix",
        nargs="?",
        default=None,
        help="Optional worker id prefix. Defaults to the local hostname.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes to launch. Defaults to one per detected CPU core.",
    )
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--request-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--heartbeat-interval-seconds", type=float, default=15.0)
    parser.add_argument(
        "--progress-log-dir",
        type=Path,
        default=None,
        help="Optional directory for per-worker progress logs.",
    )
    return parser.parse_args(argv)


def resolve_coordinator_url(explicit_url: str | None) -> str:
    configured = explicit_url or os.environ.get("TCG_AI_STANDARD_SELF_PLAY_COORDINATOR_URL")
    return (configured or DEFAULT_COORDINATOR_URL).rstrip("/")


def default_worker_count() -> int:
    return max(1, os.cpu_count() or 1)


def resolve_worker_count(explicit_count: int | None) -> int:
    if explicit_count is not None:
        return max(1, explicit_count)
    env_value = os.environ.get("TCG_AI_STANDARD_SELF_PLAY_WORKER_COUNT")
    if env_value:
        try:
            return max(1, int(env_value))
        except ValueError:
            pass
    return default_worker_count()


def resolve_worker_prefix(explicit_prefix: str | None) -> str:
    configured = explicit_prefix or os.environ.get("TCG_AI_STANDARD_SELF_PLAY_WORKER_ID_PREFIX")
    if configured:
        return configured
    hostname = socket.gethostname().strip().replace(" ", "-")
    return hostname or "worker"


def build_worker_id(prefix: str, worker_index: int, total_workers: int) -> str:
    if total_workers <= 1:
        return prefix
    width = max(2, len(str(total_workers)))
    return f"{prefix}-{worker_index + 1:0{width}d}"


def build_worker_command(
    *,
    python_executable: str,
    project_root: Path,
    coordinator_url: str,
    worker_id: str,
    machine_name: str,
    poll_seconds: float,
    request_timeout_seconds: float,
    heartbeat_interval_seconds: float,
    progress_log_dir: Path | None,
) -> list[str]:
    command = [
        python_executable,
        str(project_root / "scripts" / "run_standard_self_play_worker.py"),
        "--coordinator-url",
        coordinator_url,
        "--worker-id",
        worker_id,
        "--machine-name",
        machine_name,
        "--poll-seconds",
        str(poll_seconds),
        "--request-timeout-seconds",
        str(request_timeout_seconds),
        "--heartbeat-interval-seconds",
        str(heartbeat_interval_seconds),
    ]
    if progress_log_dir is not None:
        progress_log_dir.mkdir(parents=True, exist_ok=True)
        command.extend(
            [
                "--progress-log",
                str(progress_log_dir / f"{worker_id}.log"),
            ]
        )
    return command


def launch_worker_processes(
    *,
    coordinator_url: str,
    worker_prefix: str,
    worker_count: int,
    poll_seconds: float,
    request_timeout_seconds: float,
    heartbeat_interval_seconds: float,
    progress_log_dir: Path | None,
    python_executable: str,
    project_root: Path,
) -> int:
    worker_count = max(1, worker_count)
    print(f"[worker-launch] coordinator={coordinator_url}")
    print(f"[worker-launch] worker_prefix={worker_prefix}")
    print(f"[worker-launch] workers={worker_count}")

    processes: list[tuple[str, subprocess.Popen[str]]] = []
    for worker_index in range(worker_count):
        worker_id = build_worker_id(worker_prefix, worker_index, worker_count)
        command = build_worker_command(
            python_executable=python_executable,
            project_root=project_root,
            coordinator_url=coordinator_url,
            worker_id=worker_id,
            machine_name=worker_prefix,
            poll_seconds=poll_seconds,
            request_timeout_seconds=request_timeout_seconds,
            heartbeat_interval_seconds=heartbeat_interval_seconds,
            progress_log_dir=progress_log_dir,
        )
        print(f"[worker-launch] starting {worker_id}")
        processes.append(
            (
                worker_id,
                subprocess.Popen(
                    command,
                    cwd=project_root,
                    text=True,
                ),
            )
        )

    try:
        while processes:
            remaining: list[tuple[str, subprocess.Popen[str]]] = []
            for worker_id, process in processes:
                exit_code = process.poll()
                if exit_code is None:
                    remaining.append((worker_id, process))
                    continue
                print(f"[worker-launch] {worker_id} exited with code {exit_code}")
                if exit_code != 0:
                    _terminate_processes(remaining)
                    return exit_code
            if not remaining:
                return 0
            processes = remaining
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("[worker-launch] interrupt received; stopping workers")
        _terminate_processes(processes)
        return 130


def _terminate_processes(processes: Sequence[tuple[str, subprocess.Popen[str]]]) -> None:
    for _worker_id, process in processes:
        if process.poll() is None:
            process.terminate()
    deadline = time.time() + 5.0
    for _worker_id, process in processes:
        if process.poll() is not None:
            continue
        timeout = max(0.0, deadline - time.time())
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = Path(__file__).resolve().parents[5]
    return launch_worker_processes(
        coordinator_url=resolve_coordinator_url(args.coordinator_url),
        worker_prefix=resolve_worker_prefix(args.worker_prefix),
        worker_count=resolve_worker_count(args.workers),
        poll_seconds=args.poll_seconds,
        request_timeout_seconds=args.request_timeout_seconds,
        heartbeat_interval_seconds=args.heartbeat_interval_seconds,
        progress_log_dir=args.progress_log_dir,
        python_executable=sys.executable,
        project_root=project_root,
    )


if __name__ == "__main__":
    raise SystemExit(main())
