#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import random
import signal
import sys
import threading
from typing import Any
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.tcg_ai.game_modes.standard.ml import PlannerConfig
from backend.tcg_ai.game_modes.standard.ml.distributed_self_play import (
    DistributedSelfPlayCoordinator,
    decode_uploaded_chunk_text,
)
from backend.tcg_ai.game_modes.standard.ml.self_play_jobs import SelfPlayRunConfig

DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "standard_ml_data" / "distributed_self_play"
RUN_COMPLETE_FLAG_NAME = "RUN_COMPLETE"
DASHBOARD_ROOT = PROJECT_ROOT / "frontend" / "distributed-self-play"
DASHBOARD_ASSETS = {
    "/": ("dashboard.html", "text/html; charset=utf-8"),
    "/dashboard": ("dashboard.html", "text/html; charset=utf-8"),
    "/dashboard.css": ("dashboard.css", "text/css; charset=utf-8"),
    "/dashboard.js": ("dashboard.js", "application/javascript; charset=utf-8"),
}


@dataclass(frozen=True)
class ResolvedCoordinatorRun:
    run_id: str
    output_dir: Path
    output_root: Path
    resumed: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve distributed Standard self-play chunks over LAN.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--games", type=int, default=10000)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=random.randint(1, 999_999))
    parser.add_argument("--max-actions-per-game", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--opponent-branch-width", type=int, default=2)
    parser.add_argument("--disable-opponent-turn", action="store_true")
    parser.add_argument("--include-setup-decisions", action="store_true")
    parser.add_argument("--record-forced-actions", action="store_true")
    parser.add_argument("--oracle", choices=("heuristic", "local-model"), default="heuristic")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--lease-timeout-seconds", type=int, default=1800)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--pid-file", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    resolved_run = resolve_coordinator_run(
        run_id=args.run_id,
        output_dir=args.output_dir,
        output_root=args.output_root,
    )
    planner_config = PlannerConfig(
        max_depth=max(1, args.max_depth),
        beam_width=max(1, args.beam_width),
        opponent_branch_width=max(1, args.opponent_branch_width),
        include_opponent_turn=not args.disable_opponent_turn,
    )
    run_config = SelfPlayRunConfig(
        games=max(1, args.games),
        chunk_size=max(1, args.chunk_size),
        seed=args.seed,
        planner_config=planner_config,
        max_actions_per_game=max(1, args.max_actions_per_game),
        include_setup_decisions=args.include_setup_decisions,
        record_forced_actions=args.record_forced_actions,
        oracle=args.oracle,
        checkpoint=str(args.checkpoint.resolve()) if args.checkpoint else None,
    )
    coordinator = DistributedSelfPlayCoordinator(
        output_dir=resolved_run.output_dir,
        run_id=resolved_run.run_id,
        run_config=run_config,
        lease_timeout_seconds=max(1, args.lease_timeout_seconds),
    )
    server = ThreadingHTTPServer((args.host, args.port), make_handler(coordinator))
    pid_file = args.pid_file.resolve() if args.pid_file is not None else resolved_run.output_dir / "coordinator.pid"
    _write_pid_file(pid_file)
    print(f"[coordinator] run_id={resolved_run.run_id}")
    print(f"[coordinator] output={resolved_run.output_dir}")
    print(f"[coordinator] output_root={resolved_run.output_root}")
    print(f"[coordinator] launch_mode={'resume' if resolved_run.resumed else 'new'}")
    print(f"[coordinator] pid_file={pid_file}")
    print(f"[coordinator] serving=http://{args.host}:{args.port}")
    print(f"[coordinator] dashboard=http://{args.host}:{args.port}/dashboard")
    print(f"[coordinator] status=http://{args.host}:{args.port}/api/standard-self-play/status")
    print(f"[coordinator] lease_timeout_seconds={args.lease_timeout_seconds}")
    status = coordinator.status()
    recovery = status.get("recovery", {})
    print(
        "[coordinator] recovery "
        f"picked_up_games={status.get('aggregate', {}).get('games', 0)}/{status.get('total_games_target', 0)} "
        f"picked_up_tasks={status.get('completed_tasks', 0)}/{status.get('total_tasks', 0)} "
        f"legacy_upgrades={recovery.get('upgraded_legacy_summaries', 0)} "
        f"issues={recovery.get('integrity_issue_count', 0)}"
    )
    stop_requested = threading.Event()

    def request_shutdown(signum: int, _frame: Any) -> None:
        if stop_requested.is_set():
            return
        stop_requested.set()
        signal_name = signal.Signals(signum).name
        print(f"\n[coordinator] received {signal_name}; persisting state and stopping")
        try:
            coordinator.persist_state()
        except OSError as exc:
            print(f"[coordinator] final state save failed: {exc}", file=sys.stderr)
        threading.Thread(target=server.shutdown, name="coordinator-shutdown", daemon=True).start()

    signal.signal(signal.SIGTERM, request_shutdown)
    signal.signal(signal.SIGINT, request_shutdown)
    try:
        server.serve_forever()
    finally:
        if not stop_requested.is_set():
            try:
                coordinator.persist_state()
            except OSError as exc:
                print(f"[coordinator] final state save failed: {exc}", file=sys.stderr)
        server.server_close()
        if pid_file is not None:
            _remove_pid_file(pid_file)
        print("[coordinator] stopped")
    return 0


def resolve_coordinator_run(
    *,
    run_id: str | None,
    output_dir: Path | None,
    output_root: Path | None,
) -> ResolvedCoordinatorRun:
    if output_dir is not None:
        resolved_output_dir = output_dir.resolve()
        existing_run_id = _load_run_id_from_dir(resolved_output_dir)
        if run_id is None:
            resolved_run_id = existing_run_id or _infer_run_id_from_path(resolved_output_dir)
        else:
            if existing_run_id is not None and existing_run_id != run_id:
                raise ValueError(
                    f"Explicit run_id '{run_id}' does not match the persisted run_id '{existing_run_id}' in {resolved_output_dir}."
                )
            resolved_run_id = run_id
        return ResolvedCoordinatorRun(
            run_id=resolved_run_id,
            output_dir=resolved_output_dir,
            output_root=resolved_output_dir.parent,
            resumed=existing_run_id is not None,
        )

    resolved_output_root = (output_root or DEFAULT_OUTPUT_ROOT).resolve()
    resolved_output_root.mkdir(parents=True, exist_ok=True)
    if run_id is not None:
        candidate_dir = resolved_output_root / run_id
        existing_run_id = _load_run_id_from_dir(candidate_dir)
        if existing_run_id is not None and existing_run_id != run_id:
            raise ValueError(
                f"Explicit run_id '{run_id}' does not match the persisted run_id '{existing_run_id}' in {candidate_dir}."
            )
        return ResolvedCoordinatorRun(
            run_id=run_id,
            output_dir=candidate_dir,
            output_root=resolved_output_root,
            resumed=existing_run_id is not None,
        )

    resumable_run = _find_latest_incomplete_run(resolved_output_root)
    if resumable_run is not None:
        return resumable_run

    generated_run_id = _generate_run_id()
    return ResolvedCoordinatorRun(
        run_id=generated_run_id,
        output_dir=resolved_output_root / generated_run_id,
        output_root=resolved_output_root,
        resumed=False,
    )


def _find_latest_incomplete_run(output_root: Path) -> ResolvedCoordinatorRun | None:
    candidates: list[tuple[float, ResolvedCoordinatorRun]] = []
    for candidate_dir in output_root.glob("run_*"):
        if not candidate_dir.is_dir():
            continue
        if not _run_has_meaningful_state(candidate_dir):
            continue
        if _run_is_complete(candidate_dir):
            continue
        run_id = _load_run_id_from_dir(candidate_dir) or _infer_run_id_from_path(candidate_dir)
        candidates.append(
            (
                _run_last_activity_timestamp(candidate_dir),
                ResolvedCoordinatorRun(
                    run_id=run_id,
                    output_dir=candidate_dir.resolve(),
                    output_root=output_root.resolve(),
                    resumed=True,
                ),
            )
        )
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def _run_has_meaningful_state(run_dir: Path) -> bool:
    if any((run_dir / filename).exists() for filename in ("coordinator_state.json", "manifest.json", "summary.json")):
        return True
    for directory_name in ("decisions", "games", "summaries"):
        directory = run_dir / directory_name
        if not directory.exists():
            continue
        if any(directory.iterdir()):
            return True
    return False


def _run_is_complete(run_dir: Path) -> bool:
    if (run_dir / RUN_COMPLETE_FLAG_NAME).exists():
        return True
    summary_payload = _load_json_if_exists(run_dir / "summary.json")
    if isinstance(summary_payload, dict) and isinstance(summary_payload.get("completed_at"), str) and summary_payload.get("completed_at"):
        return True
    state_payload = _load_json_if_exists(run_dir / "coordinator_state.json")
    if not isinstance(state_payload, dict):
        return False
    tasks = state_payload.get("tasks")
    return isinstance(tasks, list) and bool(tasks) and all(
        isinstance(entry, dict) and entry.get("status") == "completed"
        for entry in tasks
    )


def _run_last_activity_timestamp(run_dir: Path) -> float:
    timestamps: list[float] = []
    for path in (
        run_dir,
        run_dir / "coordinator_state.json",
        run_dir / "manifest.json",
        run_dir / "summary.json",
        run_dir / RUN_COMPLETE_FLAG_NAME,
    ):
        try:
            timestamps.append(path.stat().st_mtime)
        except FileNotFoundError:
            continue
    for directory_name in ("decisions", "games", "summaries"):
        directory = run_dir / directory_name
        if not directory.exists():
            continue
        for child in directory.iterdir():
            try:
                timestamps.append(child.stat().st_mtime)
            except FileNotFoundError:
                continue
    return max(timestamps) if timestamps else 0.0


def _load_run_id_from_dir(run_dir: Path) -> str | None:
    for filename in ("coordinator_state.json", "manifest.json", "summary.json", RUN_COMPLETE_FLAG_NAME):
        payload = _load_json_if_exists(run_dir / filename)
        if not isinstance(payload, dict):
            continue
        run_id = payload.get("run_id")
        if isinstance(run_id, str) and run_id:
            return run_id
    return None


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _infer_run_id_from_path(path: Path) -> str:
    if path.name.startswith("run_"):
        return path.name
    return _generate_run_id()


def _generate_run_id() -> str:
    return datetime.now(UTC).strftime("run_%Y%m%dT%H%M%SZ")


def make_handler(coordinator: DistributedSelfPlayCoordinator) -> type[BaseHTTPRequestHandler]:
    class CoordinatorHandler(BaseHTTPRequestHandler):
        server_version = "StandardSelfPlayCoordinator/0.1"

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            try:
                asset = DASHBOARD_ASSETS.get(parsed.path)
                if asset is not None:
                    self._send_static_file(DASHBOARD_ROOT / asset[0], content_type=asset[1])
                    return
                if parsed.path == "/healthz":
                    self._send_json({"ok": True})
                    return
                if parsed.path == "/api/standard-self-play/status":
                    self._send_json(coordinator.status())
                    return
                self._send_json({"error": "Not found."}, status=HTTPStatus.NOT_FOUND)
            except ValueError as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            try:
                payload = self._read_json()
                if parsed.path == "/api/standard-self-play/lease-chunk":
                    worker_id = _require_worker_id(payload)
                    worker_meta = payload.get("worker_meta")
                    self._send_json(
                        coordinator.lease_chunk(
                            worker_id=worker_id,
                            worker_meta=worker_meta if isinstance(worker_meta, dict) else None,
                        )
                    )
                    return
                if parsed.path == "/api/standard-self-play/heartbeat":
                    worker_id = _require_worker_id(payload)
                    task_index = _require_int(payload, "task_index")
                    worker_meta = payload.get("worker_meta")
                    self._send_json(
                        coordinator.heartbeat(
                            worker_id=worker_id,
                            task_index=task_index,
                            worker_meta=worker_meta if isinstance(worker_meta, dict) else None,
                        )
                    )
                    return
                if parsed.path == "/api/standard-self-play/progress":
                    worker_id = _require_worker_id(payload)
                    task_index = _require_int(payload, "task_index")
                    worker_meta = payload.get("worker_meta")
                    self._send_json(
                        coordinator.record_progress(
                            worker_id=worker_id,
                            task_index=task_index,
                            progress=_require_dict(payload, "progress"),
                            worker_meta=worker_meta if isinstance(worker_meta, dict) else None,
                        )
                    )
                    return
                if parsed.path == "/api/standard-self-play/submit-chunk":
                    worker_id = _require_worker_id(payload)
                    task_index = _require_int(payload, "task_index")
                    worker_meta = payload.get("worker_meta")
                    self._send_json(
                        coordinator.submit_chunk(
                            worker_id=worker_id,
                            task_index=task_index,
                            summary=_require_dict(payload, "summary"),
                            decisions_jsonl=decode_uploaded_chunk_text(payload, "decisions"),
                            games_jsonl=decode_uploaded_chunk_text(payload, "games"),
                            worker_meta=worker_meta if isinstance(worker_meta, dict) else None,
                        )
                    )
                    return
                self._send_json({"error": "Not found."}, status=HTTPStatus.NOT_FOUND)
            except ValueError as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            return

        def _read_json(self) -> dict[str, Any]:
            content_length = int(self.headers.get("Content-Length", "0") or "0")
            if content_length <= 0:
                raise ValueError("Missing JSON payload.")
            raw = self.rfile.read(content_length)
            try:
                payload = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError as exc:
                raise ValueError("Malformed JSON payload.") from exc
            if not isinstance(payload, dict):
                raise ValueError("JSON payload must be an object.")
            return payload

        def _send_json(self, payload: dict[str, Any], *, status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_static_file(self, path: Path, *, content_type: str) -> None:
            if not path.exists():
                self._send_json({"error": "Not found."}, status=HTTPStatus.NOT_FOUND)
                return
            body = path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return CoordinatorHandler


def _require_worker_id(payload: dict[str, Any]) -> str:
    worker_id = payload.get("worker_id")
    if not isinstance(worker_id, str) or not worker_id:
        raise ValueError("Missing worker_id.")
    return worker_id


def _require_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int):
        raise ValueError(f"Missing integer field: {key}")
    return value


def _require_dict(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Missing object field: {key}")
    return value


def _write_pid_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{os.getpid()}\n", encoding="utf-8")


def _remove_pid_file(path: Path) -> None:
    try:
        recorded_pid = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return
    except OSError:
        return
    if recorded_pid and recorded_pid != str(os.getpid()):
        return
    try:
        path.unlink()
    except FileNotFoundError:
        return


if __name__ == "__main__":
    raise SystemExit(main())
