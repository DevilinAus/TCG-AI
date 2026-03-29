#!/usr/bin/env python3
from __future__ import annotations

import argparse
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

DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "standard_ml_data" / "self_play"
DASHBOARD_ROOT = PROJECT_ROOT / "frontend" / "distributed-self-play"
DASHBOARD_ASSETS = {
    "/": ("dashboard.html", "text/html; charset=utf-8"),
    "/dashboard": ("dashboard.html", "text/html; charset=utf-8"),
    "/dashboard.css": ("dashboard.css", "text/css; charset=utf-8"),
    "/dashboard.js": ("dashboard.js", "application/javascript; charset=utf-8"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve distributed Standard self-play chunks over LAN.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--run-id", default=datetime.now(UTC).strftime("run_%Y%m%dT%H%M%SZ"))
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
    parser.add_argument("--pid-file", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
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
    output_dir = (args.output_dir or (DEFAULT_OUTPUT_ROOT / args.run_id)).resolve()
    coordinator = DistributedSelfPlayCoordinator(
        output_dir=output_dir,
        run_id=args.run_id,
        run_config=run_config,
        lease_timeout_seconds=max(1, args.lease_timeout_seconds),
    )
    server = ThreadingHTTPServer((args.host, args.port), make_handler(coordinator))
    pid_file = args.pid_file.resolve() if args.pid_file is not None else None
    if pid_file is not None:
        _write_pid_file(pid_file)
    print(f"[coordinator] run_id={args.run_id}")
    print(f"[coordinator] output={output_dir}")
    print(f"[coordinator] serving=http://{args.host}:{args.port}")
    print(f"[coordinator] dashboard=http://{args.host}:{args.port}/dashboard")
    print(f"[coordinator] status=http://{args.host}:{args.port}/api/standard-self-play/status")
    print(f"[coordinator] lease_timeout_seconds={args.lease_timeout_seconds}")
    recovery = coordinator.status().get("recovery", {})
    print(
        "[coordinator] recovery "
        f"completed_on_disk={recovery.get('completed_tasks_from_artifacts', 0)} "
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
