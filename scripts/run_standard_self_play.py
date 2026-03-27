#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
import json
import math
import os
from pathlib import Path
import random
import threading
import sys
import time
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.tcg_ai.game_modes.standard.ml import (
    PlannerConfig,
)
from backend.tcg_ai.game_modes.standard.ml.self_play_jobs import (
    SelfPlayRunConfig,
    build_self_play_manifest,
    build_self_play_tasks,
    empty_self_play_summary,
    merge_chunk_summary,
    resolve_oracle_status,
    run_self_play_chunk,
    write_self_play_chunk_artifacts,
)

DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "standard_ml_data" / "self_play"
_PROGRESS_LOG_LOCK = threading.Lock()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run headless Standard self-play for Ampharos vs Lucario.")
    parser.add_argument("--games", type=int, default=1000, help="Total games to run.")
    parser.add_argument("--workers", type=int, default=max(1, min(os.cpu_count() or 1, 8)))
    parser.add_argument("--chunk-size", type=int, default=100, help="Games per worker shard.")
    parser.add_argument("--max-actions-per-game", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--opponent-branch-width", type=int, default=2)
    parser.add_argument("--disable-opponent-turn", action="store_true")
    parser.add_argument("--include-setup-decisions", action="store_true")
    parser.add_argument("--record-forced-actions", action="store_true")
    parser.add_argument("--oracle", choices=("auto", "heuristic", "local-model"), default="auto")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Checkpoint for --oracle local-model.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--progress-log",
        type=Path,
        default=None,
        help="Optional append-only log file for per-game worker progress.",
    )
    parser.add_argument("--seed", type=int, default=random.randint(1, 999_999))
    parser.add_argument("--log-every-seconds", type=float, default=5.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.games <= 0:
        raise SystemExit("--games must be positive.")
    if args.chunk_size <= 0:
        raise SystemExit("--chunk-size must be positive.")
    oracle_status = resolve_oracle_status(oracle=args.oracle, checkpoint=args.checkpoint)
    if oracle_status.get("resolved_oracle") == "local-model" and args.workers > 1:
        raise SystemExit(
            "--oracle local-model currently requires --workers 1 so we do not duplicate the checkpoint across processes."
        )

    run_id = datetime.now(UTC).strftime("run_%Y%m%dT%H%M%SZ")
    output_dir = (args.output_dir or (DEFAULT_OUTPUT_ROOT / run_id)).resolve()
    decisions_dir = output_dir / "decisions"
    games_dir = output_dir / "games"
    decisions_dir.mkdir(parents=True, exist_ok=True)
    games_dir.mkdir(parents=True, exist_ok=True)

    planner_config = PlannerConfig(
        max_depth=max(1, args.max_depth),
        beam_width=max(1, args.beam_width),
        opponent_branch_width=max(1, args.opponent_branch_width),
        include_opponent_turn=not args.disable_opponent_turn,
    )
    run_config = SelfPlayRunConfig(
        games=args.games,
        chunk_size=args.chunk_size,
        seed=args.seed,
        planner_config=planner_config,
        max_actions_per_game=args.max_actions_per_game,
        include_setup_decisions=args.include_setup_decisions,
        record_forced_actions=args.record_forced_actions,
        oracle=args.oracle,
        checkpoint=str(args.checkpoint) if args.checkpoint else None,
    )
    manifest = build_self_play_manifest(
        run_id=run_id,
        config=run_config,
        oracle_status=oracle_status,
    )
    manifest["workers"] = args.workers
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    print(f"[self-play] output={output_dir}")
    print(
        "[self-play] "
        f"games={args.games} workers={args.workers} chunk={args.chunk_size} "
        f"oracle={args.oracle} resolved={oracle_status.get('resolved_oracle', args.oracle)} seed={args.seed}"
    )
    if oracle_status:
        print(f"[self-play] oracle-status={json.dumps(oracle_status, sort_keys=True)}")

    tasks = build_self_play_tasks(run_id=run_id, config=run_config)
    aggregate = empty_self_play_summary()
    start_time = time.perf_counter()
    last_log_time = start_time

    if args.workers == 1:
        for task in tasks:
            chunk_summary = _run_self_play_chunk(
                task,
                output_dir=output_dir,
                progress_log=args.progress_log.resolve() if args.progress_log else None,
            )
            merge_chunk_summary(aggregate, chunk_summary)
            now = time.perf_counter()
            if now - last_log_time >= args.log_every_seconds or aggregate["games"] == args.games:
                _print_progress(aggregate, total_games=args.games, start_time=start_time)
                last_log_time = now
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_map = {
                executor.submit(
                    _run_self_play_chunk,
                    task,
                    output_dir=output_dir,
                    progress_log=args.progress_log.resolve() if args.progress_log else None,
                ): task
                for task in tasks
            }
            for future in as_completed(future_map):
                chunk_summary = future.result()
                merge_chunk_summary(aggregate, chunk_summary)
                now = time.perf_counter()
                if now - last_log_time >= args.log_every_seconds or aggregate["games"] == args.games:
                    _print_progress(aggregate, total_games=args.games, start_time=start_time)
                    last_log_time = now

    summary_payload = {
        "run_id": run_id,
        "completed_at": datetime.now(UTC).isoformat(),
        **aggregate,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[self-play] summary={output_dir / 'summary.json'}")
    return 0


def _run_self_play_chunk(
    task,
    *,
    output_dir: Path,
    progress_log: Path | None,
) -> dict[str, Any]:
    chunk_result = run_self_play_chunk(
        task,
        progress_callback=(
            (lambda line: _log_worker_progress(progress_log, line))
            if progress_log is not None
            else None
        ),
    )
    write_self_play_chunk_artifacts(output_dir=output_dir, result=chunk_result)
    return chunk_result.summary


def _log_worker_progress(progress_log: Path | None, line: str) -> None:
    if progress_log is None:
        return
    progress_log.parent.mkdir(parents=True, exist_ok=True)
    with _PROGRESS_LOG_LOCK:
        with progress_log.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.write("\n")


def _print_progress(aggregate: dict[str, Any], *, total_games: int, start_time: float) -> None:
    elapsed = max(time.perf_counter() - start_time, 1e-6)
    games = int(aggregate["games"])
    games_per_second = games / elapsed
    remaining_games = max(0, total_games - games)
    eta_seconds = remaining_games / games_per_second if games_per_second > 0 else float("inf")
    avg_turns = float(aggregate["turns"]) / games if games else 0.0
    deck_wins = aggregate["deck_wins"]
    total_recorded_wins = max(1, sum(int(value) for value in deck_wins.values()))
    ampharos_win_rate = 100.0 * deck_wins["ampharos-ex-battle-deck"] / total_recorded_wins
    lucario_win_rate = 100.0 * deck_wins["lucario-ex-battle-deck"] / total_recorded_wins
    eta_text = "unknown" if not math.isfinite(eta_seconds) else _format_duration(eta_seconds)
    print(
        "[self-play] "
        f"games={games}/{total_games} "
        f"games/s={games_per_second:.1f} "
        f"ampharos={ampharos_win_rate:.1f}% "
        f"lucario={lucario_win_rate:.1f}% "
        f"avg_turns={avg_turns:.1f} "
        f"samples={aggregate['samples']} "
        f"truncated={aggregate['truncated']} "
        f"eta={eta_text}"
    )


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    minutes, remaining_seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{remaining_seconds:02d}s"
    return f"{remaining_seconds}s"


if __name__ == "__main__":
    raise SystemExit(main())
