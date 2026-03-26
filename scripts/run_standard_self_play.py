#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
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
    BackendPolicyValueOracle,
    HeuristicPolicyValueOracle,
    PlannerConfig,
    SelfPlayConfig,
    play_self_play_game,
)
from backend.tcg_ai.game_modes.standard.ml.neural_policy import PolicyValueBackend

DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "standard_ml_data" / "self_play"
MATCHUP_PLAYER0_DECKS = (
    "ampharos-ex-battle-deck",
    "lucario-ex-battle-deck",
)
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
    parser.add_argument("--oracle", choices=("heuristic", "local-model"), default="heuristic")
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
    if args.oracle == "local-model" and args.workers > 1:
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

    oracle_status = _resolve_oracle_status(args)
    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "games": args.games,
        "workers": args.workers,
        "chunk_size": args.chunk_size,
        "seed": args.seed,
        "oracle": args.oracle,
        "oracle_status": oracle_status,
        "planner_config": asdict(planner_config),
        "max_actions_per_game": args.max_actions_per_game,
        "include_setup_decisions": args.include_setup_decisions,
        "record_forced_actions": args.record_forced_actions,
        "matchup_player0_decks": list(MATCHUP_PLAYER0_DECKS),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    print(f"[self-play] output={output_dir}")
    print(
        "[self-play] "
        f"games={args.games} workers={args.workers} chunk={args.chunk_size} "
        f"oracle={args.oracle} seed={args.seed}"
    )
    if oracle_status:
        print(f"[self-play] oracle-status={json.dumps(oracle_status, sort_keys=True)}")

    tasks = []
    task_count = math.ceil(args.games / args.chunk_size)
    for task_index in range(task_count):
        start_index = task_index * args.chunk_size
        game_count = min(args.chunk_size, args.games - start_index)
        tasks.append(
            {
                "task_index": task_index,
                "run_id": run_id,
                "start_index": start_index,
                "game_count": game_count,
                "base_seed": args.seed,
                "output_dir": str(output_dir),
                "planner_config": asdict(planner_config),
                "max_actions_per_game": args.max_actions_per_game,
                "include_setup_decisions": args.include_setup_decisions,
                "record_forced_actions": args.record_forced_actions,
                "oracle": args.oracle,
                "checkpoint": str(args.checkpoint) if args.checkpoint else None,
                "progress_log": str(args.progress_log.resolve()) if args.progress_log else None,
            }
        )

    aggregate = {
        "games": 0,
        "samples": 0,
        "truncated": 0,
        "deck_wins": {
            "ampharos-ex-battle-deck": 0,
            "lucario-ex-battle-deck": 0,
        },
        "turns": 0,
        "actions": 0,
    }
    start_time = time.perf_counter()
    last_log_time = start_time

    if args.workers == 1:
        for task in tasks:
            chunk_summary = _run_self_play_chunk(task)
            _merge_chunk_summary(aggregate, chunk_summary)
            now = time.perf_counter()
            if now - last_log_time >= args.log_every_seconds or aggregate["games"] == args.games:
                _print_progress(aggregate, total_games=args.games, start_time=start_time)
                last_log_time = now
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_map = {executor.submit(_run_self_play_chunk, task): task for task in tasks}
            for future in as_completed(future_map):
                chunk_summary = future.result()
                _merge_chunk_summary(aggregate, chunk_summary)
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


def _run_self_play_chunk(task: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(str(task["output_dir"]))
    decisions_path = output_dir / "decisions" / f"shard_{int(task['task_index']):06d}.jsonl"
    games_path = output_dir / "games" / f"shard_{int(task['task_index']):06d}.jsonl"
    planner_config = PlannerConfig(**dict(task["planner_config"]))
    oracle = _build_oracle(task["oracle"], task.get("checkpoint"))
    progress_log = Path(str(task["progress_log"])) if task.get("progress_log") else None

    decisions_lines: list[str] = []
    game_lines: list[str] = []
    summary = {
        "games": 0,
        "samples": 0,
        "truncated": 0,
        "turns": 0,
        "actions": 0,
        "deck_wins": {
            "ampharos-ex-battle-deck": 0,
            "lucario-ex-battle-deck": 0,
        },
    }

    for offset in range(int(task["game_count"])):
        global_index = int(task["start_index"]) + offset
        player0_deck_id = MATCHUP_PLAYER0_DECKS[global_index % len(MATCHUP_PLAYER0_DECKS)]
        config = SelfPlayConfig(
            player0_deck_id=player0_deck_id,
            planner_config=planner_config,
            max_actions_per_game=int(task["max_actions_per_game"]),
            include_setup_decisions=bool(task["include_setup_decisions"]),
            record_forced_actions=bool(task["record_forced_actions"]),
        )
        game_id = f"{task['run_id']}-g{global_index:07d}"
        seed = int(task["base_seed"]) + global_index
        game_summary, decision_records = play_self_play_game(
            game_id=game_id,
            seed=seed,
            config=config,
            oracle=oracle,
        )
        game_payload = asdict(game_summary)
        winner = game_payload.get("winner")
        if winner is not None:
            winner_deck_id = (
                game_payload["player0_deck_id"]
                if int(winner) == 0
                else game_payload["player1_deck_id"]
            )
            game_payload["winner_deck_id"] = winner_deck_id
            summary["deck_wins"][winner_deck_id] += 1
        else:
            game_payload["winner_deck_id"] = None
        game_lines.append(json.dumps(game_payload, sort_keys=True))
        decisions_lines.extend(json.dumps(record, sort_keys=True) for record in decision_records)
        _log_worker_progress(
            progress_log,
            (
                "[self-play-worker] "
                f"run={task['run_id']} shard={int(task['task_index']):06d} "
                f"local_game={offset + 1}/{int(task['game_count'])} "
                f"global_game={global_index + 1} "
                f"winner_deck={game_payload['winner_deck_id']} "
                f"turns={game_summary.turn_number} actions={game_summary.action_count} "
                f"samples={len(decision_records)} truncated={game_summary.truncated}"
            ),
        )
        summary["games"] += 1
        summary["samples"] += len(decision_records)
        summary["truncated"] += 1 if game_summary.truncated else 0
        summary["turns"] += game_summary.turn_number
        summary["actions"] += game_summary.action_count

    decisions_path.write_text("\n".join(decisions_lines) + ("\n" if decisions_lines else ""), encoding="utf-8")
    games_path.write_text("\n".join(game_lines) + ("\n" if game_lines else ""), encoding="utf-8")
    return summary


def _merge_chunk_summary(aggregate: dict[str, Any], chunk_summary: dict[str, Any]) -> None:
    aggregate["games"] += int(chunk_summary["games"])
    aggregate["samples"] += int(chunk_summary["samples"])
    aggregate["truncated"] += int(chunk_summary["truncated"])
    aggregate["turns"] += int(chunk_summary["turns"])
    aggregate["actions"] += int(chunk_summary["actions"])
    for deck_id, wins in chunk_summary["deck_wins"].items():
        aggregate["deck_wins"][deck_id] += int(wins)


def _build_oracle(oracle_name: str, checkpoint: str | None):
    if oracle_name == "local-model":
        backend = PolicyValueBackend(checkpoint_path=Path(checkpoint) if checkpoint else None)
        return BackendPolicyValueOracle(backend=backend)
    return HeuristicPolicyValueOracle()


def _resolve_oracle_status(args: argparse.Namespace) -> dict[str, Any]:
    if args.oracle != "local-model":
        return {"backend": "heuristic", "model_loaded": False}
    backend = PolicyValueBackend(checkpoint_path=args.checkpoint)
    status = backend.status
    return {
        "backend": status.backend,
        "model_loaded": status.model_loaded,
        "checkpoint_path": status.checkpoint_path,
    }


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
