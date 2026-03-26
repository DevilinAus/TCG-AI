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
import shutil
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
from backend.tcg_ai.game_modes.standard.cards import paired_deck_id_for
from backend.tcg_ai.game_modes.standard.ml.neural_policy import (
    DEFAULT_CHECKPOINT_PATH,
    PolicyValueBackend,
)

DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "standard_ml_data" / "evaluations"
EVAL_ASSIGNMENTS = (
    {"player0_deck_id": "ampharos-ex-battle-deck", "candidate_player_index": 0},
    {"player0_deck_id": "lucario-ex-battle-deck", "candidate_player_index": 1},
    {"player0_deck_id": "lucario-ex-battle-deck", "candidate_player_index": 0},
    {"player0_deck_id": "ampharos-ex-battle-deck", "candidate_player_index": 1},
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a candidate Standard checkpoint against the current champion or heuristic baseline.")
    parser.add_argument("--candidate", type=Path, required=True, help="Candidate checkpoint to evaluate.")
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="Baseline checkpoint. Defaults to standard_ml_data/champion.pt when present, otherwise heuristic.",
    )
    parser.add_argument("--games", type=int, default=400, help="Total evaluation games.")
    parser.add_argument("--workers", type=int, default=max(1, min(os.cpu_count() or 1, 8)))
    parser.add_argument("--chunk-size", type=int, default=50, help="Games per worker shard.")
    parser.add_argument("--max-actions-per-game", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--opponent-branch-width", type=int, default=2)
    parser.add_argument("--disable-opponent-turn", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--promote-path", type=Path, default=None)
    parser.add_argument("--promotion-threshold", type=float, default=0.55)
    parser.add_argument("--seed", type=int, default=random.randint(1, 999_999))
    parser.add_argument("--log-every-seconds", type=float, default=5.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.games <= 0:
        raise SystemExit("--games must be positive.")
    if args.chunk_size <= 0:
        raise SystemExit("--chunk-size must be positive.")
    if not 0.0 <= args.promotion_threshold <= 1.0:
        raise SystemExit("--promotion-threshold must be between 0.0 and 1.0.")

    candidate_spec = _resolve_model_spec(args.candidate, label="candidate", fallback_to_heuristic=False)
    baseline_spec = _resolve_baseline_spec(args.baseline)
    if args.workers > 1 and (candidate_spec["kind"] == "checkpoint" or baseline_spec["kind"] == "checkpoint"):
        raise SystemExit(
            "Checkpoint-vs-checkpoint evaluation currently requires --workers 1 so we do not duplicate torch models across processes."
        )

    run_id = datetime.now(UTC).strftime("eval_%Y%m%dT%H%M%SZ")
    output_dir = (args.output_dir or (DEFAULT_OUTPUT_ROOT / run_id)).resolve()
    games_dir = output_dir / "games"
    games_dir.mkdir(parents=True, exist_ok=True)

    planner_config = PlannerConfig(
        max_depth=max(1, args.max_depth),
        beam_width=max(1, args.beam_width),
        opponent_branch_width=max(1, args.opponent_branch_width),
        include_opponent_turn=not args.disable_opponent_turn,
    )
    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "games": args.games,
        "workers": args.workers,
        "chunk_size": args.chunk_size,
        "seed": args.seed,
        "candidate": candidate_spec,
        "baseline": baseline_spec,
        "planner_config": asdict(planner_config),
        "max_actions_per_game": args.max_actions_per_game,
        "promotion_threshold": args.promotion_threshold,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    print(f"[eval] output={output_dir}")
    print(
        "[eval] "
        f"games={args.games} workers={args.workers} chunk={args.chunk_size} "
        f"candidate={candidate_spec['label']} baseline={baseline_spec['label']} seed={args.seed}"
    )
    print(f"[eval] candidate-status={json.dumps(candidate_spec['status'], sort_keys=True)}")
    print(f"[eval] baseline-status={json.dumps(baseline_spec['status'], sort_keys=True)}")

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
                "candidate_spec": candidate_spec,
                "baseline_spec": baseline_spec,
            }
        )

    aggregate = {
        "games": 0,
        "candidate_wins": 0,
        "baseline_wins": 0,
        "draws": 0,
        "truncated": 0,
        "candidate_deck_wins": {
            "ampharos-ex-battle-deck": 0,
            "lucario-ex-battle-deck": 0,
        },
        "baseline_deck_wins": {
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
            chunk_summary = _run_evaluation_chunk(task)
            _merge_chunk_summary(aggregate, chunk_summary)
            now = time.perf_counter()
            if now - last_log_time >= args.log_every_seconds or aggregate["games"] == args.games:
                _print_progress(aggregate, total_games=args.games, start_time=start_time)
                last_log_time = now
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_map = {executor.submit(_run_evaluation_chunk, task): task for task in tasks}
            for future in as_completed(future_map):
                chunk_summary = future.result()
                _merge_chunk_summary(aggregate, chunk_summary)
                now = time.perf_counter()
                if now - last_log_time >= args.log_every_seconds or aggregate["games"] == args.games:
                    _print_progress(aggregate, total_games=args.games, start_time=start_time)
                    last_log_time = now

    decisive_games = aggregate["candidate_wins"] + aggregate["baseline_wins"]
    candidate_win_rate = (
        aggregate["candidate_wins"] / decisive_games
        if decisive_games > 0
        else 0.0
    )
    promoted = False
    promoted_path = None
    if args.promote_path is not None and decisive_games > 0 and candidate_win_rate >= args.promotion_threshold:
        args.promote_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate_spec["checkpoint_path"], args.promote_path)
        promoted = True
        promoted_path = str(args.promote_path.resolve())
        print(f"[eval] promoted new champion: {promoted_path}")
    elif args.promote_path is not None:
        print(
            "[eval] "
            f"candidate win rate {candidate_win_rate:.3%} did not clear threshold {args.promotion_threshold:.3%}; champion unchanged"
        )

    summary_payload = {
        "run_id": run_id,
        "completed_at": datetime.now(UTC).isoformat(),
        "candidate": candidate_spec,
        "baseline": baseline_spec,
        "candidate_win_rate": round(candidate_win_rate, 6),
        "decisive_games": decisive_games,
        "promoted": promoted,
        "promoted_path": promoted_path,
        **aggregate,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8")
    print(
        "[eval] "
        f"candidate={candidate_spec['label']} baseline={baseline_spec['label']} "
        f"wins={aggregate['candidate_wins']} losses={aggregate['baseline_wins']} "
        f"draws={aggregate['draws']} winrate={candidate_win_rate:.3%}"
    )
    print(f"[eval] summary={output_dir / 'summary.json'}")
    return 0


def _run_evaluation_chunk(task: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(str(task["output_dir"]))
    games_path = output_dir / "games" / f"shard_{int(task['task_index']):06d}.jsonl"
    planner_config = PlannerConfig(**dict(task["planner_config"]))
    candidate_spec = dict(task["candidate_spec"])
    baseline_spec = dict(task["baseline_spec"])
    candidate_oracle = _build_oracle(candidate_spec)
    baseline_oracle = _build_oracle(baseline_spec)

    game_lines: list[str] = []
    summary = {
        "games": 0,
        "candidate_wins": 0,
        "baseline_wins": 0,
        "draws": 0,
        "truncated": 0,
        "turns": 0,
        "actions": 0,
        "candidate_deck_wins": {
            "ampharos-ex-battle-deck": 0,
            "lucario-ex-battle-deck": 0,
        },
        "baseline_deck_wins": {
            "ampharos-ex-battle-deck": 0,
            "lucario-ex-battle-deck": 0,
        },
    }

    for offset in range(int(task["game_count"])):
        global_index = int(task["start_index"]) + offset
        assignment = EVAL_ASSIGNMENTS[global_index % len(EVAL_ASSIGNMENTS)]
        candidate_player_index = int(assignment["candidate_player_index"])
        seed = int(task["base_seed"]) + global_index
        game_summary, _ = play_self_play_game(
            game_id=f"{task['run_id']}-g{global_index:07d}",
            seed=seed,
            config=SelfPlayConfig(
                player0_deck_id=str(assignment["player0_deck_id"]),
                planner_config=planner_config,
                max_actions_per_game=int(task["max_actions_per_game"]),
                collect_training_records=False,
            ),
            oracle_by_player={
                candidate_player_index: candidate_oracle,
                1 - candidate_player_index: baseline_oracle,
            },
        )

        player0_deck_id = str(assignment["player0_deck_id"])
        player1_deck_id = paired_deck_id_for(player0_deck_id)
        candidate_deck_id = player0_deck_id if candidate_player_index == 0 else player1_deck_id
        baseline_deck_id = player1_deck_id if candidate_player_index == 0 else player0_deck_id

        candidate_result = "draw"
        if game_summary.winner is None:
            summary["draws"] += 1
        elif int(game_summary.winner) == candidate_player_index:
            candidate_result = "win"
            summary["candidate_wins"] += 1
            summary["candidate_deck_wins"][candidate_deck_id] += 1
        else:
            candidate_result = "loss"
            summary["baseline_wins"] += 1
            summary["baseline_deck_wins"][baseline_deck_id] += 1

        game_payload = asdict(game_summary)
        game_payload["candidate_player_index"] = candidate_player_index
        game_payload["candidate_deck_id"] = candidate_deck_id
        game_payload["baseline_deck_id"] = baseline_deck_id
        game_payload["candidate_result"] = candidate_result
        game_payload["candidate_label"] = candidate_spec["label"]
        game_payload["baseline_label"] = baseline_spec["label"]
        game_lines.append(json.dumps(game_payload, sort_keys=True))

        summary["games"] += 1
        summary["truncated"] += 1 if game_summary.truncated else 0
        summary["turns"] += game_summary.turn_number
        summary["actions"] += game_summary.action_count

    games_path.write_text("\n".join(game_lines) + ("\n" if game_lines else ""), encoding="utf-8")
    return summary


def _build_oracle(spec: dict[str, Any]):
    if spec["kind"] == "checkpoint":
        backend = PolicyValueBackend(checkpoint_path=Path(str(spec["checkpoint_path"])))
        return BackendPolicyValueOracle(backend=backend)
    return HeuristicPolicyValueOracle()


def _resolve_model_spec(
    checkpoint_path: Path,
    *,
    label: str,
    fallback_to_heuristic: bool,
) -> dict[str, Any]:
    resolved_path = checkpoint_path.resolve()
    if not resolved_path.exists():
        if fallback_to_heuristic:
            return {
                "kind": "heuristic",
                "label": label,
                "checkpoint_path": None,
                "status": {
                    "backend": "heuristic",
                    "model_loaded": False,
                    "checkpoint_path": None,
                },
            }
        raise SystemExit(f"{label} checkpoint does not exist: {resolved_path}")

    backend = PolicyValueBackend(checkpoint_path=resolved_path)
    status = backend.status
    if not status.model_loaded:
        raise SystemExit(
            f"{label} checkpoint exists but could not be loaded as a model: {resolved_path}. "
            "Make sure PyTorch is installed and the checkpoint matches ActionConditionedPolicyValueNet."
        )
    return {
        "kind": "checkpoint",
        "label": resolved_path.stem,
        "checkpoint_path": str(resolved_path),
        "status": {
            "backend": status.backend,
            "model_loaded": status.model_loaded,
            "checkpoint_path": status.checkpoint_path,
        },
    }


def _resolve_baseline_spec(baseline_path: Path | None) -> dict[str, Any]:
    if baseline_path is not None:
        return _resolve_model_spec(baseline_path, label="baseline", fallback_to_heuristic=False)
    if DEFAULT_CHECKPOINT_PATH.exists():
        return _resolve_model_spec(DEFAULT_CHECKPOINT_PATH, label="champion", fallback_to_heuristic=False)
    return {
        "kind": "heuristic",
        "label": "heuristic",
        "checkpoint_path": None,
        "status": {
            "backend": "heuristic",
            "model_loaded": False,
            "checkpoint_path": None,
        },
    }


def _merge_chunk_summary(aggregate: dict[str, Any], chunk_summary: dict[str, Any]) -> None:
    aggregate["games"] += int(chunk_summary["games"])
    aggregate["candidate_wins"] += int(chunk_summary["candidate_wins"])
    aggregate["baseline_wins"] += int(chunk_summary["baseline_wins"])
    aggregate["draws"] += int(chunk_summary["draws"])
    aggregate["truncated"] += int(chunk_summary["truncated"])
    aggregate["turns"] += int(chunk_summary["turns"])
    aggregate["actions"] += int(chunk_summary["actions"])
    for deck_id, wins in chunk_summary["candidate_deck_wins"].items():
        aggregate["candidate_deck_wins"][deck_id] += int(wins)
    for deck_id, wins in chunk_summary["baseline_deck_wins"].items():
        aggregate["baseline_deck_wins"][deck_id] += int(wins)


def _print_progress(aggregate: dict[str, Any], *, total_games: int, start_time: float) -> None:
    elapsed = max(time.perf_counter() - start_time, 1e-6)
    games = aggregate["games"]
    decisive_games = aggregate["candidate_wins"] + aggregate["baseline_wins"]
    candidate_win_rate = aggregate["candidate_wins"] / decisive_games if decisive_games > 0 else 0.0
    games_per_second = games / elapsed
    avg_turns = aggregate["turns"] / games if games else 0.0
    remaining_games = max(total_games - games, 0)
    eta_seconds = remaining_games / games_per_second if games_per_second > 0 else float("inf")
    eta_display = "inf" if not math.isfinite(eta_seconds) else f"{eta_seconds:.0f}s"
    print(
        "[eval] "
        f"games={games}/{total_games} "
        f"games/s={games_per_second:.1f} "
        f"candidate={candidate_win_rate:.1%} "
        f"baseline={(1.0 - candidate_win_rate):.1%} "
        f"draws={aggregate['draws']} "
        f"truncated={aggregate['truncated']} "
        f"avg_turns={avg_turns:.1f} "
        f"eta={eta_display}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
