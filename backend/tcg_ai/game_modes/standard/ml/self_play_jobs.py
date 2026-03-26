from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
import json
from pathlib import Path
import time
from typing import Any, Callable

from .oracle import BackendPolicyValueOracle, HeuristicPolicyValueOracle
from .neural_policy import PolicyValueBackend
from .planner import PlannerConfig
from .self_play import SelfPlayConfig, play_self_play_game

DEFAULT_MATCHUP_PLAYER0_DECKS = (
    "ampharos-ex-battle-deck",
    "lucario-ex-battle-deck",
)
SELF_PLAY_JOB_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SelfPlayRunConfig:
    games: int = 1000
    chunk_size: int = 100
    seed: int = 1
    planner_config: PlannerConfig = field(default_factory=PlannerConfig)
    max_actions_per_game: int = 200
    include_setup_decisions: bool = False
    record_forced_actions: bool = False
    oracle: str = "heuristic"
    checkpoint: str | None = None
    matchup_player0_decks: tuple[str, ...] = DEFAULT_MATCHUP_PLAYER0_DECKS


@dataclass(frozen=True)
class SelfPlayChunkTask:
    schema_version: int
    run_id: str
    task_index: int
    start_index: int
    game_count: int
    base_seed: int
    planner_config: PlannerConfig
    max_actions_per_game: int
    include_setup_decisions: bool
    record_forced_actions: bool
    oracle: str
    checkpoint: str | None = None
    matchup_player0_decks: tuple[str, ...] = DEFAULT_MATCHUP_PLAYER0_DECKS


@dataclass(frozen=True)
class SelfPlayChunkResult:
    task_index: int
    summary: dict[str, Any]
    decisions_jsonl: str
    games_jsonl: str


def build_self_play_tasks(
    *,
    run_id: str,
    config: SelfPlayRunConfig,
) -> list[SelfPlayChunkTask]:
    if config.games <= 0:
        raise ValueError("games must be positive.")
    if config.chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    tasks: list[SelfPlayChunkTask] = []
    task_count = (config.games + config.chunk_size - 1) // config.chunk_size
    for task_index in range(task_count):
        start_index = task_index * config.chunk_size
        game_count = min(config.chunk_size, config.games - start_index)
        tasks.append(
            SelfPlayChunkTask(
                schema_version=SELF_PLAY_JOB_SCHEMA_VERSION,
                run_id=run_id,
                task_index=task_index,
                start_index=start_index,
                game_count=game_count,
                base_seed=config.seed,
                planner_config=config.planner_config,
                max_actions_per_game=config.max_actions_per_game,
                include_setup_decisions=config.include_setup_decisions,
                record_forced_actions=config.record_forced_actions,
                oracle=config.oracle,
                checkpoint=config.checkpoint,
                matchup_player0_decks=config.matchup_player0_decks,
            )
        )
    return tasks


def build_self_play_manifest(
    *,
    run_id: str,
    config: SelfPlayRunConfig,
    oracle_status: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SELF_PLAY_JOB_SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "games": config.games,
        "chunk_size": config.chunk_size,
        "seed": config.seed,
        "oracle": config.oracle,
        "oracle_status": oracle_status,
        "planner_config": asdict(config.planner_config),
        "max_actions_per_game": config.max_actions_per_game,
        "include_setup_decisions": config.include_setup_decisions,
        "record_forced_actions": config.record_forced_actions,
        "matchup_player0_decks": list(config.matchup_player0_decks),
    }


def run_self_play_chunk(
    task: SelfPlayChunkTask,
    *,
    progress_callback: Callable[[str], None] | None = None,
    progress_event_callback: Callable[[dict[str, Any]], None] | None = None,
) -> SelfPlayChunkResult:
    oracle = _build_oracle(task.oracle, task.checkpoint)
    decisions_lines: list[str] = []
    game_lines: list[str] = []
    summary = _empty_aggregate_summary()
    for offset in range(task.game_count):
        global_index = task.start_index + offset
        player0_deck_id = task.matchup_player0_decks[global_index % len(task.matchup_player0_decks)]
        config = SelfPlayConfig(
            player0_deck_id=player0_deck_id,
            planner_config=task.planner_config,
            max_actions_per_game=task.max_actions_per_game,
            include_setup_decisions=task.include_setup_decisions,
            record_forced_actions=task.record_forced_actions,
        )
        started_at = time.perf_counter()
        game_summary, decision_records = play_self_play_game(
            game_id=f"{task.run_id}-g{global_index:07d}",
            seed=task.base_seed + global_index,
            config=config,
            oracle=oracle,
        )
        duration_seconds = time.perf_counter() - started_at
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
        summary["games"] += 1
        summary["samples"] += len(decision_records)
        summary["truncated"] += 1 if game_summary.truncated else 0
        summary["turns"] += game_summary.turn_number
        summary["actions"] += game_summary.action_count
        if progress_event_callback is not None:
            progress_event_callback(
                {
                    "run_id": task.run_id,
                    "task_index": task.task_index,
                    "local_game_index": offset + 1,
                    "task_game_count": task.game_count,
                    "global_game_index": global_index + 1,
                    "winner_deck_id": game_payload["winner_deck_id"],
                    "turns": game_summary.turn_number,
                    "actions": game_summary.action_count,
                    "samples": len(decision_records),
                    "truncated": game_summary.truncated,
                    "duration_seconds": round(duration_seconds, 6),
                    "player0_deck_id": game_payload["player0_deck_id"],
                    "player1_deck_id": game_payload["player1_deck_id"],
                }
            )
        if progress_callback is not None:
            progress_callback(
                "[self-play-worker] "
                f"run={task.run_id} shard={task.task_index:06d} "
                f"local_game={offset + 1}/{task.game_count} "
                f"global_game={global_index + 1} "
                f"winner_deck={game_payload['winner_deck_id']} "
                f"turns={game_summary.turn_number} actions={game_summary.action_count} "
                f"samples={len(decision_records)} truncated={game_summary.truncated}"
            )
    return SelfPlayChunkResult(
        task_index=task.task_index,
        summary=summary,
        decisions_jsonl="\n".join(decisions_lines) + ("\n" if decisions_lines else ""),
        games_jsonl="\n".join(game_lines) + ("\n" if game_lines else ""),
    )


def write_self_play_chunk_artifacts(
    *,
    output_dir: Path,
    result: SelfPlayChunkResult,
) -> None:
    decisions_dir = output_dir / "decisions"
    games_dir = output_dir / "games"
    decisions_dir.mkdir(parents=True, exist_ok=True)
    games_dir.mkdir(parents=True, exist_ok=True)
    (decisions_dir / f"shard_{result.task_index:06d}.jsonl").write_text(
        result.decisions_jsonl,
        encoding="utf-8",
    )
    (games_dir / f"shard_{result.task_index:06d}.jsonl").write_text(
        result.games_jsonl,
        encoding="utf-8",
    )


def merge_chunk_summary(aggregate: dict[str, Any], chunk_summary: dict[str, Any]) -> None:
    aggregate["games"] += int(chunk_summary["games"])
    aggregate["samples"] += int(chunk_summary["samples"])
    aggregate["truncated"] += int(chunk_summary["truncated"])
    aggregate["turns"] += int(chunk_summary["turns"])
    aggregate["actions"] += int(chunk_summary["actions"])
    for deck_id, wins in chunk_summary["deck_wins"].items():
        aggregate["deck_wins"][deck_id] += int(wins)


def empty_self_play_summary() -> dict[str, Any]:
    return _empty_aggregate_summary()


def self_play_task_from_payload(payload: dict[str, Any]) -> SelfPlayChunkTask:
    planner_payload = payload.get("planner_config")
    if not isinstance(planner_payload, dict):
        raise ValueError("Self-play task is missing planner_config.")
    matchup_player0_decks = payload.get("matchup_player0_decks") or list(DEFAULT_MATCHUP_PLAYER0_DECKS)
    if not isinstance(matchup_player0_decks, list) or not all(isinstance(deck_id, str) for deck_id in matchup_player0_decks):
        raise ValueError("Self-play task has invalid matchup_player0_decks.")
    return SelfPlayChunkTask(
        schema_version=int(payload.get("schema_version", SELF_PLAY_JOB_SCHEMA_VERSION)),
        run_id=str(payload["run_id"]),
        task_index=int(payload["task_index"]),
        start_index=int(payload["start_index"]),
        game_count=int(payload["game_count"]),
        base_seed=int(payload["base_seed"]),
        planner_config=PlannerConfig(**planner_payload),
        max_actions_per_game=int(payload["max_actions_per_game"]),
        include_setup_decisions=bool(payload.get("include_setup_decisions")),
        record_forced_actions=bool(payload.get("record_forced_actions")),
        oracle=str(payload.get("oracle", "heuristic")),
        checkpoint=str(payload["checkpoint"]) if payload.get("checkpoint") else None,
        matchup_player0_decks=tuple(matchup_player0_decks),
    )


def self_play_task_to_payload(task: SelfPlayChunkTask) -> dict[str, Any]:
    payload = asdict(task)
    payload["planner_config"] = asdict(task.planner_config)
    payload["matchup_player0_decks"] = list(task.matchup_player0_decks)
    return payload


def self_play_run_config_to_payload(config: SelfPlayRunConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["planner_config"] = asdict(config.planner_config)
    payload["matchup_player0_decks"] = list(config.matchup_player0_decks)
    return payload


def self_play_run_config_from_payload(payload: dict[str, Any]) -> SelfPlayRunConfig:
    planner_payload = payload.get("planner_config")
    if not isinstance(planner_payload, dict):
        raise ValueError("Self-play run config is missing planner_config.")
    matchup_player0_decks = payload.get("matchup_player0_decks") or list(DEFAULT_MATCHUP_PLAYER0_DECKS)
    if not isinstance(matchup_player0_decks, list) or not all(isinstance(deck_id, str) for deck_id in matchup_player0_decks):
        raise ValueError("Self-play run config has invalid matchup_player0_decks.")
    return SelfPlayRunConfig(
        games=int(payload["games"]),
        chunk_size=int(payload["chunk_size"]),
        seed=int(payload["seed"]),
        planner_config=PlannerConfig(**planner_payload),
        max_actions_per_game=int(payload.get("max_actions_per_game", 200)),
        include_setup_decisions=bool(payload.get("include_setup_decisions")),
        record_forced_actions=bool(payload.get("record_forced_actions")),
        oracle=str(payload.get("oracle", "heuristic")),
        checkpoint=str(payload["checkpoint"]) if payload.get("checkpoint") else None,
        matchup_player0_decks=tuple(matchup_player0_decks),
    )


def resolve_oracle_status(*, oracle: str, checkpoint: Path | None) -> dict[str, Any]:
    if oracle != "local-model":
        return {"backend": "heuristic", "model_loaded": False}
    backend = PolicyValueBackend(checkpoint_path=checkpoint)
    status = backend.status
    return {
        "backend": status.backend,
        "model_loaded": status.model_loaded,
        "checkpoint_path": status.checkpoint_path,
    }


def _build_oracle(oracle_name: str, checkpoint: str | None):
    if oracle_name == "local-model":
        backend = PolicyValueBackend(checkpoint_path=Path(checkpoint) if checkpoint else None)
        return BackendPolicyValueOracle(backend=backend)
    return HeuristicPolicyValueOracle()


def _empty_aggregate_summary() -> dict[str, Any]:
    return {
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
