from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
import json
import os
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
    oracle: str = "auto"
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


@dataclass(frozen=True)
class SelfPlayChunkArtifactPaths:
    decisions: Path
    games: Path
    summary: Path


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
    artifact_paths = self_play_chunk_artifact_paths(output_dir=output_dir, task_index=result.task_index)
    _atomic_write_text(artifact_paths.decisions, result.decisions_jsonl)
    _atomic_write_text(artifact_paths.games, result.games_jsonl)
    # Write the summary last so restart recovery can treat it as the shard commit marker.
    _atomic_write_json(artifact_paths.summary, normalize_self_play_summary(result.summary))


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


def normalize_self_play_summary(payload: Any) -> dict[str, Any]:
    summary = _empty_aggregate_summary()
    if isinstance(payload, dict):
        summary["games"] = _coerce_summary_int(payload.get("games"), default=0)
        summary["samples"] = _coerce_summary_int(payload.get("samples"), default=0)
        summary["truncated"] = _coerce_summary_int(payload.get("truncated"), default=0)
        summary["turns"] = _coerce_summary_int(payload.get("turns"), default=0)
        summary["actions"] = _coerce_summary_int(payload.get("actions"), default=0)
        for deck_id, wins in (payload.get("deck_wins") or {}).items():
            if isinstance(deck_id, str):
                summary["deck_wins"][deck_id] = _coerce_summary_int(wins, default=0)
    return summary


def self_play_chunk_artifact_paths(*, output_dir: Path, task_index: int) -> SelfPlayChunkArtifactPaths:
    shard_name = f"shard_{task_index:06d}"
    return SelfPlayChunkArtifactPaths(
        decisions=output_dir / "decisions" / f"{shard_name}.jsonl",
        games=output_dir / "games" / f"{shard_name}.jsonl",
        summary=output_dir / "summaries" / f"{shard_name}.json",
    )


def load_self_play_chunk_summary(*, output_dir: Path, task_index: int) -> dict[str, Any]:
    artifact_paths = self_play_chunk_artifact_paths(output_dir=output_dir, task_index=task_index)
    try:
        payload = json.loads(artifact_paths.summary.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Missing shard summary for task {task_index}.") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed shard summary for task {task_index}.") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Shard summary for task {task_index} must be a JSON object.")
    return normalize_self_play_summary(payload)


def derive_self_play_chunk_summary_from_artifacts(*, output_dir: Path, task_index: int) -> dict[str, Any]:
    artifact_paths = self_play_chunk_artifact_paths(output_dir=output_dir, task_index=task_index)
    summary = _empty_aggregate_summary()

    try:
        with artifact_paths.games.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Malformed games shard for task {task_index} at line {line_number}."
                    ) from exc
                if not isinstance(payload, dict):
                    raise ValueError(f"Games shard for task {task_index} contains a non-object record.")
                summary["games"] += 1
                summary["turns"] += _coerce_summary_int(payload.get("turn_number"), default=0)
                summary["actions"] += _coerce_summary_int(payload.get("action_count"), default=0)
                summary["truncated"] += 1 if bool(payload.get("truncated")) else 0
                winner_deck_id = payload.get("winner_deck_id")
                if isinstance(winner_deck_id, str):
                    summary["deck_wins"].setdefault(winner_deck_id, 0)
                    summary["deck_wins"][winner_deck_id] += 1
    except FileNotFoundError as exc:
        raise ValueError(f"Missing games shard for task {task_index}.") from exc

    try:
        with artifact_paths.decisions.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                if raw_line.strip():
                    summary["samples"] += 1
    except FileNotFoundError as exc:
        raise ValueError(f"Missing decisions shard for task {task_index}.") from exc

    return summary


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
    if oracle == "auto":
        backend = PolicyValueBackend(checkpoint_path=checkpoint)
        status = backend.status
        resolved_oracle = "local-model" if status.model_loaded else "heuristic"
        return {
            "requested_oracle": "auto",
            "resolved_oracle": resolved_oracle,
            "backend": status.backend,
            "model_loaded": status.model_loaded,
            "checkpoint_path": status.checkpoint_path,
        }
    if oracle != "local-model":
        return {
            "requested_oracle": oracle,
            "resolved_oracle": "heuristic",
            "backend": "heuristic",
            "model_loaded": False,
        }
    backend = PolicyValueBackend(checkpoint_path=checkpoint)
    status = backend.status
    return {
        "requested_oracle": oracle,
        "resolved_oracle": "local-model" if status.model_loaded else "heuristic",
        "backend": status.backend,
        "model_loaded": status.model_loaded,
        "checkpoint_path": status.checkpoint_path,
    }


def _build_oracle(oracle_name: str, checkpoint: str | None):
    if oracle_name == "auto":
        backend = PolicyValueBackend(checkpoint_path=Path(checkpoint) if checkpoint else None)
        if backend.status.model_loaded:
            return BackendPolicyValueOracle(backend=backend)
        return HeuristicPolicyValueOracle()
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


def _coerce_summary_int(value: Any, *, default: int) -> int:
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


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True))


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    temp_path.replace(path)
    _fsync_directory(path.parent)


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
