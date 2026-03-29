#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections.abc import Iterable
from datetime import UTC, datetime
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any
import zlib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.tcg_ai.game_modes.standard.ml.neural_policy import (
    ACTION_VECTOR_SIZE,
    DEFAULT_HIDDEN_SIZE,
    ENCODER_VERSION,
    STATE_VECTOR_SIZE,
    ActionConditionedPolicyValueNet,
    DEFAULT_CHECKPOINT_PATH,
    encode_action_vector,
    encode_state_vector,
    infer_checkpoint_model_dimensions,
    load_trusted_checkpoint,
)

DEFAULT_SELF_PLAY_ROOT = PROJECT_ROOT / "standard_ml_data" / "self_play"
DEFAULT_CHECKPOINT_ROOT = PROJECT_ROOT / "standard_ml_data" / "checkpoints"
torch = None
F = None


def _require_torch_modules():
    global torch, F
    if torch is not None and F is not None:
        return
    try:
        import torch as torch_module
        from torch.nn import functional as functional_module
    except Exception as exc:  # pragma: no cover - runtime dependency path
        raise SystemExit(
            "PyTorch is required for training. Install the optional standard-ml dependencies before running this script."
        ) from exc
    torch = torch_module
    F = functional_module


class SelfPlayDecisionDataset:
    def __init__(
        self,
        decision_paths: list[Path],
        *,
        state_dim: int,
        action_dim: int,
        split: str,
        validation_mod: int,
        validation_bucket: int,
        max_records: int | None = None,
    ) -> None:
        super().__init__()
        self.decision_paths = decision_paths
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.split = split
        self.validation_mod = max(2, validation_mod)
        self.validation_bucket = validation_bucket % self.validation_mod
        self.max_records = max_records
        self._warned_runtime_corruption: set[Path] = set()

    def __iter__(self):
        yielded = 0
        for path in self.decision_paths:
            try:
                handle = path.open("r", encoding="utf-8")
            except OSError as exc:
                _warn_skipped_shard(path, f"could not open shard: {exc}")
                continue
            with handle:
                for line_number, line in enumerate(handle, start=1):
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        payload = json.loads(stripped)
                    except json.JSONDecodeError as exc:
                        if path not in self._warned_runtime_corruption:
                            _warn_skipped_shard(
                                path,
                                f"encountered malformed JSON at line {line_number}: {exc}",
                            )
                            self._warned_runtime_corruption.add(path)
                        break
                    if not _record_in_split(
                        payload,
                        split=self.split,
                        validation_mod=self.validation_mod,
                        validation_bucket=self.validation_bucket,
                    ):
                        continue
                    encoded = _encode_training_record(
                        payload,
                        state_dim=self.state_dim,
                        action_dim=self.action_dim,
                    )
                    if encoded is None:
                        continue
                    yield encoded
                    yielded += 1
                    if self.max_records is not None and yielded >= self.max_records:
                        return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Standard action-conditioned model from self-play shards.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Self-play run directory. Defaults to the newest run under standard_ml_data/self_play.",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Checkpoint output directory.")
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--promote-path", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--value-loss-weight", type=float, default=0.5)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--validation-mod", type=int, default=20)
    parser.add_argument("--validation-bucket", type=int, default=0)
    parser.add_argument("--max-train-records", type=int, default=None)
    parser.add_argument("--max-eval-records", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _require_torch_modules()
    input_dir = _resolve_input_dir(args.input_dir)
    decision_paths, shard_report = _resolve_training_decision_paths(input_dir)
    if not decision_paths:
        raise SystemExit(f"No usable decision shards found under {input_dir / 'decisions'}")

    output_dir = _resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device)

    model_state_dim = STATE_VECTOR_SIZE
    model_action_dim = ACTION_VECTOR_SIZE
    checkpoint = None
    if args.resume_from is not None:
        checkpoint = load_trusted_checkpoint(args.resume_from, map_location=device)
        model_state_dim, model_action_dim = infer_checkpoint_model_dimensions(checkpoint)

    model = ActionConditionedPolicyValueNet(state_dim=model_state_dim, action_dim=model_action_dim).to(device)
    if checkpoint is not None:
        state_dict = checkpoint.get("state_dict") if isinstance(checkpoint, dict) else checkpoint
        model.load_state_dict(state_dict)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    train_dataset = SelfPlayDecisionDataset(
        decision_paths,
        state_dim=model_state_dim,
        action_dim=model_action_dim,
        split="train",
        validation_mod=args.validation_mod,
        validation_bucket=args.validation_bucket,
        max_records=args.max_train_records,
    )
    eval_dataset = SelfPlayDecisionDataset(
        decision_paths,
        state_dim=model_state_dim,
        action_dim=model_action_dim,
        split="eval",
        validation_mod=args.validation_mod,
        validation_bucket=args.validation_bucket,
        max_records=args.max_eval_records,
    )
    batch_size = max(1, args.batch_size)

    print(
        "[train] "
        f"input={input_dir} output={output_dir} device={device} "
        f"files={len(decision_paths)} batch_size={batch_size} "
        f"state_dim={model_state_dim} action_dim={model_action_dim}"
    )
    print(
        "[train] "
        f"shard_selection mode={shard_report['selection_mode']} "
        f"usable={shard_report['usable_shards']} "
        f"skipped_invalid={shard_report['skipped_invalid_shards']} "
        f"ignored_uncommitted={shard_report['ignored_uncommitted_shards']}"
    )

    global_step = 0
    latest_checkpoint_path: Path | None = None
    for epoch in range(1, max(1, args.epochs) + 1):
        epoch_start = time.perf_counter()
        epoch_metrics = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "total_loss": 0.0,
            "batches": 0,
            "samples": 0,
        }
        for batch in _iter_collated_batches(train_dataset, batch_size=batch_size):
            if batch is None:
                continue
            global_step += 1
            batch = _move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            policy_logits, values = model(batch["state_vectors"], batch["action_vectors"])
            policy_logits = policy_logits.masked_fill(~batch["action_mask"], -1e9)
            policy_loss = _soft_target_policy_loss(policy_logits, batch["policy_targets"])
            value_loss = F.smooth_l1_loss(values, batch["value_targets"])
            total_loss = policy_loss + value_loss * args.value_loss_weight
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            batch_size = int(batch["state_vectors"].shape[0])
            epoch_metrics["policy_loss"] += float(policy_loss.detach().cpu().item()) * batch_size
            epoch_metrics["value_loss"] += float(value_loss.detach().cpu().item()) * batch_size
            epoch_metrics["total_loss"] += float(total_loss.detach().cpu().item()) * batch_size
            epoch_metrics["batches"] += 1
            epoch_metrics["samples"] += batch_size

            if global_step % max(1, args.log_every) == 0:
                elapsed = max(time.perf_counter() - epoch_start, 1e-6)
                samples_per_second = epoch_metrics["samples"] / elapsed
                print(
                    "[train] "
                    f"epoch={epoch} step={global_step} "
                    f"loss={total_loss.detach().cpu().item():.4f} "
                    f"policy={policy_loss.detach().cpu().item():.4f} "
                    f"value={value_loss.detach().cpu().item():.4f} "
                    f"lr={optimizer.param_groups[0]['lr']:.2e} "
                    f"samples/s={samples_per_second:.1f}"
                )

            if global_step % max(1, args.save_every) == 0:
                latest_checkpoint_path = _save_checkpoint(
                    model=model,
                    output_dir=output_dir,
                    step=global_step,
                    epoch=epoch,
                    args=args,
                )
                print(f"[train] saved checkpoint: {latest_checkpoint_path}")

        if epoch_metrics["samples"] == 0:
            raise SystemExit("No training records were available after filtering.")

        train_policy = epoch_metrics["policy_loss"] / epoch_metrics["samples"]
        train_value = epoch_metrics["value_loss"] / epoch_metrics["samples"]
        train_total = epoch_metrics["total_loss"] / epoch_metrics["samples"]
        print(
            "[train] "
            f"epoch={epoch} complete "
            f"policy={train_policy:.4f} value={train_value:.4f} total={train_total:.4f}"
        )

        eval_metrics = _run_eval(
            model=model,
            eval_loader=_iter_collated_batches(eval_dataset, batch_size=batch_size),
            device=device,
            max_batches=max(0, args.eval_batches),
            value_loss_weight=args.value_loss_weight,
        )
        if eval_metrics["samples"] > 0:
            print(
                "[eval] "
                f"epoch={epoch} "
                f"policy={eval_metrics['policy_loss'] / eval_metrics['samples']:.4f} "
                f"value={eval_metrics['value_loss'] / eval_metrics['samples']:.4f} "
                f"total={eval_metrics['total_loss'] / eval_metrics['samples']:.4f}"
            )

    latest_checkpoint_path = _save_checkpoint(
        model=model,
        output_dir=output_dir,
        step=global_step,
        epoch=max(1, args.epochs),
        args=args,
        latest_name="final.pt",
    )
    latest_alias = output_dir / "latest.pt"
    shutil.copy2(latest_checkpoint_path, latest_alias)
    print(f"[train] final checkpoint: {latest_checkpoint_path}")
    print(f"[train] latest checkpoint alias: {latest_alias}")

    if args.promote_path is not None:
        args.promote_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(latest_checkpoint_path, args.promote_path)
        print(f"[train] promoted checkpoint copy: {args.promote_path}")
    else:
        default_hint = args.resume_from or DEFAULT_CHECKPOINT_PATH
        print(f"[train] to play against this model, point TCG_AI_STANDARD_MODEL_CHECKPOINT at {latest_alias}")
        print(f"[train] current default checkpoint location is {default_hint}")
    return 0


def _record_in_split(
    payload: dict[str, object],
    *,
    split: str,
    validation_mod: int,
    validation_bucket: int,
) -> bool:
    game_id = str(payload.get("game_id", ""))
    step_index = int(payload.get("step_index", 0) or 0)
    bucket = zlib.crc32(f"{game_id}:{step_index}".encode("utf-8")) % validation_mod
    is_eval = bucket == validation_bucket
    return is_eval if split == "eval" else not is_eval


def _encode_training_record(
    payload: dict[str, object],
    *,
    state_dim: int,
    action_dim: int,
) -> dict[str, object] | None:
    belief_state = payload.get("belief_state")
    legal_actions = payload.get("legal_actions")
    chosen_action_id = payload.get("chosen_action_id")
    value_target = payload.get("value_target")
    if not isinstance(belief_state, dict) or not isinstance(legal_actions, list):
        return None
    if not isinstance(chosen_action_id, str):
        return None
    if not isinstance(value_target, (int, float)):
        return None
    action_ids = [str(action.get("action_id", "")) for action in legal_actions if isinstance(action, dict)]
    if chosen_action_id not in action_ids:
        return None
    return {
        "state_vector": encode_state_vector(belief_state, vector_size=state_dim),
        "action_vectors": [
            encode_action_vector(action, belief_state=belief_state, vector_size=action_dim)
            for action in legal_actions
            if isinstance(action, dict)
        ],
        "chosen_action_index": action_ids.index(chosen_action_id),
        "policy_target": _build_policy_target_vector(
            payload=payload,
            action_ids=action_ids,
            chosen_action_id=chosen_action_id,
        ),
        "value_target": float(value_target),
    }


def _collate_training_batch(samples: list[dict[str, object]]):
    if not samples:
        return None
    batch_size = len(samples)
    max_actions = max(len(sample["action_vectors"]) for sample in samples)
    state_dim = max(len(sample["state_vector"]) for sample in samples)
    action_dim = max(
        (len(sample["action_vectors"][0]) for sample in samples if sample["action_vectors"]),
        default=ACTION_VECTOR_SIZE,
    )
    state_vectors = torch.zeros(batch_size, state_dim, dtype=torch.float32)
    action_vectors = torch.zeros(batch_size, max_actions, action_dim, dtype=torch.float32)
    action_mask = torch.zeros(batch_size, max_actions, dtype=torch.bool)
    chosen_indices = torch.zeros(batch_size, dtype=torch.long)
    policy_targets = torch.zeros(batch_size, max_actions, dtype=torch.float32)
    value_targets = torch.zeros(batch_size, dtype=torch.float32)

    for sample_index, sample in enumerate(samples):
        state_vectors[sample_index] = torch.tensor(sample["state_vector"], dtype=torch.float32)
        encoded_actions = torch.tensor(sample["action_vectors"], dtype=torch.float32)
        action_count = encoded_actions.shape[0]
        action_vectors[sample_index, :action_count] = encoded_actions
        action_mask[sample_index, :action_count] = True
        chosen_indices[sample_index] = int(sample["chosen_action_index"])
        policy_target = torch.tensor(sample["policy_target"], dtype=torch.float32)
        policy_targets[sample_index, :action_count] = policy_target
        value_targets[sample_index] = float(sample["value_target"])

    return {
        "state_vectors": state_vectors,
        "action_vectors": action_vectors,
        "action_mask": action_mask,
        "chosen_indices": chosen_indices,
        "policy_targets": policy_targets,
        "value_targets": value_targets,
    }


def _iter_collated_batches(dataset: Iterable[dict[str, object]], *, batch_size: int) -> Iterable[dict[str, object] | None]:
    pending: list[dict[str, object]] = []
    for sample in dataset:
        pending.append(sample)
        if len(pending) >= batch_size:
            yield _collate_training_batch(pending)
            pending = []
    if pending:
        yield _collate_training_batch(pending)


def _move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _build_policy_target_vector(
    *,
    payload: dict[str, object],
    action_ids: list[str],
    chosen_action_id: str,
) -> list[float]:
    policy_target_probs = payload.get("policy_target_probs")
    if isinstance(policy_target_probs, dict):
        weights: list[float] = []
        for action_id in action_ids:
            raw_weight = policy_target_probs.get(action_id, 0.0)
            if not isinstance(raw_weight, (int, float)):
                raw_weight = 0.0
            weights.append(max(0.0, float(raw_weight)))
        total = sum(weights)
        if total > 0:
            return [weight / total for weight in weights]
    return [1.0 if action_id == chosen_action_id else 0.0 for action_id in action_ids]


def _soft_target_policy_loss(policy_logits, policy_targets):
    normalized_targets = policy_targets / policy_targets.sum(dim=1, keepdim=True).clamp_min(1e-9)
    log_probs = F.log_softmax(policy_logits, dim=1)
    return -(normalized_targets * log_probs).sum(dim=1).mean()


def _run_eval(
    *,
    model: ActionConditionedPolicyValueNet,
    eval_loader: Iterable,
    device: torch.device,
    max_batches: int,
    value_loss_weight: float,
) -> dict[str, float]:
    if max_batches <= 0:
        return {"policy_loss": 0.0, "value_loss": 0.0, "total_loss": 0.0, "samples": 0}
    metrics = {"policy_loss": 0.0, "value_loss": 0.0, "total_loss": 0.0, "samples": 0}
    model.eval()
    with torch.no_grad():
        for batch_index, batch in enumerate(eval_loader, start=1):
            if batch is None:
                continue
            batch = _move_batch_to_device(batch, device)
            policy_logits, values = model(batch["state_vectors"], batch["action_vectors"])
            policy_logits = policy_logits.masked_fill(~batch["action_mask"], -1e9)
            policy_loss = _soft_target_policy_loss(policy_logits, batch["policy_targets"])
            value_loss = F.smooth_l1_loss(values, batch["value_targets"])
            total_loss = policy_loss + value_loss * value_loss_weight
            batch_size = int(batch["state_vectors"].shape[0])
            metrics["policy_loss"] += float(policy_loss.detach().cpu().item()) * batch_size
            metrics["value_loss"] += float(value_loss.detach().cpu().item()) * batch_size
            metrics["total_loss"] += float(total_loss.detach().cpu().item()) * batch_size
            metrics["samples"] += batch_size
            if batch_index >= max_batches:
                break
    model.train()
    return metrics


def _save_checkpoint(
    *,
    model: ActionConditionedPolicyValueNet,
    output_dir: Path,
    step: int,
    epoch: int,
    args: argparse.Namespace,
    latest_name: str | None = None,
) -> Path:
    checkpoint_path = output_dir / f"step_{step:07d}.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_config": {
                "state_dim": int(model.state_dim),
                "action_dim": int(model.action_dim),
                "hidden_size": DEFAULT_HIDDEN_SIZE,
                "encoder_version": ENCODER_VERSION,
            },
            "epoch": epoch,
            "step": step,
            "saved_at": datetime.now(UTC).isoformat(),
            "training_config": _json_safe(vars(args)),
        },
        checkpoint_path,
    )
    if latest_name is not None:
        latest_path = output_dir / latest_name
        shutil.copy2(checkpoint_path, latest_path)
    return checkpoint_path


def _resolve_training_decision_paths(input_dir: Path) -> tuple[list[Path], dict[str, int | str]]:
    decisions_dir = input_dir / "decisions"
    summaries_dir = input_dir / "summaries"
    raw_decision_paths = sorted(decisions_dir.glob("*.jsonl"))
    expected_samples_by_path: dict[Path, int] = {}
    ignored_uncommitted_shards = 0

    if summaries_dir.exists():
        summary_paths = sorted(summaries_dir.glob("shard_*.json"))
        if summary_paths:
            selected_paths: list[Path] = []
            for summary_path in summary_paths:
                decision_path = decisions_dir / f"{summary_path.stem}.jsonl"
                if not decision_path.exists():
                    _warn_skipped_shard(
                        decision_path,
                        f"missing decision shard for committed summary {summary_path.name}",
                    )
                    continue
                expected_samples = _load_expected_summary_samples(summary_path)
                if expected_samples is None:
                    continue
                expected_samples_by_path[decision_path] = expected_samples
                selected_paths.append(decision_path)
            ignored_uncommitted_shards = max(0, len(raw_decision_paths) - len(selected_paths))
            validated_paths, invalid_count = _validate_training_decision_paths(
                selected_paths,
                expected_samples_by_path=expected_samples_by_path,
            )
            return validated_paths, {
                "selection_mode": "summary_backed",
                "usable_shards": len(validated_paths),
                "skipped_invalid_shards": invalid_count,
                "ignored_uncommitted_shards": ignored_uncommitted_shards,
            }

    validated_paths, invalid_count = _validate_training_decision_paths(
        raw_decision_paths,
        expected_samples_by_path=expected_samples_by_path,
    )
    return validated_paths, {
        "selection_mode": "raw_decisions",
        "usable_shards": len(validated_paths),
        "skipped_invalid_shards": invalid_count,
        "ignored_uncommitted_shards": 0,
    }


def _load_expected_summary_samples(summary_path: Path) -> int | None:
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _warn_skipped_shard(summary_path, f"invalid shard summary: {exc}")
        return None
    if not isinstance(payload, dict):
        _warn_skipped_shard(summary_path, "invalid shard summary: expected a JSON object")
        return None
    samples = payload.get("samples")
    if not isinstance(samples, int):
        _warn_skipped_shard(summary_path, "invalid shard summary: missing integer 'samples'")
        return None
    return max(samples, 0)


def _validate_training_decision_paths(
    decision_paths: list[Path],
    *,
    expected_samples_by_path: dict[Path, int],
) -> tuple[list[Path], int]:
    valid_paths: list[Path] = []
    invalid_count = 0
    for path in decision_paths:
        if _validate_decision_shard(path, expected_samples=expected_samples_by_path.get(path)):
            valid_paths.append(path)
            continue
        invalid_count += 1
    return valid_paths, invalid_count


def _validate_decision_shard(path: Path, *, expected_samples: int | None) -> bool:
    try:
        handle = path.open("r", encoding="utf-8")
    except OSError as exc:
        _warn_skipped_shard(path, f"could not open shard: {exc}")
        return False
    record_count = 0
    with handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                json.loads(stripped)
            except json.JSONDecodeError as exc:
                _warn_skipped_shard(path, f"malformed JSON at line {line_number}: {exc}")
                return False
            record_count += 1
    if expected_samples is not None and record_count != expected_samples:
        _warn_skipped_shard(
            path,
            f"summary expected {expected_samples} decision samples but found {record_count}",
        )
        return False
    if record_count == 0:
        _warn_skipped_shard(path, "shard contained no decision records")
        return False
    return True


def _warn_skipped_shard(path: Path, detail: str) -> None:
    print(f"[train] skipping shard {path.name}: {detail}", file=sys.stderr)


def _resolve_input_dir(input_dir: Path | None) -> Path:
    if input_dir is not None:
        return input_dir.resolve()
    candidates = sorted(DEFAULT_SELF_PLAY_ROOT.glob("run_*"))
    if not candidates:
        raise SystemExit("No self-play runs found. Generate self-play data first.")
    return candidates[-1].resolve()


def _resolve_output_dir(output_dir: Path | None) -> Path:
    if output_dir is not None:
        return output_dir.resolve()
    run_name = datetime.now(UTC).strftime("run_%Y%m%dT%H%M%SZ")
    return (DEFAULT_CHECKPOINT_ROOT / run_name).resolve()


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA was requested, but no CUDA device is available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


if __name__ == "__main__":
    raise SystemExit(main())
