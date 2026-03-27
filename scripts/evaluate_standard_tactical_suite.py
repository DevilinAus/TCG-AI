#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.tcg_ai.game_modes.standard.ml.neural_policy import PolicyValueBackend
from backend.tcg_ai.game_modes.standard.ml.oracle import BackendPolicyValueOracle, HeuristicPolicyValueOracle
from backend.tcg_ai.game_modes.standard.ml.planner import PlannerConfig
from backend.tcg_ai.game_modes.standard.ml.tactical_suite import run_tactical_suite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Standard tactical regression suite against a heuristic or checkpoint-backed oracle.")
    parser.add_argument("--oracle", choices=("auto", "heuristic", "local-model"), default="auto")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--suite", choices=("core", "strategic", "all"), default="core")
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--opponent-branch-width", type=int, default=2)
    parser.add_argument("--json", action="store_true", help="Print the full result payload as JSON.")
    parser.add_argument("--require-all-pass", action="store_true", help="Exit non-zero if any scenario fails.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    oracle, status = _build_oracle(args.oracle, args.checkpoint)
    planner_config = PlannerConfig(
        max_depth=max(1, args.max_depth),
        beam_width=max(1, args.beam_width),
        opponent_branch_width=max(1, args.opponent_branch_width),
    )
    results = run_tactical_suite(oracle=oracle, planner_config=planner_config, suite=args.suite)
    passed = sum(1 for result in results if result.passed)
    payload = {
        "oracle": args.oracle,
        "suite": args.suite,
        "resolved_backend": status,
        "planner_config": {
            "max_depth": planner_config.max_depth,
            "beam_width": planner_config.beam_width,
            "opponent_branch_width": planner_config.opponent_branch_width,
        },
        "passed": passed,
        "total": len(results),
        "results": [
            {
                "name": result.name,
                "description": result.description,
                "tags": list(result.tags),
                "tier": result.tier,
                "passed": result.passed,
                "chosen_action_id": result.chosen_action_id,
                "chosen_action_type": result.chosen_action_type,
                "explanation": result.explanation,
                "acceptable_reason": result.acceptable_reason,
            }
            for result in results
        ],
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            "[tactical-suite] "
            f"oracle={args.oracle} suite={args.suite} backend={status['backend']} "
            f"model_loaded={status['model_loaded']} passed={passed}/{len(results)}"
        )
        for result in results:
            status_text = "PASS" if result.passed else "FAIL"
            print(
                f"[tactical-suite] {status_text} {result.name} tier={result.tier} "
                f"chosen={result.chosen_action_type}:{result.chosen_action_id} "
                f"expected={result.acceptable_reason}"
            )

    if args.require_all_pass and passed != len(results):
        return 1
    return 0


def _build_oracle(oracle_name: str, checkpoint: Path | None):
    if oracle_name == "heuristic":
        return HeuristicPolicyValueOracle(), {"backend": "heuristic", "model_loaded": False}
    backend = PolicyValueBackend(checkpoint_path=checkpoint)
    if oracle_name == "local-model" and not backend.status.model_loaded:
        raise SystemExit("Requested --oracle local-model, but the checkpoint could not be loaded.")
    if oracle_name == "auto" and not backend.status.model_loaded:
        return HeuristicPolicyValueOracle(), {
            "backend": backend.status.backend,
            "model_loaded": False,
            "checkpoint_path": backend.status.checkpoint_path,
        }
    return BackendPolicyValueOracle(backend=backend), {
        "backend": backend.status.backend,
        "model_loaded": backend.status.model_loaded,
        "checkpoint_path": backend.status.checkpoint_path,
    }


if __name__ == "__main__":
    raise SystemExit(main())
