from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..cards import paired_deck_id_for
from ..engine import action_id_for, apply_action_for_player, create_game, list_legal_actions
from .knowledge_state import serialize_knowledge_actions, serialize_knowledge_state
from .oracle import HeuristicPolicyValueOracle, PolicyValueOracle
from .planner import PlannerConfig, StandardTurnPlanner

SELF_PLAY_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SelfPlayConfig:
    player0_deck_id: str = "ampharos-ex-battle-deck"
    planner_config: PlannerConfig = field(default_factory=PlannerConfig)
    max_actions_per_game: int = 200
    include_setup_decisions: bool = False
    record_forced_actions: bool = False
    collect_training_records: bool = True


@dataclass(frozen=True)
class SelfPlayGameSummary:
    schema_version: int
    game_id: str
    seed: int
    player0_deck_id: str
    player1_deck_id: str
    winner: int | None
    truncated: bool
    turn_number: int
    action_count: int
    decision_samples: int


def play_self_play_game(
    *,
    game_id: str,
    seed: int,
    config: SelfPlayConfig | None = None,
    oracle: PolicyValueOracle | None = None,
    oracle_by_player: dict[int, PolicyValueOracle] | None = None,
) -> tuple[SelfPlayGameSummary, list[dict[str, Any]]]:
    config = config or SelfPlayConfig()
    player1_deck_id = paired_deck_id_for(config.player0_deck_id)
    state = create_game(
        seed=seed,
        ai_name="Self-Play P1",
        human_deck_id=config.player0_deck_id,
    )
    planners = _build_player_planners(
        planner_config=config.planner_config,
        default_oracle=oracle or HeuristicPolicyValueOracle(),
        oracle_by_player=oracle_by_player,
    )
    recorded_samples: list[dict[str, Any]] = []
    action_count = 0

    # Player 1's opening active is normally preselected by the session runtime. Self-play does it here.
    player1_opening_actions = list_legal_actions(state, player_index=1)
    if player1_opening_actions:
        opening_result = _resolve_action_choice(
            state,
            player_index=1,
            legal_actions=player1_opening_actions,
            planner=planners[1],
        )
        if _should_record_decision(
            legal_actions=player1_opening_actions,
            setup_phase=state.setup_phase,
            config=config,
        ):
            recorded_samples.append(
                _build_decision_record(
                    game_id=game_id,
                    seed=seed,
                    state=state,
                    player_index=1,
                    player0_deck_id=config.player0_deck_id,
                    player1_deck_id=player1_deck_id,
                    action_count=action_count,
                    legal_actions=player1_opening_actions,
                    chosen_action=opening_result["action"],
                    planner_diagnostics=opening_result["diagnostics"],
                )
            )
        apply_action_for_player(state, opening_result["action"], 1)
        action_count += 1

    while state.winner is None and action_count < config.max_actions_per_game:
        player_index = state.current_player
        legal_actions = list_legal_actions(state, player_index=player_index)
        if not legal_actions:
            break
        decision = _resolve_action_choice(
            state,
            player_index=player_index,
            legal_actions=legal_actions,
            planner=planners[player_index],
        )
        if _should_record_decision(
            legal_actions=legal_actions,
            setup_phase=state.setup_phase,
            config=config,
        ):
            recorded_samples.append(
                _build_decision_record(
                    game_id=game_id,
                    seed=seed,
                    state=state,
                    player_index=player_index,
                    player0_deck_id=config.player0_deck_id,
                    player1_deck_id=player1_deck_id,
                    action_count=action_count,
                    legal_actions=legal_actions,
                    chosen_action=decision["action"],
                    planner_diagnostics=decision["diagnostics"],
                )
            )
        apply_action_for_player(state, decision["action"], player_index)
        action_count += 1

    truncated = state.winner is None
    if truncated:
        recorded_samples = []
    else:
        for sample in recorded_samples:
            acting_player_index = int(sample["acting_player_index"])
            perspective_result = _winner_to_value_target(state.winner, acting_player_index)
            sample["winner"] = state.winner
            sample["result_for_player"] = _winner_to_result_label(state.winner, acting_player_index)
            sample["value_target"] = perspective_result

    summary = SelfPlayGameSummary(
        schema_version=SELF_PLAY_SCHEMA_VERSION,
        game_id=game_id,
        seed=seed,
        player0_deck_id=config.player0_deck_id,
        player1_deck_id=player1_deck_id,
        winner=state.winner,
        truncated=truncated,
        turn_number=state.turn_number,
        action_count=action_count,
        decision_samples=len(recorded_samples),
    )
    return summary, recorded_samples


def _resolve_action_choice(
    state,
    *,
    player_index: int,
    legal_actions: list[dict[str, Any]],
    planner: StandardTurnPlanner,
) -> dict[str, Any]:
    if len(legal_actions) == 1:
        return {
            "action": legal_actions[0],
            "diagnostics": {
                "planner": "forced_action",
                "top_candidates": [
                    {
                        "action_id": action_id_for(legal_actions[0]),
                        "score": None,
                        "delta": None,
                        "line": [action_id_for(legal_actions[0])],
                    }
                ],
            },
        }
    decision = planner.plan(
        state,
        acting_player_index=player_index,
        legal_actions=legal_actions,
    )
    return {
        "action": decision["chosen_action"],
        "diagnostics": decision["diagnostics"],
    }


def _should_record_decision(
    *,
    legal_actions: list[dict[str, Any]],
    setup_phase: Any,
    config: SelfPlayConfig,
) -> bool:
    if not config.collect_training_records:
        return False
    if not config.record_forced_actions and len(legal_actions) <= 1:
        return False
    if not config.include_setup_decisions and setup_phase is not None:
        return False
    return True


def _build_player_planners(
    *,
    planner_config: PlannerConfig,
    default_oracle: PolicyValueOracle,
    oracle_by_player: dict[int, PolicyValueOracle] | None,
) -> dict[int, StandardTurnPlanner]:
    planners: dict[int, StandardTurnPlanner] = {}
    oracle_by_player = oracle_by_player or {}
    for player_index in (0, 1):
        planners[player_index] = StandardTurnPlanner(
            config=planner_config,
            oracle=oracle_by_player.get(player_index, default_oracle),
        )
    return planners


def _build_decision_record(
    *,
    game_id: str,
    seed: int,
    state,
    player_index: int,
    player0_deck_id: str,
    player1_deck_id: str,
    action_count: int,
    legal_actions: list[dict[str, Any]],
    chosen_action: dict[str, Any],
    planner_diagnostics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SELF_PLAY_SCHEMA_VERSION,
        "game_id": game_id,
        "seed": seed,
        "step_index": action_count,
        "turn_number": state.turn_number,
        "acting_player_index": player_index,
        "player0_deck_id": player0_deck_id,
        "player1_deck_id": player1_deck_id,
        "setup_phase": state.setup_phase,
        "belief_state": serialize_knowledge_state(state, perspective_player_index=player_index),
        "legal_actions": serialize_knowledge_actions(
            state,
            acting_player_index=player_index,
            legal_actions=legal_actions,
        ),
        "chosen_action_id": action_id_for(chosen_action),
        "chosen_action_type": str(chosen_action.get("type", "")),
        "planner_diagnostics": planner_diagnostics,
    }


def _winner_to_value_target(winner: int | None, acting_player_index: int) -> float:
    if winner is None:
        return 0.0
    return 100.0 if winner == acting_player_index else -100.0


def _winner_to_result_label(winner: int | None, acting_player_index: int) -> str:
    if winner is None:
        return "draw"
    return "win" if winner == acting_player_index else "loss"
