from __future__ import annotations

from dataclasses import dataclass, field
from math import exp
from typing import Any

from ..cards import paired_deck_id_for
from ..engine import action_id_for, apply_action_for_player, create_game, list_legal_actions
from .knowledge_state import serialize_knowledge_actions, serialize_knowledge_state
from .oracle import HeuristicPolicyValueOracle, PolicyValueOracle
from .planner import PlannerConfig, StandardTurnPlanner

SELF_PLAY_SCHEMA_VERSION = 3
POLICY_TARGET_SCORE_TEMPERATURE = 6.0
TERMINAL_OUTCOME_WEIGHT = 0.65
PRIZE_PROGRESS_WEIGHT = 0.35
PRIZE_REWARD_PER_CARD = 100.0 / 6.0
VALUE_REWARD_DISCOUNT = 0.99


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
            record = _build_decision_record(
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
            recorded_samples.append(record)
        else:
            record = None
        apply_action_for_player(state, opening_result["action"], 1)
        if record is not None:
            record["transition_summary"] = _build_transition_summary(state, player_index=1, pre_action_record=record)
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
            record = _build_decision_record(
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
            recorded_samples.append(record)
        else:
            record = None
        apply_action_for_player(state, decision["action"], player_index)
        if record is not None:
            record["transition_summary"] = _build_transition_summary(
                state,
                player_index=player_index,
                pre_action_record=record,
            )
        action_count += 1

    truncated = state.winner is None
    if truncated:
        recorded_samples = []
    else:
        for sample_index, sample in enumerate(recorded_samples):
            acting_player_index = int(sample["acting_player_index"])
            terminal_outcome_target = _winner_to_value_target(state.winner, acting_player_index)
            discounted_prize_progress_target = _discounted_prize_progress_target(
                recorded_samples,
                start_index=sample_index,
                perspective_player_index=acting_player_index,
            )
            perspective_result = round(
                terminal_outcome_target * TERMINAL_OUTCOME_WEIGHT
                + discounted_prize_progress_target * PRIZE_PROGRESS_WEIGHT,
                6,
            )
            sample["winner"] = state.winner
            sample["result_for_player"] = _winner_to_result_label(state.winner, acting_player_index)
            sample["terminal_outcome_target"] = terminal_outcome_target
            sample["discounted_prize_progress_target"] = round(discounted_prize_progress_target, 6)
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
                "policy_target_scores": [
                    {
                        "action_id": action_id_for(legal_actions[0]),
                        "score": 0.0,
                        "source": "forced",
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
    policy_target_probs = _build_policy_target_probs(
        legal_actions=legal_actions,
        chosen_action=chosen_action,
        planner_diagnostics=planner_diagnostics,
    )
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
        "policy_target_probs": policy_target_probs,
        "player_prize_count_before": _player_prize_count_from_state(state, player_index),
        "opponent_prize_count_before": _player_prize_count_from_state(state, 1 - player_index),
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


def _build_policy_target_probs(
    *,
    legal_actions: list[dict[str, Any]],
    chosen_action: dict[str, Any],
    planner_diagnostics: dict[str, Any],
) -> dict[str, float]:
    action_ids = [action_id_for(action) for action in legal_actions]
    score_entries = planner_diagnostics.get("policy_target_scores", [])
    if not isinstance(score_entries, list):
        return _one_hot_policy_target(action_ids, action_id_for(chosen_action))

    score_by_action_id: dict[str, float] = {}
    for entry in score_entries:
        if not isinstance(entry, dict):
            continue
        action_id = entry.get("action_id")
        score = entry.get("score")
        if not isinstance(action_id, str) or not isinstance(score, (int, float)):
            continue
        if action_id not in action_ids:
            continue
        score_by_action_id[action_id] = float(score)

    if len(score_by_action_id) != len(action_ids):
        return _one_hot_policy_target(action_ids, action_id_for(chosen_action))

    max_score = max(score_by_action_id.values(), default=0.0)
    weights = {
        action_id: exp((score_by_action_id[action_id] - max_score) / POLICY_TARGET_SCORE_TEMPERATURE)
        for action_id in action_ids
    }
    total = sum(weights.values())
    if total <= 0:
        return _one_hot_policy_target(action_ids, action_id_for(chosen_action))
    probabilities: dict[str, float] = {}
    running_total = 0.0
    for action_id in action_ids[:-1]:
        probability = round(weights[action_id] / total, 6)
        probabilities[action_id] = probability
        running_total += probability
    final_action_id = action_ids[-1]
    probabilities[final_action_id] = round(max(0.0, 1.0 - running_total), 6)
    return probabilities


def _one_hot_policy_target(action_ids: list[str], chosen_action_id: str) -> dict[str, float]:
    if not action_ids:
        return {}
    return {
        action_id: 1.0 if action_id == chosen_action_id else 0.0
        for action_id in action_ids
    }


def _build_transition_summary(
    state,
    *,
    player_index: int,
    pre_action_record: dict[str, Any],
) -> dict[str, Any]:
    player_prize_count_before = int(pre_action_record.get("player_prize_count_before", 6))
    opponent_prize_count_before = int(pre_action_record.get("opponent_prize_count_before", 6))
    player_prize_count_after = _player_prize_count_from_state(state, player_index)
    opponent_prize_count_after = _player_prize_count_from_state(state, 1 - player_index)
    prizes_taken_by_actor = max(0, opponent_prize_count_before - opponent_prize_count_after)
    prizes_lost_by_actor = max(0, player_prize_count_before - player_prize_count_after)
    return {
        "player_prize_count_before": player_prize_count_before,
        "player_prize_count_after": player_prize_count_after,
        "opponent_prize_count_before": opponent_prize_count_before,
        "opponent_prize_count_after": opponent_prize_count_after,
        "prizes_taken_by_actor": prizes_taken_by_actor,
        "prizes_lost_by_actor": prizes_lost_by_actor,
    }


def _discounted_prize_progress_target(
    recorded_samples: list[dict[str, Any]],
    *,
    start_index: int,
    perspective_player_index: int,
) -> float:
    total = 0.0
    for offset, sample in enumerate(recorded_samples[start_index:]):
        transition = sample.get("transition_summary")
        if not isinstance(transition, dict):
            continue
        acting_player_index = int(sample.get("acting_player_index", perspective_player_index))
        sign = 1.0 if acting_player_index == perspective_player_index else -1.0
        prizes_taken = int(transition.get("prizes_taken_by_actor", 0) or 0)
        prizes_lost = int(transition.get("prizes_lost_by_actor", 0) or 0)
        step_reward = (prizes_taken - prizes_lost) * PRIZE_REWARD_PER_CARD * sign
        total += step_reward * (VALUE_REWARD_DISCOUNT**offset)
    return total


def _player_prize_count_from_state(state, player_index: int) -> int:
    return len(state.players[player_index].prizes)
