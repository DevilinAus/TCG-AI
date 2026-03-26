from __future__ import annotations

from copy import deepcopy
from typing import Any

from .engine import apply_action, list_legal_actions
from .learning import RewardLearner, calculate_reward, extract_action_features, summarize_state
from .models import GameState


def choose_action(
    state: GameState,
    player_index: int,
    learner: RewardLearner | None = None,
    runtime: Any | None = None,
) -> dict[str, Any] | None:
    del runtime
    if state.current_player != player_index or state.winner is not None:
        return None

    legal_actions = list_legal_actions(state)
    if not legal_actions:
        return None

    baseline = summarize_state(state, player_index)
    scored_actions = []
    for action in legal_actions:
        heuristic_score = _score_action_with_simulation(state, player_index, action, baseline)
        features = extract_action_features(state, player_index, action)
        learned_score = learner.score_features(features) if learner is not None else 0.0
        scored_actions.append((heuristic_score + learned_score, heuristic_score, action))

    if learner is not None and _should_explore(state, learner):
        exploratory_actions = [entry for entry in scored_actions if entry[2]["type"] != "end_turn"]
        candidates = exploratory_actions or scored_actions
        return state.rng.choice(candidates)[2]

    return max(
        scored_actions,
        key=lambda entry: (
            round(entry[0], 6),
            round(entry[1], 6),
            -_action_priority(entry[2]["type"]),
        ),
    )[2]


def _score_action_with_simulation(
    state: GameState,
    player_index: int,
    action: dict[str, Any],
    baseline: Any,
) -> float:
    simulated_state = deepcopy(state)
    apply_action(simulated_state, action)
    reward = calculate_reward(baseline, summarize_state(simulated_state, player_index), action)

    if action["type"] == "end_turn":
        other_actions = [candidate for candidate in list_legal_actions(state) if candidate["type"] != "end_turn"]
        if other_actions:
            reward -= 2.0

    return reward


def _should_explore(state: GameState, learner: RewardLearner) -> bool:
    return state.rng.random() < learner.current_exploration_rate()


def _action_priority(action_type: str) -> int:
    priorities = {
        "promote": 0,
        "attack": 1,
        "evolve": 2,
        "play_energy": 3,
        "play_switch": 4,
        "play_potion": 5,
        "bench_basic": 6,
        "end_turn": 7,
    }
    return priorities.get(action_type, 99)
