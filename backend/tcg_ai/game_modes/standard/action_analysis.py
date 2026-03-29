from __future__ import annotations

from copy import deepcopy
from typing import Any

from .action_metadata import build_action_metadata
from .engine import action_id_for, apply_action_for_player, get_top_card_definition, list_legal_actions
from .models import GameState, PokemonInPlay

INTENT_TAGS: tuple[str, ...] = (
    "finish_game",
    "take_prize",
    "pivot_ready_attacker",
    "save_endangered_active",
    "preserve_board_investment",
    "power_attacker",
    "develop_board",
    "recover_resource",
    "refresh_hand",
    "hand_thinning",
)

QUALITY_FLAGS: tuple[str, ...] = (
    "dominated_optional_play",
    "wastes_item",
    "wastes_supporter",
    "low_value_retreat",
    "misses_immediate_prize",
    "misses_immediate_win",
)

REASON_PRIORITY: tuple[str, ...] = (
    "finish_game",
    "take_prize",
    "pivot_ready_attacker",
    "save_endangered_active",
    "recover_resource",
    "refresh_hand",
    "hand_thinning",
    "develop_board",
)

AI_REASON_LOG_PREFIX = "AI Reason: "
_FOLLOWUP_TYPES: tuple[str, ...] = (
    "attack",
    "play_energy",
    "evolve",
    "retreat",
    "bench_basic",
    "play_basic_to_active",
    "play_item",
    "play_supporter",
)

_PLANNER_SCORES = {
    "wins_game_now_bonus": 600.0,
    "takes_prize_now_bonus": 180.0,
    "creates_same_turn_prize_line_bonus": 140.0,
    "pivot_ready_attacker_bonus": 45.0,
    "power_attacker_bonus": 50.0,
    "save_endangered_active_bonus": 55.0,
    "misses_immediate_win_penalty": -520.0,
    "misses_immediate_prize_penalty": -190.0,
    "dominated_optional_play_penalty": -70.0,
    "low_value_retreat_penalty": -95.0,
    "wastes_item_penalty": -55.0,
    "wastes_supporter_penalty": -80.0,
    "hand_thinning_bonus": 8.0,
}
_HAND_THINNING_JUSTIFICATION_SCORE = 14.0


def analyze_legal_actions(
    state: GameState,
    *,
    acting_player_index: int,
    legal_actions: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    if not legal_actions:
        return {}

    opponent_index = 1 - acting_player_index
    before_active = state.players[acting_player_index].active
    before_opponent_active = state.players[opponent_index].active
    immediate_win_available = False
    immediate_prize_available = False

    raw_entries: dict[str, dict[str, Any]] = {}
    grouped_action_ids: dict[tuple[Any, ...], list[str]] = {}

    for action in legal_actions:
        action_id = _safe_action_id(action)
        metadata = build_action_metadata(state, acting_player_index, action)
        simulated_state = deepcopy(state)
        apply_action_for_player(simulated_state, action, acting_player_index)
        prizes_taken_now = _prizes_taken_now(
            state,
            simulated_state,
            acting_player_index=acting_player_index,
        )
        wins_game_now = simulated_state.winner == acting_player_index
        takes_prize_now = prizes_taken_now > 0
        continuation = _best_same_turn_outcomes(
            simulated_state,
            acting_player_index=acting_player_index,
            depth=2,
        )
        changes_active = bool(metadata["expected_state_delta"].get("active_changes"))
        reduces_active_ko_risk = _reduces_active_ko_risk(
            before_state=state,
            before_active=before_active,
            before_opponent_active=before_opponent_active,
            after_state=simulated_state,
            after_active=simulated_state.players[acting_player_index].active,
            after_opponent_active=simulated_state.players[opponent_index].active,
        )
        saves_board_investment = (
            reduces_active_ko_risk
            and _pokemon_board_investment(state, before_active)
            >= _pokemon_board_investment(simulated_state, simulated_state.players[acting_player_index].active)
        )
        resolution_facts = {
            "optional_choice_empty": _is_optional_choice_empty(action, metadata),
            "productive_variant_exists": False,
            "net_known_hand_delta": int(metadata["expected_state_delta"].get("hand_count_delta_known", 0) or 0),
            "net_known_bench_delta": int(metadata["expected_state_delta"].get("bench_count_delta", 0) or 0),
            "net_known_discard_delta": int(metadata["expected_state_delta"].get("discard_count_delta_known", 0) or 0),
        }
        tactical_outcomes = {
            "wins_game_now": wins_game_now,
            "takes_prize_now": takes_prize_now,
            "prizes_taken_now": prizes_taken_now,
            "creates_same_turn_prize_line": continuation["takes_prize"],
            "creates_live_attack_this_turn": continuation["can_attack"],
            "changes_active": changes_active,
            "saves_board_investment": saves_board_investment,
            "reduces_active_ko_risk": reduces_active_ko_risk,
        }
        base_score = _immediate_usefulness_score(
            action=action,
            tactical_outcomes=tactical_outcomes,
            resolution_facts=resolution_facts,
            metadata=metadata,
        )
        raw_entries[action_id] = {
            "action": action,
            "action_id": action_id,
            "metadata": metadata,
            "simulated_state": simulated_state,
            "tactical_outcomes": tactical_outcomes,
            "resolution_facts": resolution_facts,
            "base_score": base_score,
        }
        grouped_action_ids.setdefault(_variant_group_key(action), []).append(action_id)
        immediate_win_available = immediate_win_available or wins_game_now
        immediate_prize_available = immediate_prize_available or takes_prize_now

    analyses: dict[str, dict[str, Any]] = {}
    for action_id, entry in raw_entries.items():
        action = entry["action"]
        metadata = entry["metadata"]
        tactical_outcomes = dict(entry["tactical_outcomes"])
        resolution_facts = dict(entry["resolution_facts"])
        same_group_action_ids = grouped_action_ids.get(_variant_group_key(action), [])
        best_variant_score = max(
            (
                float(raw_entries[candidate_id]["base_score"])
                for candidate_id in same_group_action_ids
                if candidate_id != action_id
            ),
            default=float(entry["base_score"]),
        )
        best_variant_score_gap = max(0.0, round(best_variant_score - float(entry["base_score"]), 6))
        better_variant_ids = [
            candidate_id
            for candidate_id in same_group_action_ids
            if candidate_id != action_id and raw_entries[candidate_id]["base_score"] > entry["base_score"] + 1.0
        ]
        resolution_facts["productive_variant_exists"] = bool(better_variant_ids)

        quality_flags: set[str] = set()
        misses_immediate_win = immediate_win_available and not tactical_outcomes["wins_game_now"]
        misses_immediate_prize = (
            immediate_prize_available
            and not tactical_outcomes["wins_game_now"]
            and not tactical_outcomes["takes_prize_now"]
            and not tactical_outcomes["creates_same_turn_prize_line"]
        )
        if misses_immediate_win:
            quality_flags.add("misses_immediate_win")
        if misses_immediate_prize:
            quality_flags.add("misses_immediate_prize")

        hand_thinning_value = _hand_thinning_value(
            action=action,
            metadata=metadata,
            tactical_outcomes=tactical_outcomes,
            resolution_facts=resolution_facts,
            best_variant_score_gap=best_variant_score_gap,
            misses_immediate_win=misses_immediate_win,
            misses_immediate_prize=misses_immediate_prize,
        )
        hand_thinning = _qualifies_hand_thinning(
            action=action,
            metadata=metadata,
            tactical_outcomes=tactical_outcomes,
            resolution_facts=resolution_facts,
            hand_thinning_value=hand_thinning_value,
            misses_immediate_win=misses_immediate_win,
            misses_immediate_prize=misses_immediate_prize,
        )
        if _is_dominated_optional_play(
            resolution_facts=resolution_facts,
            hand_thinning_value=hand_thinning_value,
            best_variant_score_gap=best_variant_score_gap,
        ):
            quality_flags.add("dominated_optional_play")
        if _is_low_value_retreat(
            action=action,
            tactical_outcomes=tactical_outcomes,
        ):
            quality_flags.add("low_value_retreat")
        if _is_wasted_play(
            action=action,
            metadata=metadata,
            tactical_outcomes=tactical_outcomes,
            hand_thinning=hand_thinning,
        ):
            quality_flags.add(
                "wastes_supporter" if action.get("type") == "play_supporter" else "wastes_item"
            )

        intent_tags = _build_intent_tags(
            action=action,
            metadata=metadata,
            tactical_outcomes=tactical_outcomes,
            hand_thinning=hand_thinning,
        )
        reason_tags = [tag for tag in REASON_PRIORITY if tag in intent_tags]
        reason_summary = _reason_summary_for_action(
            action=action,
            reason_tags=reason_tags,
            tactical_outcomes=tactical_outcomes,
        )
        dominance_context = {
            "immediate_win_available": immediate_win_available,
            "immediate_prize_available": immediate_prize_available,
            "better_variant_action_ids": better_variant_ids,
            "productive_variant_exists": resolution_facts["productive_variant_exists"],
            "best_variant_score_gap": best_variant_score_gap,
        }
        penalty_breakdown = _planner_penalty_breakdown(
            tactical_outcomes=tactical_outcomes,
            intent_tags=intent_tags,
            quality_flags=quality_flags,
        )
        analyses[action_id] = {
            "tactical_outcomes": tactical_outcomes,
            "resolution_facts": resolution_facts,
            "intent_tags": intent_tags,
            "quality_flags": [flag for flag in QUALITY_FLAGS if flag in quality_flags],
            "reason_tags": reason_tags,
            "reason_summary": reason_summary,
            "dominance_context": dominance_context,
            "penalty_breakdown": penalty_breakdown,
            "planner_adjustment": round(sum(float(value) for value in penalty_breakdown.values()), 6),
        }
    return analyses


def planner_adjustment_for_analysis(analysis: dict[str, Any]) -> tuple[float, dict[str, float]]:
    penalty_breakdown = analysis.get("penalty_breakdown")
    if not isinstance(penalty_breakdown, dict):
        return 0.0, {}
    normalized = {
        str(key): round(float(value), 6)
        for key, value in penalty_breakdown.items()
        if isinstance(value, (int, float))
    }
    return round(sum(normalized.values()), 6), normalized


def ai_reason_log_line(actor_name: str, reason_summary: str | None) -> str | None:
    if not reason_summary:
        return None
    return f"{AI_REASON_LOG_PREFIX}{actor_name} {reason_summary}"


def _prizes_taken_now(
    before_state: GameState,
    after_state: GameState,
    *,
    acting_player_index: int,
) -> int:
    opponent_index = 1 - acting_player_index
    before_prizes = len(before_state.players[opponent_index].prizes)
    after_prizes = len(after_state.players[opponent_index].prizes)
    return max(0, before_prizes - after_prizes)


def _best_same_turn_outcomes(
    state: GameState,
    *,
    acting_player_index: int,
    depth: int,
) -> dict[str, bool]:
    can_attack_now = False
    takes_prize = False
    wins_game = state.winner == acting_player_index
    if wins_game:
        return {"can_attack": False, "takes_prize": True, "wins_game": True}
    if depth <= 0 or state.current_player != acting_player_index:
        return {"can_attack": False, "takes_prize": False, "wins_game": False}

    legal_actions = list_legal_actions(state, player_index=acting_player_index)
    if not legal_actions:
        return {"can_attack": False, "takes_prize": False, "wins_game": False}

    ordered_actions = sorted(
        [
            action
            for action in legal_actions
            if str(action.get("type", "")) in _FOLLOWUP_TYPES
        ],
        key=lambda action: (-_followup_priority(action), _safe_action_id(action)),
    )[:8]
    for action in ordered_actions:
        simulated_state = deepcopy(state)
        apply_action_for_player(simulated_state, action, acting_player_index)
        if action.get("type") == "attack":
            can_attack_now = True
            if _prizes_taken_now(state, simulated_state, acting_player_index=acting_player_index) > 0:
                takes_prize = True
            if simulated_state.winner == acting_player_index:
                wins_game = True
        child_outcomes = _best_same_turn_outcomes(
            simulated_state,
            acting_player_index=acting_player_index,
            depth=depth - 1,
        )
        can_attack_now = can_attack_now or child_outcomes["can_attack"]
        takes_prize = takes_prize or child_outcomes["takes_prize"]
        wins_game = wins_game or child_outcomes["wins_game"]
        if can_attack_now and takes_prize and wins_game:
            break
    return {
        "can_attack": can_attack_now,
        "takes_prize": takes_prize,
        "wins_game": wins_game,
    }


def _reduces_active_ko_risk(
    *,
    before_state: GameState,
    before_active: PokemonInPlay | None,
    before_opponent_active: PokemonInPlay | None,
    after_state: GameState,
    after_active: PokemonInPlay | None,
    after_opponent_active: PokemonInPlay | None,
) -> bool:
    before_risk = _likely_knockout_next_turn(
        state=before_state,
        defender=before_active,
        attacker=before_opponent_active,
    )
    after_risk = _likely_knockout_next_turn(
        state=after_state,
        defender=after_active,
        attacker=after_opponent_active,
    )
    return before_risk and not after_risk


def _likely_knockout_next_turn(
    *,
    state: GameState,
    defender: PokemonInPlay | None,
    attacker: PokemonInPlay | None,
) -> bool:
    defender_hp = _remaining_hp(state, defender)
    if defender_hp <= 0:
        return False
    if _turns_until_ready(state, attacker) not in {0, 1}:
        return False
    return _max_attack_damage(state, attacker, max_remaining_cost=1) >= defender_hp


def _immediate_usefulness_score(
    *,
    action: dict[str, Any],
    tactical_outcomes: dict[str, Any],
    resolution_facts: dict[str, Any],
    metadata: dict[str, Any],
) -> float:
    resource_costs = metadata.get("resource_costs", {})
    expected_state_delta = metadata.get("expected_state_delta", {})
    score = 0.0
    if tactical_outcomes["wins_game_now"]:
        score += 1_000.0
    score += float(tactical_outcomes["prizes_taken_now"]) * 160.0
    if tactical_outcomes["creates_same_turn_prize_line"]:
        score += 130.0
    if tactical_outcomes["creates_live_attack_this_turn"]:
        score += 50.0
    if tactical_outcomes["reduces_active_ko_risk"]:
        score += 45.0
    if tactical_outcomes["saves_board_investment"]:
        score += 35.0
    score += float(resource_costs.get("recover_from_discard_count", 0) or 0) * 16.0
    score += float(resource_costs.get("search_selection_count", 0) or 0) * 14.0
    score += float(expected_state_delta.get("cards_drawn_known", 0) or 0) * 10.0
    score += max(0.0, float(resolution_facts.get("net_known_bench_delta", 0) or 0)) * 8.0
    score += max(0.0, -float(resolution_facts.get("net_known_hand_delta", 0) or 0)) * 3.0
    if resolution_facts["optional_choice_empty"]:
        score -= 4.0
    if action.get("type") == "retreat":
        score += 4.0 if tactical_outcomes["changes_active"] else 0.0
    return round(score, 6)


def _is_optional_choice_empty(action: dict[str, Any], metadata: dict[str, Any]) -> bool:
    resource_costs = metadata.get("resource_costs", {})
    if int(resource_costs.get("recover_from_discard_count", 0) or 0) == 0 and "recover_from_discard_ids" in action:
        return True
    if int(resource_costs.get("search_selection_count", 0) or 0) == 0 and "search_deck_ids" in action:
        return True
    return False


def _qualifies_hand_thinning(
    *,
    action: dict[str, Any],
    metadata: dict[str, Any],
    tactical_outcomes: dict[str, Any],
    resolution_facts: dict[str, Any],
    hand_thinning_value: float,
    misses_immediate_win: bool,
    misses_immediate_prize: bool,
) -> bool:
    if not resolution_facts["optional_choice_empty"]:
        return False
    if misses_immediate_win or misses_immediate_prize:
        return False
    if action.get("type") == "play_supporter":
        return False
    if action.get("type") not in {"play_item", "play_energy"}:
        return False
    if action.get("type") == "play_energy":
        return False
    if tactical_outcomes["wins_game_now"] or tactical_outcomes["takes_prize_now"]:
        return False
    if tactical_outcomes["creates_live_attack_this_turn"] or tactical_outcomes["creates_same_turn_prize_line"]:
        return False
    if int(resolution_facts.get("net_known_hand_delta", 0) or 0) >= 0:
        return False
    return hand_thinning_value > 0.0


def _hand_thinning_value(
    *,
    action: dict[str, Any],
    metadata: dict[str, Any],
    tactical_outcomes: dict[str, Any],
    resolution_facts: dict[str, Any],
    best_variant_score_gap: float,
    misses_immediate_win: bool,
    misses_immediate_prize: bool,
) -> float:
    if not resolution_facts.get("optional_choice_empty"):
        return 0.0
    if misses_immediate_win or misses_immediate_prize:
        return 0.0
    if tactical_outcomes["wins_game_now"] or tactical_outcomes["takes_prize_now"]:
        return 0.0
    if tactical_outcomes["creates_live_attack_this_turn"] or tactical_outcomes["creates_same_turn_prize_line"]:
        return 0.0
    if action.get("type") != "play_item":
        return 0.0

    expected_state_delta = metadata.get("expected_state_delta", {})
    hand_delta = int(resolution_facts.get("net_known_hand_delta", 0) or 0)
    discard_delta = int(resolution_facts.get("net_known_discard_delta", 0) or 0)
    cards_drawn_known = int(expected_state_delta.get("cards_drawn_known", 0) or 0)
    if hand_delta >= 0 or cards_drawn_known > 0:
        return 0.0

    thinning_value = 10.0
    if discard_delta > 0:
        thinning_value += 2.0
    if resolution_facts.get("productive_variant_exists") and best_variant_score_gap > _HAND_THINNING_JUSTIFICATION_SCORE:
        thinning_value = 0.0
    return round(thinning_value, 6)


def _is_dominated_optional_play(
    *,
    resolution_facts: dict[str, Any],
    hand_thinning_value: float,
    best_variant_score_gap: float,
) -> bool:
    return bool(
        resolution_facts.get("optional_choice_empty")
        and resolution_facts.get("productive_variant_exists")
        and best_variant_score_gap > max(1.0, hand_thinning_value)
    )


def _is_low_value_retreat(
    *,
    action: dict[str, Any],
    tactical_outcomes: dict[str, Any],
) -> bool:
    if action.get("type") != "retreat":
        return False
    if tactical_outcomes["wins_game_now"] or tactical_outcomes["takes_prize_now"]:
        return False
    if tactical_outcomes["creates_same_turn_prize_line"] or tactical_outcomes["creates_live_attack_this_turn"]:
        return False
    if tactical_outcomes["reduces_active_ko_risk"] or tactical_outcomes["saves_board_investment"]:
        return False
    return True


def _is_wasted_play(
    *,
    action: dict[str, Any],
    metadata: dict[str, Any],
    tactical_outcomes: dict[str, Any],
    hand_thinning: bool,
) -> bool:
    action_type = str(action.get("type", ""))
    if action_type not in {"play_item", "play_supporter"}:
        return False
    if hand_thinning:
        return False
    if tactical_outcomes["wins_game_now"] or tactical_outcomes["takes_prize_now"]:
        return False
    if tactical_outcomes["creates_same_turn_prize_line"] or tactical_outcomes["creates_live_attack_this_turn"]:
        return False
    resource_costs = metadata.get("resource_costs", {})
    expected_state_delta = metadata.get("expected_state_delta", {})
    gains_anything = any(
        (
            int(resource_costs.get("recover_from_discard_count", 0) or 0) > 0,
            int(resource_costs.get("search_selection_count", 0) or 0) > 0,
            int(expected_state_delta.get("cards_drawn_known", 0) or 0) > 0,
            int(expected_state_delta.get("bench_count_delta", 0) or 0) > 0,
            bool(expected_state_delta.get("active_changes")),
        )
    )
    return not gains_anything


def _build_intent_tags(
    *,
    action: dict[str, Any],
    metadata: dict[str, Any],
    tactical_outcomes: dict[str, Any],
    hand_thinning: bool,
) -> list[str]:
    intent_tags: list[str] = []
    if tactical_outcomes["wins_game_now"]:
        intent_tags.append("finish_game")
    if tactical_outcomes["takes_prize_now"] or tactical_outcomes["creates_same_turn_prize_line"]:
        intent_tags.append("take_prize")
    if action.get("type") == "retreat" and (
        tactical_outcomes["creates_live_attack_this_turn"] or tactical_outcomes["creates_same_turn_prize_line"]
    ):
        intent_tags.append("pivot_ready_attacker")
    if tactical_outcomes["reduces_active_ko_risk"]:
        intent_tags.append("save_endangered_active")
    if tactical_outcomes["saves_board_investment"]:
        intent_tags.append("preserve_board_investment")
    if action.get("type") in {"play_energy", "evolve"} and tactical_outcomes["creates_live_attack_this_turn"]:
        intent_tags.append("power_attacker")
    if int(metadata["expected_state_delta"].get("bench_count_delta", 0) or 0) > 0 or action.get("type") in {"bench_basic", "play_basic_to_active", "evolve"}:
        intent_tags.append("develop_board")
    if int(metadata["resource_costs"].get("recover_from_discard_count", 0) or 0) > 0 or int(metadata["resource_costs"].get("search_selection_count", 0) or 0) > 0:
        intent_tags.append("recover_resource")
    if int(metadata["expected_state_delta"].get("cards_drawn_known", 0) or 0) > 0:
        intent_tags.append("refresh_hand")
    if hand_thinning:
        intent_tags.append("hand_thinning")
    return [tag for tag in INTENT_TAGS if tag in intent_tags]


def _reason_summary_for_action(
    *,
    action: dict[str, Any],
    reason_tags: list[str],
    tactical_outcomes: dict[str, Any],
) -> str | None:
    primary_reason = next((tag for tag in REASON_PRIORITY if tag in reason_tags), None)
    if primary_reason is None:
        return None
    if primary_reason == "finish_game":
        return "acted to close out the game."
    if primary_reason == "take_prize":
        if tactical_outcomes["takes_prize_now"]:
            return "acted to take a prize immediately."
        return "set up a same-turn prize line."
    if primary_reason == "pivot_ready_attacker":
        return "pivoted into a ready attacker."
    if primary_reason == "save_endangered_active":
        return "moved the threatened Active Pokemon out of danger."
    if primary_reason == "recover_resource":
        return "recovered resources for the next attack."
    if primary_reason == "refresh_hand":
        return "refreshed the hand for a stronger follow-up."
    if primary_reason == "hand_thinning":
        return "thinned a low-value card out of hand."
    if primary_reason == "develop_board":
        if action.get("type") == "evolve":
            return "developed the board with an evolution."
        return "developed the board for future turns."
    return None


def _planner_penalty_breakdown(
    *,
    tactical_outcomes: dict[str, Any],
    intent_tags: list[str],
    quality_flags: set[str],
) -> dict[str, float]:
    breakdown: dict[str, float] = {}
    if tactical_outcomes["wins_game_now"]:
        breakdown["wins_game_now_bonus"] = _PLANNER_SCORES["wins_game_now_bonus"]
    if tactical_outcomes["takes_prize_now"]:
        breakdown["takes_prize_now_bonus"] = (
            _PLANNER_SCORES["takes_prize_now_bonus"] * max(1.0, float(tactical_outcomes["prizes_taken_now"]))
        )
    elif tactical_outcomes["creates_same_turn_prize_line"]:
        breakdown["creates_same_turn_prize_line_bonus"] = _PLANNER_SCORES["creates_same_turn_prize_line_bonus"]
    if "pivot_ready_attacker" in intent_tags:
        breakdown["pivot_ready_attacker_bonus"] = _PLANNER_SCORES["pivot_ready_attacker_bonus"]
    if "power_attacker" in intent_tags:
        breakdown["power_attacker_bonus"] = _PLANNER_SCORES["power_attacker_bonus"]
    if "save_endangered_active" in intent_tags:
        breakdown["save_endangered_active_bonus"] = _PLANNER_SCORES["save_endangered_active_bonus"]
    if "hand_thinning" in intent_tags and "dominated_optional_play" not in quality_flags:
        breakdown["hand_thinning_bonus"] = _PLANNER_SCORES["hand_thinning_bonus"]
    for quality_flag in quality_flags:
        key = f"{quality_flag}_penalty"
        score_key = f"{quality_flag}_penalty"
        if score_key in _PLANNER_SCORES:
            breakdown[key] = _PLANNER_SCORES[score_key]
    return breakdown


def _variant_group_key(action: dict[str, Any]) -> tuple[Any, ...]:
    target_zone = action.get("target_zone")
    target_bench_index = action.get("target_bench_index")
    return (
        action.get("type"),
        action.get("hand_card_id"),
        action.get("attack_index"),
        action.get("bench_index"),
        target_zone,
        target_bench_index,
        action.get("target_player_index"),
    )


def _followup_priority(action: dict[str, Any]) -> int:
    priorities = {
        "attack": 80,
        "evolve": 60,
        "play_energy": 50,
        "retreat": 45,
        "play_item": 40,
        "play_supporter": 25,
        "bench_basic": 20,
        "play_basic_to_active": 15,
    }
    return priorities.get(str(action.get("type", "")), 0)


def _safe_action_id(action: dict[str, Any]) -> str:
    try:
        return action_id_for(action)
    except Exception:
        return str(action)


def _remaining_hp(state: GameState, pokemon: PokemonInPlay | None) -> int:
    if pokemon is None:
        return 0
    definition = get_top_card_definition(state, pokemon)
    if definition is None or definition.hp is None:
        return 0
    return max(0, int(definition.hp) - int(pokemon.damage))


def _turns_until_ready(state: GameState, pokemon: PokemonInPlay | None) -> int | None:
    definition = get_top_card_definition(state, pokemon)
    if definition is None or not definition.attacks:
        return None
    attached_energy = len(pokemon.attached_energy) if pokemon is not None else 0
    remaining_costs = [
        max(0, int(attack.cost or 0) - attached_energy)
        for attack in definition.attacks
    ]
    return min(remaining_costs) if remaining_costs else None


def _max_attack_damage(state: GameState, pokemon: PokemonInPlay | None, *, max_remaining_cost: int) -> int:
    definition = get_top_card_definition(state, pokemon)
    if definition is None:
        return 0
    attached_energy = len(pokemon.attached_energy) if pokemon is not None else 0
    max_damage = 0
    for attack in definition.attacks:
        remaining_cost = max(0, int(attack.cost or 0) - attached_energy)
        if remaining_cost > max_remaining_cost:
            continue
        digits = "".join(character for character in str(attack.damage) if character.isdigit())
        max_damage = max(max_damage, int(digits or 0))
    return max_damage


def _pokemon_board_investment(state: GameState, pokemon: PokemonInPlay | None) -> int:
    definition = get_top_card_definition(state, pokemon)
    if definition is None:
        return 0
    return (
        max(0, _remaining_hp(state, pokemon))
        + len(pokemon.attached_energy) * 25
        + int(definition.prize_card_value or 0) * 40
    )
