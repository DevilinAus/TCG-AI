from __future__ import annotations

from dataclasses import dataclass
from inspect import signature
from typing import Any, Callable

from ..engine import card_definition, create_game, list_legal_actions
from ..models import PokemonInPlay
from .oracle import PolicyValueOracle
from .planner import PlannerConfig, StandardTurnPlanner


@dataclass(frozen=True)
class TacticalScenario:
    name: str
    description: str
    tags: tuple[str, ...]
    builder: Callable[[], tuple[Any, int]]
    expectation: Callable[..., bool]
    explanation: str
    tier: str = "core"
    planner_config: PlannerConfig | None = None


@dataclass(frozen=True)
class TacticalScenarioResult:
    name: str
    description: str
    tags: tuple[str, ...]
    tier: str
    passed: bool
    chosen_action_id: str
    chosen_action_type: str
    explanation: str
    acceptable_reason: str
    diagnostics: dict[str, Any]


def core_tactical_scenarios() -> list[TacticalScenario]:
    return [
        TacticalScenario(
            name="attack_for_immediate_win",
            description="If an attack wins the game now, choose the attack over ending the turn.",
            tags=("attack", "finisher"),
            builder=_build_attack_for_immediate_win_state,
            expectation=lambda action: action.get("type") == "attack",
            explanation="The active can take the last knockout immediately.",
        ),
        TacticalScenario(
            name="attack_for_immediate_win_over_supporter",
            description="If an attack wins the game now, do not spend the turn on a draw supporter first.",
            tags=("attack", "finisher", "supporter_timing"),
            builder=_build_attack_for_immediate_win_with_supporter_state,
            expectation=lambda action: action.get("type") == "attack",
            explanation="A draw supporter is still legal, but the immediate winning attack is the uniquely clean conversion.",
        ),
        TacticalScenario(
            name="attach_to_enable_immediate_win",
            description="If attaching energy creates a forced winning attack this turn, attach first.",
            tags=("energy", "tempo", "attack_setup"),
            builder=_build_attach_to_enable_immediate_win_state,
            expectation=lambda action: action.get("type") == "play_energy",
            explanation="The active is one energy short of a winning attack.",
        ),
        TacticalScenario(
            name="retreat_into_ready_attacker",
            description="If retreating into a ready bench attacker wins the game, retreat first.",
            tags=("retreat", "pivot", "attack_setup"),
            builder=_build_retreat_into_ready_attacker_state,
            expectation=lambda action: action.get("type") == "retreat" and action.get("target_bench_index") == 0,
            explanation="The active cannot convert damage, but the bench attacker can win immediately after retreat.",
        ),
        TacticalScenario(
            name="retreat_into_ready_attacker_over_supporter",
            description="If retreating into a ready attacker wins immediately, do that before spending the turn on draw.",
            tags=("retreat", "pivot", "finisher", "supporter_timing"),
            builder=_build_retreat_into_ready_attacker_with_supporter_state,
            expectation=lambda action: action.get("type") == "retreat" and action.get("target_bench_index") == 0,
            explanation="The game can be won now by retreating; a draw supporter is legal but unnecessary.",
        ),
    ]


def strategic_tactical_scenarios() -> list[TacticalScenario]:
    return [
        TacticalScenario(
            name="call_for_family_over_supporter_draw",
            description="If an attack immediately develops two benched basics, prefer that board conversion over a pure draw supporter.",
            tags=("attack", "board_development", "supporter_timing"),
            builder=_build_call_for_family_over_supporter_draw_state,
            expectation=lambda action: action.get("type") == "attack" and action.get("attack_index") == 0,
            explanation="Call for Family creates immediate board development that should outrank a simple draw line in this state.",
            tier="strategic",
            planner_config=PlannerConfig(max_depth=3, beam_width=6, opponent_branch_width=2),
        ),
        TacticalScenario(
            name="attach_to_bench_for_retreat_win",
            description="If attaching to the bench unlocks retreat into a winning attack this turn, invest there first.",
            tags=("energy", "retreat", "attack_setup", "pivot"),
            builder=_build_attach_to_bench_for_retreat_win_state,
            expectation=lambda action: action.get("type") == "play_energy" and action.get("target_zone") == "bench" and action.get("target_bench_index") == 0,
            explanation="The active only supplies retreat; the bench attacker needs the attachment to convert the turn into an immediate knockout.",
            tier="strategic",
            planner_config=PlannerConfig(max_depth=3, beam_width=6, opponent_branch_width=2),
        ),
        TacticalScenario(
            name="attach_to_bench_from_doomed_active_for_retreat_win",
            description="If the active is about to be knocked out anyway, invest in the bench line that converts immediately.",
            tags=("energy", "retreat", "attack_setup", "pivot", "sacrifice"),
            builder=_build_attach_to_bench_from_doomed_active_for_retreat_win_state,
            expectation=lambda action: action.get("type") == "play_energy" and action.get("target_zone") == "bench" and action.get("target_bench_index") == 0,
            explanation="The active is effectively a sacrifice piece here; the only meaningful attachment is the one that powers the bench attacker for the winning retreat line.",
            tier="strategic",
            planner_config=PlannerConfig(max_depth=4, beam_width=6, opponent_branch_width=2),
        ),
        TacticalScenario(
            name="overattach_to_active_for_retreat_win",
            description="If an extra attachment on the active is only valuable because it pays retreat, still choose it when that line wins.",
            tags=("energy", "retreat", "attack_setup", "pivot", "overattach"),
            builder=_build_overattach_to_active_for_retreat_win_state,
            expectation=lambda action: action.get("type") == "play_energy" and action.get("target_zone") == "active",
            explanation="This is a deliberate over-attachment: the active already has enough energy to attack, but the extra energy is what pays retreat and unlocks the real winning attacker.",
            tier="strategic",
            planner_config=PlannerConfig(max_depth=4, beam_width=8, opponent_branch_width=2),
        ),
        TacticalScenario(
            name="nest_ball_unique_target_for_retreat_win",
            description="If only one Nest Ball target converts into a winning attach-retreat-attack line, choose that target.",
            tags=("search", "nest_ball", "retreat", "attack_setup"),
            builder=_build_nest_ball_unique_target_for_retreat_win_state,
            expectation=lambda state, action: action.get("type") == "play_item"
            and _action_search_target_names(state, action) == ["Mankey"],
            explanation="Only Mankey converts this Nest Ball into an immediate knockout sequence; the other targets still leave the damage short.",
            tier="strategic",
            planner_config=PlannerConfig(max_depth=4, beam_width=8, opponent_branch_width=2),
        ),
        TacticalScenario(
            name="ultra_ball_lucario_ex_for_immediate_conversion",
            description="If Ultra Ball can uniquely convert into Lucario ex, choose that search bundle and continue the line.",
            tags=("search", "ultra_ball", "evolution", "resource_conversion"),
            builder=_build_ultra_ball_lucario_ex_for_immediate_conversion_state,
            expectation=lambda state, action, decision: action.get("type") == "play_item"
            and _action_search_target_names(state, action) == ["Lucario ex"]
            and _action_discard_names(state, action) == ["Potion", "Switch"]
            and (
                not decision
                or (
                    any(action_id.startswith("evolve:") for action_id in decision.get("planned_action_sequence", [])[1:])
                    and any(action_id.startswith("attack:") for action_id in decision.get("planned_action_sequence", [])[1:])
                )
            ),
            explanation="Here Ultra Ball is not just a legal search: Lucario ex is the only target that immediately converts the turn into evolve-plus-attack pressure, and the discard bundle is uniquely low-value.",
            tier="strategic",
            planner_config=PlannerConfig(max_depth=4, beam_width=12, opponent_branch_width=2),
        ),
        TacticalScenario(
            name="ultra_ball_ampharos_ex_for_immediate_conversion",
            description="If Ultra Ball can uniquely convert into Ampharos ex, choose that search bundle and continue the line.",
            tags=("search", "ultra_ball", "evolution", "resource_conversion"),
            builder=_build_ultra_ball_ampharos_ex_for_immediate_conversion_state,
            expectation=lambda state, action, decision: action.get("type") == "play_item"
            and _action_search_target_names(state, action) == ["Ampharos ex"]
            and _action_discard_names(state, action) == ["Potion", "Switch"]
            and (
                not decision
                or (
                    any(action_id.startswith("evolve:") for action_id in decision.get("planned_action_sequence", [])[1:])
                    and any(action_id.startswith("attack:") for action_id in decision.get("planned_action_sequence", [])[1:])
                )
            ),
            explanation="This is the Ampharos mirror of the Lucario ex conversion test: the best Ultra Ball line is the one that immediately evolves and cashes in the turn's damage.",
            tier="strategic",
            planner_config=PlannerConfig(max_depth=4, beam_width=12, opponent_branch_width=2),
        ),
        TacticalScenario(
            name="youngster_over_low_value_ultra_ball",
            description="If the hand is functionally dead, prefer the refresh supporter over forcing a low-value Ultra Ball line.",
            tags=("supporter", "resource_conversion", "ultra_ball"),
            builder=_build_youngster_over_low_value_ultra_ball_state,
            expectation=lambda state, action, decision: action.get("type") == "play_supporter"
            and _action_hand_card_name(state, action) == "Youngster"
            and (
                not decision
                or any(
                    action_id.startswith(("bench_basic:", "play_energy:", "attack:"))
                    for action_id in decision.get("planned_action_sequence", [])[1:]
                )
            ),
            explanation="Youngster is only correct here because it turns a dead hand into real follow-up board development; forcing Ultra Ball would burn resources for a weak line.",
            tier="strategic",
            planner_config=PlannerConfig(max_depth=4, beam_width=10, opponent_branch_width=2),
        ),
    ]


def default_tactical_scenarios(suite: str = "core") -> list[TacticalScenario]:
    if suite == "core":
        return core_tactical_scenarios()
    if suite == "strategic":
        return strategic_tactical_scenarios()
    if suite == "all":
        return [*core_tactical_scenarios(), *strategic_tactical_scenarios()]
    raise ValueError(f"Unsupported tactical suite '{suite}'.")


def run_tactical_suite(
    *,
    oracle: PolicyValueOracle,
    planner_config: PlannerConfig | None = None,
    scenarios: list[TacticalScenario] | None = None,
    suite: str = "core",
) -> list[TacticalScenarioResult]:
    results: list[TacticalScenarioResult] = []
    default_config = planner_config or PlannerConfig(max_depth=2, beam_width=4, opponent_branch_width=2)
    for scenario in scenarios or default_tactical_scenarios(suite):
        planner = StandardTurnPlanner(config=scenario.planner_config or default_config, oracle=oracle)
        state, acting_player_index = scenario.builder()
        legal_actions = list_legal_actions(state, player_index=acting_player_index)
        decision = planner.plan(
            state,
            acting_player_index=acting_player_index,
            legal_actions=legal_actions,
        )
        chosen_action = decision["chosen_action"]
        results.append(
            TacticalScenarioResult(
                name=scenario.name,
                description=scenario.description,
                tags=scenario.tags,
                tier=scenario.tier,
                passed=bool(_scenario_matches(scenario, state, chosen_action, decision)),
                chosen_action_id=str(decision["chosen_action_id"]),
                chosen_action_type=str(chosen_action.get("type", "")),
                explanation=scenario.explanation,
                acceptable_reason=_acceptable_reason_for_scenario(scenario, state, legal_actions),
                diagnostics=dict(decision["diagnostics"]),
            )
        )
    return results


def _acceptable_reason_for_scenario(
    scenario: TacticalScenario,
    state,
    legal_actions: list[dict[str, Any]],
) -> str:
    matching = [action for action in legal_actions if _scenario_matches(scenario, state, action)]
    if not matching:
        return "Scenario expectation did not match any legal action."
    labels = list(dict.fromkeys(_action_reason_label(state, action) for action in matching))
    return "; ".join(labels)


def _scenario_matches(
    scenario: TacticalScenario,
    state,
    action: dict[str, Any],
    decision: dict[str, Any] | None = None,
) -> bool:
    parameter_count = len(signature(scenario.expectation).parameters)
    if parameter_count <= 1:
        return bool(scenario.expectation(action))
    if parameter_count == 2:
        return bool(scenario.expectation(state, action))
    return bool(scenario.expectation(state, action, decision or {}))


def _action_reason_label(state, action: dict[str, Any]) -> str:
    base_label = str(action.get("label") or action.get("type", ""))
    search_targets = _action_search_target_names(state, action)
    if search_targets and str(action.get("type", "")) in {"play_item", "play_supporter"}:
        return f"{base_label} -> {', '.join(search_targets)}"
    discard_ids = [instance_id for instance_id in action.get("discard_from_hand_ids", []) if isinstance(instance_id, str)]
    if discard_ids:
        discard_names = ", ".join(card_definition(state, instance_id).name for instance_id in discard_ids)
        return f"{base_label} (discard {discard_names})"
    return base_label


def _build_attack_for_immediate_win_state():
    state = create_game(seed=1, human_deck_id="ampharos-ex-battle-deck")
    _reset_to_midgame_turn(state, current_player=0)
    _set_named_active_pokemon(state, 0, "Mareep")
    _set_named_active_pokemon(state, 1, "Squawkabilly")
    state.players[0].active.attached_energy = [_take_named_card(state, 0, "Basic Lightning Energy")]
    state.players[1].active.damage = 60
    _set_exact_hand(state, 0, [])
    _set_exact_hand(state, 1, [])
    return state, 0


def _build_attach_to_enable_immediate_win_state():
    state = create_game(seed=1, human_deck_id="ampharos-ex-battle-deck")
    _reset_to_midgame_turn(state, current_player=0)
    _set_named_active_pokemon(state, 0, "Mareep")
    _set_named_active_pokemon(state, 1, "Squawkabilly")
    state.players[1].active.damage = 60
    energy_id = _move_named_card_to_hand(state, 0, "Basic Lightning Energy")
    _set_exact_hand(state, 0, [energy_id])
    _set_exact_hand(state, 1, [])
    return state, 0


def _build_attack_for_immediate_win_with_supporter_state():
    state, acting_player_index = _build_attack_for_immediate_win_state()
    nemona_id = _move_named_card_to_hand(state, 0, "Nemona")
    _set_exact_hand(state, 0, [nemona_id])
    return state, acting_player_index


def _build_retreat_into_ready_attacker_state():
    state = create_game(seed=1, human_deck_id="lucario-ex-battle-deck")
    _reset_to_midgame_turn(state, current_player=0)
    _set_named_active_pokemon(state, 0, "Lechonk")
    _set_named_bench_pokemon(state, 0, "Riolu")
    _set_named_active_pokemon(state, 1, "Mareep")
    state.players[0].active.attached_energy = [_take_named_card(state, 0, "Basic Fighting Energy")]
    state.players[0].bench[0].attached_energy = [_take_named_card(state, 0, "Basic Fighting Energy")]
    state.players[1].active.damage = 50
    _set_exact_hand(state, 0, [])
    _set_exact_hand(state, 1, [])
    return state, 0


def _build_retreat_into_ready_attacker_with_supporter_state():
    state, acting_player_index = _build_retreat_into_ready_attacker_state()
    nemona_id = _move_named_card_to_hand(state, 0, "Nemona")
    _set_exact_hand(state, 0, [nemona_id])
    return state, acting_player_index


def _build_call_for_family_over_supporter_draw_state():
    state = create_game(seed=1, human_deck_id="lucario-ex-battle-deck")
    _reset_to_midgame_turn(state, current_player=0)
    _set_named_active_pokemon(state, 0, "Squawkabilly")
    _set_named_active_pokemon(state, 1, "Mareep")
    state.players[0].active.attached_energy = [_take_named_card(state, 0, "Basic Fighting Energy")]
    nemona_id = _move_named_card_to_hand(state, 0, "Nemona")
    _set_exact_hand(state, 0, [nemona_id])
    _set_exact_hand(state, 1, [])
    return state, 0


def _build_attach_to_bench_for_retreat_win_state():
    state = create_game(seed=1, human_deck_id="lucario-ex-battle-deck")
    _reset_to_midgame_turn(state, current_player=0)
    _set_named_active_pokemon(state, 0, "Lechonk")
    _set_named_bench_pokemon(state, 0, "Riolu")
    _set_named_active_pokemon(state, 1, "Mareep")
    state.players[0].active.attached_energy = [_take_named_card(state, 0, "Basic Fighting Energy")]
    state.players[1].active.damage = 50
    energy_id = _move_named_card_to_hand(state, 0, "Basic Fighting Energy")
    _set_exact_hand(state, 0, [energy_id])
    _set_exact_hand(state, 1, [])
    return state, 0


def _build_attach_to_bench_from_doomed_active_for_retreat_win_state():
    state = create_game(seed=1, human_deck_id="lucario-ex-battle-deck")
    _reset_to_midgame_turn(state, current_player=0)
    _set_named_active_pokemon(state, 0, "Lechonk")
    _set_named_bench_pokemon(state, 0, "Riolu")
    _set_named_active_pokemon(state, 1, "Mareep")
    state.players[0].active.damage = 50
    state.players[0].active.attached_energy = [_take_named_card(state, 0, "Basic Fighting Energy")]
    state.players[1].active.damage = 50
    energy_id = _move_named_card_to_hand(state, 0, "Basic Fighting Energy")
    _set_exact_hand(state, 0, [energy_id])
    _set_exact_hand(state, 1, [])
    return state, 0


def _build_overattach_to_active_for_retreat_win_state():
    state = create_game(seed=1, human_deck_id="ampharos-ex-battle-deck")
    _reset_to_midgame_turn(state, current_player=0)
    _set_named_active_pokemon(state, 0, "Ampharos ex")
    _set_named_bench_pokemon(state, 0, "Flaaffy")
    _set_named_active_pokemon(state, 1, "Cyclizar")
    state.players[0].active.attached_energy = [_take_named_card(state, 0, "Basic Lightning Energy")]
    state.players[0].bench[0].attached_energy = [
        _take_named_card(state, 0, "Basic Lightning Energy"),
        _take_named_card(state, 0, "Basic Lightning Energy"),
        _take_named_card(state, 0, "Basic Lightning Energy"),
    ]
    state.players[1].active.damage = 30
    energy_id = _move_named_card_to_hand(state, 0, "Basic Lightning Energy")
    _set_exact_hand(state, 0, [energy_id])
    _set_exact_hand(state, 1, [])
    return state, 0


def _build_nest_ball_unique_target_for_retreat_win_state():
    state = create_game(seed=1, human_deck_id="lucario-ex-battle-deck")
    _reset_to_midgame_turn(state, current_player=0)
    _set_named_active_pokemon(state, 0, "Lechonk")
    _set_named_active_pokemon(state, 1, "Mareep")
    state.players[0].active.attached_energy = [_take_named_card(state, 0, "Basic Fighting Energy")]
    state.players[1].active.damage = 30
    nest_ball_id = _move_named_card_to_hand(state, 0, "Nest Ball")
    energy_id = _move_named_card_to_hand(state, 0, "Basic Fighting Energy")
    _set_exact_hand(state, 0, [nest_ball_id, energy_id])
    _set_exact_hand(state, 1, [])
    return state, 0


def _build_ultra_ball_lucario_ex_for_immediate_conversion_state():
    state = create_game(seed=1, human_deck_id="lucario-ex-battle-deck")
    _reset_to_midgame_turn(state, current_player=0)
    _set_named_active_pokemon(state, 0, "Riolu")
    _set_named_active_pokemon(state, 1, "Flamigo")
    state.players[0].active.attached_energy = [
        _take_named_card(state, 0, "Basic Fighting Energy"),
        _take_named_card(state, 0, "Basic Fighting Energy"),
    ]
    state.players[1].active.damage = 50
    ultra_ball_id = _move_named_card_to_hand(state, 0, "Ultra Ball")
    potion_id = _move_named_card_to_hand(state, 0, "Potion")
    switch_id = _move_named_card_to_hand(state, 0, "Switch")
    _set_exact_hand(state, 0, [ultra_ball_id, potion_id, switch_id])
    _set_exact_hand(state, 1, [])
    return state, 0


def _build_ultra_ball_ampharos_ex_for_immediate_conversion_state():
    state = create_game(seed=1, human_deck_id="ampharos-ex-battle-deck")
    _reset_to_midgame_turn(state, current_player=0)
    _set_named_active_pokemon(state, 0, "Flaaffy")
    _set_named_active_pokemon(state, 1, "Lucario ex")
    state.players[0].active.attached_energy = [
        _take_named_card(state, 0, "Basic Lightning Energy"),
        _take_named_card(state, 0, "Basic Lightning Energy"),
    ]
    state.players[1].active.damage = 130
    ultra_ball_id = _move_named_card_to_hand(state, 0, "Ultra Ball")
    potion_id = _move_named_card_to_hand(state, 0, "Potion")
    switch_id = _move_named_card_to_hand(state, 0, "Switch")
    _set_exact_hand(state, 0, [ultra_ball_id, potion_id, switch_id])
    _set_exact_hand(state, 1, [])
    return state, 0


def _build_youngster_over_low_value_ultra_ball_state():
    state = create_game(seed=1, human_deck_id="ampharos-ex-battle-deck")
    _reset_to_midgame_turn(state, current_player=0)
    _set_named_active_pokemon(state, 0, "Flaaffy")
    _set_named_active_pokemon(state, 1, "Lucario ex")
    ultra_ball_id = _move_named_card_to_hand(state, 0, "Ultra Ball")
    potion_id = _move_named_card_to_hand(state, 0, "Potion")
    youngster_id = _move_named_card_to_hand(state, 0, "Youngster")
    _set_exact_hand(state, 0, [ultra_ball_id, potion_id, youngster_id])
    _set_exact_hand(state, 1, [])
    return state, 0


def _reset_to_midgame_turn(state, *, current_player: int) -> None:
    state.setup_phase = None
    state.current_player = current_player
    state.turn_number = 2
    state.pending_promotion_for = None
    state.winner = None
    for player in state.players:
        player.turns_taken = 2
        player.supporter_played_this_turn = False
        player.energy_attached_this_turn = False
        player.retreated_this_turn = False
        player.bench = []


def _set_named_active_pokemon(state, player_index: int, card_name: str) -> None:
    instance_id = _find_instance_id(state, player_index, card_name)
    player = state.players[player_index]
    _remove_instance_from_player_zones(player, instance_id)
    player.active = PokemonInPlay(stack=[instance_id])


def _set_named_bench_pokemon(state, player_index: int, card_name: str) -> None:
    instance_id = _find_instance_id(state, player_index, card_name)
    player = state.players[player_index]
    _remove_instance_from_player_zones(player, instance_id)
    player.bench.append(PokemonInPlay(stack=[instance_id]))


def _move_named_card_to_hand(state, player_index: int, card_name: str) -> str:
    player = state.players[player_index]
    for zone_name in ("hand", "deck", "discard", "prizes"):
        zone = getattr(player, zone_name)
        for instance_id in list(zone):
            if card_definition(state, instance_id).name != card_name:
                continue
            if zone_name != "hand":
                zone.remove(instance_id)
                player.hand.append(instance_id)
            return instance_id
    raise AssertionError(f"Could not find {card_name} for player {player_index}")


def _take_named_card(state, player_index: int, card_name: str) -> str:
    player = state.players[player_index]
    for zone_name in ("hand", "deck", "discard", "prizes"):
        zone = getattr(player, zone_name)
        for instance_id in list(zone):
            if card_definition(state, instance_id).name != card_name:
                continue
            zone.remove(instance_id)
            return instance_id
    raise AssertionError(f"Could not take {card_name} for player {player_index}")


def _set_exact_hand(state, player_index: int, ordered_instance_ids: list[str]) -> None:
    player = state.players[player_index]
    kept = set(ordered_instance_ids)
    extras = [instance_id for instance_id in player.hand if instance_id not in kept]
    player.hand = list(ordered_instance_ids)
    player.deck.extend(extras)


def _find_instance_id(state, player_index: int, card_name: str) -> str:
    player = state.players[player_index]
    for zone_name in ("hand", "deck", "discard", "prizes"):
        zone = getattr(player, zone_name)
        for instance_id in zone:
            if card_definition(state, instance_id).name == card_name:
                return instance_id
    raise AssertionError(f"Could not find {card_name} for player {player_index}")


def _remove_instance_from_player_zones(player, instance_id: str) -> None:
    for zone_name in ("hand", "deck", "discard", "prizes"):
        zone = getattr(player, zone_name)
        if instance_id in zone:
            zone.remove(instance_id)
            return


def _action_search_target_names(state, action: dict[str, Any]) -> list[str]:
    return [
        card_definition(state, instance_id).name
        for instance_id in action.get("search_deck_ids", [])
        if isinstance(instance_id, str)
    ]


def _action_hand_card_name(state, action: dict[str, Any]) -> str | None:
    instance_id = action.get("hand_card_id")
    if not isinstance(instance_id, str):
        return None
    return card_definition(state, instance_id).name


def _action_discard_names(state, action: dict[str, Any]) -> list[str]:
    return sorted(
        card_definition(state, instance_id).name
        for instance_id in action.get("discard_from_hand_ids", [])
        if isinstance(instance_id, str)
    )
