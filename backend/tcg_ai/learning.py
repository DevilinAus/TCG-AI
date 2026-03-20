from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import threading
from typing import Any, Iterable

from .engine import card_definition, get_top_card_definition
from .models import GameState, PokemonInPlay

MAX_RECENT_EPISODES = 20


@dataclass(frozen=True)
class BoardSummary:
    prizes_taken: int
    total_remaining_hp: int
    active_remaining_hp: int
    energy_count: int
    bench_count: int
    ready_attacker_count: int
    total_attack_potential: int
    active_attack_potential: int
    best_bench_attack_potential: int


@dataclass(frozen=True)
class LearningSnapshot:
    player_index: int
    player: BoardSummary
    opponent: BoardSummary
    winner: int | None


@dataclass(frozen=True)
class EpisodeStep:
    features: tuple[str, ...]
    action_type: str
    reward: float


class RewardLearner:
    def __init__(
        self,
        learning_rate: float = 0.12,
        discount: float = 0.9,
        exploration_rate: float = 0.18,
        min_exploration_rate: float = 0.05,
    ) -> None:
        self.learning_rate = learning_rate
        self.discount = discount
        self.exploration_rate = exploration_rate
        self.min_exploration_rate = min_exploration_rate
        self._lock = threading.Lock()
        self._feature_weights: defaultdict[str, float] = defaultdict(float)
        self._action_counts: defaultdict[str, int] = defaultdict(int)
        self._action_totals: defaultdict[str, float] = defaultdict(float)
        self._recent_episode_rewards: deque[float] = deque(maxlen=MAX_RECENT_EPISODES)
        self._games_played = 0
        self._wins = 0
        self._losses = 0

    def current_exploration_rate(self) -> float:
        decay = 0.995**self._games_played
        return max(self.min_exploration_rate, self.exploration_rate * decay)

    def score_features(self, features: Iterable[str]) -> float:
        with self._lock:
            return sum(self._feature_weights.get(feature, 0.0) for feature in features)

    def record_step_reward(
        self,
        features: Iterable[str],
        action_type: str,
        reward: float,
    ) -> None:
        feature_list = tuple(features)
        with self._lock:
            self._apply_weight_update(feature_list, reward)
            self._action_counts[action_type] += 1
            self._action_totals[action_type] += reward

    def record_episode_result(
        self,
        steps: list[EpisodeStep],
        terminal_reward: float,
        winner: int | None,
        learner_player_index: int,
        total_episode_reward: float,
        skip_last_step: bool = False,
    ) -> None:
        with self._lock:
            self._games_played += 1
            if winner == learner_player_index:
                self._wins += 1
            elif winner is not None:
                self._losses += 1

            history = steps[:-1] if skip_last_step else steps
            discounted_reward = terminal_reward
            for step in reversed(history):
                self._apply_weight_update(step.features, discounted_reward)
                discounted_reward *= self.discount

            self._recent_episode_rewards.append(total_episode_reward)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            action_types = {
                feature.partition(":")[2]
                for feature in self._feature_weights
                if feature.startswith("action:")
            }
            action_types.update(self._action_counts.keys())
            action_biases = []
            for action_type in sorted(action_types):
                sample_count = self._action_counts.get(action_type, 0)
                total_reward = self._action_totals.get(action_type, 0.0)
                average_reward = total_reward / sample_count if sample_count else 0.0
                action_biases.append(
                    {
                        "action_type": action_type,
                        "bias": round(self._feature_weights.get(f"action:{action_type}", 0.0), 3),
                        "samples": sample_count,
                        "average_reward": round(average_reward, 3),
                    }
                )

            return {
                "games_played": self._games_played,
                "wins": self._wins,
                "losses": self._losses,
                "current_epsilon": round(self.current_exploration_rate(), 3),
                "learning_rate": self.learning_rate,
                "discount": self.discount,
                "tracked_feature_count": len(self._feature_weights),
                "recent_episode_rewards": [round(reward, 3) for reward in self._recent_episode_rewards],
                "action_biases": action_biases,
            }

    def export_state(self) -> dict[str, Any]:
        with self._lock:
            return {
                "learning_rate": self.learning_rate,
                "discount": self.discount,
                "exploration_rate": self.exploration_rate,
                "min_exploration_rate": self.min_exploration_rate,
                "feature_weights": dict(self._feature_weights),
                "action_counts": dict(self._action_counts),
                "action_totals": dict(self._action_totals),
                "recent_episode_rewards": list(self._recent_episode_rewards),
                "games_played": self._games_played,
                "wins": self._wins,
                "losses": self._losses,
            }

    def load_state(self, payload: dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            return

        with self._lock:
            self.learning_rate = float(payload.get("learning_rate", self.learning_rate))
            self.discount = float(payload.get("discount", self.discount))
            self.exploration_rate = float(payload.get("exploration_rate", self.exploration_rate))
            self.min_exploration_rate = float(
                payload.get("min_exploration_rate", self.min_exploration_rate)
            )
            self._feature_weights = defaultdict(
                float,
                {
                    str(feature): float(weight)
                    for feature, weight in payload.get("feature_weights", {}).items()
                },
            )
            self._action_counts = defaultdict(
                int,
                {
                    str(action_type): int(count)
                    for action_type, count in payload.get("action_counts", {}).items()
                },
            )
            self._action_totals = defaultdict(
                float,
                {
                    str(action_type): float(total)
                    for action_type, total in payload.get("action_totals", {}).items()
                },
            )
            self._recent_episode_rewards = deque(
                [float(reward) for reward in payload.get("recent_episode_rewards", [])],
                maxlen=MAX_RECENT_EPISODES,
            )
            self._games_played = int(payload.get("games_played", self._games_played))
            self._wins = int(payload.get("wins", self._wins))
            self._losses = int(payload.get("losses", self._losses))

    def _apply_weight_update(self, features: Iterable[str], reward: float) -> None:
        feature_list = tuple(features)
        if not feature_list:
            return
        delta = self.learning_rate * reward / len(feature_list)
        for feature in feature_list:
            self._feature_weights[feature] += delta


def summarize_state(state: GameState, player_index: int) -> LearningSnapshot:
    return LearningSnapshot(
        player_index=player_index,
        player=_summarize_player(state, player_index),
        opponent=_summarize_player(state, 1 - player_index),
        winner=state.winner,
    )


def calculate_reward(
    before: LearningSnapshot,
    after: LearningSnapshot,
    action: dict[str, Any],
) -> float:
    reward = 0.0
    reward += (after.player.prizes_taken - before.player.prizes_taken) * 12.0
    reward -= (after.opponent.prizes_taken - before.opponent.prizes_taken) * 12.0
    reward += (before.opponent.total_remaining_hp - after.opponent.total_remaining_hp) * 0.22
    reward -= max(0, before.player.total_remaining_hp - after.player.total_remaining_hp) * 0.2
    reward += max(0, after.player.active_remaining_hp - before.player.active_remaining_hp) * 0.12
    reward += (after.player.ready_attacker_count - before.player.ready_attacker_count) * 2.5
    reward -= (after.opponent.ready_attacker_count - before.opponent.ready_attacker_count) * 2.5
    reward += (after.player.energy_count - before.player.energy_count) * 2.0
    reward -= (after.opponent.energy_count - before.opponent.energy_count) * 2.0
    reward += (after.player.bench_count - before.player.bench_count) * 0.8
    reward -= (after.opponent.bench_count - before.opponent.bench_count) * 0.8
    reward += (after.player.active_attack_potential - before.player.active_attack_potential) * 0.18
    reward -= (after.opponent.active_attack_potential - before.opponent.active_attack_potential) * 0.18
    reward += (after.player.best_bench_attack_potential - before.player.best_bench_attack_potential) * 0.03
    reward -= (after.opponent.best_bench_attack_potential - before.opponent.best_bench_attack_potential) * 0.03

    if action["type"] == "attack":
        reward += 1.0
    elif action["type"] == "end_turn":
        reward -= 1.5

    if after.winner == after.player_index:
        reward += 30.0
    elif after.winner is not None:
        reward -= 30.0

    return reward


def extract_action_features(
    state: GameState,
    player_index: int,
    action: dict[str, Any],
) -> tuple[str, ...]:
    player = state.players[player_index]
    opponent = state.players[1 - player_index]
    action_type = action["type"]
    features = [
        f"action:{action_type}",
        f"state:energy:{min(len(player.energy_zone), 4)}",
        f"state:bench:{min(len(player.bench), 3)}",
        f"state:prize:{_prize_status(player.prize_tokens_remaining, opponent.prize_tokens_remaining)}",
    ]

    if _active_in_danger(state, player_index):
        features.append("state:active_in_danger")
    if _opponent_active_in_ko_range(state, player_index):
        features.append("state:opponent_in_ko_range")

    hand_card_id = action.get("hand_card_id")
    if hand_card_id is not None:
        card = card_definition(state, hand_card_id)
        features.append(f"card:{card.card_id}")
        features.append(f"card_kind:{card.kind}")

    if action_type == "attack":
        active_card = get_top_card_definition(state, player.active)
        if active_card is not None:
            attack = active_card.attacks[action["attack_index"]]
            features.append(f"attack:{attack.name.lower().replace(' ', '_')}")
            features.append(f"attack_cost:{attack.cost}")
            if _attack_would_knock_out(state, player_index, action["attack_index"]):
                features.append("attack:knock_out")

    if action_type == "promote":
        promoted = player.bench[action["bench_index"]]
        promoted_card = get_top_card_definition(state, promoted)
        if promoted_card is not None:
            features.append(f"card:{promoted_card.card_id}")

    target = action.get("target")
    if isinstance(target, str):
        features.append(f"target:{target.partition(':')[0]}")

    if action_type == "play_switch":
        current_attack = _pokemon_attack_potential(state, player.active, len(player.energy_zone))
        bench_attack = _pokemon_attack_potential(
            state,
            player.bench[action["bench_index"]],
            len(player.energy_zone),
        )
        if bench_attack > current_attack:
            features.append("switch:improves_active")

    return tuple(sorted(set(features)))


def _summarize_player(state: GameState, player_index: int) -> BoardSummary:
    player = state.players[player_index]
    available_energy = len(player.energy_zone)
    active_attack_potential = _pokemon_attack_potential(state, player.active, available_energy)
    bench_attack_potentials = [
        _pokemon_attack_potential(state, pokemon, available_energy) for pokemon in player.bench
    ]
    total_attack_potential = active_attack_potential + sum(bench_attack_potentials)
    total_remaining_hp = _pokemon_remaining_hp(state, player.active) + sum(
        _pokemon_remaining_hp(state, pokemon) for pokemon in player.bench
    )

    return BoardSummary(
        prizes_taken=3 - player.prize_tokens_remaining,
        total_remaining_hp=total_remaining_hp,
        active_remaining_hp=_pokemon_remaining_hp(state, player.active),
        energy_count=available_energy,
        bench_count=len(player.bench),
        ready_attacker_count=(1 if active_attack_potential > 0 else 0)
        + sum(1 for potential in bench_attack_potentials if potential > 0),
        total_attack_potential=total_attack_potential,
        active_attack_potential=active_attack_potential,
        best_bench_attack_potential=max(bench_attack_potentials, default=0),
    )


def _pokemon_remaining_hp(state: GameState, pokemon: PokemonInPlay | None) -> int:
    card = get_top_card_definition(state, pokemon)
    if card is None or card.hp is None or pokemon is None:
        return 0
    return max(0, card.hp - pokemon.damage)


def _pokemon_attack_potential(
    state: GameState,
    pokemon: PokemonInPlay | None,
    available_energy: int,
) -> int:
    card = get_top_card_definition(state, pokemon)
    if card is None:
        return 0

    best = 0
    for attack in card.attacks:
        if attack.cost > available_energy:
            continue

        expected_damage = attack.damage
        if attack.effect == "coin_flip_bonus_20":
            expected_damage += 10
        elif attack.effect == "coin_flip_bonus_30":
            expected_damage += 15
        elif attack.effect == "coin_flip_bonus_40":
            expected_damage += 20
        elif attack.effect == "coin_flip_fail":
            expected_damage //= 2
        elif attack.effect == "bonus_per_benched_matching_element_20":
            expected_damage += 20 * _count_benched_matching_element(state, pokemon, card.element)
        best = max(best, expected_damage)

    return best


def _prize_status(player_prizes_remaining: int, opponent_prizes_remaining: int) -> str:
    if player_prizes_remaining < opponent_prizes_remaining:
        return "ahead"
    if player_prizes_remaining > opponent_prizes_remaining:
        return "behind"
    return "tied"


def _active_in_danger(state: GameState, player_index: int) -> bool:
    player = state.players[player_index]
    opponent = state.players[1 - player_index]
    remaining_hp = _pokemon_remaining_hp(state, player.active)
    threat = _pokemon_attack_potential(state, opponent.active, len(opponent.energy_zone))
    return remaining_hp > 0 and threat >= remaining_hp


def _opponent_active_in_ko_range(state: GameState, player_index: int) -> bool:
    player = state.players[player_index]
    opponent = state.players[1 - player_index]
    remaining_hp = _pokemon_remaining_hp(state, opponent.active)
    pressure = _pokemon_attack_potential(state, player.active, len(player.energy_zone))
    return remaining_hp > 0 and pressure >= remaining_hp


def _attack_would_knock_out(
    state: GameState,
    player_index: int,
    attack_index: int,
) -> bool:
    player = state.players[player_index]
    opponent = state.players[1 - player_index]
    active_card = get_top_card_definition(state, player.active)
    defending_card = get_top_card_definition(state, opponent.active)
    if active_card is None or defending_card is None or opponent.active is None:
        return False

    attack = active_card.attacks[attack_index]
    damage = attack.damage
    if attack.effect == "coin_flip_bonus_20":
        damage += 10
    elif attack.effect == "coin_flip_bonus_30":
        damage += 15
    elif attack.effect == "coin_flip_bonus_40":
        damage += 20
    elif attack.effect == "coin_flip_fail":
        damage //= 2
    elif attack.effect == "bonus_per_benched_matching_element_20":
        damage += 20 * _count_benched_matching_element(state, player.active, active_card.element)

    return damage >= max(0, defending_card.hp - opponent.active.damage)


def _count_benched_matching_element(
    state: GameState,
    pokemon: PokemonInPlay | None,
    element: str | None,
) -> int:
    if pokemon is None or not element:
        return 0

    for player in state.players:
        if pokemon is player.active:
            return sum(
                1
                for benched in player.bench
                if (benched_card := get_top_card_definition(state, benched)) is not None
                and benched_card.element == element
            )
    return 0
