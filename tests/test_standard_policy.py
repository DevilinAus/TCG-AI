from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from backend.tcg_ai.game_modes.standard.decision_payload import build_decision_request
from backend.tcg_ai.game_modes.standard.engine import card_definition, create_game, list_legal_actions
from backend.tcg_ai.game_modes.standard.policy import (
    DecisionRequest,
    FallbackStandardDecisionProvider,
    LocalStandardDecisionProvider,
    StandardPolicyConfig,
)
from backend.tcg_ai.game_modes.standard.policy_store import StandardPolicyStore


class StandardPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.policy_path = Path(self.temp_dir.name) / "standard_policy_progress.json"
        self.store = StandardPolicyStore(state_path=self.policy_path)
        self.config = StandardPolicyConfig(
            remote_enabled=False,
            remote_url=None,
            remote_timeout_ms=2_000,
            exploration_rate=0.20,
            min_exploration_rate=0.05,
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_local_provider_is_seeded_when_only_exploration_is_available(self) -> None:
        state_a = create_game(seed=1, human_deck_id="ampharos-ex-battle-deck", ai_name="Brock")
        state_b = create_game(seed=1, human_deck_id="ampharos-ex-battle-deck", ai_name="Brock")
        provider_a = LocalStandardDecisionProvider(
            trainer_id="brock",
            ai_deck_id="lucario-ex-battle-deck",
            session_id="session-a",
            policy_store=self.store,
            config=self.config,
        )
        provider_b = LocalStandardDecisionProvider(
            trainer_id="brock",
            ai_deck_id="lucario-ex-battle-deck",
            session_id="session-b",
            policy_store=self.store,
            config=self.config,
        )
        actions_a = list_legal_actions(state_a, player_index=1)
        actions_b = list_legal_actions(state_b, player_index=1)

        result_a = provider_a.choose_action(
            DecisionRequest(
                state=state_a,
                acting_player_index=1,
                decision_type="opening_active",
                decision_id="session-a:opening_active",
                legal_actions=actions_a,
                payload=build_decision_request(
                    state_a,
                    session_id="session-a",
                    decision_id="session-a:opening_active",
                    decision_type="opening_active",
                    acting_player_index=1,
                    ai_trainer_id="brock",
                    ai_deck_id="lucario-ex-battle-deck",
                    legal_actions=actions_a,
                ),
            )
        )
        result_b = provider_b.choose_action(
            DecisionRequest(
                state=state_b,
                acting_player_index=1,
                decision_type="opening_active",
                decision_id="session-b:opening_active",
                legal_actions=actions_b,
                payload=build_decision_request(
                    state_b,
                    session_id="session-b",
                    decision_id="session-b:opening_active",
                    decision_type="opening_active",
                    acting_player_index=1,
                    ai_trainer_id="brock",
                    ai_deck_id="lucario-ex-battle-deck",
                    legal_actions=actions_b,
                ),
            )
        )

        self.assertEqual(result_a.action_id, result_b.action_id)
        self.assertEqual(result_a.diagnostics["selection_mode"], "explore")

    def test_policy_store_stats_are_isolated_by_trainer_and_deck(self) -> None:
        self.store.record_opener_outcome(
            trainer_id="brock",
            ai_deck_id="lucario-ex-battle-deck",
            chosen_card_id="sv1-124",
            terminal_reward=30.0,
            did_win=True,
        )
        self.store.record_opener_outcome(
            trainer_id="misty",
            ai_deck_id="lucario-ex-battle-deck",
            chosen_card_id="sv1-124",
            terminal_reward=-30.0,
            did_win=False,
        )
        self.store.record_opener_outcome(
            trainer_id="brock",
            ai_deck_id="zapdos-ex-battle-deck",
            chosen_card_id="sv1-124",
            terminal_reward=-15.0,
            did_win=False,
        )

        brock_lucario = self.store.stats_for_deck("brock", "lucario-ex-battle-deck")
        misty_lucario = self.store.stats_for_deck("misty", "lucario-ex-battle-deck")
        brock_zapdos = self.store.stats_for_deck("brock", "zapdos-ex-battle-deck")

        self.assertEqual(brock_lucario["sv1-124"].wins, 1)
        self.assertEqual(brock_lucario["sv1-124"].resolved_samples, 1)
        self.assertEqual(misty_lucario["sv1-124"].wins, 0)
        self.assertEqual(misty_lucario["sv1-124"].total_terminal_reward, -30.0)
        self.assertEqual(brock_zapdos["sv1-124"].resolved_samples, 1)
        self.assertEqual(brock_zapdos["sv1-124"].total_terminal_reward, -15.0)

    def test_local_provider_exploits_best_recorded_opener(self) -> None:
        state = create_game(seed=1, human_deck_id="ampharos-ex-battle-deck", ai_name="Brock")
        actions = list_legal_actions(state, player_index=1)
        card_ids = [card_definition(state, action["hand_card_id"]).card_id for action in actions]
        self.assertGreaterEqual(len(card_ids), 2)
        self.store.record_opener_outcome(
            trainer_id="brock",
            ai_deck_id="lucario-ex-battle-deck",
            chosen_card_id=card_ids[0],
            terminal_reward=30.0,
            did_win=True,
        )
        self.store.record_opener_outcome(
            trainer_id="brock",
            ai_deck_id="lucario-ex-battle-deck",
            chosen_card_id=card_ids[1],
            terminal_reward=-30.0,
            did_win=False,
        )
        provider = LocalStandardDecisionProvider(
            trainer_id="brock",
            ai_deck_id="lucario-ex-battle-deck",
            session_id="session-a",
            policy_store=self.store,
            config=StandardPolicyConfig(
                remote_enabled=False,
                remote_url=None,
                remote_timeout_ms=2_000,
                exploration_rate=0.0,
                min_exploration_rate=0.0,
            ),
        )

        result = provider.choose_action(
            DecisionRequest(
                state=state,
                acting_player_index=1,
                decision_type="opening_active",
                decision_id="session-a:opening_active",
                legal_actions=actions,
                payload=build_decision_request(
                    state,
                    session_id="session-a",
                    decision_id="session-a:opening_active",
                    decision_type="opening_active",
                    acting_player_index=1,
                    ai_trainer_id="brock",
                    ai_deck_id="lucario-ex-battle-deck",
                    legal_actions=actions,
                ),
            )
        )

        self.assertEqual(result.diagnostics["selection_mode"], "exploit")
        self.assertEqual(card_definition(state, result.chosen_action["hand_card_id"]).card_id, card_ids[0])

    def test_decision_payload_exposes_public_and_private_state_without_hidden_opponent_hand(self) -> None:
        state = create_game(seed=1, human_deck_id="ampharos-ex-battle-deck", ai_name="Brock")
        actions = list_legal_actions(state, player_index=1)

        payload = build_decision_request(
            state,
            session_id="session-a",
            decision_id="session-a:opening_active",
            decision_type="opening_active",
            acting_player_index=1,
            ai_trainer_id="brock",
            ai_deck_id="lucario-ex-battle-deck",
            legal_actions=actions,
        )

        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["decision_type"], "opening_active")
        self.assertEqual(payload["acting_player_index"], 1)
        self.assertEqual(payload["public_state"]["setup_phase"], "choose_active")
        self.assertNotIn("hand", payload["public_state"]["players"][0])
        self.assertNotIn("deck", payload["public_state"]["players"][0])
        self.assertIsNone(payload["public_state"]["players"][0]["active"])
        self.assertEqual(len(payload["player_private_state"]["hand"]), len(state.players[1].hand))
        self.assertEqual(
            sorted(action["action_id"] for action in payload["legal_actions"]),
            sorted(f"play_basic_to_active:{action['hand_card_id']}" for action in actions),
        )

    def test_fallback_provider_uses_local_when_remote_response_is_invalid(self) -> None:
        state = create_game(seed=1, human_deck_id="ampharos-ex-battle-deck", ai_name="Brock")
        actions = list_legal_actions(state, player_index=1)
        provider = FallbackStandardDecisionProvider(
            trainer_id="brock",
            ai_deck_id="lucario-ex-battle-deck",
            session_id="session-a",
            policy_store=self.store,
            config=StandardPolicyConfig(
                remote_enabled=True,
                remote_url="http://example.invalid/policy",
                remote_timeout_ms=2_000,
                exploration_rate=0.0,
                min_exploration_rate=0.0,
            ),
        )
        request = DecisionRequest(
            state=state,
            acting_player_index=1,
            decision_type="opening_active",
            decision_id="session-a:opening_active",
            legal_actions=actions,
            payload=build_decision_request(
                state,
                session_id="session-a",
                decision_id="session-a:opening_active",
                decision_type="opening_active",
                acting_player_index=1,
                ai_trainer_id="brock",
                ai_deck_id="lucario-ex-battle-deck",
                legal_actions=actions,
            ),
        )

        with patch(
            "backend.tcg_ai.game_modes.standard.policy.urllib_request.urlopen",
            return_value=_FakeResponse({"decision_id": "session-a:opening_active", "chosen_action_id": "illegal"}),
        ):
            result = provider.choose_action(request)

        self.assertEqual(result.source, "local")
        self.assertIn("fallback_reason", result.diagnostics)


class _FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")
