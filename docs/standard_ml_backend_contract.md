# Standard ML Contract

## Platform Choice

Use Linux for the production ML worker.

Why:
- CUDA, PyTorch, and background service orchestration are simpler and more stable on Linux than Windows.
- The `i7-9700K`, `32 GB RAM`, and `RTX 4070 12 GB` are strong enough for a hybrid system:
  - CPU handles game simulation, legal action generation, and rollout orchestration.
  - GPU handles batched position/value inference and later supervised or reinforcement learning updates.
- We can keep one local inference worker always warm and batch evaluations without fighting Windows driver quirks.

## Core Design Decision

The engine should teach the AI the rules. The AI should learn strategy.

That means:
- The core game engine remains the single source of truth for legality and card resolution.
- The ML system should not be asked to infer what `Iono`, `Nest Ball`, or `Drakloak` do from sparse win/loss signals alone.
- Blind trial-and-error is far too sample-inefficient if early training comes from human games.

So the backend must send structured card and action semantics, not just card names or raw text.

## What The ML Worker Needs

For every decision request, send a full canonical game snapshot, not only the public board.

Minimum required information:
- Match metadata: `session_id`, `decision_id`, `turn_number`, `acting_player_index`
- Exact game state: both players, all zones, exact deck order, hand contents, discard, active, bench, prizes, turn flags, setup flags, and winner if terminal
- Card definitions for every referenced card: kind, stage, HP, attacks, energy requirements, and machine-readable effect metadata
- Legal actions for the acting player, including exact action ids
- Per-turn flags: supporter used, manual attachment used, retreat used, stadium in play, abilities locked, special conditions, pending prompts
- Hidden-information context: which cards are known, revealed, searched, shuffled, bottom-decked, or selected privately

## Important Rule About Card Knowledge

Do not make the AI learn card rules from names alone.

For cards like:
- `Iono`
- `Nest Ball`
- `Ultra Ball`
- `Rare Candy`
- `Drakloak`

the ML worker should receive structured effect annotations such as:
- `effect_tags`: `hand_refresh`, `disruption`, `deck_search`, `evolution`, `look_at_top_cards`, `choose_one`, `bottom_deck_remaining`
- `draw_count`
- `shuffle_required`
- `search_filters`
- `selection_count`
- `destination_zone`
- `source_zone`
- `reveals_cards`
- `changes_hidden_information`

Natural-language text can be included for debugging, but it should not be the main source of truth.

## Recommended Request Shape

The new ML endpoint should receive a payload shaped like this:

```json
{
  "schema_version": 2,
  "decision_id": "session-123:turn4:p1:d7",
  "decision_type": "turn_action",
  "session_id": "session-123",
  "acting_player_index": 1,
  "search_config": {
    "max_depth": 3,
    "beam_width": 8,
    "opponent_branch_width": 4,
    "include_opponent_turn": true
  },
  "state": {
    "schema_version": 2,
    "turn_number": 4,
    "current_player": 1,
    "setup_phase": null,
    "winner": null,
    "players": []
  }
}
```

The current implementation in this repo already includes a richer canonical serializer/deserializer for Standard game state and a local ML service entrypoint.

## Action Metadata Requirements

Each legal action should carry:
- `action_id`
- `type`
- `label`
- `source`
- `target`
- `card_instance_id` when relevant
- `effect_tags`
- `resource_costs`
- `consumes_supporter_for_turn`
- `consumes_attachment_for_turn`
- `reveals_hidden_cards`

Best practice:
- Add an `expected_state_delta` block for every legal action.

This is not strictly required once the ML worker can reconstruct the engine state locally, but it is extremely helpful for debugging, feature generation, and eventually for training labels.

## Turn Loop Integration

Recommended flow:

1. Backend updates the canonical ML state after every human or AI action.
2. When it is the AI player's turn and there is a meaningful choice, backend sends the full state to the ML worker.
3. ML worker searches candidate lines and returns one `chosen_action_id`.
4. Backend applies the action in the engine and repeats until the turn ends.
5. At game end, backend sends an outcome payload so the ML worker can log the full episode result.

## Learning Plan

Phase 1:
- Learn from human-vs-AI games using logged decisions and outcomes.
- Start with supervised preference signals plus value targets from final outcome.

Phase 2:
- Add self-play and batched rollout search once the Standard rules engine is deeper.

Phase 3:
- Train a GPU value-policy model that replaces most heuristic ranking while the engine still guarantees legality.

## Current Code Added

This repo now contains:
- a canonical Standard state serializer/deserializer
- a beam-search turn planner scaffold
- a local HTTP ML service
- experience logging for decision and outcome data

Run the ML worker with:

```bash
python3 -m backend.tcg_ai.game_modes.standard.ml_server
```

Then point the backend policy URL at:

```text
http://127.0.0.1:8100/api/standard-ml/decision
```
