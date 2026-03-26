# TCG AI MVP

This project is a browser-playable Pokemon TCG sandbox with two separate game modes:

- `My First Battle`, which is the current fully playable local experience
- `Standard`, which is an in-progress `ex Battle Deck` rules prototype on this branch

The app uses a Python rules engine, a board-first browser UI, and local AI/trainer systems that are being expanded toward a future remote ML service.

## Current MVP

- Play a full local `My First Battle` match in the browser with real card art and a board-first UI.
- Start a `Standard` prototype game with imported `ex Battle Deck` data and playable early-turn flow.
- Choose a gym leader opponent and a deck before starting a new game.
- Let the backend enforce legal actions, attacks, knockouts, promotion flow, prizes, and win conditions.
- Watch AI turns replay in the battle log instead of manually driving both sides.
- Track each leader's battle XP and level as a fun "we've fought a lot" progress indicator.

## What Is Working

- Pure Python game engine for setup, turn order, legal action generation, energy attachment, evolution, attacks, coin-flip effects, knockout handling, bench promotion, prize taking, and game-end detection.
- Session-based HTTP API for creating games, restoring the current board state, submitting player actions, and replaying AI turns.
- Browser UI with:
  - contextual click targets for legal plays
  - floating active, bench, hand, discard, and energy zones
  - real local card art
  - floating selected-card preview
  - targeted-attack drag indicator for bench-sniping attacks
  - battle log and status banner
  - hidden dev panel with the raw legal action list
  - game mode switcher, deck picker, gym leader picker, and XP progress bar
- `My First Battle` mode:
  - playable local matches with the starter deck ruleset kept separate from Standard logic
  - support for the four `My First Battle` decks in the project assets/data lane
  - prize coin artwork by deck
- `Standard` mode on this branch:
  - shuffled 60-card deck loading from imported `ex Battle Deck` data
  - opening hands, mulligans, active selection, bench setup, and end-setup flow
  - benching, supporter/item plays already wired in the Standard engine, energy attachment, evolution, and attacking
  - attached-energy rendering on Pokemon and targeted attack selection for attacks like `Linear Attack`
  - a Standard-only local planner / decision layer with a remote-ready state payload boundary
- In-memory trainer profiles for the original Kanto gym leaders:
  - Brock
  - Misty
  - Lt. Surge
  - Erika
  - Koga
  - Sabrina
  - Blaine
  - Giovanni
- Automated tests covering core engine rules, API behavior, bot behavior, targeting, and promotion edge cases.

## Gym Leaders Learning To Play

Each gym leader has their own `RewardLearner` profile. When you start a game against Brock, Misty, or another leader, that match uses that leader's personal learner rather than a single shared AI brain.

Right now that learning loop is used by `My First Battle`.

`Standard` is being built on a separate decision path so it can later call a remote ML service with serialized public state, private acting-player state, and legal actions without disturbing the `My First Battle` ruleset.

During AI turns, the backend:

- summarizes the game state
- extracts features for the chosen action
- calculates a shaped reward for how helpful that action appears to have been
- records the step into the selected leader's learner

When the game ends, the leader's learner also receives the episode result. That means repeated games against the same leader should slowly nudge their policy toward stronger play for this tiny ruleset.

The XP bar and level are meant to be a fun progression layer, not a perfect skill rating. Leaders gain XP from damage dealt and prizes taken, so level mainly answers "how much battle experience has this leader built up in this run?" The real learning signal comes from the reward learner weights behind the scenes.

## How To Run

```bash
python3 -m backend.tcg_ai.server
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000).

## Remote Standard ML Worker

The Standard AI can now call a remote policy/value worker over HTTP while the main game continues to run locally.

Backend environment variables:

```bash
export TCG_AI_STANDARD_REMOTE_ENABLED=1
export TCG_AI_STANDARD_REMOTE_URL=http://<linux-box>:8100/api/standard-ml/decision
export TCG_AI_STANDARD_REMOTE_BATCH_EVAL_URL=http://<linux-box>:8100/api/standard-ml/batch-eval
export TCG_AI_STANDARD_REMOTE_TIMEOUT_MS=1500
export TCG_AI_STANDARD_REMOTE_API_TOKEN=<shared-token>
```

Worker launch options:

```bash
python3 -m backend.tcg_ai.game_modes.standard.ml_server
```

Or, with the optional `standard-ml` extras installed:

```bash
python3 -m backend.tcg_ai.game_modes.standard.ml_fastapi
```

Quick launcher for the Linux NN box:

```bash
bash scripts/start_standard_ml_worker.sh --checkpoint /home/<you>/models/champion.pt --token <shared-token>
```

## How To Test

```bash
python3 -m unittest discover -s tests
node --test tests/test_selection_state.js
```

## Card Art Assets

You can re-download the `My First Battle` card images into the UI asset folder with:

```bash
bash scripts/download_my_first_battle_assets.sh
```

The assets are stored in `frontend/assets/cards/my-first-battle`, and the filename mapping is tracked in `frontend/assets/cards/my-first-battle/manifest.json`.

## Current API Shape

- `GET /api/lobby`
  - Returns the current lobby snapshot, available game modes, decks, and trainer metadata.
- `POST /api/new-game`
  - Creates a new session.
  - Accepts optional `game_mode`, `trainer_id`, `human_deck_id`, `human_first`, and `seed`.
- `GET /api/game?session_id=...`
  - Returns the current serialized game state for that session.
- `POST /api/action`
  - Expects `session_id` and a raw engine action object.
- `POST /api/ai-step`
  - Advances exactly one AI action and returns replay metadata for that step.
- `POST /api/ai-turn`
  - Expects `session_id` and replays the AI turn for that session.

Serialized state also includes the selected trainer snapshot, available game modes, the available trainer/deck lists, and the AI-learning or AI-decision snapshot used by the frontend.

All API errors use a small JSON shape:

```json
{
  "error": "Unknown session ID.",
  "code": "session_not_found"
}
```

## Manual Smoke Path

Use this quick browser path to sanity-check the MVP:

1. Start the server and open the browser UI.
2. Choose a game mode, a deck, and a gym leader.
3. Start a new game and confirm the same session resumes after a page reload.
4. Click a hand card and verify only legal contextual actions appear.
5. Bench a Pokemon, attach Energy, and attack.
6. Let the AI complete its turn through the replay flow.
7. Play through a knockout and confirm bench promotion works correctly.
8. Finish a full game and check that the selected leader's XP bar updates.

For the `Standard` branch flow specifically:

1. Switch to `Standard`.
2. Start a new game and choose your Active Pokemon.
3. Bench any extra Basics during setup, then click `End Setup`.
4. Attach energy, play simple trainer cards, and attack.
5. Verify targeted attacks only ask for a target when the attack genuinely needs one.

## Important Limitations

- `My First Battle` is the stable playable mode today.
- `Standard` is not yet a full rules-complete Pokemon TCG implementation.
- Several `ex Battle Deck` attacks and card-specific effects are still being added incrementally.
- Some Standard effects that need richer choice UX or deeper rules support are still intentionally unimplemented.
- Sessions, trainer XP, and AI learning all live in server memory right now.
- Restarting the server resets active sessions and gym leader progression.
- Higher-level gym leaders should trend stronger over time, but level is still only a proxy for battle experience, not a guaranteed measure of decision quality.
- The backend remains the single source of truth for legality and game progression.
