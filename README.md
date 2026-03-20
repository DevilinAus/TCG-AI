# TCG AI MVP

This project is a browser-playable MVP of a small Pokemon TCG experience built around the `My First Battle` ruleset. Right now the game supports a single matchup, `Charmander Deck` vs `Squirtle Deck`, with a Python rules engine, a lightweight browser board, and gym leaders that gradually adapt over repeated games.

## Current MVP

- Play a full local match in the browser with real card art and a board-first UI.
- Choose a gym leader opponent before starting a new game.
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
  - battle log and status banner
  - hidden dev panel with the raw legal action list
  - gym leader picker and XP progress bar
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

## How To Test

```bash
python3 -m unittest discover -s tests
```

## Card Art Assets

You can re-download the `My First Battle` card images into the UI asset folder with:

```bash
bash scripts/download_my_first_battle_assets.sh
```

The assets are stored in `frontend/assets/cards/my-first-battle`, and the filename mapping is tracked in `frontend/assets/cards/my-first-battle/manifest.json`.

## Current API Shape

- `POST /api/new-game`
  - Creates a new session.
  - Accepts optional `trainer_id`, `human_first`, and `seed`.
- `GET /api/game?session_id=...`
  - Returns the current serialized game state for that session.
- `POST /api/action`
  - Expects `session_id` and a raw engine action object.
- `POST /api/ai-turn`
  - Expects `session_id` and replays the AI turn for that session.

Serialized state also includes the selected trainer snapshot, the available trainer list, and the AI learning snapshot used by the frontend.

All API errors use a small JSON shape:

```json
{
  "error": "Unknown session ID.",
  "code": "session_not_found"
}
```

## Manual Smoke Path

Use this quick browser path to sanity-check the MVP:

1. Start a new game and choose a gym leader.
2. Confirm the same session resumes after a page reload.
3. Click a hand card and verify only legal contextual actions appear.
4. Bench a Pokemon, attach Energy, and attack.
5. Let the AI complete its turn through the replay flow.
6. Play through a knockout and confirm bench promotion works correctly.
7. Finish a full game and check that the selected leader's XP bar updates.

## Important Limitations

- The current build only supports the `Charmander Deck` vs `Squirtle Deck` `My First Battle` matchup.
- Sessions, trainer XP, and AI learning all live in server memory right now.
- Restarting the server resets active sessions and gym leader progression.
- Higher-level gym leaders should trend stronger over time, but level is still only a proxy for battle experience, not a guaranteed measure of decision quality.
- The backend remains the single source of truth for legality and game progression.
