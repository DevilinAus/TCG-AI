# TCG AI Starter

This project is a playable prototype of a very small Pokemon TCG experience based on the `My First Battle` rules. The current supported matchup is still `Charmander Deck` vs `Squirtle Deck`, but the app now has real card art, browser sessions, and a guided action flow that is much closer to a usable demo.

## What is implemented

- A pure Python rules engine for setup, turns, legal actions, attacking, KO flow, Prize tokens, Potion, Switch, and first-turn evolution.
- Session-based in-memory game management with structured API errors.
- A presentation layer that turns engine state into display-ready JSON with `card_id`, `instance_id`, image URLs, and UI hints.
- A lightweight web UI with local card art, contextual interaction, a rules help drawer, and a styled battle log.
- Card art assets stored locally in `frontend/assets/cards/my-first-battle`.
- Automated tests for engine rules, targeting, promotion flow, coin-flip attacks, and API session behavior.

## How to run

```bash
python3 -m backend.tcg_ai.server
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000).

## How to test

```bash
python3 -m unittest discover -s tests
```

## Card art assets

You can re-download the `My First Battle` card images into the UI asset folder with:

```bash
bash scripts/download_my_first_battle_assets.sh
```

The assets are stored in `frontend/assets/cards/my-first-battle`, and the filename mapping is tracked in `frontend/assets/cards/my-first-battle/manifest.json`.

## Current API shape

- `POST /api/new-game`
  - Creates a new browser session and returns the initial game state with `session_id`.
- `GET /api/game?session_id=...`
  - Returns the current serialized game state for that session.
- `POST /api/action`
  - Expects `session_id` and a raw engine action object.
- `POST /api/ai-turn`
  - Expects `session_id` and runs the AI for that session only.

All API errors use a small JSON shape:

```json
{
  "error": "Unknown session ID.",
  "code": "session_not_found"
}
```

## Manual smoke path

Use this quick browser path when you want to sanity-check the full demo:

1. Start a new game.
2. Confirm the same session resumes after a page reload.
3. Click a hand card and verify only legal contextual actions appear.
4. Bench a Pokemon, play Energy, and attack.
5. Let the AI take its turn automatically.
6. Finish a full game without using the debug action panel.

## Important assumptions

- The current build still supports only `Charmander Deck` vs `Squirtle Deck`.
- Sessions live only in server memory for now.
- Local browser persistence only stores the `session_id`; if the server restarts, the saved session becomes invalid and the UI creates a new game.
- The backend remains the single source of truth for rule legality and game progression.
