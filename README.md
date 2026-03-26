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

Quick launcher for the remote NN worker host:

```bash
bash scripts/start_standard_ml_worker.sh --checkpoint /home/<you>/models/champion.pt --token <shared-token>
```

## Standard Self-Play And Training

The first self-play/training scripts on this branch target only the current `Ampharos ex Battle Deck` vs `Lucario ex Battle Deck` matchup.

Generate self-play data locally on the training host:

```bash
python3 scripts/run_standard_self_play.py --games 100000 --workers 8 --chunk-size 250
```

Or run the whole pipeline in one go:

```bash
bash scripts/run_standard_training_pipeline.sh --games 100000 --workers 8 --train-device cuda --epochs 1
```

To watch detailed self-play/training/evaluation progress in a second terminal while the main pipeline terminal stays quiet:

```bash
bash scripts/tail_standard_training_progress.sh
```

Train a checkpoint from the newest self-play run:

```bash
python3 scripts/train_standard_model.py --device cuda --batch-size 128 --epochs 1
```

Evaluate a new checkpoint against the current champion and promote it if it clears the threshold:

```bash
python3 scripts/evaluate_standard_checkpoints.py --candidate standard_ml_data/checkpoints/run_20260326T010203Z/final.pt --promote-path standard_ml_data/champion.pt
```

If you want the trained model to become the worker's default checkpoint immediately:

```bash
python3 scripts/train_standard_model.py --device cuda --promote-path standard_ml_data/champion.pt
```

The self-play script prints rolling console progress, the training script prints live loss/checkpoint updates, and the evaluation script prints live head-to-head win-rate progress before optionally promoting a new champion checkpoint.

## Distributed Standard Self-Play

If you just want the shortest launch instructions, start with [START_HERE.md](/Users/andrew/Documents/projects/TCG-AI/START_HERE.md).

If one machine is handling accelerated training/inference and several others are mostly idle CPUs, the best MVP scale-up path is distributed self-play:

- one main machine runs the coordinator and stores shards
- extra machines poll for chunks and simulate games locally
- the main training host trains and evaluates checkpoints after the run

Why this shape:

- the Standard simulator is symbolic Python logic and is a poor fit for "move the whole engine onto the accelerator"
- CPU workers are still useful for generating lots of games
- the accelerated training host is better used for training now, and later for batched model inference

### What Runs Where

Coordinator / training box:

- machine that will run training, evaluation, and optional accelerated inference
- full repo checkout and Python environment
- runs the coordinator
- serves the dashboard
- stores the self-play shards
- later runs training and evaluation

Worker machines:

- any Mac or Linux machine with the repo checkout and Python environment
- no dedicated accelerator required
- run self-play workers only
- request chunk leases from the coordinator and upload results back

### Requirements And Dependencies

Coordinator / training host requirements:

- Python `3.13+`
- full repo checkout
- project dependencies installed in a virtual environment
- network visibility from worker machines to the chosen coordinator host/port

Coordinator / training host Python dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Accelerated training / remote NN worker extras:

```bash
pip install -e '.[standard-ml]'
```

That extra installs the optional ML stack from [pyproject.toml](/Users/andrew/Documents/projects/TCG-AI/pyproject.toml):

- `torch`
- `fastapi`
- `uvicorn`
- `numpy`
- `polars`
- `pyarrow`
- `tensorboard`

Worker machine requirements:

- Python `3.13+`
- full repo checkout
- project dependencies installed with `pip install -e .`
- no accelerated ML extras required for heuristic distributed self-play workers

Dashboard requirements:

- no separate frontend build step
- the coordinator serves `/dashboard`, `/dashboard.css`, and `/dashboard.js` directly

### How To Launch A Distributed Run

1. On the coordinator machine, start the coordinator.

Minimal:

```bash
bash scripts/start_standard_self_play_coordinator.sh
```

Recommended first run:

```bash
export TCG_AI_STANDARD_SELF_PLAY_RUN_ID=run_$(date -u '+%Y%m%dT%H%M%SZ')
TCG_AI_STANDARD_SELF_PLAY_GAMES=20000 \
TCG_AI_STANDARD_SELF_PLAY_CHUNK_SIZE=50 \
TCG_AI_STANDARD_SELF_PLAY_HOST=0.0.0.0 \
TCG_AI_STANDARD_SELF_PLAY_PORT=8787 \
bash scripts/start_standard_self_play_coordinator.sh
```

The coordinator prints:

- the server URL
- the dashboard URL
- the status endpoint
- the output directory for this run

2. Open the dashboard in your browser.

```text
http://<linux-box>:8787/dashboard
```

3. Optionally open a second terminal on the coordinator machine for the text status watcher.

```bash
bash scripts/watch_standard_self_play_status.sh http://<linux-box>:8787
```

4. On each worker machine, launch a worker.

Minimal worker setup:

```bash
export TCG_AI_STANDARD_SELF_PLAY_COORDINATOR_URL=http://<linux-box>:8787
export TCG_AI_STANDARD_SELF_PLAY_WORKER_ID=<unique-worker-name>
bash scripts/start_standard_self_play_worker.sh
```

Recommended worker setup:

```bash
export TCG_AI_STANDARD_SELF_PLAY_COORDINATOR_URL=http://<linux-box>:8787
export TCG_AI_STANDARD_SELF_PLAY_WORKER_ID=macbook-m1
export TCG_AI_STANDARD_SELF_PLAY_POLL_SECONDS=2
export TCG_AI_STANDARD_SELF_PLAY_HEARTBEAT_INTERVAL_SECONDS=15
export TCG_AI_STANDARD_SELF_PLAY_PROGRESS_LOG=standard_ml_data/progress/self_play_worker.log
bash scripts/start_standard_self_play_worker.sh
```

Important:

- every worker should use a unique `TCG_AI_STANDARD_SELF_PLAY_WORKER_ID`
- every worker should be on the same repo revision
- every worker needs the Python dependencies installed
- if a worker disappears, the coordinator will eventually reclaim the lease and reissue that chunk

5. Confirm the machines appear in the dashboard.

What you should see:

- workers marked as `busy`, `idle`, `stalled`, or `offline`
- current shard progress per machine
- total distributed throughput and recent games-per-minute
- per-worker pace so you can compare machines against each other
- dropout visibility when a worker stops checking in

### Dashboard Notes

The dashboard shows two slightly different progress views:

- `reported` totals update as soon as a worker finishes a game and reports progress
- `aggregate` totals update only after a full shard is submitted and written to disk

So during an active run it is normal for `reported.games` to be slightly ahead of `aggregate.games`.

Worker states mean:

- `busy`: worker is alive and currently holds a shard lease
- `idle`: worker is alive but waiting for work
- `stalled`: worker still holds a lease but has stopped checking in
- `offline`: worker has no active lease and has stopped checking in

### Output Layout

The coordinator writes a normal self-play dataset under `standard_ml_data/distributed_self_play/<run_id>`:

- `manifest.json`
- `summary.json`
- `decisions/shard_XXXXXX.jsonl`
- `games/shard_XXXXXX.jsonl`

That output is intentionally compatible with the training/evaluation scripts that already exist.

If you want to control the output folder explicitly:

```bash
export TCG_AI_STANDARD_SELF_PLAY_OUTPUT_DIR=standard_ml_data/distributed_self_play/<run_id>
```

### Parameter Guidance

These are not fixed:

- `games`
- `chunk-size`
- `workers`

Practical guidance:

- `games` mostly changes how much data you collect
- `chunk-size` mostly changes scheduler behavior, time-to-visible-progress, and shard count
- `chunk-size` does not make the model smarter by itself
- `max-depth`, `beam-width`, and `opponent-branch-width` affect teacher quality much more directly

Good early coordinator defaults:

- `games=5000` for the first end-to-end distributed smoke run
- `chunk-size=25` or `50` so work is rebalanced often and progress appears quickly
- keep the current search defaults until the distributed loop is stable

### After The Run Finishes

Train on the training host using the completed distributed run folder:

```bash
python3 scripts/train_standard_model.py --input-dir standard_ml_data/distributed_self_play/<run_id> --device cuda --epochs 1
```

Evaluate the candidate checkpoint and promote it if it wins:

```bash
python3 scripts/evaluate_standard_checkpoints.py --candidate standard_ml_data/checkpoints/<run_id>/final.pt --promote-path standard_ml_data/champion.pt
```

If you want to play against the promoted model in the UI afterward, launch the remote NN worker against the champion checkpoint:

```bash
bash scripts/start_standard_ml_worker.sh --checkpoint standard_ml_data/champion.pt --token <shared-token>
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
