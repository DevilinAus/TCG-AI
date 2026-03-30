# TCG AI MVP

Browser-playable Pokemon TCG sandbox with:

- `My First Battle` for the current local playable mode
- `Standard` for the in-progress ex Battle Deck rules + ML work

If you only care about getting distributed self-play running fast, start with [START_HERE.md](/Users/andrew/Documents/projects/TCG-AI/START_HERE.md).

## Run The App

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
python3 -m backend.tcg_ai.server
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000).

## Distributed Self-Play

Mental model:

- `start_standard_self_play_coordinator.sh` starts the coordinator and the dashboard
- `start_standard_self_play_worker.sh` starts a worker
- the dashboard lives at `/dashboard`

You do **not** launch the dashboard separately.

### 1. Start The Coordinator

Run this on the main machine that will collect results and later train.
In your current setup, the worker machines should target `192.168.0.175`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

bash scripts/start_standard_self_play_coordinator.sh
```

Optional example with explicit settings:

```bash
export TCG_AI_STANDARD_SELF_PLAY_GAMES=20000
export TCG_AI_STANDARD_SELF_PLAY_CHUNK_SIZE=50
export TCG_AI_STANDARD_SELF_PLAY_HOST=0.0.0.0
export TCG_AI_STANDARD_SELF_PLAY_PORT=8787

bash scripts/start_standard_self_play_coordinator.sh
```

### 2. Open The Dashboard

Open this in a browser on the coordinator machine:

```text
http://127.0.0.1:8787/dashboard
```

From another machine on the LAN:

```text
http://<coordinator-ip>:8787/dashboard
```

### 3. Start Workers

The worker launchers now do the repetitive part for you:

- default coordinator: `http://192.168.0.175:8787`
- default worker prefix: machine hostname
- default worker count: one process per detected CPU core

Linux / macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

bash scripts/start_standard_self_play_worker.sh
```

Linux / macOS with explicit overrides:

```bash
bash scripts/start_standard_self_play_worker.sh http://192.168.0.175:8787 macbook-m1 --workers 4
```

Windows:

```bat
scripts\start_standard_self_play_worker.cmd
```

Windows with explicit overrides:

```bat
scripts\start_standard_self_play_worker.cmd http://192.168.0.175:8787 windows-box --workers 4
```

If you do not supply a prefix, the launcher uses the hostname.

### 4. Optional Text Status Watcher

On the coordinator machine:

```bash
bash scripts/watch_standard_self_play_status.sh http://127.0.0.1:8787
```

### What The Dashboard Means

- `busy`: worker is alive and currently holds a shard
- `idle`: worker is alive but waiting for more work
- `stalled`: worker still holds a shard but stopped checking in
- `offline`: worker has no shard and stopped checking in

You may see:

- `reported` totals move first
- `aggregate` totals move after a shard is fully submitted

That is normal.

## Requirements

For coordinator and workers:

- Python `3.13+`
- full repo checkout
- `pip install -e .`

For training and the remote NN worker:

```bash
pip install -e '.[standard-ml]'
```

That installs the optional ML stack from [pyproject.toml](/Users/andrew/Documents/projects/TCG-AI/pyproject.toml), including:

- `torch`
- `fastapi`
- `uvicorn`
- `numpy`
- `polars`
- `pyarrow`
- `tensorboard`

Important:

- workers should be on the same repo revision
- workers do not need the optional ML extras for heuristic distributed self-play
- the dashboard has no separate frontend build step

## After A Distributed Run

Train on the completed run:

```bash
python3 scripts/train_standard_model.py --input-dir standard_ml_data/distributed_self_play/<run_id> --device cuda --epochs 1
```

Evaluate and optionally promote:

```bash
python3 scripts/evaluate_standard_checkpoints.py --candidate standard_ml_data/checkpoints/<run_id>/final.pt --promote-path standard_ml_data/champion.pt
```

## Remote NN Worker

If you want to play against a trained model through the UI, run the remote Standard ML worker on the machine hosting the model checkpoint.

Backend environment variables for the main app:

```bash
export TCG_AI_STANDARD_REMOTE_ENABLED=1
export TCG_AI_STANDARD_REMOTE_URL=http://<worker-host>:8100/api/standard-ml/decision
export TCG_AI_STANDARD_REMOTE_BATCH_EVAL_URL=http://<worker-host>:8100/api/standard-ml/batch-eval
export TCG_AI_STANDARD_REMOTE_TIMEOUT_MS=1800000
export TCG_AI_STANDARD_REMOTE_API_TOKEN=<shared-token>
```

If your terminal or VS Code session keeps old `TCG_AI_*` exports around, you can put the current values in `.env.local` at the repo root. The backend now loads `.env` and then `.env.local` on startup, and those project-local values override stale shell values.

Launch the worker:

```bash
bash scripts/start_standard_ml_worker.sh --checkpoint standard_ml_data/champion.pt --token <shared-token>
```

## Useful Scripts

- [scripts/start_standard_self_play_coordinator.sh](/Users/andrew/Documents/projects/TCG-AI/scripts/start_standard_self_play_coordinator.sh)
- [scripts/start_standard_self_play_worker.sh](/Users/andrew/Documents/projects/TCG-AI/scripts/start_standard_self_play_worker.sh)
- [scripts/start_standard_self_play_worker.cmd](/Users/andrew/Documents/projects/TCG-AI/scripts/start_standard_self_play_worker.cmd)
- [scripts/start_standard_self_play_workers.py](/Users/andrew/Documents/projects/TCG-AI/scripts/start_standard_self_play_workers.py)
- [scripts/watch_standard_self_play_status.sh](/Users/andrew/Documents/projects/TCG-AI/scripts/watch_standard_self_play_status.sh)
- [scripts/run_standard_training_pipeline.sh](/Users/andrew/Documents/projects/TCG-AI/scripts/run_standard_training_pipeline.sh)
- [scripts/start_standard_ml_worker.sh](/Users/andrew/Documents/projects/TCG-AI/scripts/start_standard_ml_worker.sh)

## Testing

Fast local preflight:

```bash
npm run preflight
```

`preflight` is intentionally lightweight now. It runs:
- a small JS smoke subset
- Python syntax compilation
- a small Python smoke subset around Standard AI planning/policy

Full CI-equivalent suite:

```bash
npm run test:full
```

Python tests only:

```bash
python3 -m pytest -q
```
