# START HERE

This file is the shortest path to launching the distributed Standard self-play system.

If you want the longer explanation, use [README.md](/Users/andrew/Documents/projects/TCG-AI/README.md).

## 1. Start The Coordinator

Run this on the main machine that will collect results.
In your current LAN setup, the worker machines should point at `192.168.0.175`.

```bash
git checkout main
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

bash scripts/start_standard_self_play_coordinator.sh
```

That starts:

- the coordinator
- the dashboard server
- the chunk lease API for workers

Open the dashboard here:

```text
http://127.0.0.1:8787/dashboard
```

If you are opening it from another machine on your LAN, replace `127.0.0.1` with the coordinator machine IP.

## 2. Start Workers

Run this on each helper machine.
By default, the launcher:

- connects to `http://192.168.0.175:8787`
- uses the machine hostname as the worker prefix
- launches one worker process per detected CPU core

Linux / macOS:

```bash
git checkout main
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

bash scripts/start_standard_self_play_worker.sh
```

If you want to override the defaults:

```bash
bash scripts/start_standard_self_play_worker.sh http://192.168.0.175:8787 macbook-m1 --workers 4
```

Windows:

```bat
scripts\start_standard_self_play_worker.cmd
```

Windows with explicit prefix and worker cap:

```bat
scripts\start_standard_self_play_worker.cmd http://192.168.0.175:8787 windows-box --workers 4
```

Use a different worker prefix on each machine only if you want names that are easier to read than the hostname.

## 3. Optional Text Status View

On the coordinator machine:

```bash
bash scripts/watch_standard_self_play_status.sh http://127.0.0.1:8787
```

## 4. What You Built

- `start_standard_self_play_coordinator.sh`
  Starts the coordinator and serves the dashboard.
- `start_standard_self_play_worker.sh`
  Starts one worker per detected CPU core on Linux / macOS.
- `start_standard_self_play_worker.cmd`
  Starts one worker per detected CPU core on Windows.
- `/dashboard`
  Shows live workers, shard progress, throughput, and dropouts.

## 5. After The Run

Train on the completed run:

```bash
python3 scripts/train_standard_model.py --input-dir standard_ml_data/distributed_self_play/<run_id> --device cuda --epochs 1
python3 scripts/evaluate_standard_checkpoints.py --candidate standard_ml_data/checkpoints/<run_id>/final.pt --promote-path standard_ml_data/champion.pt
```
