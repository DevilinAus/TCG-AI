# Standard ML Distributed Plan

## Goal

Scale Standard self-play beyond a single CPU while keeping training centralized on the main accelerated training host.

The target matchup is still:

- `ampharos-ex-battle-deck`
- `lucario-ex-battle-deck`

The central design is:

- many CPU machines simulate games
- one main Linux machine coordinates chunks, stores results, trains, evaluates, and promotes checkpoints
- the rules engine remains authoritative
- the existing remote Standard NN worker continues to serve promoted checkpoints for live play

## Why This Direction

This repo's Standard engine is symbolic Python logic:

- legal action generation
- hidden-information handling
- deck search and zone movement
- branching search
- `deepcopy`-heavy rollouts

That is a poor fit for "move the whole simulator onto the accelerator" as an MVP.

The better fit for the available hardware is:

- CPU for simulation
- accelerator-backed training for model updates
- later, accelerator-backed batched neural inference during self-play

## Hardware Strategy

Recommended split:

- accelerated training host: coordinator, trainer, evaluator, champion storage
- extra CPU machines: self-play workers

Candidate worker machines:

- M1 MacBook
- 5700X3D
- 2600K
- i5-2400
- 9700K

## Phases

### Phase 1: Distributed Heuristic Self-Play

Build a coordinator that:

- creates a self-play run
- splits it into chunk leases
- hands chunks to workers over HTTP on the LAN
- accepts chunk submissions
- writes standard self-play shard outputs compatible with existing training scripts

Build workers that:

- poll for chunks
- simulate games locally using the current heuristic/search teacher
- upload decision/game shard results
- retry until the run is complete

Notes:

- this is the fastest practical way to use all available CPUs now
- this avoids making the accelerated training host a bottleneck for first-pass data generation

### Phase 2: Centralized Training And Promotion

Keep training on the accelerated training host only.

The central machine should:

- train on merged shard outputs
- evaluate candidate checkpoints against the current champion
- promote stronger checkpoints to `standard_ml_data/champion.pt`

This phase reuses the scripts already built:

- `scripts/train_standard_model.py`
- `scripts/evaluate_standard_checkpoints.py`

### Phase 3: Generation-Based Learning

Current limitation:

- the existing pipeline runs one heuristic self-play batch, then trains once

Desired improvement:

- if `standard_ml_data/champion.pt` exists, future generations should initialize from it
- self-play should optionally use the champion as guidance instead of always falling back to heuristic bootstrap
- training should support `resume-from champion`

Generation loop:

1. generate data
2. train candidate from current champion
3. evaluate vs champion
4. promote if better
5. repeat

### Phase 4: Shared Accelerated Inference For Self-Play

This is the right way to use the accelerated training host more often.

Do not try to move full game simulation onto the accelerator.

Instead:

- keep simulation on CPU workers
- add a shared batched inference path on the main Linux box
- let workers request policy/value evaluations from the central inference service

This phase only makes sense after there is already a useful champion checkpoint.

### Phase 5: Mixed Quality Tiers

Not all self-play needs the same cost.

Preferred long-term split:

- `fast` tier: lower search cost, high throughput
- `quality` tier: stronger search or champion-guided games, lower throughput

This prevents the project from getting trapped in either:

- huge weak datasets
- tiny expensive datasets

## Coordinator MVP Requirements

The coordinator should expose:

- `GET /healthz`
- `GET /api/standard-self-play/status`
- `POST /api/standard-self-play/lease-chunk`
- `POST /api/standard-self-play/heartbeat`
- `POST /api/standard-self-play/submit-chunk`

It should:

- persist run state to disk
- survive restart without losing run progress
- reclaim expired chunk leases
- ignore duplicate or invalid submissions safely
- write shard outputs under the same format the training scripts already read

## Worker MVP Requirements

The worker should:

- identify itself with a stable worker ID
- request a chunk
- run the chunk locally
- heartbeat while working
- submit decisions, games, and summary
- continue until the coordinator reports the run is complete

It should support:

- configurable coordinator URL
- configurable poll interval
- local worker progress logging

## Data Format Requirements

Worker submissions must preserve compatibility with the current training pipeline.

Required artifacts per chunk:

- `decisions/shard_XXXXXX.jsonl`
- `games/shard_XXXXXX.jsonl`
- merged `summary.json`
- run `manifest.json`

Hidden-information constraints still apply:

- no leaking prize positions
- no leaking opponent hand contents
- no leaking opponent deck order

## Operational Guidance

Recommended early chunk sizes:

- `25`
- `50`
- `100`

Do not treat `100000` games or `250` chunk size as fixed.

Practical guidance:

- `games` changes how much data you collect
- `chunk-size` mostly changes operational feel, scheduling, and time-to-visible-progress
- search settings change label quality much more than chunk size does

## MVP Build Order

1. Factor self-play chunk execution into reusable code.
2. Implement a file-backed coordinator state machine.
3. Implement an HTTP coordinator server script.
4. Implement a worker script that polls, heartbeats, runs, and submits.
5. Add targeted tests for lease/submit persistence.
6. Document Linux coordinator plus multi-machine worker setup.
7. Add generation-based champion reuse after the coordinator MVP works.
8. Add shared accelerated inference only after profiling shows the next real bottleneck.

## Things To Avoid

- Do not try to fully accelerator-port the Standard simulator for MVP.
- Do not make every worker depend on central accelerated inference immediately.
- Do not build online live-weight mutation into human matches.
- Do not assume more games automatically means better data.
- Do not block the whole system on the slowest worker; reclaim leases.

## Current Recommendation

The best next engineering move is:

- distributed heuristic self-play first
- centralized training/evaluation second
- generation-based champion reuse third
- shared accelerated inference fourth

That sequence is the most practical use of the available machines and keeps the project moving toward a stronger playable Standard model.
