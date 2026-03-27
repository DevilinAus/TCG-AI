# Standard ML Quality TODO

This file is the working checklist for the Standard neural-quality branch.

## Merge Readiness

These items are complete and make the codebase mergeable from a stability/tooling point of view:

- [x] Implement native retreat in the Standard engine
- [x] Expose retreat through the API/presentation/frontend flow
- [x] Add retreat-aware backend/frontend tests
- [x] Add legal-action metadata for ML-facing action serialization
- [x] Upgrade the Standard neural encoder with richer board/action features
- [x] Upgrade training targets to use soft policy targets and denser value targets
- [x] Add champion-aware self-play/training pipeline defaults
- [x] Add strategic tactical regression suite
- [x] Add objective `Nest Ball` / `Ultra Ball` search-target scenarios
- [x] Add objective resource-investment scenarios for:
  - [x] bench attachment from a doomed active
  - [x] useful over-attachment to pay retreat
- [x] Run focused tactical verification
- [x] Run full test suite (`175` tests passing at last local verification)

## Before Judging Model Strength

These are not merge blockers for the code, but they do need to happen before we judge whether the model now plays better:

- [ ] Regenerate self-play data from scratch with the updated engine/action space
- [ ] Train a fresh checkpoint from the new data
- [ ] Evaluate the new checkpoint against the current champion/baseline
- [ ] Confirm the checkpoint passes the strategic tactical suite during promotion
- [ ] Smoke-test live play against the promoted model through the remote worker

## Recommended Next Quality Work

These are the next worthwhile improvements after merge if the retrained model still shows quirks:

- [ ] Add more objective sacrifice scenarios that survive opponent-turn search cleanly
- [ ] Add stronger supporter/resource-conversion scenarios when a uniquely live search line exists
- [ ] Improve training-time search quality further without making live play slower
- [ ] Add better distributed run -> train -> evaluate -> promote automation
- [ ] Add shared accelerated inference for model-guided self-play workers
- [ ] Reassess whether the current MLP encoder is enough after a fresh full retrain

## Notes

- The code now has better tooling to learn stronger play.
- The currently trained checkpoint is outdated relative to the engine/encoder/target changes.
- The next meaningful quality signal is a fresh retrain, not more manual playtesting of the old checkpoint.
