# FB-10 extra covered-data training — predeclaration, 2026-09-28

The user requested continuation through the extra training after the frozen
PN-input readout passed the original gates and fresh assay, while its separately
added covered-data error gate remained failed. The previous 500-update fit and
its result are closed and must stay unchanged.

## Single hypothesis

The original 500-update loss curve was still declining at update 500.
Extending the **same initialization, data, model, objective, and Adam learning
rate** to 750 total updates will reduce covered turn MSE from 0.0235447047 to
at most the pre-existing 0.0222527315 ceiling while retaining original-data
turn MSE at or below that ceiling and the original matched navigation gates.
This tests optimization time, not a new representation or training method.

## Locked fit scope and budget

- One new 20→16→2, 370-scalar PN-input/endpoint head, exact seed 1010.
- The already recorded 15,864 canonical train-only rows and their targets.
  The first 1,008 original rows and all coverage rows remain in fixed order.
- Full-batch Adam, learning rate 0.01, objective forward MSE plus turn MSE,
  **750 optimizer updates maximum**, one candidate, no width/encoder/seed or
  learning-rate sweep. Initial state comes from the same seed, rather than a
  partly restored optimizer, which was not saved in the old experiment.
- Stop immediately on nonfinite model, gradient, target or loss. A finite run
  stops at exactly 750 updates; do not tune after inspecting its error.
- Before embodied checks, require the update-500 curve row to match the
  preserved original fit exactly. A mismatch closes this attempt as invalid;
  it cannot be treated as a same-method extension.
- Only if fit numerics are finite, run the same eight original matched training
  starts plus one exact loaded-checkpoint replay: at most 25 episodes, 15,000
  world ticks and 300,000 neural steps. No fresh case enters optimization.
- Training gates are unchanged from the covered fit: aggregate and original
  turn MSE ≤0.0222527315; aggregate ≤25% of the train-only mean-command MSE;
  trained arrivals ≥7/8 with strict arrival and all-case progress gains over
  identical untrained and fixed BANC controls; all integrity checks pass.
- Preserve every old source and artifact byte. Prepare a single-use protocol
  and receipt before fitting. Record software/device, seed, hashes, objective,
  actual updates, losses, gradients, changed parameters, checkpoint, cases,
  summaries, counters and failure under `experiments/fb10-extra-training/`.

## Stop and evaluation rule

If any training gate fails, record the closed negative attempt and stop this
experiment. Do not automatically add another update budget or a different
method. The accepted frozen 500-update controller and its fresh result remain
the project's established navigation result.

If every training gate passes, a separate zero-update fresh comparison may
proceed only after locking a new single-use evaluation protocol. Proposed
identities 730000–730015 validation and 731000–731031 final were checked against
recorded history before preparation and had no overlap. Use the original
annulus, heading distribution, 600-tick horizon, identical controls and
13/16 validation, 26/32 final, paired gain ≥8, one-sided p≤0.025, strict
all-case progress and frozen-integrity gates. Stop before final if validation
fails. All new cases become historical once executed. Any learned-navigation
claim for the 750-update model requires these new outcomes.

## Outcome — completed 2026-09-28

The single fit passed: covered turn MSE 0.0213424152 and original-data turn
MSE 0.0100253166; original train arrivals 8/8 versus 0/8 and 0/8. All nine
integrity checks passed within 750 updates / 25 episodes / 10,778 ticks /
215,560 neural steps. Its separately locked fresh assay passed validation
16/16 versus 0/16 and 2/16, and final 31/32 versus 2/32 and 2/32, including
both paired gains +29, p=1.862645149230957e-9 each and all-case progress.
It consumed 144 episodes / 64,106 ticks / 1,282,120 neural steps, zero
updates. The 25-check [audit](../../experiments/fb10-extra-evaluation/audit.json)
passes and the [plot](../../experiments/fb10-extra-evaluation/extra_training_summary.png)
was inspected. Both receipts are consumed; no more fits or assays are implied.
The historical 500-update covered failure remains unchanged.
