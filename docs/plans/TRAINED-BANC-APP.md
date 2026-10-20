# Frozen trained BANC app integration — 2026-09-28

The user's continuation authorizes the next app integration task. The extra
covered-training threshold remains a recorded failure. No training or scientific
evaluation belongs to this task.

## Implementation

- Add `python -m flybrain.trained_app --control banc-trained --seed 7`.
  The new entry point defaults to the trained option; its `manual`, `neural`
  and `banc` options delegate to the original app.
- Keep every existing protocol-bound Python source and experiment byte intact.
  Reuse the original world, controller, projection decoder and brain drawing.
  Add HUD subclasses with accurate frozen-readout labels.
- Verify five pinned training/assay files, all 64 assay-recorded sources and
  four accepted data files before strict checkpoint loading. Keep the failed
  fit result pinned as well as the separately successful accepted assay.
- Freeze the existing 370 parameters and preserve exact 20 ms world ticks,
  twenty 1 ms neural steps per tick, sensory-only inputs and episode resets.

## Verification and remaining limit

Behavior tests cover checkpoint/evidence/source corruption, RNG preservation,
zero-step loading, frozen parameters, reset replay, timing, motor disabling,
rendering independence, CLI modes, failure before display initialization and
injected reset/exit events. A 120-tick headless smoke and
[inspected renderer frame](../verification/trained-app-frame.png) show world and
brain at tick120 / neural step2400. These are software/renderer checks.

Live keyboard and desktop-window review is pending: the Windows Computer Use
helper could not connect to its native pipe after retry and kernel reset.
Complete that review when the helper is available; do not rerun an assay.

## Small PyTorch example

Suppose a PN receives one weighted spike event during a 20 ms interval. Its
budget-normalized rate is `1 / (3 * .020) = 16.67 Hz`. Starting from zero,
the fixed 100 ms history becomes `(1 - exp(-.020/.100)) * 16.67 = 3.02 Hz`.
Together with the other PN and endpoint histories it contributes to a `[20]`
feature tensor. Pair means/differences and saved normalization precede
`W_hidden @ x + b`, with `W_hidden` shaped `[16,20]`; tanh produces `[16]`.
The `[2,16]` output matrix produces two values, bounded with sigmoid for
forward and tanh for turn. Neural state and histories change each tick;
the saved matrices and biases stay fixed. This is forward inference, with
no autograd graph, gradient calculation or optimizer update.


## Verified 750-update integration - 2026-09-28

The user's request to fix remaining work authorizes exposing the separately
verified 750-update checkpoint. `python -m flybrain --seed 7` now selects it;
`flybrain.verified_app` is also directly runnable. The old `trained_app` stays
on its historical 500-update model. Both share the original world and brain
views; the current HUD/caption explicitly says 750 updates.

The new loader pins five extra-fit/fresh-assay artifacts and validates 68
recorded sources and four accepted data files before strict loading. It makes
no optimizer updates, consumes no receipt, and rejects missing/changed evidence.
Tests cover actual saved weights, RNG, reset replay, render independence,
20:1 timing, motor gating, all CLI modes, lazy manual dependencies, error cleanup,
R/Esc/window-close event handling, and truthful interrupted-smoke reporting.

The 120-tick smoke and [inspected frame](../verification/verified-app-frame.png)
pass. Native desktop review remains blocked by the missing helper pipe; injected
events and the saved frame do not substitute for it. Next: with the helper
available, inspect movement and both panels, press R, then Esc, and compare the
fixed mode's labels. No additional training belongs to this task.

Final software verification: **32 focused tests passed** in 9.68 s; **523 full-suite tests passed** in 101.56 s, zero skips, one existing upstream Pygame deprecation warning per run. Reports: `.verified-app-targeted.xml`, `.verified-app-full.xml`.
