# Smooth after-spike color fade — verified 2026-10-05

The user requested a visible fade back to cold blue after each neuron finishes
firing. The old exponential envelope faded quickly, then switched its remaining
4.9787% strength off at 600 ms. The new display envelope eases class RGBA to idle
over the same **600 ms** window, without that visible cutoff. Each neuron's own
last spike refreshes its pulse; pause freezes it. Explicit class-off/scope choices
remain immediate. The 200 ms rate average and neural calculations are unchanged.

## Calculation and boundaries

For `u=clip(age/.600,0,1)`, strength is `(1-u)^2*(1+2u)` (one minus cubic
smoothstep). It is 1 at 0 ms, .740741 at 200 ms, .5 at 300 ms, .259259 at
400 ms and 0 at 600 ms. Both ends have zero slope. The original 1e-12-second
expiry tolerance removes floating-point tails, so a mathematically expired owner
does not remain in the colored-pulse count. Never-fired/reset cells stay idle.

At 300 ms, an odor branch interpolates halfway between soft idle
(.24,.30,.55,.22) and amber (.902,.706,.314,.43), giving RGBA approximately
(.571,.503,.432,.325). Its GPU pixel over black is approximately
(.186,.163,.140). These are display calculations, separate from PyTorch neuron
dynamics or optimizer learning. A faded color does not erase recorded spikes.

## Fresh verification

One agent, local cached geometry; no experiment fitting or reserved assay. Venv
Python 3.12.10, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, serial required commands:

| Command | Result | Pytest duration | Exit |
|---|---|---:|---:|
| `python -m pytest -q tests -k flight3d` | 87 passed, 567 deselected, 2 warnings | 96.73 s | 0 |
| `python -m pytest -q` | 654 passed, zero skips, 2 warnings | 179.22 s | 0 |

Warnings are the existing Pygame `pkg_resources` and Torch `jit.script`
deprecations. Wrapper durations 112.16/189.13 s.
Logs, XML and exact commands/exits:
[targeted](../../.superpowers/activity-fade-2026-10-04/final-targeted.log),
[full](../../.superpowers/activity-fade-2026-10-04/final-full.log),
[exits](../../.superpowers/activity-fade-2026-10-04/final-tests-exits.json).

Four new behavioral checks cover independent last-spike ages, a visible intermediate
fade, negligible endpoint jump, floating-point expiry, refreshed firing, pause,
class mask off/on and reset. Real GPU pixels verify intermediate class color and
alpha plus the final return to idle. Complete history/palette outputs at shared
simulation times agree for 30/60/120 display Hz during silence. Existing
recorded/unrecorded teacher/fixed/disposable frozen-readout snapshot, neural state,
parameter and RNG checks pass in the required suites. No product/test edits
followed the final checks. Local code review completed without subagents.

Audit: 2,372 pre-existing source/data/experiment files;
**2,370 protected files byte-identical**. Only display pulse
math and help text changed in existing product sources. Recorder, scientific
snapshot, neural/sensory/controller/physics, class index, static geometry/camera,
cached sources and interrupted 1,050-update fit remain unchanged. No Git repository.

## Visual evidence and limits

Actual GPU one-spike fade frames, a contact sheet and looping GIF were generated
and inspected. This explicitly labeled **synthetic display fixture** checks
rendering; it is not a neural experiment or a biological result. The real session
does not advance while rendering it. See
[contact sheet](../../.superpowers/activity-fade-2026-10-04/fade-contact-sheet.png),
[GIF](../../.superpowers/activity-fade-2026-10-04/fade-demo.gif) and
[samples](../../.superpowers/activity-fade-2026-10-04/fade-samples.json).

Native computer automation is disabled. The updated teacher viewer was reopened
with seed 970000, launcher PID 4408; user confirmation of the new fade is pending.
This review instance uses All activity and opens Display settings while running;
the normal product launch still enables Odor inputs only.
The native window (PID 21396) is responding. Launch recorded a foreground-activation
warning, so automatic foregrounding is not claimed.
Earlier class-switch live acceptance is retained. This change adds no new navigation
timing claim; the prior isolated timing evidence remains historical.

## Problems and next step

Test-first numerical and GPU checks failed against the old envelope, then passed
after repair. A boundary check found a residual 8.34e-26 strength from floating
time subtraction; restoring the expiry tolerance fixed the count boundary without
a visible cutoff. The first broad run passed 87 targeted tests, but its full-suite
process was absent on resumption with only a partial log and no final result.
Its termination cause is unknown; it is not passing evidence. Final suites were
run fresh after the boundary fix and are the results above. Initial logs remain
preserved, not overwritten.

**One next task / Sol Medium:** review the fade in the native viewer together with
remaining selected-scope, pause/reset, cord and navigation checks. No training or
new scientific milestone begins here.
