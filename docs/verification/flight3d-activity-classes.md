# Independent activity classes — verification 2026-10-04

Implementation and automated/saved-frame verification use one agent and cached
geometry. Independent classes replace exclusive role presets. New launches enable
Odor inputs only. The user confirms revised native class switches work; other live checks remain pending; no training or scientific
assay was performed.

## Behavior and preservation

All 589 cells belong to exactly one display class: 69 odor, 96 vision, 20 optic
flow, 16 motion, 8 taste, 8 touch, 2 odor relays, 88 motor, 41 descending and 241
structural relays. Original roles remain unchanged. Cached direct adjacency includes
the selected cell and deduplicated non-self incoming/outgoing partners, regardless
of outgoing degree, firing, weight or geometry. Class switches intersect that
scope without altering recorder coverage. Sensory parent off/on/mixed behavior,
independent switches, all-on/all-off, invalid selection fallback, disabled scope,
and reset retention are exercised through actual viewer commands.

The existing static owned batches use the same explicit-RGBA GPU texture. Real
rendered pixels check every specified class RGB endpoint with alpha .43, and check
that a same-frame disabled mask returns one texel to idle. Numerical checks cover
200 ms decay, 600 ms expiry, silent owners and unchanged cumulative spike totals.
The legacy optional-argument-free warm behavior remains verified. A separate pixel
test checks actual legend swatches at both requested sizes. Cyan selection is
independent of switches/scope.

Viewer regressions cover no-outgoing cells, class/circuit intersections, silence,
missing exports, cord-only and mixed/deferred branches, replay/new trial, geometry
batch reuse, camera preservation, unchanged snapshots and exact neural voltages.
The existing recorded/unrecorded session checks continue covering teacher, fixed
and disposable frozen-readout modes. At matched simulation times, masked class
colors and complete display frames agree at 30/60/120 display Hz; the tested mask
contains actual firing odor cells, rather than trivially empty activity.

The preservation audit inventoried 2,371 pre-existing source/data/experiment files:
three authorized display files changed; **2,368 protected files remain byte-identical**.
The new helper is additive. Scientific sources/contracts, sensors, neural rules,
all cached source geometry and the interrupted 1,050-update fit remain unchanged.
No experiment fitting, reserved evaluation or continuation ran. The required test
suite still exercises its disposable numerical/learning fixtures. No Git repository
exists, so there is no commit hash.

## Fresh suites

Python 3.12.10 in `.venv`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`. Executed
serially with JUnit output via `PYTEST_ADDOPTS`. No product/test edits followed.

| Exact command (venv Python) | Result | Pytest duration | Exit |
|---|---|---:|---:|
| `python -m pytest -q tests -k flight3d` | 83 passed, 567 deselected, 2 warnings | 80.49 s | 0 |
| `python -m pytest -q` | 650 passed, zero skips, 2 warnings | 144.48 s | 0 |

Warnings are the existing Pygame `pkg_resources` and Torch `jit.script`
deprecations. Wrapper times were 94.08 s and 152.89 s. Commands, exits, logs and
JUnit XML are under `.superpowers/activity-classes-2026-10-04/`:
[targeted.log](../../.superpowers/activity-classes-2026-10-04/review-targeted.log),
[full.log](../../.superpowers/activity-classes-2026-10-04/review-full.log),
[tests-exits.json](../../.superpowers/activity-classes-2026-10-04/review-tests-exits.json).

## Isolated navigation timing

Four serial, separate processes at 1440×900, seed 970000 teacher, Intel Iris Xe.
Cached geometry drained; 12 warmup and 120 measured frames per condition, 60 Hz
pacing. No live FlyBrain process or other check ran concurrently. Command pattern:

```text
python scripts/benchmark_flight3d_viewer.py --view repaired --frames 120 --activity {all,odor} --cord {hidden,shown} --output <new-json-path>
```

| Classes / cord | Paused p95 ms | Running | Orbit | Selection | Visible pulses, every condition |
|---|---:|---:|---:|---:|---:|
| All / hidden | 3.93 | 8.10 | 4.74 | 11.76 | 359 |
| All / shown | 7.72 | 7.57 | 5.27 | 14.10 | 527 |
| Odor / hidden | 8.24 | 11.53 | 8.76 | 10.72 | 69 |
| Odor / shown | 4.98 | 10.11 | 8.62 | 13.02 | 69 |

All sixteen p95 frame-work values pass **<33 ms**; maximum 14.10 ms. All-mode
setup gates confirm hundreds of actual resident pulses. That setup gate is
intentionally inapplicable to the 69-cell odor-only mode. JSONs separately report
full-model firing and enabled firing min/max for every condition. In the initial
paused frame all modes have **462 latest firing / 530 fired since reset**; enabled
firing is 462 for All and 69 for odor-only. Visible count describes unique resident
owners in visible anatomy layers, not screen pixels or latest-interval firing.

Actual raw timings, counts, renderer metadata and exits:
[all-hidden](../../.superpowers/activity-classes-2026-10-04/review-timing-all-hidden.json),
[all-shown](../../.superpowers/activity-classes-2026-10-04/review-timing-all-shown.json),
[odor-hidden](../../.superpowers/activity-classes-2026-10-04/review-timing-odor-hidden.json),
[odor-shown](../../.superpowers/activity-classes-2026-10-04/review-timing-odor-shown.json),
[timings-exits.json](../../.superpowers/activity-classes-2026-10-04/review-timings-exits.json).
Paced display measured 57.94–58.65 FPS and running simulation 1.013–1.015 simulated
seconds per wall second. These observations do not establish uniformly low latency or steady 60 FPS.
Historical failed shown-cord timings remain in the prior verification record.

## Saved visual review and separate live review

At 1440×900 and 1000×700, saved GPU screenshots cover odor-only launch, expanded
and collapsed settings, scrolled switches, All activity, selected circuit/cyan,
activity off with cyan retained, and both class modes with the cord shown. The
two-column legend precedes help; the right inspector/settings remains scrollable.
Frames and counts are in `.superpowers/activity-classes-2026-10-04/frames/` and
[capture-summary.json](../../.superpowers/activity-classes-2026-10-04/capture-summary.json).

Native computer automation is disabled. Actual native pointer/keyboard/window
review cannot be inferred from injected commands or screenshots. The separate
live reviewer should check independent switches and sensory parent, all/off,
palette/legend, selected scope/cyan, paused recoloring, Space/R/Backspace, cord and
camera navigation. The updated teacher preview was launched with seed 970000,
PID 27060 and feedback requested. This review instance is paused at two simulated
seconds with Display settings already open, so the class buttons and a firing
frame are immediately available. Its launch logged a foreground-activation warning;
native visibility/interaction acceptance still depends on the reviewer. **One
next task / Sol Medium:** obtain and record that review.

## Problems and teaching note

The first viewer fixture assumed a no-outgoing cell would have brain-visible
branches; the manifest/cache places those cells in the cord. The fixture now selects
from full model membership. Saved-frame QA caught invisible legend swatches:
DirectFrame with no relief omits its fill. A real-pixel regression failed against
that state and passed after selecting flat relief. Both failures and fixes are
retained in red/green logs. No unexplained product/test failures remain.

Color is a display calculation, separate from PyTorch's neural state. After 200 ms
without a spike, strength is `exp(-1) ≈ .368`: a soft-idle odor branch interpolates
from RGB (.24,.30,.55) toward (.902,.706,.314), giving approximately
(.484,.449,.463). Switching that class off substitutes zero display strength;
the underlying spikes, neuron voltages and scientific statistics still exist.


### Live-review feedback and discoverability repair

The user reported difficulty locating switches for the other colors. Initial
settings placed them beneath introductory text and shortcuts, outside the initial
viewport. The compact two-column class grid now precedes shortcuts/scope and
uses visible button backgrounds. Every class button is initially visible at both
requested sizes; an actual rendered-layout regression failed against the old
arrangement and passed after repair. The first test version inspected GUI transforms
before the layout task ran; it was corrected to step/render the real GUI before
checking viewport membership. User feedback is preserved; the revised class switches are now user-confirmed. Other native interactions
remain unconfirmed.

Required targeted/full suites and four isolated navigation processes were rerun
after that final product/test change. The fresh evidence linked above uses the
`review-` prefix; initial passing logs/timings and pre-repair frames are retained
under their original names and `frames-before-discoverability/`. Final screenshots
cover expanded/collapsed settings, the top class grid and remaining controls.


### User confirmation

After the discoverability repair and native relaunch, the user answered
**"Yes, the switches work now"** when asked whether another class color can be
turned on/off with the visible controls. This closes class-switch discoverability
and live color-toggle review. It does not establish that the reviewer also tested
Sensory parent, legend comprehension, selected scope/cyan, pause/replay/new trial,
cord or camera navigation. Those remain the one next live-review task / Sol Medium.
The responding native window is PID 25144 (venv launcher PID 27060); the recorded
foreground warning did not prevent this user review.
