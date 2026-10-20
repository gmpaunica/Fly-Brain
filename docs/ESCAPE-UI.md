# Explore the fly and its brain

Run from the project root:

```powershell
.\.venv\Scripts\python.exe -m flybrain.escape_ui
```

The modern dark dashboard keeps the world and brain side by side, with the
Groups, Inspect and Help tabs in a persistent right sidebar. It uses the
original verified edge controller; the historical launcher and source-bound
scientific receipts remain unchanged.

## Simulation controls

The top toolbar contains Spawn flower, Spawn predator, Pause/Resume, New trial
and Replay. Primary controls have 46-pixel targets. Space pauses/resumes; R starts
a fresh no-object trial; Backspace reconstructs the recorded interaction to its
endpoint and stays paused; Esc exits. Replay performs real simulation work.

Spawning replaces the one current object and retains the pause preference.
Resume to see movement if you spawn while paused. Flower flight is teacher-guided.
Predator encounters use neural spikes to select a turn, followed by engineered
IMU flight/braking/edge hover. No online training occurs.

Move the predator with W/S along arena +Y/-Y, A/D along -X/+X and Q/E down/up.
These use arena axes even if the camera rotates. Pause/focus loss releases held
movement input.

## Neuron group checkboxes

All ten project-assigned role groups are visible without scrolling at the two
reviewed sizes, 1000x700 and 1440x900. Each has a real tick, color swatch, readable
name and model-cell count. Clicking a row changes its visibility and explains
that group at the bottom of the sidebar.

| Group | Model cells |
|---|---:|
| Smell | 69 |
| Vision | 96 |
| Visual motion | 20 |
| Body movement sensing | 16 |
| Taste | 8 |
| Touch | 8 |
| Smell relays | 2 |
| Wing and landing control | 88 |
| Movement commands | 41 |
| Other relays | 241 |

**Other relays** are extra intermediate cells retained on selected connection
paths between sensing and movement groups. This is a structural selection label,
not one biological neuron type or proof of each cell's exact function. Think of
them as the intermediate links between the named groups. Their dashboard firing
color is muted steel-blue (#5B8FB9), replacing the overwhelming white; this
changes neither their simulation nor the source-bound historical palette.

Show all and Hide all control every group. Only this keeps the group named in the
description and hides the others; Vision is the initially described group.
Unchecked branches cannot be picked and their selection/connection highlights
are hidden. Inspector information remains available with a hidden-by-filter note.

**Every simulated neuron keeps running and recording.** For example, choosing
only Vision selects 96 cells for display while all 589 remain simulated. This
is a viewing filter, not a neural ablation or a change to the controller.
The swatches identify firing colors; actual recorded spikes drive the pulses.

Faint brain outline is a separate display checkbox, on initially. Include nerve
cord is off initially. The outline is source context rather than another neuron
group. Some cells have nerve-cord branches; enable the cord to see those parts.
Four modeled cells have no source skeleton in the local cache.

## Inspect, Help and navigation

Click a visible branch to open Inspect. It starts with a readable role, source
cell type and recorded activity. Long source IDs, connection counts, provenance
and pathway measurements are behind Technical details. Without a selection,
Inspect offers a paged list of source types/cells in the described group.
Help explains controls and what the visualization can establish.

Drag either viewport to orbit, Shift-drag to pan, and wheel to zoom. Sidebar
interaction does not move the camera behind it. Wheel over Inspect/Help scrolls
that page. Fit visible frames enabled anatomy and visible full selection; when
all groups are hidden, the camera stays stable and Show all groups is offered.
Front/Side/Top change brain orientation. Follow tracks the fly; Reset view in the
world header resets its camera. F follows, B resets brain view, V toggles nerve
cord, and Tab toggles the outline.

Filters, tab, selection and outline/cord preferences persist through spawn,
reset, replay and resizing. Available Windows installations use native Segoe UI
for readable text, with Panda3D's font as a portable fallback. No font is copied
or distributed by the project.

## Captures and evidence

`python -m flybrain.escape_ui --capture preview.png` saves one paused, zero-step
initial frame. For loaded geometry and all review states at both sizes:

```powershell
.\.venv\Scripts\python.exe scripts/review_escape_ui.py artifacts/my-dashboard-review
```

The helper refuses to overwrite frames, captures each size in a fresh process,
and verifies zero simulation/neural steps plus original source preservation.
Saved-frame/programmatic-widget review is separate from native mouse/keyboard
acceptance. See [verification](verification/dashboard-redesign.md),
[scientific results and limits](ESCAPE-EDGE-RESULTS.md), and the
[standing learning roadmap](LEARNING_ROADMAP.md).
