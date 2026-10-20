# Switchable neuron activity classes — 2026-10-04

The user's implementation specification supersedes the exclusive group presets
and All-activity launch default proposed below. Independent display switches now
control colored activity; every neuron remains simulated and recorded. One agent,
cached geometry, no training. Recommended implementation effort: Sol Medium.

## Current display contract

Fresh launches enable **Odor inputs** only, in **Whole model** scope. Display
settings contains independent class switches and an expandable Sensory group.
All ten switches appear at the top in two columns without scrolling, with visible
button backgrounds. Scope and other settings follow below them. This layout
addresses the user's live-review feedback that other class controls were hard to find.
Its parent reports off/on/mixed; clicking off or mixed enables all six children,
while on disables them. Expansion changes no switch state. Counts come from the
fixed manifest, without hardcoded UI membership.

| Display class | Manifest roles | Current count | Activation RGB |
|---|---|---:|---|
| Odor inputs | odor | 69 | #E6B450 |
| Vision | vision | 96 | #86D97A |
| Optic flow | optic_flow | 20 | #3A9D6D |
| Motion sensing | motion | 16 | #E889B5 |
| Taste | taste | 8 | #E75C5C |
| Touch | touch | 8 | #C7DC66 |
| Odor relays | odor_projection | 2 | #C59BE8 |
| Motor | wing_power, wing_steering, wing_tension, landing_motor | 88 | #F18A48 |
| Descending cells | flight_power_dn, flight_steering_dn, landing_dn | 41 | #B378D9 |
| Structural relays | structural_corridor | 241 | #D7D7DB |

These are display categories based on model assignments, not newly established
biological classifications. Original roles stay intact. **All activity** enables
all classes and Whole model. **Activity off** disables all while retaining scope
and history. Selected circuit includes the selected modeled cell and direct
non-self incoming/outgoing partners, without ranking or truncation. Class switches
apply within that membership. Selected circuit is disabled without a modeled
selection. Clearing selection or choosing a background-only cell returns to Whole
model and retains switches. Replay/new trial retain choices and reset history;
fresh launches have no persisted preferences.

The two-column legend leads the left information panel, followed by navigation
help. Each entry has a swatch and on/off state; disabled entries dim. Blue/violet
is idle anatomy, cyan is independent inspection. Full-model firing, enabled/scoped
firing and visible colored pulses are separate counts. Glow off never means
neuronal silence. Missing/cord-only geometry is explained, with an explicit Show
nerve cord action that neither automatically reveals anatomy nor changes framing.

`activity_classes.py` caches complete/disjoint membership and direct adjacency.
A frozenset of enabled classes produces an immutable tuple mask intersected with
scope. `BranchActivityTable` accepts optional RGB endpoints and mask, preserving
original warm behavior for callers omitting both. Both upload/display tokens
include the mask; paused switches recolor immediately using existing static owned
batches and explicit RGBA upload. Existing brightness, camera, geometry, inspector
statistics and scientific snapshots remain unchanged. The user's later fade
request supersedes the initial exponential pulse envelope: strength now eases
smoothly from one to zero over .600 s using `(1-u)^2*(1+2u)` for
`u=clip(age/.600,0,1)`. Active endpoint alpha remains .43. New spikes refresh color,
pause freezes the fade, and the separate 200 ms firing-rate average is unchanged.

Verification covers exact categories, sensory parent states, independent/all/off
switches, scope intersections, no-outgoing/silent/missing/cord-only cells,
mixed/deferred branches, replay, unchanged neural state, 30/60/120 Hz equality,
rendered palette/alpha/channel ordering and actual legend swatches. Saved layouts
at 1440×900 and 1000×700, fresh required suites and isolated All/odor hidden/shown
timings are recorded in the [verification report](../verification/flight3d-activity-classes.md).

**One next task / Sol Medium:** separate live review of switches, colors, legend,
selection, pause, replay/new trial, cord and navigation. Saved frames/injected
commands do not replace it. The interrupted 1,050-update fit remains incomplete;
no learning task begins here.

---

## Historical exclusive-mode proposal — superseded

**Status:** Planning only. The user approves the existing full-firing view as an
option and selected a defined circuit for a smaller alternative. This proposal
does not authorize a new scientific experiment or claim the mode selector exists.
One agent, Sol Medium recommended. No product code changed for this follow-up.

## Intended behavior

Add an **Activity view** control within the existing Display settings:

- **All activity:** the current uncapped firing display, retained exactly.
- **Focused circuit:** warm activity only for the selected modeled neuron and its
  direct non-self incoming/outgoing modeled partners. Membership is deduplicated,
  fixed by retained topology, and independent of current firing or edge strength.
  Without a modeled selection, explain that a neuron must be selected; do not
  silently invent a subset. Clear selection returns to All activity.
- Within Focused circuit, offer **Model role** presets using exact existing
  manifest roles, such as wing steering (24 cells), wing power (24), visual
  inputs (96), odor inputs (69), and landing motor (28). These are justified by
  declared model membership, not by a special recommended number. Counts were
  read from the current cached `data/flight3d/circuit.json`; compute them at runtime,
  never hardcode the count. Label roles as model assignments rather than proven
  biological mediation. A role preset needs no selected cell.

Keep All activity as the launch default to preserve the accepted view. Selection
updates the focused neighborhood when that source is active; it does not move
the camera. Cyan still denotes selection. Unfocused branches keep blue/violet
idle color, preserving spatial context and existing brightness. Visible loaded
branches, missing exports and cord visibility retain their current meanings.
The selected neighborhood can be large: show its exact membership count rather
than quietly truncating it. The mode is a way to inspect a circuit, not proof
that its members caused the fly's behavior.

Show separate counts for full-model latest firing, focus membership, firing
within the focus, and resident warm owners. State that activity outside the focus
is still recorded. Inspector statistics stay complete in every mode.

This design follows a query-based exploration approach: neuPrint supports common
connectivity and shortest-path queries, while FlyWire Codex offers queried-cell
partners and annotation grouping. Those tools justify explicit membership as an
exploration pattern; they do not prescribe a universal highlight count. Sources:
[neuPrint tutorials](https://neuprint.janelia.org/help/videos),
[FlyWire Codex FAQ](https://codex.flywire.ai/faq).

## Implementation outline after design review

1. Add a pure display-only membership helper in
   `src/flybrain/flight3d/activity_focus.py`. Consume manifest IDs/edges/roles and
   return owner indices for All, selected/direct partners or one exact role.
   Ignore self-edges for partner expansion and deduplicate shared roles/partners.
2. Extend `BranchActivityTable.update` in `branch_activity.py` with an optional
   immutable membership mask and a focus revision in its upload cache token.
   Compute all-neuron pulses from the existing frame, then keep excluded owners
   idle. No geometry rebuilding, per-active-cell overlays or recorder changes.
3. Add mode/source controls to `brain_controls.py` and integrate membership/counts
   in `anatomical_viewer.py`. Selection, clear, loading, brightness and cord
   changes invalidate only the necessary display caches. Missing/source-only
   selections explain their limitation. Preserve camera and panel scroll.
4. Regress exact neighbor/role membership, a >12-member focus, no-outgoing cells,
   self-edges, silence, missing geometry, mixed/deferred cord, paused mode switches,
   same-time color invalidation, replay and all-neuron cumulative count invariance.
   Verify focused owner colors through the GPU, plus 30/60/120 Hz matched times.
5. Run `python -m pytest -q tests -k flight3d`, then fresh `python -m pytest -q`;
   repeat isolated 1440×900 hidden/shown timings in both activity modes, preserving
   the current full-mode benchmark. Review live mode changes, cyan selection,
   Space/V/R/Backspace and navigation. Update records without advancing training.

**Review:** Every subset has an explicit membership rule; All remains available;
silent and omitted activity are distinguishable; every neuron remains recorded;
no scientific contracts, source cache, dynamics or interrupted experiment change.
Implementation is pending the user's requested plan review.
