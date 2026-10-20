# Understanding FlyBrain and discussing it in interviews

Standing user goal, recorded 2026-10-12: finish the UI, then understand the
whole project well enough to discuss its implementation, algorithms and results
as an AI-assisted project. This is part of AGENTS.md and applies to future
sessions. Writing this roadmap does not establish that any topic is understood.

## How we will work

One manageable topic per session. Read the relevant real code together, trace a
small example, predict an output or state change, then make a small exercise or
explain the design in the user's own words. Record demonstrated understanding
and remaining questions here. Never substitute memorized interview answers for
understanding. Start with the current demo; work backward through its dependencies,
then cover the earlier 2D app, training experiments and supporting scripts.

For every distinct script, identify its purpose, entry points, dependencies,
inputs/outputs, units and shapes, persistent state, algorithm, failure modes and
tests. Build a file-by-file study checklist as each subsystem is covered, including
scripts, tests and configuration. Archived source copies should be linked to
their originals and any real differences, rather than taught as new algorithms.

## Sequence

1. **Run and map the app.** Start at `escape_ui/__main__.py`: verified candidate
   loading, viewer construction, Panda3D event loop and shutdown. Follow the
   inherited viewer/session chain. Explain the UI, physics worker and immutable
   display snapshot, including why drawing must not advance simulation time.
2. **Python and simulation.** Modules, classes, dataclasses, type hints, arrays,
   threads and worker barriers. Fixed timesteps, seeds, reset/replay, coordinate
   conventions, forces/velocity/orientation, collision and sensory observations.
   Follow one input event all the way to an updated body and displayed frame.
3. **Rendering and interaction.** Pygame/SDL in the original 2D app; Panda3D,
   scene graphs, cameras, projection, DirectGUI, picking and source anatomy in
   the current 3D app. Explain activity colors and what their brightness means.
4. **Neural forward computation.** Sensory encoding, sourced versus synthetic
   graphs, connection direction and signs, tensor ordering/shapes, neural state,
   time units, leak/threshold/reset, spike propagation and paired reference
   simulation. Trace one concrete numerical timestep before a larger tensor.
5. **Behavior and control.** Trace the threat measurement, temporal paired-spike
   turn decision, IMU estimate, authored turn/flight/braking/hover logic and flower
   teacher. Explain which outputs come from neural computation and which come
   from engineered rules. Review the fixed-start calibration and its limits.
6. **Machine learning.** Follow the historical 2D teacher data and readout
   training code: forward pass, loss, autograd graph, gradients, optimizer
   updates, frozen topology, checkpoints and reload. Calculate one tiny update
   by hand and compare trained, untrained and fixed baselines. Cover failed
   experiments as well as the accepted result.
7. **Evidence and reproducibility.** Training/validation/final splits, previously
   inspected benchmarks, confounds, metrics, gates, budgets, saved audits,
   source-bound receipts and exact replay. Read actual configs and result files;
   distinguish tests, visual review and scientific measurements.
8. **Complete the script inventory and interview practice.** Cover remaining
   utilities, historical launchers, tests, dependencies and experiment tooling.
   Give a short project explanation and a deeper technical walkthrough; answer
   questions about design tradeoffs, failures, limitations and AI assistance.

## Facts the walkthrough must keep distinct

- The current demo renders with **Panda3D**, while the earlier 2D app uses
  **Pygame**. They are different applications with shared project history.
- **PyTorch** performs tensor calculations. Our neuron equations and much of the
  behavior are authored rules; running tensors does not itself establish learning.
- The current 589-cell model uses a sourced BANC subset with modeled dynamics.
  Earlier synthetic networks and this sourced subset must be described separately.
- Current edge escape has a neural turn decision and engineered flight/hover.
  Opening the demo does not train it. Its 48 intact successes concern the declared
  protocol; the historical 94/96 directional result still has two failures.
- The accepted 2D learned readout and its measured food approach-and-stop results
  are separate from the current 3D behavior and the failed 3D learning prerequisite.
- AI assistance is part of the development process. Record actual human and AI
  contributions; claim only the knowledge and work the user can substantiate.

## Learning record

- 2026-10-12: goal and sequence recorded. Comprehensive walkthrough not started;
  no topic has been marked understood.
- Next session: lesson 1, trace launch -> source verification -> viewer -> worker
  -> snapshot -> rendered frame. Preferred Sol Medium; one agent. End with the
  user explaining why a UI redraw cannot count as a neural timestep.
