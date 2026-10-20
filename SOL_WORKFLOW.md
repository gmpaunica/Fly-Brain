# Building FlyBrain with Sol

Use **Sol Medium for most tasks** and **Sol High for the few tasks involving neural dynamics, learning, or hard debugging**. The task queue makes that choice explicit. This is an engineering recommendation, not a benchmark proving equal performance or a particular token saving.

## Start the first build

Open a fresh chat in this project, select **GPT-5.6 Sol / Medium**, and send:

```text
Implement FB-01 from docs/plans/M0-M1.md. Follow AGENTS.md and the current project status. Complete Milestones 0 and 1, run the specified checks, and update the project records. Stop before odor sensing. Work with one agent and make routine implementation decisions yourself.
```

The application model/effort picker controls the model for that chat. Editing AGENTS.md or typing a model preference does not itself switch it. The official [model guide](https://learn.chatgpt.com/docs/models) documents the picker and CLI model selection; higher reasoning effort uses more tokens.

If you prefer PowerShell, from this folder:

```powershell
.\scripts\Start-Sol.ps1
.\scripts\Start-Sol.ps1 -Effort high
.\scripts\Start-Sol.ps1 -Preview
```

The launcher opens an interactive Codex CLI session, pins `gpt-5.6-sol` and the requested effort, and supplies a short startup prompt. It changes no persistent configuration and does not override your approval/sandbox settings. It does not alter this existing chat. `-Preview` shows the arguments without launching Codex.

Equivalent direct launch:

```powershell
codex --model gpt-5.6-sol -c 'model_reasoning_effort="medium"'
```

These flags were checked against installed Codex CLI 0.154.0 and the official [CLI reference](https://learn.chatgpt.com/docs/developer-commands?surface=cli) / [configuration reference](https://learn.chatgpt.com/docs/config-file/config-reference). Local model metadata supports both Medium and High. Authenticated model execution and billing were not tested during preparation. Choose the effort explicitly: local metadata currently defaults to Low.

## Normal working cycle

1. Read status to see the active task and recommended effort. Complete one task with its tests.
2. Inspect the demo/result. A passing test suite does not prove the visual presentation or scientific conclusion.
3. Start the next bounded task from the saved handoff. Use a fresh chat after a completed task or a large change of topic; continue the current chat while resolving its failures.
4. Keep the same Sol model; raise effort to High for the designated tasks. Return to Medium for subsequent routine work. No automatic Astra fallback or parallel agents.

Continuation prompt:

```text
Build the next task in PROJECT_STATUS.md. Read its task entry and only the relevant code and references. Finish its acceptance checks and update the status/log. Use one agent, stay within the task, and explain one useful concept from the implementation.
```

Focused High review, when the queue calls for one:

```text
Review the current milestone against its acceptance criteria and docs/architecture.md. Focus on numerical correctness, sensory leakage, state reset, determinism, and (when applicable) valid learning/evaluation. Reproduce suspected failures, fix confirmed in-scope bugs, run the affected checks, and record evidence. Do not redesign working components.
```

## How the project stays economical

`AGENTS.md` is the compact always-needed guide. Status points to one active task; the chronological log preserves evidence. The full original brief is an on-demand reference. `docs/architecture.md` fixes shared assumptions so a later agent need not invent them again. `docs/BUILD_QUEUE.md` describes future gates, not an instruction to implement everything at once.

Do not paste the full brief into every new chat. Do not read every plan, log entry, source file, and skill on startup. Do not create a planning/review/delegation ceremony for each small edit. Run the checks that establish correctness and retain their results.

Measure savings using actual client usage, if exposed, alongside completed acceptance gates and rework. Token count, subscription allowance, cached input, and API dollar cost are different quantities. Smaller prompts and Medium are intended to reduce avoidable work; no saving percentage or hard token cap is promised. Test failures must still be resolved.

## Recovery and handoff

When stuck, preserve the minimal reproduction, exact failing command, observed output, and hypotheses already ruled out in the log. Investigate on High with this focused context rather than asking for a whole-project rewrite. Seek a user decision only for missing facts or a material scope/science tradeoff. An unavailable dependency is an environment problem, not evidence that a more expensive model is needed.

Use this final response shape: **Changed / Verified / Remaining / Next**. Keep the learning explanation to a short paragraph. If a GUI check or experiment could not run, leave that criterion pending in status.

## Current work

The simulator, synthetic neural loop, live brain diagram and evaluation tools
are implemented. The charge readout passed the approved fresh navigation gate
on 2026-09-23. FB-08's intervention experiments and plots are also complete.
FB-09's BANC v888 subset is packaged and connected to the PyTorch LIF core.
The first two-gain dynamics probe failed selectivity and washout; a bounded
dynamics recovery is next. The live app still uses its fixed synthetic decoder. Follow the single
next task in `PROJECT_STATUS.md` and the recorded scope/limits in
`docs/plans/M6-CHARGE-REPAIR.md`. FB-01 is historical;
FB-07K is a superseded proposal, not the next implementation task.
