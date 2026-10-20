# Escape diagnostic execution ledger — 2026-10-07

Binding spec: the user's escape recovery handoff, preserved in
`artifacts/escape-diagnostic-2026-10-07/request.md`.

1. Passive instrumentation, bounded synthetic verification and read-only preflight.
2. One unchanged-model run: the 18 ordered historical cases, 27 seconds,
   54,000 individual kernel steps across the two brains; no retry.
3. Saved-only audit, diagnostic report and one repair decision; stop for approval.

Pre-flight interfaces: recorder -> runner supplies complete 1 ms tensors and
original diagnostics; runner -> audit supplies compressed interval chunks,
immutable configuration, graph, source archives and output receipt.

Ruling: this folder has no Git repository. Use additive package and fresh output
directories in the authorized workspace; no branch, worktree or commit.
Ruling: user explicitly requires one agent and excludes a full simulation suite.
Use local acceptance review and only bounded synthetic checks plus offline audits.
Ruling: preserve the entire 2,720-file snapshot (including historical launchers).
Inspection confirms it contains no status/decision/experiment documentation;
the requested documentation updates do not conflict with its byte locks.
Hash all protected inputs at preflight and completion, hash executable sources
each interval, and check protected size/mtime each interval (rehash any change).

Task 1: complete. 10 preflight tests pass, final 6 serialization/infrastructure checks pass; 1,747/2,000 cumulative synthetic steps. Original audit and 2,510 preservation /2,720 snapshot hashes pass. Zero scientific presentations.
Task 2: closed incomplete at first baseline interval. One start, zero completed cases/threat intervals, 0.02 s /40 individual steps. Runtime tuple/list comparison produced a false pathway mismatch. Receipt sealed failed; no retry.
Task 3: complete as a failure handoff. Guarded partial audit verifies all 20 paired baseline steps and exact saved-original equality; hypothesis unresolved. Tables and saved timeline reviewed; one comparator-only recovery proposal prepared, approval pending. No neural repair can be selected from absent threat measurements.

Local acceptance review: the stage-1 tests missed runtime tuple/list versus JSON comparison. Numerical parity remains verified, but the scientific gate stopped on the first false mismatch. Do not treat this as completed diagnosis or successful escape. Proposed correction is not implemented; preserve the receipt-bound v1 sources.
