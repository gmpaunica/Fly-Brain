# Physical validation verification — 2026-10-08

One-agent execution; no Git repository, commit or source changes to frozen
packages. Authority and prospective thresholds: [execution plan](../plans/ESCAPE-FLIGHT-VALIDATION.md).

Commands use `.venv/Scripts/python.exe`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`.
The scientific worker is a consumed action, not a rerun instruction.

- Targeted: `python -m pytest tests -k "escape_flight_validation or escape_recovery or turnflee or retreat3d" -q -p flybrain.escape_recovery.pytest_plugin --junitxml=artifacts/escape-flight-validation-2026-10-08/targeted-final.xml`.
  **117 passed**, zero skips, 724 deselected, four upstream warning instances;
  60.18 s. Test ledgers use `FLYBRAIN_TEST_LEDGERS=artifacts/escape-flight-validation-2026-10-08/engineering-ledgers/targeted-final`.
- Full: `python -m pytest -q -p flybrain.escape_recovery.pytest_plugin --junitxml=artifacts/escape-flight-validation-2026-10-08/full.xml`.
  **841 passed**, zero skips, four upstream warning instances; 281.86 s.
  Separate `engineering-ledgers/full` output; original diagnostic ledger remains
  1,747 steps and its bytes are hash-preserved.
- Saved original: `flybrain.escape_recovery.audit.audit(Path("experiments/escape-recovery-execution-2026-10-07/run"))`.
  Audit passed; original status remains closed-direction-failed, 94/96, zero
  new neural steps. Saved JSON: `directional-saved-audit.json` in current artifacts.
- UI preflight: `python artifacts/escape-flight-validation-2026-10-08/viewer-preflight.py`.
  New viewer construction/offscreen render passed, world time and both brain
  steps zero. Saved frame visually reviewed. This is not a live interaction
  review, activated app or scientific flight case.

New behavior tests cover exact ordering/576,000-step boundary, parent-scoped
consumption, candidate/exception/file tampering, missing/nonfinite measurements,
stationary/wrong-turn/premature-flee/insufficient-separation/unstable-recovery/
contact examples, JSON/CSV/action storage, exact short intact/silent replay,
child/start/I/O failures, sealing, full synthetic saved audit, fabricated success
and delivery records from a different receipt. The 48-case synthetic audit
fixture advances no neurons; it is explicitly engineering data.

4,858 original files are hash-protected, including consumed scientific records
and engineering proofs. Final preservation and scientific/audit/replay outcomes
will be appended after the one physical run.

Final checks after the saved CSV adapter: `python -m pytest -q tests artifacts/escape-flight-validation-2026-10-08/test_serialized_audit.py -p flybrain.escape_recovery.pytest_plugin --junitxml=artifacts/escape-flight-validation-2026-10-08/post-audit-full.xml`:
**847 passed**, zero skips, four upstream warning instances, **414.03 s**.
Fresh `FLYBRAIN_TEST_LEDGERS=artifacts/escape-flight-validation-2026-10-08/engineering-ledgers/post-audit-full`;
original and preflight proofs are unchanged. Adapter targeted command:
`python -m pytest artifacts/escape-flight-validation-2026-10-08/test_serialized_audit.py -q --junitxml=artifacts/escape-flight-validation-2026-10-08/serialized.xml`:
**6 passed**, zero skips, 0.23 s, zero neural calls. Red logs are retained.

Consumed scientific launch: hidden detached `python -m flybrain.escape_flight_validation.worker`.
Normal child/supervisor exit0; scientific status **closed-physical-failed**,
exactly48 cases/288 s/576,000 individual neural steps. This is not a passing
behavioral result. Intact 19/24 (rest6/8, hover7/8, cruise6/8), silent triggers 0;
max downstream215Hz, state finite. Complete raw archives/seals remain unchanged.

Saved analysis command: `python artifacts/escape-flight-validation-2026-10-08/analyze-saved.py`.
The initial strict audit rejects NumPy scalar-wrapper CSV representation; retain
`analysis.log`. Corrected source-bound saved audit validates exact finite float64
CSV/JSON vector values, then all original independent numerical/gate checks:
**audit-passed**, **zero neural calls**,14,400 intervals. Original source bytes,
gates and receipts are not modified/monkeypatched. Public strict raw audit remains
limited to its expected CSV text representation; corrected read-only command:
`python artifacts/escape-flight-validation-2026-10-08/audit_serialized.py`.

Consumed exact replay launch: hidden `python artifacts/escape-flight-validation-2026-10-08/replay_driver.py --supervise`.
**All 48 cases/14,400 intervals exactly match every recorded field**, including
silent interventions; both child/supervisor exit0. Replay576,000 steps are in a
separate append-only verification journal, **zero new scientific cases**.
Science+replay1,152,000 steps; unit/preflight computation is separate. Repeated
new diagnostic-test ledgers each contain529 explicit steps; protected original
proof stays 1,747. The new physical-validation neural fixtures execute680 paired
steps per complete suite, recorded separately in preflight acceptance metadata.

All three paired 8-direction trajectory figures and representative physical
measurement figure were visually reviewed. Four figure hashes and receipt binding
are in `visual-review.json`; live review pending/unavailable, new app unlaunched.
Final CLI `python -m flybrain.escape_flight_validation --headless --steps 0`
correctly exits1/refuses missing activation. No scientific exposure from refusal.

Final independent receipt/budget/source/evidence check:
[final-verification.json](../../artifacts/escape-flight-validation-2026-10-08/final-verification.json)
passes, including4,858 protected files, unchanged original 1,747-step proof,
science/replay reservation boundaries,847-test XML and disabled activation.
See [measured results](../ESCAPE-FLIGHT-VALIDATION-RESULTS.md) for all five failures.
No skipped verification; live interaction unavailable and conditionally withheld.
No scientific retry, source/model/gain/policy tuning, training, commit or push.
