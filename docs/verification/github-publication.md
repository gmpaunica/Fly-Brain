# GitHub publication security review - 2026-10-20

Requested scope: import the entire FlyBrain project into the owner's GitHub
account, review security risks, and preserve existing scientific evidence.
Repository: https://github.com/gmpaunica/Fly-Brain (private).

## Changes and scope

- Initialized a new `main` branch; there was no existing Git history to inspect.
- Included source, tests, documentation, data, archived source and experiment
  records. The initial selection was 6,739 files / 1,586,043,521 bytes; publication
  documentation and audit reports add a few more files.
- Excluded virtual environments, Python/pytest caches, local credentials, private
  keys and machine/editor settings. No secret was added to the remote URL.
- Set Git author identity to the account's GitHub noreply address.
- Disabled line-ending conversion through `.gitattributes` and local Git config;
  exact source/data bytes are part of the scientific receipt contract.
- Added audited dependency constraints and raised the setuptools build floor.
  Enabled GitHub Dependabot vulnerability alerts and automated security fixes
  (both API calls returned HTTP 204), plus weekly dependency update checks.

## Dependency review

`pip-audit 2.10.1` initially reported 16 advisory entries in two local build
tools: pip 25.0.1 and setuptools 78.1.0. Some entries duplicate an advisory.
The revised snapshot uses pip 26.2.2 and setuptools 84.2.0. All other installed
dependency versions are unchanged. No virtual environment is published.

```powershell
python -m pip_audit -r constraints-audited.txt --no-deps --disable-pip --format json
```

Observed: **30 dependencies audited, zero known vulnerabilities, zero skipped
packages**. All dependencies were explicitly supplied, including transitives.
These pins are a version snapshot, not a hash-locked package distribution.
An editable install dry run with `.[dev,neural,flight3d,plots]`, binary external
packages and these constraints also succeeded.

## Source review

Bandit 1.9.4 scanned active `src` and `scripts`, with no scan errors. It reported
zero high, one medium and 107 low findings. The medium B310 finding is
`flight3d/assets.py:62`: its downloader uses a fixed HTTPS Google Cloud Storage
base; neuron IDs are validated numeric strings. It does not expose an arbitrary
URL to a remote client. Low findings cover assertions, seeded simulation RNG
and fixed local subprocess commands. No application network listener was found.

A supplementary AST check parsed all **1,571** Python files, including archives,
with zero parse errors. It found no `eval`, `exec`, `os.system`, pickle/marshal
load calls, `shell=True`, `allow_pickle=True`, or `torch.load` calls missing
explicit `weights_only=True`. This is a pattern check, not a formal proof.

## Secret review and regression evidence

Gitleaks 8.30.1 was downloaded from the official project release and its archive
SHA-256 matched that release's checksum manifest. Default detection rules were
extended only to exclude unpublished environments, Git internals and caches.

One existing interrupted experiment contains a truncated gzip:
`experiments/escape-diagnostic-recovery-2026-10-07/07-1100008/intervals.jsonl.gz`.
Gitleaks reported unexpected EOF while traversing it. A separate zlib recovery
read all 142,040 compressed bytes and extracted 1,771,585 readable bytes; its
gzip end marker is absent. Scanning that recovered text found zero secrets.
The original file remains byte-for-byte unchanged. Missing bytes cannot be
reviewed or reconstructed.

Whole-tree command:

```powershell
gitleaks dir . --config .gitleaks.toml --redact --no-banner --no-color --max-archive-depth 3 --report-format json
```

Observed: **zero detected secrets, 4,496,200,962 bytes scanned, exit 0**, 10m36s.
The truncated-gzip warning and its separate readable-content scan are described
above. A supplementary scan of the new publication/configuration/audit files also
found zero secrets. Machine-readable evidence is kept in
`artifacts/github-publication-2026-10-20`.

Regression command (OMP/MKL threads one, unbuffered Python, fresh
`FLYBRAIN_TEST_LEDGERS` outside historical evidence):

```powershell
python -m pytest -q tests artifacts/escape-flight-validation-2026-10-08/test_serialized_audit.py -p flybrain.escape_recovery.pytest_plugin --junitxml=full.xml
```

Observed: **928 passed**, zero failures/errors/skips, three upstream warnings,
891.09s, exit 0. The archive scan and staging ran concurrently for much of this
check. It completed before the local build-tool update. Only pip and setuptools
were then updated in the excluded local environment; all runtime dependency
versions remain unchanged. The installed 30-package snapshot matches the audited
constraints and a second audit reports zero known vulnerabilities, zero skips.
The full suite did not run a new scientific experiment or reopen consumed assays.

Post-update command:

```powershell
python -m pytest -q tests/test_app.py tests/test_dashboard_ui.py --junitxml=after-tools.xml
```

Observed: **17 passed**, zero skips/failures/errors, one upstream warning,
19.29s, exit 0. These exercise application launch and the real dashboard after
the build-tool update. No new native input or aesthetic review is claimed.
Final remote verification is pending the initial push.

## Limits

This review covers detected secrets, known dependency advisories, selected
unsafe-code patterns and existing regression tests. It cannot establish zero
security risk, certify arbitrary third-party model/data files, predict future
advisories or validate every lower version permitted by package metadata.
Use the audited constraints and rerun checks after changes. Historical source
copies remain evidence, rather than supported alternative applications.

Private visibility limits exposure; it does not replace local scanning. GitHub
secret scanning availability differs by repository/account plan. This review
does not claim that GitHub secret scanning was enabled. See
[GitHub secret-scanning documentation](https://docs.github.com/en/code-security/concepts/secret-security/secret-scanning).
