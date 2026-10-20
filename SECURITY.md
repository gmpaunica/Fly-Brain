# Security

This is a local educational simulation, not a network service. Use checkpoints,
scientific records and source data from a trusted checkout. Restricted PyTorch
loading and hash receipts reduce risk; they do not make arbitrary downloaded
files safe.

Create a virtual environment and update its build tools before installation:

```powershell
python -m pip install --upgrade "pip>=26.2.2" "setuptools>=84.2.0"
python -m pip install -r requirements.txt
```

`constraints-audited.txt` records the dependency versions reviewed on 2026-10-20
on Windows AMD64 with Python 3.12.10. Other environments need a fresh resolution
and audit.
Pass `-c constraints-audited.txt` when installing optional extras too. The torch
pin omits its local `+cpu` suffix, allowing the matching official CPU wheel.
An audit applies to those versions and the advisories available at that time;
rerun it when dependencies change. Virtual environments are excluded from Git.

Keep credentials in local environment variables or ignored `.env` files. Never
commit tokens, private keys or personal datasets. Run Gitleaks before publishing
new content. Preserve exact source/data bytes: scientific receipts depend on
them, and `.gitattributes` disables automatic line-ending conversion.

Report a suspected vulnerability privately to the repository owner. Do not open
a public issue containing credentials or other private information. Rotate any
credential that has been exposed, even after removing it from Git.

See [the initial publication review](docs/verification/github-publication.md)
for the checks actually performed and their limitations.
