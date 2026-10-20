from dataclasses import asdict
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from flybrain.escape_repair.session import source_hashes
from flybrain.escape_repair.validation import ART
from flybrain.escape_recovery.config import CandidateConfig
from flybrain.escape_flight_validation.runner import preflight as previous_preflight, source_hashes as previous_sources
from flybrain.escape_diagnostic_recovery.runner import verify_hashes, write
from flybrain.flight3d.anatomy import digest

for name in ('targeted', 'full'):
    suite = ET.parse(ART / f'{name}.xml').getroot().find('testsuite')
    assert suite is not None and int(suite.attrib['tests']) > 100
    assert all(int(suite.attrib[k]) == 0 for k in ('failures', 'errors', 'skipped'))
old = previous_preflight()
candidate = CandidateConfig.create()
protected = old['protected'].copy()
for directory in ('experiments/escape-flight-validation-2026-10-08', 'artifacts/escape-flight-validation-2026-10-08'):
    protected.update({p.as_posix(): digest(p) for p in Path(directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts})
verify_hashes(protected)
bound = {**dict(candidate.source_sha256), **previous_sources(), **source_hashes(),
         'Launch Escape.cmd': digest(Path('Launch Escape.cmd'))}
proofs = {str(ART / name): digest(ART / name) for name in ('targeted.xml', 'full.xml', 'viewer-preflight.png', 'viewer-check.py')}
proofs['tests/test_escape_repair.py'] = digest(Path('tests/test_escape_repair.py'))
write(ART / 'preflight.json', dict(status='passed', candidate=asdict(candidate),
    bound=bound, proofs=proofs, protected=protected, repair_sources=source_hashes(),
    original_direction='94/96 accepted previously; original gate still failed',
    original_physical='19/24 failed; preserved', optimizer_updates=0))
print(f'Preflight sealed: {len(protected)} protected files; one fixed 96-case evaluation permitted.')
