"""Freeze the candidate only after fresh passing engineering evidence."""
from dataclasses import asdict
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from flybrain.escape_diagnostic_recovery.runner import write, verify_hashes
from flybrain.escape_recovery.config import CandidateConfig
from flybrain.flight3d.anatomy import digest
from flybrain.escape_edge.session import source_hashes

art = Path('artifacts/escape-edge-2026-10-11')
proofs = {}
counts = {}
for name in ('targeted','final-full','spawn','edge'):
    p = art/f'{name}.xml'
    suites = list(ET.parse(p).getroot().iter('testsuite'))
    assert suites and all(int(s.attrib.get(k,0))==0 for s in suites for k in ('failures','errors','skipped'))
    counts[name] = sum(int(s.attrib['tests']) for s in suites)
    assert counts[name]>0
    proofs[p.as_posix()] = digest(p)
frame = json.loads((art/'offscreen.json').read_text())
assert frame['status']=='passed' and frame['neural_steps']==0
for name in ('offscreen.json','viewer-preflight.png'):
    p = art/name
    proofs[p.as_posix()] = digest(p)
protected = json.loads((art/'protected.json').read_text())
verify_hashes(protected)
candidate = CandidateConfig.create()
bound = {p.as_posix():digest(p) for p in sorted(Path('src/flybrain').rglob('*.py'))}
bound.update(dict(candidate.source_sha256))
write(art/'preflight.json',dict(status='passed',candidate=asdict(candidate),bound=bound,
    edge_sources=source_hashes(),protected=protected,proofs=proofs,test_counts=counts,
    historical_direction=dict(successes=94,cases=96,original_gate='failed',known_failures=[1100018,1100021]),
    optimizer_updates=0))
print(json.dumps(dict(status='source-frozen',test_counts=counts,protected_files=len(protected)),indent=2))
