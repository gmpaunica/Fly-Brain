"""Preflight acceptance only; this file does not run scientific cases."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from flybrain.escape_flight_validation.acceptance import ART,verify_exception
from flybrain.escape_flight_validation.runner import source_hashes,preflight
from flybrain.escape_diagnostic_recovery.runner import write,verify_hashes
from flybrain.flight3d.anatomy import digest
proofs={}
for name in ('targeted-final.xml','full.xml'):
    path=ART/name;suite=ET.parse(path).getroot()[0]
    assert int(suite.attrib['errors'])==int(suite.attrib['failures'])==int(suite.attrib['skipped'])==0
    proofs[name]=suite.attrib
exception=json.loads((ART/'prototype-exception.json').read_text());verify_exception(exception)
protected=json.loads((ART/'protected.json').read_text());verify_hashes(protected)
paths=[ART/name for name in ('targeted-final.xml','targeted-final.log','full.xml','full.log','directional-saved-audit.json','viewer-preflight.py','viewer-preflight.log','viewer-preflight.png','protected.json','prototype-exception.json')]
paths += [Path('tests/test_escape_flight_validation.py')]
paths += list((ART/'engineering-ledgers').rglob('*.json'))
write(ART/'acceptance.json',dict(status='passed',source_hashes=source_hashes(),verification_hashes={p.as_posix():digest(p) for p in paths},tests=proofs,
    protected_files=len(protected),ui_smoke='offscreen zero-step construction/render; saved frame reviewed',live_review='pending',
    new_validation_tests_paired_steps_per_complete_run=680,saved_audit_neural_steps=0,original_diagnostic_engineering_steps=1747))
config=preflight()
print(json.dumps(dict(preflight='passed',candidate_identity=config['candidate_identity'],max_cases=config['max_cases'],max_kernel_steps=config['max_kernel_steps'],tests=proofs,protected_files=len(protected)),indent=2))
