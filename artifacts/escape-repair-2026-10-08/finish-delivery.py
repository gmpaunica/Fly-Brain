"""Activate only after independently checked scientific and replay receipts."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from flybrain.escape_repair.validation import ART, RUN
from flybrain.escape_repair.session import source_hashes
from flybrain.escape_diagnostic_recovery.runner import verify_hashes, write
from flybrain.flight3d.anatomy import digest

audit = json.loads((ART / 'audit.json').read_text())
replay = json.loads((ART / 'replay.json').read_text())
visual = json.loads((ART / 'visual-review.json').read_text())
config = json.loads((RUN / 'config.json').read_text())
result = json.loads((RUN / 'results.json').read_text())
preflight = json.loads((ART / 'preflight.json').read_text())
assert audit['status'] == 'audit-passed' and audit['experiment_status'] == result['status'] == 'passed'
assert result['cases'] == 96 and result['intact_successes'] == 48 and result['silent_triggers'] == 0
assert replay['status'] == 'exact-replay-passed' and replay['cases'] == 4 and replay['replay_neural_steps'] == 48000
assert visual['saved_review'] == 'reviewed'
receipt_hash = digest(RUN / 'receipt.json')
assert audit['receipt_sha256'] == replay['receipt_sha256'] == visual['receipt_sha256'] == receipt_hash
verify_hashes(preflight['bound'])
verify_hashes(preflight['proofs'])
verify_hashes(preflight['protected'])
for name in ('targeted', 'full'):
    suite = ET.parse(ART / f'{name}.xml').getroot().find('testsuite')
    assert all(int(suite.attrib[k]) == 0 for k in ('failures', 'errors', 'skipped'))
proof_paths = [ART / name for name in ('audit.json', 'replay.json', 'visual-review.json', 'preflight.json', 'targeted.xml', 'full.xml')]
proof_paths += [RUN / 'receipt.json', RUN.parent / 'seal.json']
proofs = {p.as_posix(): digest(p) for p in proof_paths}
proofs.update(visual['figures'])
write(ART / 'delivery.json', dict(status='passed', candidate=config['candidate'],
    repair_sources=source_hashes(), proofs=proofs, receipt=(RUN / 'receipt.json').as_posix(),
    audit=(ART / 'audit.json').as_posix(), replay=(ART / 'replay.json').as_posix(),
    receipt_sha256=receipt_hash, directional_limit='Historical 94/96 exception retained; 1100018/1100021 not repaired or rerun.'))
from flybrain.escape_repair.delivery import verified_candidate
assert verified_candidate() is not None
write(ART / 'final-verification.json', dict(status='passed', preserved_files=len(preflight['protected']),
    science_steps=result['reserved_kernel_steps'], replay_steps=replay['replay_neural_steps'],
    original_neural_candidate_unchanged=True, original_physical_failure_preserved=True,
    full_tests=ET.parse(ART / 'full.xml').getroot().find('testsuite').attrib,
    activation='enabled', optimizer_updates=0))
print('Repaired app activated after scientific, saved audit, exact replay, visual and preservation checks.')
