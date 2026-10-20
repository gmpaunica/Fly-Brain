"""Read saved gate evidence without rerunning historical scientific records."""
import json
from pathlib import Path
import numpy as np
from flybrain.flight3d.anatomy import CACHE, digest

RUN=Path('experiments/encounters3d-characterization-2026-10-05')


def test_characterization_closes_gate_and_records_nonexclusive_dnp03_contribution():
    config=json.loads((RUN/'config.json').read_text())
    result=json.loads((RUN/'results.json').read_text())
    assert config['circuit_sha256']==digest(CACHE/'circuit.json')
    assert config['network_cells']==589 and config['network_edges']==4139
    assert config['optimizer_updates']==0 and config['checkpoint'] is None
    physical=[r for r in result['rows'] if 'scene' in r]
    assert len(physical)==48 and all(r['motor_spikes']==0 for r in physical)
    assert not result['directional_escape_supported']
    full={r['intervention']:r for r in result['rows'] if 'scene' not in r}
    assert full['intact']['motor_spikes']>full['DNp03-silent']['motor_spikes']>0
    assert full['visual-silent']['motor_spikes']==0
    assert full['DNp03-silent']['DNp03_spikes']==[0,0]
    assert all(r['visual_spikes']==0 and not any(r['counts']) for r in physical if r['intervention']=='visual-silent')
    # Identical images are paired across interventions; all source cells are retained.
    for label in {r['label'] for r in physical}:
        rows=[r for r in physical if r['label']==label]
        assert len({r['eyes_sha256'] for r in rows})==1
        assert all(len(r['counts'])==589 for r in rows)


def test_closed_loop_geometry_reports_hover_without_claiming_retreat():
    result=json.loads((RUN/'closed_loop_results.json').read_text())
    assert len(result['rows'])==24 and not result['avoidance_supported']
    for row in result['rows']:
        assert row['duration_seconds']==5
        assert row['separation_change_m']==0 and row['final_speed_m_per_s']==0
        assert not row['collision'] and row['escape_response_delay_seconds'] is None
        assert row['first_horizontal_command_seconds'] is None
