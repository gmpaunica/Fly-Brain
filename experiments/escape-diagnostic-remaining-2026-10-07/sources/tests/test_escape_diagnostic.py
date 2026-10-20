"""Bounded synthetic checks; never render a scientific threat presentation."""
import importlib
from pathlib import Path
import numpy as np
import pytest
import torch


def api():
    assert importlib.util.find_spec('flybrain.escape_diagnostic'), 'passive package required'
    return importlib.import_module('flybrain.escape_diagnostic.recorder')


def test_kernel_leak_reset_directed_delay_and_all_edge_measurements():
    m = api()
    # A -> B, B -> A, plus a discarded incoming edge to sensory A.
    v = torch.tensor([0., 1.02, .5], dtype=torch.float64)
    p = torch.tensor([1., 0., 0.], dtype=torch.float64)
    ph = torch.tensor([.999, 0., 0.], dtype=torch.float64)
    rates = torch.tensor([2., 0., 0.], dtype=torch.float64)
    pre = torch.tensor([0, 1, 1]); post = torch.tensor([1, 2, 0])
    w = torch.tensor([.3, .2, .4], dtype=torch.float64)
    inputs = torch.tensor([True, False, False]); live = torch.ones(3, dtype=torch.bool)
    m.engineering_reserve('literal dynamics', 2)
    r = m.advance_recorded(v, p, ph, rates, pre, post, w, inputs, live, 2)
    trace = r[11:]
    assert float(trace[0][0, 1]) == 1.02*.9900498337491681+.3
    assert float(trace[1][0, 1]) == float(trace[0][0, 1])-1
    assert float(trace[0][0, 2]) == .5*.9900498337491681
    assert float(trace[0][1, 2]) == float(trace[1][0, 2])*.9900498337491681+.2
    assert trace[2][0].tolist() == [True, True, False]
    assert trace[5][0].tolist() == [.3, 0., 0.]
    assert trace[5][1].tolist() == [.3, .2, .4]
    assert r[4].tolist() == [.6, .2, 0.]
    assert float(r[7][0]) == .4
    m.engineering_reserve('exact threshold equality',1)
    empty=torch.empty(0,dtype=torch.int64)
    threshold=m.advance_recorded(torch.tensor([1/.9900498337491681],dtype=torch.float64),
        torch.zeros(1,dtype=torch.float64),torch.zeros(1,dtype=torch.float64),
        torch.zeros(1,dtype=torch.float64),empty,empty,torch.empty(0,dtype=torch.float64),
        torch.tensor([False]),torch.tensor([True]),1)
    assert threshold[11][0,0]==1 and threshold[13][0,0] and threshold[12][0,0]==0


@pytest.mark.parametrize('steps', [1, 2, 20, 99])
def test_bit_identical_original_measured_and_recorded_kernels(steps):
    m = api()
    from flybrain.flight3d.neural import _advance
    from flybrain.encounters3d.diagnostics import _advance_measured
    rng = np.random.default_rng(2718+steps)
    n, e = 17, 70
    args = (torch.from_numpy(rng.uniform(0, 1.7, n)),
            torch.from_numpy(rng.integers(0, 2, n).astype(np.float64)),
            torch.from_numpy(rng.uniform(0, 1, n)),
            torch.from_numpy(rng.uniform(0, 240, n)),
            torch.from_numpy(rng.integers(0, n, e)),
            torch.from_numpy(rng.integers(0, n, e)),
            torch.from_numpy(rng.uniform(0, .8, e)),
            torch.from_numpy(rng.integers(0, 2, n).astype(bool)),
            torch.from_numpy(rng.integers(0, 2, n).astype(bool)))
    results=[]
    for name, kernel in [('original', _advance), ('measured', _advance_measured), ('recorded', m.advance_recorded)]:
        m.engineering_reserve(f'parity {name} {steps}', steps)
        results.append(kernel(*(a.clone() for a in args), steps))
    for i in range(5):
        assert torch.equal(results[0][i], results[1][i])
        assert torch.equal(results[0][i], results[2][i])
    for i in range(11):
        assert torch.equal(results[1][i], results[2][i])
    audit=importlib.import_module('flybrain.escape_diagnostic.audit')
    arrays={'rgb':np.zeros((2,24,48,3),dtype=np.float32),'step_index':np.arange(1,steps+1)}
    for label in ('actual','reference'):
        for name,tensor in zip(('pre_voltage','post_voltage','spikes','pre_phase','post_phase','contribution'),results[2][11:]):
            arrays[label+'_'+name]=tensor.numpy()
        for name,tensor in zip(('initial_voltage','initial_previous','initial_phase','rates'),args[:4]):
            arrays[label+'_'+name]=tensor.numpy()
    graph={k:t.numpy() for k,t in zip(('pre','post','weights','inputs','live'),args[4:])}
    audited=audit.check_chunk(arrays,graph,1)
    for label in ('actual','reference'):
        for name,i in [('counts',3),('delivered',4),('current_sum',5),('current_peak',6),('discarded_current',7),('peak_voltage',8),('peak_phase',9),('first',10)]:
            assert np.array_equal(audited[label][name],results[2][i].numpy())


def test_passive_brain_interval_parity_reset_and_serialization(tmp_path):
    m = api()
    from flybrain.flight3d.anatomy import load_manifest
    from flybrain.flight3d.contracts import FlightObservation
    from flybrain.turnflee.neural import ResponseConfig, ResponsePair
    manifest=load_manifest(); config=ResponseConfig.create(manifest,(4,2))
    original=ResponsePair(manifest,config); recorded=m.RecordedPair(manifest,config)
    obs=FlightObservation(0.,0.,np.zeros((2,24,48,3),dtype=np.float32),(.1,-.2,.3),(0.,0.,9.81),False,False)
    m.engineering_reserve('pair wrapper parity, two pairs two intervals',160)
    for k in range(2):
        assert np.array_equal(original.advance(obs),recorded.advance(obs))
        for b,c in [(original.actual,recorded.actual),(original.reference,recorded.reference)]:
            for name in ('voltage','previous','phase','history','last_counts'):
                assert np.array_equal(getattr(b,name),getattr(c,name))
            assert b.edge_transmissions==c.edge_transmissions
        assert original.actual.diagnostic==recorded.actual.diagnostic
        arrays=m.chunk_arrays(recorded,obs.eyes)
        assert arrays['salience'].dtype==arrays['growth'].dtype==np.float64
        p=tmp_path/f'{k}.npz'; np.savez_compressed(p,**arrays)
        with np.load(p,allow_pickle=False) as saved:
            for name,value in arrays.items(): assert np.array_equal(saved[name],value)
        assert arrays['step_index'].tolist()==list(range(k*20+1,k*20+21))
    recorded.reset()
    assert recorded.actual.steps==0 and recorded.actual.trace is None
    assert not recorded.actual.phase.any() and not recorded.actual.previous.any()
    assert recorded.actual.diagnostic is None and not recorded.window


def test_consumed_run_tamper_and_budget_guards(tmp_path):
    api()
    r=importlib.import_module('flybrain.escape_diagnostic.runner')
    path=tmp_path/'one'; r.consume(path)
    with pytest.raises(FileExistsError): r.consume(tmp_path/'renamed')
    receipt=r.StepBudget(tmp_path/'budget.json',limit=40)
    receipt.reserve_interval()
    with pytest.raises(ValueError,match='budget'): receipt.reserve_interval()
    assert receipt.steps==40
    p=tmp_path/'protected'; p.write_bytes(b'original')
    bound={str(p):r.sha(p)}; r.verify_hashes(bound)
    p.write_bytes(b'changed')
    with pytest.raises(ValueError,match='hash'): r.verify_hashes(bound)


def test_audit_rejects_voltage_edge_window_and_rgb_tampering(tmp_path):
    m=api()
    audit=importlib.import_module('flybrain.escape_diagnostic.audit')
    # A hand-computable 1-step fixture. No simulated kernel call.
    arrays=dict(step_index=np.array([1]),rgb=np.zeros((2,24,48,3),dtype=np.float32))
    for b in ('actual','reference'):
        arrays.update({f'{b}_{k}':v for k,v in dict(
            initial_voltage=np.array([0.,0.]),initial_previous=np.array([1.,0.]),
            initial_phase=np.array([0.,0.]),rates=np.array([0.,0.]),
            pre_voltage=np.array([[0.,.4]]),post_voltage=np.array([[0.,.4]]),
            spikes=np.array([[False,False]]),pre_phase=np.array([[0.,0.]]),
            post_phase=np.array([[0.,0.]]),contribution=np.array([[.4]])).items()})
    graph=dict(pre=np.array([0]),post=np.array([1]),weights=np.array([.4]),
               inputs=np.array([True,False]),live=np.array([True,True]))
    audit.check_chunk(arrays,graph,1)
    arrays['actual_contribution'][0,0]=.5
    with pytest.raises(ValueError): audit.check_chunk(arrays,graph,1)


def test_exact_window_expiration_cumulative_counts_and_downstream_stop():
    m=api()
    from flybrain.flight3d.anatomy import load_manifest
    from flybrain.turnflee.neural import ResponseConfig
    manifest=load_manifest(); pair=m.RecordedPair(manifest,ResponseConfig.create(manifest,(4,2)))
    a=np.zeros(589,dtype=np.int64); z=a.copy(); a[pair.indices[0]]=1
    pair.record_counts(a,z)
    for _ in range(9): pair.record_counts(z,z)
    assert pair.last_vector[0]==1
    pair.record_counts(z,z)
    assert pair.last_vector[0]==0
    a[:]=0; a[np.flatnonzero(~pair.actual.inputs.numpy())[0]]=5
    for _ in range(10): pair.record_counts(a,z)
    assert pair.max_downstream_hz==250 and not pair.stable


def test_interruption_seals_consumed_run_without_any_neural_steps(tmp_path,monkeypatch):
    api(); r=importlib.import_module('flybrain.escape_diagnostic.runner')
    # Infrastructure fault before brain construction; no mocked neuron behavior.
    monkeypatch.setattr(r,'preflight',lambda: {})
    monkeypatch.setattr(r,'historical_records',lambda: {})
    import json
    directory=tmp_path/'interrupted'
    with pytest.raises(KeyError): r.run(directory)
    receipt=json.loads((directory/'receipt.json').read_text())
    assert receipt['status']=='failed' and receipt['reserved_kernel_steps']==0
    assert not receipt['retry_authorized']
    assert json.loads((tmp_path/'escape-diagnostic-seal.json').read_text())['receipt_sha256']==r.sha(directory/'receipt.json')
    with pytest.raises(FileExistsError): r.consume(tmp_path/'retry')

