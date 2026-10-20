"""Numerical instrumentation parity and actual substep measurements."""
import math
import numpy as np
import pytest
import torch
from flybrain.flight3d.anatomy import load_manifest
from flybrain.flight3d.neural import FlightBrain
from flybrain.encounters3d.neural import EscapeBrain
from flybrain.encounters3d.scene import EncounterWorld, SceneSpec
from flybrain.encounters3d.diagnostics import DiagnosticEscapeBrain, DiagnosticFlightBrain


@pytest.mark.parametrize('original,instrumented', [(EscapeBrain, DiagnosticEscapeBrain),
                                                  (FlightBrain, DiagnosticFlightBrain)])
def test_instrumented_stepping_is_bit_identical(original, instrumented):
    manifest = load_manifest()
    a, b = original(manifest), instrumented(manifest)
    rng = np.random.default_rng(20261005)
    for intervention in ('intact', 'vision', 'DNp03', 'DNp26', 'both'):
        a.reset(); b.reset()
        silent = {n['id'] for n in a.nodes if
                  (intervention == 'vision' and 'vision' in n['roles']) or
                  n['metadata'].get('cell_type') in
                  ({'DNp03', 'DNp26'} if intervention == 'both' else {intervention})}
        a.silenced = silent; b.silenced = silent
        for steps in (1, 2, 20, 57, 100):
            rates = rng.uniform(0, 240, len(a.ids))
            np.testing.assert_array_equal(a.advance_rates(rates, steps), b.advance_rates(rates, steps))
            for field in ('voltage', 'previous', 'phase', 'history'):
                assert torch.equal(getattr(a, field), getattr(b, field)), field
            assert a.edge_transmissions == b.edge_transmissions
            assert a.steps == b.steps
            assert b.diagnostic.spike_counts == tuple(int(x) for x in a.last_counts)
        a.reset(); b.reset()
        for kind in ('flower', 'predator'):
            obs = EncounterWorld(SceneSpec(5, kind, (.32, .43, .1))).observe()
            np.testing.assert_array_equal(a.advance(obs), b.advance(obs))
            assert torch.equal(a.voltage, b.voltage)
            assert torch.equal(a.history, b.history)
            assert a.edge_transmissions == b.edge_transmissions
            np.testing.assert_array_equal(a.previous_eyes, b.previous_eyes)


def test_exact_leak_threshold_reset_delay_currents_and_spike_time():
    b = DiagnosticEscapeBrain(load_manifest())
    d = next(i for i, n in enumerate(b.nodes) if n['metadata'].get('cell_type') == 'DNp03')
    zeros = np.zeros(len(b.ids))
    b.voltage[d] = .5
    b.advance_rates(zeros, 1)
    assert b.diagnostic.peak_voltage[d] == .5 * math.exp(-.01)
    b.reset(); b.voltage[d] = 1.1
    b.advance_rates(zeros, 1)
    assert b.diagnostic.spike_counts[d] == 1
    assert b.diagnostic.first_spike_seconds[d] == .001
    assert float(b.voltage[d]) == 1.1 * math.exp(-.01) - 1
    k = next(k for k, (pre, post) in enumerate(zip(b.pre, b.post))
             if int(post) == d and 'vision' in b.nodes[int(pre)]['roles'])
    pre = int(b.pre[k]); weight = float(b.weights[k])
    b.reset(); b.phase[pre] = .99
    rates = zeros.copy(); rates[pre] = 240
    b.advance_rates(rates, 1)
    assert b.diagnostic.peak_phase[pre] == 1.23
    assert b.diagnostic.first_spike_seconds[pre] == .001
    assert b.diagnostic.current_sum[d] == 0
    b.advance_rates(zeros, 1)
    assert b.diagnostic.current_sum[d] == weight
    assert b.diagnostic.current_peak[d] == weight
    assert b.diagnostic.peak_voltage[d] == weight
    assert math.isclose(sum(b.diagnostic.current_sum), sum(b.edge_transmissions), abs_tol=1e-13)
    b.reset(); b.silenced = {b.ids[d]}; b.previous[pre] = 1
    b.advance_rates(zeros, 1)
    assert b.diagnostic.current_sum[d] == 0
    assert b.diagnostic.discarded_current[d] == weight
    assert b.diagnostic.spike_counts[d] == 0 and float(b.voltage[d]) == 0


def test_completed_intervals_replay_pause_and_frozen_publication():
    from flybrain.encounters3d.session import EncounterSession
    from flybrain.flight3d.simulation_pump import SimulationPump
    from flybrain.flight3d.activity import SpikeRecorder
    s = EncounterSession(SceneSpec(11, 'predator', (.34, .42, .1)))
    assert s.snapshot.pathway is None
    pump = SimulationPump(s, SpikeRecorder(s.snapshot))
    try:
        snapshots = [pump.advance_fixed(1) for _ in range(12)]
        assert len(s.brain.diagnostic_intervals) == 12
        frame = pump.display_frame
        assert frame.snapshot.pathway.end_seconds == .24
        old = s.brain.voltage.clone()
        for _ in range(20): pump.request(.1)
        assert frame is pump.display_frame and torch.equal(old, s.brain.voltage)
        pump.reset(s.replay)
        assert s.snapshot.pathway is None and s.brain.diagnostic_intervals == []
        assert snapshots == [pump.advance_fixed(1) for _ in range(12)]
        pump.reset(lambda: s.spawn('flower'))
        assert s.snapshot.pathway is None and s.brain.diagnostic_intervals == []
    finally: pump.close()
