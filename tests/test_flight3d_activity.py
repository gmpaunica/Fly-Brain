"""Completed-step display history is independent of drawing and neural state."""
from dataclasses import replace, FrozenInstanceError
import math
import random
import numpy as np
import pytest
import torch


def snapshot(time=0., counts=(0, 0)):
    from flybrain.flight3d.world import FlightWorld, make_trial
    return replace(FlightWorld(make_trial(970000)).snapshot(), time=time,
                   neuron_ids=tuple(str(i) for i in range(len(counts))), spike_counts=counts)


def test_recorder_counts_every_interval_and_keeps_prior_frames_immutable():
    from flybrain.flight3d.activity import SpikeRecorder, pulse_strengths
    recorder = SpikeRecorder(snapshot())
    first = recorder.observe(snapshot(.02, (3, 0)))
    second = recorder.observe(snapshot(.04, (0, 2)))
    assert first.cumulative_counts == (3, 0)
    assert second.cumulative_counts == (3, 2)
    assert second.snapshot.spike_counts == (0, 2)
    assert second.last_active_times == (.02, .04)
    assert second.step_index == 2
    assert first.firing_rates_hz[0] == pytest.approx(150*(1-math.exp(-.1)))
    assert second.firing_rates_hz[0] == pytest.approx(first.firing_rates_hz[0]*math.exp(-.1))
    np.testing.assert_allclose(pulse_strengths(second), (13456/13500, 1))
    with pytest.raises(FrozenInstanceError):
        first.step_index = 9
    reset = recorder.reset(snapshot())
    assert reset.episode_id == first.episode_id+1 and reset.step_index == 0
    assert reset.cumulative_counts == (0, 0) and reset.last_active_times == (None, None)
    assert not pulse_strengths(reset).any()


@pytest.mark.parametrize('age,expected', [(0, 1), (.2, 20/27),
                                         (.599, 899/108000000), (.6, 0), (.8, 0)])
def test_pulse_cutoff_and_never_fired(age, expected):
    from flybrain.flight3d.activity import SpikeRecorder, pulse_strengths
    recorder = SpikeRecorder(snapshot())
    frame = recorder.observe(snapshot(.02, (1, 0)))
    frame = replace(frame, snapshot=replace(frame.snapshot, time=.02+age))
    np.testing.assert_allclose(pulse_strengths(frame), (expected, 0), atol=1e-12)


def test_observation_validation_does_not_mutate_history():
    from flybrain.flight3d.activity import SpikeRecorder
    recorder = SpikeRecorder(snapshot())
    frame = recorder.observe(snapshot(.02, (1, 0)))
    for bad in [snapshot(.02), snapshot(.01), snapshot(float('nan')),
                snapshot(.04, (-1, 0)), snapshot(.04, (1.5, 0)),
                replace(snapshot(.04), neuron_ids=('x', 'y')),
                replace(snapshot(.04), seed=2)]:
        with pytest.raises(ValueError): recorder.observe(bad)
        assert recorder.frame is frame


@pytest.mark.parametrize('mode', ['teacher', 'fixed', 'trained'])
def test_recorded_stepping_preserves_all_scientific_state_and_rng(mode, tmp_path):
    from flybrain.flight3d.activity import SpikeRecorder
    from flybrain.flight3d.simulation_pump import SimulationPump
    from flybrain.flight3d.session import FlightSession
    from flybrain.flight3d.world import make_trial
    torch.set_num_threads(1)
    checkpoint=None
    if mode=='trained':
        from flybrain.flight3d.neural import FlightReadout,MotorFeatures,save_readout
        from flybrain.flight3d.anatomy import load_manifest,CACHE,digest
        checkpoint=tmp_path/'disposable-unfitted-readout.pt'
        save_readout(checkpoint,FlightReadout(MotorFeatures(load_manifest()).sizes),digest(CACHE/'circuit.json'))
    ordinary = FlightSession(make_trial(970000), mode, checkpoint)
    recorded = FlightSession(make_trial(970000), mode, checkpoint)
    initial_parameters=({key:value.clone() for key,value in recorded.controller.model.state_dict().items()}
                        if mode=='trained' else {})
    pump = SimulationPump(recorded, SpikeRecorder(recorded.snapshot))
    try:
        for operation in (None, 'replay', 'new_trial'):
            if operation:
                getattr(ordinary, operation)()
                pump.reset(getattr(recorded, operation))
            for _ in range(5):
                ordinary.advance()
                py_rng = random.getstate(); np_rng = np.random.get_state(); torch_rng = torch.get_rng_state()
                pump.advance_fixed(1)
                assert random.getstate() == py_rng
                after = np.random.get_state()
                assert np_rng[0] == after[0] and np_rng[2:] == after[2:]
                np.testing.assert_array_equal(np_rng[1], after[1])
                assert torch.equal(torch.get_rng_state(), torch_rng)
                assert recorded.snapshot == ordinary.snapshot
                for name in ('voltage', 'previous', 'phase', 'history', 'weights', 'pre', 'post'):
                    assert torch.equal(getattr(recorded.brain, name), getattr(ordinary.brain, name))
                for name in ('previous_eyes', 'last_counts'):
                    np.testing.assert_array_equal(getattr(recorded.brain, name), getattr(ordinary.brain, name))
                assert recorded.brain.steps == ordinary.brain.steps
                np.testing.assert_array_equal(recorded.transmission_trace.values, ordinary.transmission_trace.values)
                assert recorded.brain.edge_transmissions == ordinary.brain.edge_transmissions
                for key,value in initial_parameters.items():
                    assert torch.equal(recorded.controller.model.state_dict()[key],value)
                    assert torch.equal(ordinary.controller.model.state_dict()[key],value)
                # Teacher's embodied state is also untouched by observation.
                if mode == 'teacher':
                    assert vars(recorded.controller).keys() == vars(ordinary.controller).keys()
                    for key, value in vars(ordinary.controller).items():
                        np.testing.assert_equal(vars(recorded.controller)[key], value)
    finally: pump.close()


def test_pump_publication_skipped_displays_pause_replay_and_failure():
    from flybrain.flight3d.activity import SpikeRecorder
    from flybrain.flight3d.simulation_pump import SimulationPump
    class Intervals:
        def __init__(self): self.snapshot = snapshot(); self.fail = False
        def advance(self, command=None):
            if self.fail: raise ValueError('interval failed')
            self.snapshot = snapshot(self.snapshot.time+.02, (1, 2))
            return self.snapshot
        def replay(self): self.snapshot = snapshot()
    session = Intervals(); pump = SimulationPump(session, SpikeRecorder(session.snapshot))
    try:
        pump.request(.1); pump.wait_idle()
        frame = pump.display_frame
        assert frame.cumulative_counts == (5, 10) and frame.step_index == 5
        assert frame.snapshot is session.snapshot
        pump.pause(); pump.request(.1); pump.wait_idle()
        assert pump.display_frame is frame
        pump.reset(session.replay)
        assert pump.display_frame.episode_id == frame.episode_id+1
        assert pump.display_frame.cumulative_counts == (0, 0)
        pump.advance_fixed(2)
        assert pump.display_frame.cumulative_counts == (2, 4)
        pump.resume(); session.fail = True; frame = pump.display_frame
        pump.request(.02)
        with pytest.raises(RuntimeError, match='worker failed'): pump.wait_idle()
        assert pump.display_frame is frame
    finally: pump.close()
