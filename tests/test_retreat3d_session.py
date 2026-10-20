"""Fixed-step actions, barrier resets and exact interaction replay."""
import importlib
import json
import numpy as np
import pytest
import torch


def api():
    assert importlib.util.find_spec('flybrain.retreat3d.session') is not None
    return importlib.import_module('flybrain.retreat3d.session')


def test_continuous_spawn_neural_origin_and_complete_reset():
    m = api(); s = m.RetreatSession(71)
    for _ in range(25): s.advance()
    before = s.snapshot; estimate = s.controller.idle.estimate
    s.spawn('predator', (.3, .47, .1))
    assert s.snapshot.position == before.position and s.snapshot.velocity == before.velocity
    assert s.controller.idle.estimate is estimate
    assert s.brain.steps == s.pair.reference.steps == 0
    assert not any(s.snapshot.spike_counts) and s.snapshot.pathway is None
    assert s.snapshot.neural_origin_seconds == .5
    s.advance()
    assert s.snapshot.pathway.end_seconds+s.snapshot.neural_origin_seconds == pytest.approx(s.snapshot.time)
    s.new_trial()
    assert s.seed == 72 and s.snapshot.time == 0 and s.world.objects == ()
    assert s.log.steps == [] and s.log.spawns == []
    assert s.brain.steps == s.pair.reference.steps == 0


def test_action_log_replay_all_spawns_commands_and_endpoint_spawn(tmp_path):
    m = api()
    from flybrain.retreat3d.world import PredatorInput
    s = m.RetreatSession(73)
    for _ in range(4): s.advance()
    s.spawn('predator', (.3, .48, .1))
    for _ in range(13): s.advance(PredatorInput.from_keys({'w', 'a', 'e'}))
    s.spawn('flower', (.4, .45, .1))
    for _ in range(3): s.advance()
    s.spawn('predator', (.38, .44, .1))
    expected = s.snapshot; voltage = s.brain.voltage.clone()
    path = tmp_path/'actions.json'; s.log.save(path)
    assert len(json.loads(path.read_text())['steps']) == 20
    s.replay()
    assert s.snapshot == expected and torch.equal(s.brain.voltage, voltage)
    assert len(s.log.steps) == 20 and len(s.log.spawns) == 3
    loaded = m.RetreatSession.from_log(path)
    assert loaded.snapshot == expected


def test_pump_replay_preserves_endpoint_histories_and_pauses():
    m = api()
    from flybrain.retreat3d.pump import InteractionPump
    from flybrain.retreat3d.world import PredatorInput
    from flybrain.flight3d.activity import SpikeRecorder
    s = m.RetreatSession(74); p = InteractionPump(s, SpikeRecorder(s.snapshot))
    try:
        p.advance_fixed(5)
        p.reset(lambda: s.spawn('predator', (.3, .48, .1)))
        p.advance_fixed(12, PredatorInput.from_keys({'d'}))
        before = p.display_frame
        p.replay_interaction()
        after = p.display_frame
        assert after.snapshot == before.snapshot
        assert after.cumulative_counts == before.cumulative_counts
        assert after.last_active_times == before.last_active_times
        assert after.firing_rates_hz == before.firing_rates_hz
        p.request(.1, PredatorInput.from_keys({'w'})); p.wait_idle()
        assert after is p.display_frame
    finally: p.close()


def test_frozen_decoder_is_diagnostic_only_after_failed_scientific_gate():
    m = api(); s = m.RetreatSession(75)
    assert s.decoder is not None
    for _ in range(10): s.advance()
    assert s.snapshot.decoder_norm == np.linalg.norm(s.snapshot.wing_increment)
    assert not s.snapshot.escape_supported
    assert s.snapshot.behavior_state in ('takeoff', 'cruising')
    assert s.decoder.wing_ids == tuple(s.brain.ids[i] for i in s.pair.indices)


def test_spawn_handoff_uses_current_contact_observation_not_previous_tick():
    m = api(); s = m.RetreatSession(76)
    s.world.position[2] = .1; s.world.foot_contact = False
    # Controller still has its initial resting estimate of contact. The event
    # must sample contact only, without transferring our evaluator pose.
    s.spawn('flower')
    assert not s.controller.flower_takeoff
