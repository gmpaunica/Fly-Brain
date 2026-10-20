"""New response logs bind dynamics; old logs reconstruct historical behavior."""
import importlib
import json
import numpy as np
import pytest
import torch
from flybrain.flight3d.anatomy import load_manifest
from flybrain.turnflee.neural import ResponseConfig
from flybrain.retreat3d.session import RetreatSession
from flybrain.retreat3d.pump import InteractionPump
from flybrain.flight3d.activity import SpikeRecorder


def api():
    assert importlib.util.find_spec('flybrain.turnflee.session'), 'response-bound session required'
    return importlib.import_module('flybrain.turnflee.session')


def test_new_log_replay_resets_state_pauses_at_exact_endpoint(tmp_path):
    m=api(); s=m.TurnFleeSession(41,ResponseConfig.create(load_manifest(),(2,1)))
    p=InteractionPump(s,SpikeRecorder(s.snapshot))
    try:
        p.advance_fixed(4); p.reset(lambda:s.spawn('predator',(.3,.38,.04)))
        p.advance_fixed(15); before=p.display_frame
        path=tmp_path/'actions.json'; s.log.save(path)
        record=json.loads(path.read_text()); assert record['schema']=='turnflee-actions-v2'
        assert record['response_config']['gains']==[2,1]
        voltage=s.brain.voltage.clone()
        p.replay_interaction(); after=p.display_frame
        assert after.snapshot==before.snapshot and torch.equal(s.brain.voltage,voltage)
        assert after.cumulative_counts==before.cumulative_counts
        assert after.last_active_times==before.last_active_times
        assert after.firing_rates_hz==before.firing_rates_hz
        p.request(.1); p.wait_idle(); assert p.display_frame==after
        restored=m.TurnFleeSession.from_log(path)
        assert restored.snapshot==before.snapshot
        assert s.snapshot.response_config.gains==(2,1)
        assert s.snapshot.spike_counts==tuple(s.brain.last_counts)
        record['response_config']['gains']=[4,2]
        path.write_text(json.dumps(record))
        with pytest.raises(ValueError): m.TurnFleeSession.from_log(path)
    finally: p.close()


def test_old_logs_keep_original_brain_and_endpoint(tmp_path):
    m=api(); old=RetreatSession(42)
    for _ in range(3): old.advance()
    path=tmp_path/'old.json'; old.log.save(path)
    replay=m.TurnFleeSession.from_log(path)
    assert type(replay) is RetreatSession and replay.snapshot==old.snapshot


def test_spawn_retains_imu_but_clears_all_response_state():
    m=api(); s=m.TurnFleeSession(43,ResponseConfig.create(load_manifest(),(4,2)))
    for _ in range(20): s.advance()
    estimate=s.controller.idle.estimate; pose=s.snapshot.position
    s.controller.escape.pending_until=100; s.controller.escape.side='left'
    s.spawn('predator',(.3,.45,.1))
    assert s.controller.idle.estimate is estimate and s.snapshot.position==pose
    assert s.controller.escape.pending_until==-1 and s.controller.escape.side is None
    assert not s.pair.window and s.brain.steps==0 and s.pair.reference.steps==0
    assert s.snapshot.pathway is None and not any(s.snapshot.wing_increment)
    s.new_trial(); assert s.snapshot.time==0 and s.log.response_config==s.config


def test_no_default_candidate_without_final_verification(tmp_path,monkeypatch):
    module=api(); monkeypatch.setattr(module,'RUN',tmp_path)
    assert module.promoted_config() is None
    (tmp_path/'receipt.json').write_text(json.dumps(dict(status='closed-calibration-failed',sha256={})))
    assert module.promoted_config() is None
