from dataclasses import asdict
import numpy as np
from flybrain.flight3d.session import ActivityTrace, FlightSession
from flybrain.flight3d.world import make_trial


def test_activity_fades_in_simulation_time_and_replay_clears():
    trace=ActivityTrace(('a','b'))
    trace.update(.02,(1,0))
    np.testing.assert_allclose(trace.values(.22),(np.exp(-1),0))
    np.testing.assert_array_equal(trace.values(.22),trace.values(.22))
    trace.update(0,(0,0)); assert not trace.values(0).any()


def test_camera_free_session_replay_and_new_trial():
    session=FlightSession(make_trial(10001),'teacher',simulate_brain=False)
    first=[session.advance() for _ in range(25)]
    session.replay()
    assert first==[session.advance() for _ in range(25)]
    session.new_trial(); assert session.world.time==0
    assert session.world.case!=make_trial(10001)
    assert session.world.snapshot().position==(.3,.3,.1)
