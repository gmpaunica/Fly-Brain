"""Exclusive trial resets, teacher parity and transparent failed escape gate."""
import numpy as np
from flybrain.encounters3d.scene import make_scene, SceneSpec


def test_flower_preserves_exact_teacher_trajectory_and_neural_activity():
    from flybrain.encounters3d.session import EncounterSession
    from flybrain.flight3d.session import FlightSession
    from flybrain.flight3d.world import make_trial
    old=FlightSession(make_trial(970000),'teacher')
    new=EncounterSession(make_scene(970000,'flower'))
    for _ in range(100):
        a=old.advance(); b=new.advance()
        for field in ('time','position','velocity','orientation','command','food_contact','dwell',
                      'spike_counts','edge_transmissions','edge_activity'):
            assert getattr(a,field)==getattr(b,field)


def test_spawn_new_trial_replay_and_all_controller_state_reset():
    from flybrain.encounters3d.session import EncounterSession
    s=EncounterSession(make_scene(71,'flower'))
    for _ in range(10): s.advance()
    s.spawn('predator')
    assert s.scene.kind=='predator' and s.scene.seed==72
    assert not any(s.snapshot.spike_counts) and s.snapshot.time==0
    assert s.brain.steps==0 and s.brain.previous_eyes is None
    a=[s.advance() for _ in range(60)]
    voltage=s.brain.voltage.clone()
    s.replay()
    assert s.scene.seed==72 and s.brain.steps==0
    assert not s.controller.stabilizer.velocity.any()
    assert a==[s.advance() for _ in range(60)]
    assert (voltage==s.brain.voltage).all()
    s.new_trial()
    assert s.scene.seed==73 and s.scene.kind=='predator'
    s.spawn('flower')
    assert s.scene.seed==74 and s.mode=='teacher'


def test_failed_gate_never_substitutes_rgb_steering_and_hover_is_stable():
    from flybrain.encounters3d.session import EncounterSession
    from flybrain.encounters3d.controller import PredatorController
    from flybrain.flight3d.anatomy import load_manifest
    from flybrain.flight3d.contracts import FlightObservation
    controller=PredatorController(load_manifest())
    assert not controller.avoidance_supported
    commands=[]
    for eyes in (np.zeros((2,24,48,3)),np.ones((2,24,48,3))*[.05,.95,.95]):
        controller.reset()
        commands.append(controller.command(FlightObservation(0,0,eyes,(0,0,0),(0,0,9.81),False,False)))
    assert commands[0]==commands[1] and commands[0].power==.5
    s=EncounterSession(SceneSpec(7,'predator',(.3,.42,.1)))
    for _ in range(250): s.advance()
    assert not s.world.crashed
    np.testing.assert_array_equal(s.world.position,(.3,.3,.1))
    assert 'unavailable' in s.mode


def test_pause_pump_publication_and_render_observation_independence():
    from flybrain.encounters3d.session import EncounterSession
    from flybrain.flight3d.simulation_pump import SimulationPump
    from flybrain.flight3d.activity import SpikeRecorder, pulse_strengths
    s=EncounterSession(make_scene(82,'predator'))
    pump=SimulationPump(s,SpikeRecorder(s.snapshot))
    try:
        pump.advance_fixed(30)
        frame=pump.display_frame; voltage=s.brain.voltage.clone()
        for _ in range(20):
            s.world.observe(); pulse_strengths(frame); pump.request(.1)
        assert frame is pump.display_frame and (voltage==s.brain.voltage).all()
        pump.reset(lambda:s.spawn('flower'))
        assert pump.display_frame.snapshot.object_kind=='flower'
        assert not any(pump.display_frame.cumulative_counts)
        assert pump.display_frame.episode_id!=frame.episode_id
    finally: pump.close()
