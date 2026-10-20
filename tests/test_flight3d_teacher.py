"""A sensory-only teacher must infer control from pixels and local feedback."""

import numpy as np
import pytest
from flybrain.flight3d.contracts import FlightCommand, TrialSpec
from flybrain.flight3d.world import FlightWorld
from flybrain.flight3d.teacher import FlightTeacher, visual_measurement


def test_visual_measurement_changes_with_target_elevation():
    low=FlightWorld(TrialSpec(1,(.3,.45,.04))).observe()
    high=FlightWorld(TrialSpec(1,(.3,.45,.18))).observe()
    assert visual_measurement(low.eyes)[0][2]<0<visual_measurement(high.eyes)[0][2]
    assert FlightTeacher().command(high).power>FlightTeacher().command(low).power


def test_inertial_sensor_can_integrate_own_motion_without_pose_input():
    w=FlightWorld(TrialSpec(2,(.5,.5,.1)))
    t=FlightTeacher()
    t.command(w.observe())
    for _ in range(5):
        w.step(FlightCommand(power=.55,pitch=.2,yaw=.1))
        t.command(w.observe())
    assert np.allclose(t.orientation,w.orientation,atol=1e-10)
    assert np.allclose(t.velocity,w.velocity,atol=1e-10)


def test_teacher_stops_on_real_food_contact_and_resets_history():
    w=FlightWorld(TrialSpec(2,(.3,.3,.06)))
    w.position[:]=(.3,.3,.063); w.velocity[:]=(0,0,-.01)
    w.step(FlightCommand(power=.5,legs=1))
    t=FlightTeacher(); c=t.command(w.observe())
    assert c.power==0 and c.legs==1
    t.reset()
    assert np.array_equal(t.velocity,np.zeros(3))
