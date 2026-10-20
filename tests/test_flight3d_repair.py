"""Numerical regressions for RGB-only acquisition and bounded lost-target search."""

import numpy as np
import pytest
from flybrain.flight3d import sensors
from flybrain.flight3d.teacher import FlightTeacher, visual_measurement
from flybrain.flight3d.world import FlightWorld, make_trial
from flybrain.flight3d.contracts import FlightObservation


def test_historical_thin_flower_is_detected_in_immutable_rgb():
    world = FlightWorld(make_trial(960000, sector=0, height=0))
    observation = world.observe()
    assert observation.eyes.shape == (2, 24, 48, 3)
    assert not observation.eyes.flags.writeable
    assert visual_measurement(observation.eyes) is not None


def test_shared_contrast_rejects_sky_ground_stem_and_accepts_mixed_petals():
    contrast = getattr(sensors, 'flower_contrast', None)
    assert contrast is not None, 'shared RGB contrast is required'
    background = np.array([[.035,.055,.09],[.11,.17,.085],[.135,.20,.105],[.09,.38,.12]])
    assert np.all(contrast(background) == 0)
    petals = np.array([.82,.12,.48])
    assert np.all(contrast((background * 7 + petals) / 8)[:3] > 0)
    assert contrast(np.array([.98,.69,.06])) > 0


def test_lost_target_search_is_bounded_periodic_and_resettable():
    teacher = FlightTeacher()
    world = FlightWorld(make_trial(970000, sector=0, height=0))
    heights = []
    blank = np.broadcast_to(np.array([.035,.055,.09]),(2,24,48,3))
    for _ in range(800):
        obs = world.observe()
        hidden = FlightObservation(obs.odor_left,obs.odor_right,blank,
            obs.angular_velocity,obs.specific_force,False,False)
        world.step(teacher.command(hidden)); heights.append(world.position[2])
    assert max(heights)-.1 <= .016
    assert .1-min(heights) <= .016
    assert max(heights)-min(heights) > .02
    teacher.reset()
    assert teacher.search_elapsed == 0
    assert teacher.search_entry is None


@pytest.mark.parametrize('flower',[(.3,.1,.1),(.30001,.5,.1),(.29999,.5,.1),(.3,.45,.19),(.3,.45,.04)])
def test_image_seams_above_below_and_behind_have_rgb_detection(flower):
    from flybrain.flight3d.contracts import TrialSpec
    world=FlightWorld(TrialSpec(970001,flower))
    measured=visual_measurement(world.observe().eyes)
    assert measured is not None
    expected=np.array(flower)-world.position
    assert np.dot(measured[0],expected)>0


def test_subpixel_occlusion_resolves_stem_before_averaging():
    from flybrain.flight3d.contracts import TrialSpec
    world=FlightWorld(TrialSpec(970001,(.3,.4,.2)))
    # Looking straight into the thick near stem: centre pixels contain green,
    # never a flower hit leaking through it from above.
    world.position[:]=(.3,.395,.1)
    eyes=world.observe().eyes
    middle=eyes[:,11:13,[0,-1]]
    assert np.any(np.all(np.isclose(middle,[.09,.38,.12],atol=1e-6),axis=-1))
    assert np.any((eyes[...,0]>.14)&(eyes[...,0]<.8)), 'partial silhouette RGB must survive averaging'


def test_reacquisition_stops_search_and_replay_reproduces_commands():
    world=FlightWorld(make_trial(970000,sector=0,height=0)); teacher=FlightTeacher()
    blank=np.zeros((2,24,48,3))
    for _ in range(20):
        obs=world.observe()
        hidden=FlightObservation(obs.odor_left,obs.odor_right,blank,obs.angular_velocity,obs.specific_force,False,False)
        world.step(teacher.command(hidden))
    assert teacher.search_entry is not None
    teacher.command(world.observe()); assert teacher.search_entry is None and teacher.search_elapsed==0
    def replay():
        world.reset();teacher.reset();commands=[]
        for _ in range(20):
            cmd=teacher.command(world.observe());commands.append(cmd);world.step(cmd)
        return commands
    assert replay()==replay()
