"""Numerical edge cases for the isolated 3D subsystem."""
import numpy as np
from flybrain.flight3d.contracts import FlightCommand, TrialSpec
from flybrain.flight3d.world import FlightWorld
from flybrain.flight3d.teacher import visual_measurement


def test_behind_flower_is_visible_without_spectator_camera():
    world=FlightWorld(TrialSpec(33,(.3,.2,.1)))
    relative,pixels=visual_measurement(world.observe().eyes)
    assert pixels>0 and relative[1]<0


def test_nectar_occluded_from_below_but_visible_from_above():
    world=FlightWorld(TrialSpec(34,(.3,.3,.12)))
    world.position[:]=(.3,.3,.16)
    top=world.observe().eyes
    world.position[:]=(.3,.3,.08)
    bottom=world.observe().eyes
    nectar=lambda eyes: (eyes[...,0]>.9)&(eyes[...,1]>.5)&(eyes[...,2]<.2)
    assert nectar(top).sum()>0
    assert not nectar(bottom).any()


def test_stem_impact_and_missed_landing_never_count_food():
    world=FlightWorld(TrialSpec(35,(.4,.4,.16)))
    world.position[:]=(.398,.4,.10); world.step(FlightCommand(.5))
    assert world.crashed and not world.food_contact
    world.reset(); world.position[:]=(.43,.4,.165); world.velocity[:]=(0,0,-.05)
    for _ in range(20): world.step(FlightCommand(.45,legs=1))
    assert not world.food_contact and world.dwell==0


def test_repeat_reset_clears_motion_contacts_and_time():
    world=FlightWorld(TrialSpec(36,(.4,.4,.16)))
    initial=world.snapshot()
    for _ in range(4):
        for _ in range(100): world.step(FlightCommand(.7,.2,-.2,.1,1))
        world.reset(); assert world.snapshot()==initial
        np.testing.assert_array_equal(world.specific_force,0)
