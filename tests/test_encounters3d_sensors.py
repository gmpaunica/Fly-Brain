"""Real geometry and RGB boundary checks for exclusive encounters."""
import math
import numpy as np
import pytest


def test_exclusive_scene_and_rgb_visibility_expansion():
    from flybrain.encounters3d.scene import SceneSpec, encounter_eyes
    from flybrain.encounters3d.neural import threat_contrast
    origin=np.array((.3,.3,.1)); angles=np.zeros(3)
    for angle in (0,math.pi/2,math.pi,3*math.pi/2):
        delta=np.array((math.sin(angle),math.cos(angle),0))
        far=SceneSpec(7,'predator',tuple(origin+.15*delta))
        near=SceneSpec(7,'predator',tuple(origin+.07*delta))
        a=threat_contrast(encounter_eyes(origin,angles,far))
        b=threat_contrast(encounter_eyes(origin,angles,near))
        assert a.max()>0 and b.sum()>a.sum()
    flower=SceneSpec(7,'flower',(.3,.45,.1))
    assert not threat_contrast(encounter_eyes(origin,angles,flower)).any()
    # Environment test puts the camera below the floor: nearest floor hides predator.
    assert not threat_contrast(encounter_eyes(np.array((.3,.3,-.1)),angles,far)).any()
    with pytest.raises(ValueError): SceneSpec(1,'both',(.3,.4,.1))


def test_world_contains_only_selected_object_and_observation_has_no_identity():
    from flybrain.encounters3d.scene import SceneSpec, EncounterWorld
    from flybrain.flight3d.contracts import FlightObservation
    from flybrain.encounters3d.neural import threat_contrast
    for kind in ('flower','predator'):
        world=EncounterWorld(SceneSpec(9,kind,(.3,.42,.1)))
        obs=world.observe()
        assert isinstance(obs,FlightObservation)
        assert not hasattr(obs,'kind') and not hasattr(obs,'position')
        assert len(world.objects)==1 and world.objects[0].kind==kind
        assert (obs.odor_left>0)==(kind=='flower')
        assert bool(threat_contrast(obs.eyes).any())==(kind=='predator')


def test_predator_collision_and_background_rejection():
    from flybrain.encounters3d.scene import SceneSpec, EncounterWorld
    from flybrain.encounters3d.neural import threat_contrast
    from flybrain.flight3d.contracts import FlightCommand
    world=EncounterWorld(SceneSpec(1,'predator',(.3,.42,.1)))
    world.position[:]=(.3,.42,.1)
    world.step(FlightCommand(.5))
    assert world.crashed and not world.food_contact
    rgb=np.array([(.035,.055,.09),(.11,.17,.085),(.09,.38,.12),(.82,.12,.48),(.98,.69,.06)])
    assert not threat_contrast(rgb).any()
