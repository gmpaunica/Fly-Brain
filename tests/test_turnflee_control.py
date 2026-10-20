"""Exercise spike-only gating, exact ties, grounded flight and IMU recovery."""
import importlib
import math
import numpy as np
import pytest
from flybrain.flight3d.anatomy import load_manifest
from flybrain.flight3d.contracts import FlightObservation
from flybrain.retreat3d.controller import KinematicEstimate
from flybrain.retreat3d.threat import ThreatMeasurement


def api():
    assert importlib.util.find_spec('flybrain.turnflee.control'), 'spike escape control required'
    return importlib.import_module('flybrain.turnflee.control')


def fixture():
    m=api(); manifest=load_manifest(); scores=m.WingScores(manifest)
    ids=[n for n in manifest['nodes'] if set(n['roles']) & {'wing_power','wing_steering'}]
    def vector(side,cell='i1'):
        v=np.zeros(48,dtype=np.int64)
        v[next(i for i,n in enumerate(ids) if n['metadata']['cell_type']==cell and n['metadata']['side']==side)]=1
        return v
    obs=FlightObservation(0.,0.,np.zeros((2,24,48,3)),(0.,0.,0.),(0.,0.,9.81),True,False)
    return m,scores,vector,obs


def test_wing_signs_and_power_spikes_cannot_steer():
    m,s,v,o=fixture()
    assert s(v('left')).left==1 and s(v('right')).right==1
    assert s(v('left','b1')).right==1 and s(v('right','b1')).left==1
    power=np.zeros(48); power[0]=100
    # Explicitly use a power endpoint, independently of its manifest rank.
    wings=[n for n in load_manifest()['nodes'] if set(n['roles'])&{'wing_power','wing_steering'}]
    power[:]=0; power[next(i for i,n in enumerate(wings) if 'wing_power' in n['roles'])]=100
    assert s(power).strength==0


def test_strong_event_waits_for_spikes_but_expires_at_500ms():
    m,s,v,o=fixture(); e=KinematicEstimate(); p=m.EscapePolicy(31)
    assert p.command(e,o,s(np.zeros(48)),ThreatMeasurement(strong=True)) is None
    for _ in range(24): assert p.command(e,o,s(np.zeros(48)),ThreatMeasurement()) is None
    assert p.command(e,o,s(v('left')),ThreatMeasurement()) is not None
    assert p.side=='left'
    p.reset()
    p.command(e,o,s(np.zeros(48)),ThreatMeasurement(strong=True))
    for _ in range(25): p.command(e,o,s(np.zeros(48)),ThreatMeasurement())
    assert p.command(e,o,s(v('left')),ThreatMeasurement()) is None


def test_grounded_trigger_retains_turn_across_neural_gaps():
    m,s,v,o=fixture(); e=KinematicEstimate(); p=m.EscapePolicy(32)
    command=p.command(e,o,s(v('right')),ThreatMeasurement(strong=True))
    assert command.power>.5 and command.yaw>0 and command.legs==1
    assert p.target_heading==pytest.approx(math.pi/2)
    for _ in range(30):
        command=p.command(e,o,s(np.zeros(48)),ThreatMeasurement())
        assert command.yaw>0 and p.side=='right'
    e.orientation[2]=math.pi/2; e.altitude=.1
    command=p.command(e,o,s(np.zeros(48)),ThreatMeasurement())
    assert p.state=='fleeing' and command.pitch>0


def test_bilateral_tie_is_seeded_and_reset_reproducible():
    m,s,v,o=fixture(); e=KinematicEstimate(); p=m.EscapePolicy(33)
    tied=s(v('left')+v('right'))
    p.command(e,o,tied,ThreatMeasurement(strong=True)); side=p.side
    assert p.symmetry_breaking and side in ('left','right')
    p.reset(); p.command(e,o,tied,ThreatMeasurement(strong=True))
    assert p.side==side
    p.reset(); assert p.command(e,o,s(np.zeros(48)),ThreatMeasurement(strong=True)) is None


def test_weak_braking_waits_for_both_speed_components():
    m,s,v,o=fixture(); e=KinematicEstimate(); p=m.EscapePolicy(34)
    p.command(e,o,s(v('left')),ThreatMeasurement(strong=True))
    e.orientation[2]=-math.pi/2; e.altitude=.1; e.velocity[:]=(.08,0.,.03)
    p.command(e,o,s(np.zeros(48)),ThreatMeasurement(weak_continuous=True))
    assert p.state=='braking'
    e.velocity[:]=(.01,0.,.03)
    p.command(e,o,s(np.zeros(48)),ThreatMeasurement())
    assert p.state=='braking'
    e.velocity[:]=(.01,0.,.01)
    assert p.command(e,o,s(np.zeros(48)),ThreatMeasurement()) is None
    assert p.state=='idle' and p.recovered


def test_home_guard_hysteresis_uses_only_imu_estimate():
    m,s,v,o=fixture(); e=KinematicEstimate(); p=m.EscapePolicy(35)
    p.command(e,o,s(v('right')),ThreatMeasurement(strong=True)); e.orientation[2]=math.pi/2; e.altitude=.1
    e.displacement[:2]=(.23,0.)
    p.command(e,o,s(np.zeros(48)),ThreatMeasurement())
    assert p.home_guard and p.desired_velocity[0]<0 and p.desired_velocity[1]<0
    e.displacement[0]=.2; p.command(e,o,s(np.zeros(48)),ThreatMeasurement())
    assert p.home_guard
    e.displacement[0]=.17; p.command(e,o,s(np.zeros(48)),ThreatMeasurement())
    assert not p.home_guard
