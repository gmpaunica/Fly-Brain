"""Behavior regressions for sustained flight, edge hover and honest delivery."""
import importlib
import math
import numpy as np
import pytest

from flybrain.retreat3d.controller import KinematicEstimate
from flybrain.retreat3d.threat import ThreatMeasurement
from flybrain.retreat3d.world import RetreatWorld
from flybrain.turnflee.control import WingResponse
from flybrain.flight3d.contracts import FlightObservation


def api(module='control'):
    name = 'flybrain.escape_edge.' + module
    assert importlib.util.find_spec(name), f'{name} implementation required'
    return importlib.import_module(name)


def observation():
    return FlightObservation(0., 0., np.zeros((2, 24, 48, 3)),
                             (0., 0., 0.), (0., 0., 9.81), False, False)


def trigger(policy, estimate):
    return policy.command(estimate, observation(), WingResponse(right=1., strength=1.),
                          ThreatMeasurement(strong=True))


def test_weak_threat_cannot_cancel_turn_or_flight_or_enable_home_guard():
    p = api().EdgeEscapePolicy(1)
    e = KinematicEstimate()
    e.altitude = .1
    trigger(p, e)
    p.command(e, observation(), WingResponse(), ThreatMeasurement(weak_continuous=True))
    assert p.state == 'turning'
    e.orientation[2] = math.pi/2
    p.command(e, observation(), WingResponse(), ThreatMeasurement(weak_continuous=True))
    assert p.state == 'fleeing' and p.desired_velocity[0] == pytest.approx(.12)
    e.displacement[0] = .23
    p.command(e, observation(), WingResponse(), ThreatMeasurement())
    assert p.desired_velocity[0] > 0 and not p.home_guard
    assert p.endpoint == pytest.approx([.285, 0.])


def test_endpoint_hover_latches_and_reset_clears_complete_escape_state():
    p = api().EdgeEscapePolicy(1)
    e = KinematicEstimate()
    e.altitude = .1
    trigger(p, e)
    e.orientation[2] = math.pi/2
    e.displacement[:2] = p.endpoint
    for _ in range(400):
        p.command(e, observation(), WingResponse(right=2., strength=2.),
                  ThreatMeasurement(strong=True, weak_continuous=True))
    assert p.state == 'edge_hover' and p.trigger_count == 1
    assert not p.recovered and np.linalg.norm(p.desired_velocity[:2]) < 1e-9
    p.reset()
    assert p.state == 'idle' and p.endpoint is None and p.trigger_count == 0


def motor_episode(state, angle, *, start=(.3, .3), heading=0., seconds=20.):
    """Real physics with a synthetic neural impulse: motor evidence only."""
    from dataclasses import asdict
    world = RetreatWorld(42)
    controller = api().EdgeEscapeBehavior(42)
    controller.calibrated_start(state)
    world.position[:2] = start
    world.orientation[2] = heading
    e = controller.idle.estimate
    e.displacement[:2] = np.array(start) - .3
    e.orientation[2] = heading
    if state != 'resting':
        world.position[2] = .1
        world.foot_contact = False
        if state == 'cruising':
            world.velocity[1] = .06
    # The estimate is supplied by this isolated calibrated motor test, never by
    # an actual session handoff. Runtime updates use only IMU observations.
    controller.encounter('predator', world.observe())
    side = 'left' if angle <= 180 else 'right'
    rows = []
    retinal = np.zeros((2,24,48,3),dtype=np.float32)
    for tick in range(round(seconds/.02)):
        # Motor-only checks use real IMU/contact from physics. Rendering the
        # retina adds no evidence when the triggering impulse is synthetic.
        obs = FlightObservation(0.,0.,retinal,tuple(world.angular_velocity),
                                tuple(world.specific_force),world.foot_contact,False)
        impulse = WingResponse(left=side == 'left', right=side == 'right', strength=1.) if tick == 14 else WingResponse()
        before = world.position.copy(), world.orientation.copy()
        threat = ThreatMeasurement(strong=tick == 14, weak_continuous=tick > 15)
        command = controller.respond(obs, impulse, threat)
        world.step(command)
        p = controller.escape
        rows.append(dict(pre_position=before[0].tolist(), pre_orientation=before[1].tolist(),
            position=world.position.tolist(), orientation=world.orientation.tolist(), velocity=world.velocity.tolist(),
            predator_position=[.3, .375, .021], foot_contact=world.foot_contact,
            collided=world.crashed, behavior_state=controller.state, escape_state=p.state,
            recovered=p.recovered, triggers=p.trigger_count, selected_side=p.side,
            endpoint_estimate=None if p.endpoint is None else p.endpoint.tolist(),
            estimated_displacement=e.displacement.tolist(), desired_velocity=p.desired_velocity.tolist(),
            braking_reason=p.braking_reason, threat=asdict(threat), end_seconds=(tick+1)*.02))
    return world, controller, rows


@pytest.mark.parametrize('state', ['resting', 'hovering', 'cruising'])
@pytest.mark.parametrize('angle', range(0, 360, 45))
def test_real_motor_escape_reaches_edge_and_hovers_twenty_seconds(state, angle):
    world, controller, rows = motor_episode(state, angle)
    assert not any(r['collided'] for r in rows)
    assert controller.state == 'edge_hover'
    distances = [min(*r['position'][:2], .6-r['position'][0], .6-r['position'][1]) for r in rows[-100:]]
    assert min(distances) >= .005 and max(distances) <= .025
    assert max(np.linalg.norm(r['velocity'][:2]) for r in rows[-100:]) <= .01
    assert max(abs(r['position'][2]-.1) for r in rows[-100:]) <= .015
    assert next(r['end_seconds'] for r in rows if r['escape_state'] == 'edge_hover') <= 10
    assert controller.escape.trigger_count == 1
    if state == 'resting':
        assert any(r['position'][2] >= .008 and not r['foot_contact'] for r in rows)


@pytest.mark.parametrize('start,heading', [((.45,.42), .7), ((.12,.15), -.7),
    ((.3,.3), math.pi/4), ((.3,.3), -math.pi/4)])
def test_off_center_encounter_and_corner_approach(start, heading):
    world, controller, rows = motor_episode('hovering', 270, start=start, heading=heading)
    assert not any(r['collided'] for r in rows)
    assert controller.state == 'edge_hover'
    assert min(*world.position[:2], .6-world.position[0], .6-world.position[1]) == pytest.approx(.015, abs=.004)
    controller.encounter('none', world.observe())
    assert controller.escape.state == 'idle' and controller.escape.endpoint is None


def test_short_session_replay_and_source_tampering(tmp_path):
    import json
    from flybrain.escape_recovery.config import CandidateConfig
    m = api('session')
    s = m.EdgeSession(1110000, CandidateConfig.create(), 'resting')
    s.spawn('predator', (.3, .375, .021))
    for _ in range(4):
        s.advance()
    path = tmp_path/'actions.json'
    s.log.save(path)
    replay = m.EdgeSession.from_log(path)
    assert replay.snapshot == s.snapshot
    assert s.log.schema == 'escape-edge-actions-v1'
    s.new_trial()
    assert s.controller.escape.endpoint is None and s.brain.steps == 0
    r = json.loads(path.read_text())
    r['edge_sources'][next(iter(r['edge_sources']))] = 'bad'
    path.write_text(json.dumps(r))
    with pytest.raises(ValueError, match='binding'):
        m.EdgeSession.from_log(path)


def test_metrics_reject_early_stop_inward_reversal_and_forged_hover():
    m = api('metrics')
    _, _, rows = motor_episode('hovering', 270, seconds=12)
    assert m.edge_metrics({'state':'hovering'}, .3, 'right', rows)['physical_success']
    import copy
    forged = copy.deepcopy(rows)
    for r in forged:
        if r['escape_state'] == 'edge_hover':
            r['position'][:2] = [.3, .3]
    assert not m.edge_metrics({'state':'hovering'}, .3, 'right', forged)['physical_success']
    forged = copy.deepcopy(rows)
    i = next(i for i, r in enumerate(forged) if r['escape_state'] == 'fleeing') + 30
    forged[i]['position'][0] -= .02
    assert not m.edge_metrics({'state':'hovering'}, .3, 'right', forged)['physical_success']
    forged = copy.deepcopy(rows)
    forged[i]['velocity'][:2] = [0., 0.]
    assert not m.edge_metrics({'state':'hovering'}, .3, 'right', forged)['physical_success']


def test_delivery_fails_closed_and_requires_every_case(tmp_path):
    m = api('delivery')
    assert m.verified_candidate(tmp_path) is None
    (tmp_path/'delivery.json').write_text('{"status":"failed"}')
    assert m.verified_candidate(tmp_path) is None
    protocol = api('validation')
    cases = protocol.cases('development')
    rows = [dict(**c, success=c['intervention']=='intact', triggers=int(c['intervention']=='intact')) for c in cases]
    assert protocol.gate(cases, rows)
    assert not protocol.gate(cases, rows[:-1])
    rows[0]['success'] = False
    assert not protocol.gate(cases, rows)


def test_independent_motion_audit_rejects_bad_hover():
    import copy
    metrics, audit = api('metrics'), api('audit')
    _, _, rows = motor_episode('hovering',270,seconds=12)
    a = metrics.edge_metrics({'state':'hovering'},.3,'right',rows)
    b = audit.reconstruct_motion({'state':'hovering'},.3,'right',rows)
    assert a == b and b['physical_success']
    forged = copy.deepcopy(rows)
    forged[-1]['velocity'][0] = .03
    assert not audit.reconstruct_motion({'state':'hovering'},.3,'right',forged)['physical_success']


def test_runner_seals_reservation_failure_without_neural_steps(tmp_path,monkeypatch):
    import json
    from dataclasses import asdict
    from flybrain.escape_recovery.config import CandidateConfig
    from flybrain.escape_diagnostic_journal import ReservationJournal
    validation, session = api('validation'), api('session')
    art = tmp_path/'art'
    art.mkdir()
    bound = session.source_hashes()
    (art/'preflight.json').write_text(json.dumps(dict(status='passed',bound=bound,proofs={},protected={},
        edge_sources=bound,candidate=asdict(CandidateConfig.create()))))
    class BrokenBudget(ReservationJournal):
        def reserve_interval(self):
            raise RuntimeError('pre-neural reservation fault')
    monkeypatch.setattr(validation,'ReservationJournal',BrokenBudget)
    with pytest.raises(RuntimeError,match='reservation fault'):
        validation.run('development',tmp_path/'candidate',art)
    value = json.loads((tmp_path/'candidate/development/results.json').read_text())
    assert value['reserved_kernel_steps']==0 and value['status']=='failed'
    assert (tmp_path/'candidate/development-seal.json').exists()
    with pytest.raises(FileExistsError):
        validation.run('development',tmp_path/'candidate',art)


def test_short_exact_record_replay_rejects_forged_position(tmp_path):
    from flybrain.escape_recovery.config import CandidateConfig
    from flybrain.escape_diagnostic_journal import ReservationJournal
    v, replay = api('validation'), api('replay')
    case = v.cases('development')[4]
    s = v.setup_case(case,CandidateConfig.create())
    rows = [v.record_interval(s,case,i) for i in range(2)]
    path = tmp_path/'actions.json'
    s.log.save(path)
    budget = ReservationJournal(tmp_path/'budget.json',120)
    try:
        replay.verify_case(path,case,rows,budget)
        rows[0]['position'][0] += .001
        with pytest.raises(ValueError,match='replay mismatch'):
            replay.verify_case(path,case,rows,budget)
        assert budget.steps==120
    finally:
        budget.close()


def test_viewer_predator_spawn_remains_valid_at_the_edge():
    from flybrain.encounters3d.scene import PREDATOR_RADIUS
    m = api('viewer')
    assert hasattr(m,'review_predator_position'), 'edge-safe review spawn required'
    for pos in ((.015,.3,.1),(.585,.585,.1),(.3,.3,.003)):
        target = m.review_predator_position(pos)
        assert all(PREDATOR_RADIUS <= v <= limit-PREDATOR_RADIUS
                   for v,limit in zip(target,(.6,.6,.3)))
        assert np.linalg.norm(np.array(target)-pos) >= .05


def test_paused_capture_runs_frame_tasks_without_advancing_neurons(tmp_path):
    from flybrain.escape_recovery.config import CandidateConfig
    m = api('viewer')
    assert hasattr(m.EdgeViewer,'capture_paused'), 'normal-frame zero-step capture required'
    app = m.EdgeViewer(1110000,CandidateConfig.create(),offscreen=True)
    try:
        before = app.session.snapshot
        app.capture_paused(tmp_path/'frame.png')
        assert (tmp_path/'frame.png').stat().st_size > 10000
        assert app.session.snapshot == before
        assert app.session.brain.steps == app.session.pair.reference.steps == 0
        assert app.session.world.time == 0 and app.paused
    finally:
        app.destroy()
