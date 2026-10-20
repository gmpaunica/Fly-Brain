"""Reproductions for pixel-jitter braking and outward recovery drift."""
import math
import numpy as np
import pytest

from flybrain.escape_repair.threat import ClearanceMonitor as ThreatMonitor
from flybrain.escape_repair.control import SafeEscapeBehavior as EscapeBehavior
from flybrain.turnflee.control import WingResponse
from flybrain.retreat3d.world import RetreatWorld


def jitter_eyes(high):
    """A small cyan patch flickering at a sampled silhouette edge."""
    eyes = np.zeros((2, 24, 48, 3), dtype=np.float32)
    eyes[0, 11:12, 20:22, 1:] = .9 if high else .6
    return eyes


def test_small_stationary_pixel_jitter_eventually_allows_braking():
    monitor = ThreatMonitor()
    values = [monitor.update(jitter_eyes(i % 2)) for i in range(80)]
    assert all(v.size_degrees < 10 for v in values)
    assert values[-1].weak_continuous


def recovery_fixture(controller_type):
    """Literal recorded resting-0 recovery, independent of runtime evaluation."""
    world = RetreatWorld(1110000)
    world.position[:] = (.053, .295, .1)
    world.velocity[:] = (-.017, -.005, 0.)
    world.orientation[:] = (0., 0., -1.7)
    world.foot_contact = False
    controller = controller_type(1110000)
    controller.calibrated_start('hovering')
    e = controller.idle.estimate
    e.displacement[:] = (-.247, -.005, .097)
    e.orientation[:] = world.orientation
    e.velocity[:] = world.velocity
    controller.encounter('predator', world.observe())
    controller.idle.heading = -1.7
    controller.idle.transition('cruising')
    return world, controller


def test_recovered_cruise_does_not_hit_wall_while_turning_home():
    world, controller = recovery_fixture(EscapeBehavior)
    from flybrain.retreat3d.threat import ThreatMeasurement
    for _ in range(100):
        command = controller.respond(world.observe(), WingResponse(), ThreatMeasurement(weak_continuous=True))
        world.step(command)
        assert not world.crashed
    assert .015 < world.position[0] < .585


def test_large_or_sustained_growing_image_does_not_clear():
    monitor = ThreatMonitor()
    eyes = np.zeros((2, 24, 48, 3), dtype=np.float32)
    eyes[:, 8:16, 16:32, 1:] = .9
    for _ in range(30):
        assert not monitor.update(eyes).weak_continuous
    # Increasing occupancy inside a small patch: real RGB inputs, no mocks.
    monitor.reset()
    for level in np.linspace(.06, .89, 20):
        eyes[:] = 0
        eyes[0, 11:12, 20:22, 1:] = level
        assert not monitor.update(eyes).weak_continuous


def test_repaired_clearance_preserves_strong_onset_exactly_and_resets():
    from flybrain.retreat3d.threat import ThreatMonitor as Original
    original, repaired = Original(), ThreatMonitor()
    for i in range(50):
        eyes = jitter_eyes(i % 2)
        if 15 <= i < 30:
            eyes[:, 8:16, 16:32, 1:] = .9
        a, b = original.update(eyes), repaired.update(eyes)
        assert (a.strong, a.size_degrees, a.positive_growth_degrees_s) == (
            b.strong, b.size_degrees, b.positive_growth_degrees_s)
    repaired.reset()
    assert not repaired.update(jitter_eyes(False)).weak_continuous


@pytest.mark.parametrize('state', ['resting', 'hovering', 'cruising'])
@pytest.mark.parametrize('angle', range(0, 360, 45))
def test_motor_escape_completes_with_real_vision_physics_and_synthetic_wing_impulse(state, angle):
    """Motor integration only: does not establish a neural side decision."""
    from dataclasses import asdict
    from flybrain.escape_flight_validation.metrics import physical_metrics
    world = RetreatWorld(42)
    controller = EscapeBehavior(42)
    controller.calibrated_start(state)
    if state != 'resting':
        world.position[2] = .1
        world.foot_contact = False
        if state == 'cruising':
            world.velocity[1] = .06
    az = math.radians(angle)
    world.spawn('predator', (.3 + .075 * math.sin(az), .3 + .075 * math.cos(az), max(.021, world.position[2])))
    controller.encounter('predator', world.observe())
    monitor = ThreatMonitor()
    side = 'left' if angle <= 180 else 'right'
    rows = []
    for tick in range(300):
        observation = world.observe()
        threat = monitor.update(observation.eyes)
        impulse = WingResponse(left=side == 'left', right=side == 'right', strength=1.) if tick == 14 else WingResponse()
        before = world.position.copy(), world.orientation.copy()
        command = controller.respond(observation, impulse, threat)
        world.step(command)
        rows.append(dict(pre_position=before[0].tolist(), pre_orientation=before[1].tolist(),
            position=world.position.tolist(), orientation=world.orientation.tolist(), velocity=world.velocity.tolist(),
            predator_position=list(world.scene.position), foot_contact=world.foot_contact,
            collided=world.crashed, behavior_state=controller.state, escape_state=controller.escape.state,
            recovered=controller.escape.recovered, triggers=controller.escape.trigger_count,
            threat=asdict(threat), end_seconds=(tick + 1) * .02))
    result = physical_metrics(dict(state=state), .3, side, rows)
    assert result['physical_success'], result


@pytest.mark.parametrize('heading', [-1.7, 1.7])
def test_recovery_remains_collision_free_for_twenty_seconds(heading):
    from flybrain.retreat3d.threat import ThreatMeasurement
    world, controller = recovery_fixture(EscapeBehavior)
    if heading > 0:
        world.position[0] = .547
        world.velocity[0] *= -1
        world.orientation[2] = heading
        controller.idle.estimate.orientation[2] = heading
        controller.idle.estimate.displacement[0] *= -1
        controller.idle.estimate.velocity[0] *= -1
        controller.idle.heading = heading
    for _ in range(1000):
        world.step(controller.respond(world.observe(), WingResponse(), ThreatMeasurement(weak_continuous=True)))
        assert not world.crashed


def test_session_keeps_neurons_unchanged_before_clearance_and_replays(tmp_path):
    from flybrain.escape_repair.session import RepairSession, RepairLog
    from flybrain.escape_recovery.session import RecoverySession
    from flybrain.escape_recovery.config import CandidateConfig
    candidate = CandidateConfig.create()
    old, new = RecoverySession(1200010, candidate), RepairSession(1200010, candidate)
    for session in (old, new):
        session.spawn('predator', (.3, .375, .021))
    for _ in range(12):
        a, b = old.advance(), new.advance()
        assert a == b
    path = tmp_path / 'actions.json'
    assert isinstance(new.log, RepairLog)
    new.log.save(path)
    replay = RepairSession.from_log(path)
    assert replay.snapshot == new.snapshot
    new.new_trial()
    assert isinstance(new.log, RepairLog)
    assert new.brain.steps == 0 and new.controller.escape.trigger_count == 0
    assert not new.threat.clearance_sizes
    import json
    record = json.loads(path.read_text())
    record['repair_sources'][next(iter(record['repair_sources']))] = 'bad'
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        RepairSession.from_log(path)


def test_repair_protocol_requires_complete_success_and_silent_controls():
    from flybrain.escape_repair.validation import cases, gate
    rows = [dict(**case, success=case['intervention'] == 'intact', triggers=case['intervention'] == 'intact') for case in cases()]
    assert len(rows) == 96 and gate(rows)
    assert not gate(rows[:-1])
    rows[0]['success'] = False
    assert not gate(rows)
    rows[0]['success'] = True
    rows[1]['triggers'] = 1
    assert not gate(rows)


def test_launcher_refuses_missing_or_failed_receipts(tmp_path):
    from flybrain.escape_repair.delivery import verified_candidate
    assert verified_candidate(tmp_path) is None
    (tmp_path / 'delivery.json').write_text('{"status":"failed"}')
    assert verified_candidate(tmp_path) is None


def test_real_runner_seals_pre_neural_failure_and_cannot_retry(tmp_path, monkeypatch):
    import json
    from dataclasses import asdict
    from flybrain.escape_recovery.config import CandidateConfig
    from flybrain.escape_repair import validation
    from flybrain.escape_repair.session import source_hashes
    from flybrain.escape_diagnostic_journal import ReservationJournal
    art = tmp_path / 'art'
    art.mkdir()
    (art / 'preflight.json').write_text(json.dumps(dict(status='passed', bound=source_hashes(),
        proofs={}, protected={}, repair_sources=source_hashes(), candidate=asdict(CandidateConfig.create()))))
    class StopBeforeNeural(ReservationJournal):
        def reserve_interval(self):
            raise RuntimeError('controlled pre-neural stop')
    monkeypatch.setattr(validation, 'ART', art)
    monkeypatch.setattr(validation, 'ReservationJournal', StopBeforeNeural)
    with pytest.raises(RuntimeError, match='controlled'):
        validation.run(tmp_path / 'run')
    result = json.loads((tmp_path / 'run/results.json').read_text())
    assert result['status'] == 'failed' and result['reserved_kernel_steps'] == 0
    assert (tmp_path / 'seal.json').exists()
    with pytest.raises(FileExistsError):
        validation.run(tmp_path / 'renamed')


def test_independent_audit_rejects_forged_success_and_weak_clearance():
    from dataclasses import asdict
    from flybrain.escape_repair.audit import check_case
    from flybrain.escape_repair.validation import cases
    from flybrain.escape_flight_validation.runner import summary
    from flybrain.escape_recovery.readout import Decision
    from flybrain.flight3d.anatomy import load_manifest
    import json
    case = cases()[0]
    records = []
    for i in range(300):
        records.append(dict(**case, interval=i, end_seconds=(i+1)*.02,
            actual_counts=[0]*589, reference_counts=[0]*589, wing_vector=[0]*48,
            finite=True, max_downstream_hz=0., temporal=json.loads(json.dumps(asdict(Decision()))),
            pre_position=[.3,.3,.003], pre_orientation=[0.,0.,0.], position=[.3,.3,.003],
            orientation=[0.,0.,0.], velocity=[0.,0.,0.], predator_position=[.3,.375,.021],
            foot_contact=True, collided=False, recovered=False, triggers=0, selected_side=None,
            escape_state='idle', behavior_state='resting', clearance_growth_degrees_s=0.,
            threat=dict(strong=False, weak_continuous=i>=14, size_degrees=8.)))
    saved = summary(case, records)
    check_case(case, records, saved, load_manifest())
    saved['success'] = True
    with pytest.raises(ValueError, match='summary'):
        check_case(case, records, saved, load_manifest())
    saved['success'] = False
    records[0]['threat']['weak_continuous'] = True
    with pytest.raises(ValueError, match='clearance'):
        check_case(case, records, saved, load_manifest())


@pytest.mark.parametrize('condition', ['intact', 'visual-silent'])
def test_named_start_replay_matches_every_field_and_rejects_tampering(tmp_path, condition):
    from flybrain.escape_repair.validation import setup_case, record_interval, cases
    from flybrain.escape_repair.replay import verify_case
    from flybrain.escape_recovery.config import CandidateConfig
    from flybrain.escape_diagnostic_journal import ReservationJournal
    case = dict(cases()[4], intervention=condition)
    session = setup_case(case, CandidateConfig.create())
    records = [record_interval(session, case, i) for i in range(2)]
    path = tmp_path / 'actions.json'
    session.log.save(path)
    budget = ReservationJournal(tmp_path / 'progress.json', 120)
    verify_case(path, case, records, budget)
    records[0]['position'][0] += .001
    with pytest.raises(ValueError, match='replay mismatch'):
        verify_case(path, case, records, budget)
    assert budget.steps == 120
    budget.close()

