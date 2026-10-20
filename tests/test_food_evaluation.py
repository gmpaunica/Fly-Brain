"""Full-horizon behavior scoring cannot equate touch with a settled stop."""

from flybrain.training.food_evaluation import score_trace
from flybrain.world.fly import MotorCommand


def test_score_requires_approach_brake_exact_stop_and_remaining_dwell():
    moving = [(False, MotorCommand(.7, 0), 70., 100.)] * 800
    approach = [(False, MotorCommand(.2, 0), 20., 20.)] * 25
    contact = [(True, MotorCommand(.2, 0), 20., 15.)]
    stopped = [(True, MotorCommand(), 0., 15.)] * 174
    result = score_trace(moving + approach + contact + stopped)
    assert result['success']
    assert result['first_contact_seconds'] == 16.52
    assert result['stop_latency_seconds'] == .02
    assert result['approach_speed'] <= 30
    assert result['stationary_dwell_seconds'] >= 2
    departing = score_trace(moving + approach + contact + stopped[:-1]
                            + [(False, MotorCommand(.7, 0), 70., 30.)])
    assert not departing['success'] and departing['departures'] == 1


def test_wall_stall_or_turn_in_place_does_not_count_as_stop():
    trace = [(False, MotorCommand(.2, 0), 20., 100.)] * 500
    trace += [(True, MotorCommand(.7, 0), 0., 15.)] * 250
    trace += [(True, MotorCommand(0, .2), 0., 15.)] * 250
    result = score_trace(trace)
    assert not result['success']
    assert result['stationary_dwell_seconds'] == 0
