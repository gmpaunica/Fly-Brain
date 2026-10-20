"""FB-07J: one matched actuator turn-gain intervention."""

import csv
import json

import pytest
import torch

from flybrain.neural.network import NeuralActivity, NeuralSnapshot, SyntheticNetwork
from flybrain.neural.readout import LearnedMotorDecoder, TrainableMotorReadout
from flybrain.training.evaluation import (
    EvaluationConfig, evaluate_cases, fixed_controller, locked_cases,
    run_episode, write_run,
)
from flybrain.training.fb07h_training import run_experiment
from flybrain.training.fb07j_gain import (
    TurnGainDecoder, gain_controller_factory, run_intervention, validation_gate,
)
from flybrain.world.sensing import SensoryObservation


def test_turn_gain_preserves_forward_clamps_turn_and_delegates_reset():
    network = SyntheticNetwork()
    model = TrainableMotorReadout(7)
    with torch.no_grad():
        model.linear.bias.copy_(torch.tensor([0.5, 1.0]))
    base = LearnedMotorDecoder(network.topology, model)
    gained = TurnGainDecoder(base, gain=2.0)
    activity = NeuralActivity(
        0, 20, 0.02, torch.zeros(100, dtype=torch.int64),
        NeuralSnapshot(20, torch.zeros(100), torch.zeros(100, dtype=torch.bool)),
    )

    command = gained.decode(activity)
    assert command.forward == pytest.approx(torch.sigmoid(torch.tensor(0.5)).item())
    assert command.turn == 1.0
    gained.reset()
    assert gained.decode(activity) == command
    with pytest.raises(ValueError, match='gain'):
        TurnGainDecoder(base, gain=0)


def test_gain_one_replays_fixed_controller_exactly():
    case = locked_cases()['validation'][0]
    config = EvaluationConfig(horizon_ticks=4)
    observation = SensoryObservation(0.02, 0.03)
    reference = fixed_controller(config.network_seed)
    candidate = gain_controller_factory('fixed', None, config.network_seed, gain=1.0)()

    assert candidate.decide(observation).command == reference.decide(observation).command
    assert run_episode(case, config,
                       gain_controller_factory('fixed', None, config.network_seed,
                                               gain=1.0)) == run_episode(case, config)


def test_gate_requires_arrivals_and_progress_above_both_controls():
    nine_1 = {'successes': 0, 'mean_progress_fraction': -0.3}
    fixed_2 = {'successes': 1, 'mean_progress_fraction': 0.0}
    nine_2 = {'successes': 2, 'mean_progress_fraction': 0.2}

    assert validation_gate(nine_2, nine_1, fixed_2)['passed'] is True
    assert validation_gate({**nine_2, 'successes': 1}, nine_1, fixed_2)['passed'] is False
    assert validation_gate({**nine_2, 'mean_progress_fraction': -0.1},
                           nine_1, fixed_2)['passed'] is False


def test_short_intervention_records_matched_cases_and_seals_final(tmp_path):
    config = EvaluationConfig(horizon_ticks=2)
    cases = locked_cases()
    baseline, _ = write_run(tmp_path / 'baseline', config, cases,
                            evaluate_cases(cases['validation'] + cases['final'], config))
    trained = run_experiment(tmp_path / 'trained', baseline, config=config,
                             sample_ticks=2, steps=2)
    path = run_intervention(tmp_path / 'gain', trained, baseline, config=config)
    result = json.loads(path.read_text(encoding='utf-8'))
    with (tmp_path / 'gain' / 'validation.csv').open(newline='', encoding='utf-8') as file:
        validation = list(csv.DictReader(file))

    assert result['experiment'] == 'fb07j_turn_gain'
    assert result['turn_gain'] == 2.0
    assert len(result['results']['train']['nine_2']) == 8
    assert len(result['results']['validation']['nine_2']) == 4
    assert [row['case_id'] for row in validation] == [case.case_id for case in cases['validation']]
    assert result['gain_one_replay_exact'] is True
    assert result['topology_unchanged'] is True
    assert result['training_run'] is False
    assert result['final_cases_evaluated'] is False
    assert 'final' not in result['results']
    assert not (tmp_path / 'gain' / 'final.csv').exists()
