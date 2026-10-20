import pytest
import torch
import hashlib
from dataclasses import replace

import flybrain.training.banc_readiness as readiness

from flybrain.training.banc_readiness import summarize_features
from flybrain.training.banc_readiness import TeacherRecorder
from flybrain.neural.banc_network import BancDynamicsConfig
from flybrain.world.sensing import SensoryObservation
from flybrain.training.evaluation import EvaluationConfig, locked_cases, run_episode


def test_readiness_reports_constant_baseline_and_unavoidable_exact_alias_error():
    features = torch.tensor([[0.] * 8, [0.] * 8, [1.] * 8, [1.] * 8])
    targets = torch.tensor([[.7, -1.], [.7, 1.], [.7, .5], [.7, .5]])
    result = summarize_features(features, targets)
    assert result['mean_command'][1] == pytest.approx(.25)
    assert result['constant_mse'][1] == pytest.approx(.5625)
    assert result['exact_alias_turn_mse_lower_bound'] == pytest.approx(.5)
    assert result['unique_feature_rows'] == 2
    assert result['feature_std'] == pytest.approx([.5] * 8)


@pytest.mark.parametrize('features,targets', [
    (torch.zeros((0, 8)), torch.zeros((0, 2))),
    (torch.zeros((2, 7)), torch.zeros((2, 2))),
    (torch.full((2, 8), float('nan')), torch.zeros((2, 2))),
])
def test_readiness_rejects_invalid_training_tensors(features, targets):
    with pytest.raises(ValueError):
        summarize_features(features, targets)


def test_teacher_recorder_uses_sensory_teacher_and_endpoint_features():
    rows = []
    controller = TeacherRecorder(BancDynamicsConfig(2.0), rows, 0)
    decision = controller.decide(SensoryObservation(.08, .12))
    assert decision.command.forward == pytest.approx(.7)
    assert decision.command.turn == 1
    assert decision.activity.end_step == 20
    assert rows[0]['left_odor'] == .08
    assert len(controller.decoder.feature_names) == 8


def test_teacher_reset_restarts_tick_and_preserves_shared_recorded_rows():
    rows = []
    controller = TeacherRecorder(BancDynamicsConfig(2.0), rows, 0)
    observation = SensoryObservation(.08, .12)
    first = controller.decide(observation)
    controller.decide(observation)
    controller.reset()
    replay = controller.decide(observation)
    assert [row['tick'] for row in rows] == [0, 1, 0]
    assert rows[0] == rows[2]
    assert replay.activity.end_step == first.activity.end_step == 20


def provenance_files(tmp_path):
    package = tmp_path / 'src/flybrain'
    (package / 'data').mkdir(parents=True)
    for name in ('banc_dynamics_config.json', 'banc_dynamics_results.json',
                 'banc_dynamics_protocol.json', 'banc_v888_va2_dna02.json'):
        (package / 'data' / name).write_bytes(b'{}\n')
    for relative in ('training/banc_readiness.py', 'training/evaluation.py',
                     'training/readout_experiment.py', 'world/sensing.py',
                     'neural/controller.py', 'neural/sensory.py'):
        path = package / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'# preserved code\n')
    previous = tmp_path / 'experiments/previous/results.json'
    previous.parent.mkdir(parents=True)
    previous.write_bytes(b'{"passed": false}\n')
    return tmp_path / 'experiments/readiness'


@pytest.mark.parametrize('relative', [
    'src/flybrain/data/banc_dynamics_config.json',
    'src/flybrain/data/banc_dynamics_results.json',
    'src/flybrain/data/banc_dynamics_protocol.json',
    'src/flybrain/data/banc_v888_va2_dna02.json',
    'src/flybrain/training/readout_experiment.py',
    'src/flybrain/world/sensing.py',
    'src/flybrain/neural/controller.py',
    'src/flybrain/neural/sensory.py',
    'experiments/previous/results.json',
])
def test_provenance_detects_changed_bundle_source_subset_and_prior_files(tmp_path, relative):
    output = provenance_files(tmp_path)
    before = readiness.capture_provenance(tmp_path, output)
    assert readiness.verify_provenance(before)
    (tmp_path / relative).write_bytes(b'changed')
    assert not readiness.verify_provenance(before)


def test_provenance_ignores_new_run_artifacts_and_detects_deleted_evidence(tmp_path):
    output = provenance_files(tmp_path)
    output.mkdir()
    (output / 'protocol.json').write_bytes(b'new run')
    before = readiness.capture_provenance(tmp_path, output)
    assert str(output / 'protocol.json') not in before['previous_files_sha256']
    (output / 'protocol.json').write_bytes(b'new result')
    assert readiness.verify_provenance(before)
    (tmp_path / 'experiments/previous/results.json').unlink()
    assert not readiness.verify_provenance(before)


def test_effective_weight_digest_detects_real_weight_change_without_state_dependence():
    controller = readiness.make_banc_controller(BancDynamicsConfig(2.0))
    weights = controller.network.population.weights
    before = readiness.weights_sha256(controller)
    assert before == hashlib.sha256(weights.numpy().tobytes()).hexdigest()
    controller.decide(SensoryObservation(.08, .12))
    assert readiness.weights_sha256(controller) == before
    altered = weights.clone()
    altered[0, 1] += 1
    controller.network.population = controller.network.population_for_weights(altered)
    assert readiness.weights_sha256(controller) != before


def test_readiness_refuses_existing_destination_before_loading_evidence(tmp_path):
    output = tmp_path / 'readiness'
    output.mkdir()
    saved = output / 'results.json'
    saved.write_bytes(b'preserve this')
    with pytest.raises(FileExistsError):
        readiness.run_readiness(output)
    assert saved.read_bytes() == b'preserve this'


def test_exclusive_json_write_does_not_overwrite_partial_run(tmp_path):
    path = tmp_path / 'protocol.json'
    readiness.save_new_json(path, {'attempt': 1})
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        readiness.save_new_json(path, {'attempt': 2})
    assert path.read_bytes() == original


def test_execution_accounting_counts_actual_ticks_and_rejects_missing_samples_or_excess():
    # Two real ticks provide the measured record; copies isolate the audit's
    # eight-case accounting without executing a scientific readiness run.
    cases = locked_cases()['train']
    measured = run_episode(cases[0], EvaluationConfig(horizon_ticks=2),
                           lambda: readiness.make_banc_controller(BancDynamicsConfig(2.0)))
    episodes = tuple(replace(measured, case_id=case.case_id, seed=case.seed) for case in cases)
    result = readiness.execution_accounting(episodes, episodes, 16)
    assert result['actual_world_ticks'] == 32
    assert result['teacher_world_ticks'] == result['fixed_banc_world_ticks'] == 16
    assert result['execution_budget_passed']
    assert not readiness.execution_accounting(episodes, episodes, 15)['execution_budget_passed']
    too_long = tuple(replace(episode, ticks=601) for episode in episodes)
    assert not readiness.execution_accounting(too_long, too_long, 4808)['execution_budget_passed']
    assert not readiness.execution_accounting(episodes[:-1], episodes, 14)['execution_budget_passed']
    duplicate = (episodes[0],) * 8
    assert not readiness.execution_accounting(duplicate, episodes, 16)['execution_budget_passed']
