"""FB-07D: spike observability and a single encoder-scale intervention."""

import csv
import json

import pytest
import torch

from flybrain.neural.network import SyntheticNetwork
from flybrain.training.evaluation import EvaluationConfig, locked_cases
from flybrain.training.fb07d_diagnosis import (
    controlled_probe, feature_aliasing, paired_probe, run_diagnosis,
)
from flybrain.training.readout_experiment import load_checkpoint, save_checkpoint, train_readout
from flybrain.world.sensing import SensoryObservation


def test_encoder_scale_changes_current_and_spikes_without_changing_topology():
    observation = SensoryObservation(0.025, 0.05)
    original = SyntheticNetwork(seed=7)
    wider = SyntheticNetwork(seed=7, encoder_scale=0.05)

    assert torch.equal(original.topology.weights, wider.topology.weights)
    assert original.encode(observation)[0].item() == pytest.approx(2.0)
    assert wider.encode(observation)[0].item() == pytest.approx(1.0)
    assert wider.encode(observation)[10].item() == pytest.approx(2.0)
    assert original.advance(observation).spike_counts[:10].sum().item() > (
        wider.advance(observation).spike_counts[:10].sum().item()
    )
    with pytest.raises(ValueError, match='scale'):
        SyntheticNetwork(encoder_scale=0)


def test_alias_metric_counts_distinct_inputs_with_same_feature_and_turn_conflict():
    observations = [
        SensoryObservation(0.02, 0.03),
        SensoryObservation(0.03, 0.02),
        SensoryObservation(0.02, 0.03),
        SensoryObservation(0.03, 0.03),
    ]
    features = torch.tensor([[1.0] * 7] * 3 + [[2.0] * 7])

    result = feature_aliasing(observations, features, encoder_scale=0.025)

    assert result['ticks'] == 4
    assert result['unique_raw_pairs'] == 3
    assert result['unique_feature_vectors'] == 2
    assert result['aliased_feature_vectors'] == 1
    assert result['aliased_ticks'] == 3
    assert result['conflicting_turn_ticks'] == 3
    assert result['turn_mse_floor'] == pytest.approx(2 / 3)
    assert result['both_saturated_ticks'] == 1
    with pytest.raises(ValueError, match='length'):
        feature_aliasing(observations, features[:2], encoder_scale=0.025)


def test_paired_probe_replays_identical_locked_observations_at_both_scales():
    config = EvaluationConfig(horizon_ticks=3)
    cases = locked_cases()['train'][:2]
    probe = paired_probe(cases, config, ticks=3)

    assert [row['case_id'] for row in probe['cases']] == ['train-0', 'train-1']
    assert all(row['ticks'] == 3 for row in probe['cases'])
    assert probe['original']['ticks'] == probe['wider']['ticks'] == 6
    assert probe['original']['unique_raw_pairs'] == probe['wider']['unique_raw_pairs']
    assert probe['topology_unchanged'] is True
    with pytest.raises(ValueError, match='locked train or validation'):
        paired_probe(locked_cases()['final'][:1], config, ticks=1)


def test_controlled_probe_exposes_saturation_and_spike_discrimination():
    probe = controlled_probe(network_seed=7)

    assert probe['observations'] == [[0.03, 0.04], [0.04, 0.03]]
    assert probe['original']['currents'] == [[2.0, 2.0], [2.0, 2.0]]
    assert probe['original']['sensory_spike_totals'] == [[10, 10], [10, 10]]
    assert probe['wider']['sensory_spike_totals'] == [[0, 10], [10, 0]]


def test_checkpoint_rejects_wrong_encoder_scale(tmp_path):
    features = torch.tensor([[0.5] * 7], dtype=torch.float32)
    targets = torch.tensor([[0.7, 0.4]], dtype=torch.float32)
    model = train_readout(features, targets, steps=1).model
    path = tmp_path / 'readout.pt'

    save_checkpoint(path, model, network_seed=7, alpha=0.5, encoder_scale=0.05)

    with pytest.raises(ValueError, match='encoder'):
        load_checkpoint(path, network_seed=7, encoder_scale=0.025)
    assert load_checkpoint(path, network_seed=7, encoder_scale=0.05) is not None


def test_short_diagnosis_writes_validation_only_matched_rows(tmp_path):
    config = EvaluationConfig(horizon_ticks=2)
    result_path = run_diagnosis(tmp_path, config=config, probe_ticks=2,
                                sample_ticks=1, training_steps=2)
    result = json.loads(result_path.read_text(encoding='utf-8'))
    with (tmp_path / 'validation.csv').open(newline='', encoding='utf-8') as file:
        rows = list(csv.DictReader(file))

    assert result['encoder_scales'] == {'original': 0.025, 'wider': 0.05}
    assert result['train_sample_count'] == 8
    assert result['topology_unchanged'] is True
    assert result['reload_replay_exact'] is True
    assert result['probe']['train']['original']['ticks'] == 8
    assert result['probe']['validation']['wider']['ticks'] == 8
    assert [row['case_id'] for row in rows] == [
        case.case_id for case in locked_cases()['validation']
    ]
    assert len(result['validation_results']) == 4
    assert 'final_results' not in result
    assert (tmp_path / 'readout.pt').is_file()
