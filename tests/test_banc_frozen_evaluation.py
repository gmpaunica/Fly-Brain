"""Acceptance exceptions must preserve failures and cannot bypass frozen evidence."""

import json

import pytest
import torch


def test_actual_candidate_reproduces_original_gate_and_preserves_covered_failure():
    from flybrain.training.banc_frozen_evaluation import training_evidence

    protocol, digest, models, evidence = training_evidence()
    assert digest == '769db7336a0adfb10e28d395dd2c64eb2a8da3aef091011e478c827a3fa91290'
    assert evidence['original_gates']['passed'] is True
    assert evidence['covered_gates']['passed'] is False
    assert evidence['covered_gates']['aggregate_absolute_turn_fit'] is False
    assert evidence['original_per_output_mse'][1] == pytest.approx(.010884924333201046)
    assert evidence['covered_per_output_mse'][1] == pytest.approx(.0235447046758735)
    assert evidence['matched_initialization_exact'] is True
    assert all(not p.requires_grad for m in models.values() for p in m.parameters())
    assert protocol['optimizer_step_ceiling'] == 500


def test_missing_integrity_check_cannot_be_treated_as_passing():
    from flybrain.training.banc_frozen_evaluation import validate_checks, REQUIRED_CHECKS

    with pytest.raises(ValueError):
        validate_checks({})
    with pytest.raises(ValueError):
        validate_checks({name: True for name in REQUIRED_CHECKS if name != 'checkpoint_replay_exact'})
    with pytest.raises(ValueError):
        validate_checks({name: 1 for name in REQUIRED_CHECKS})


def test_history_includes_partial_journals_csv_and_other_evaluation_directories(tmp_path):
    from flybrain.training.banc_frozen_evaluation import historical_seeds

    own = tmp_path / 'current'
    own.mkdir()
    (own / 'protocol.json').write_text(json.dumps({'seed': 720000}))
    (tmp_path / 'prior.json').write_text(json.dumps({'cases': [{'seed': 11}], 'seeds': [12]}))
    (tmp_path / 'partial.jsonl').write_text('{"result":{"seed":13}}\n')
    (tmp_path / 'episodes.csv').write_text('seed,outcome\n14,timeout\n')
    assert historical_seeds(tmp_path, exclude=own) == {11, 12, 13, 14}
    assert historical_seeds(tmp_path) == {11, 12, 13, 14, 720000}


def test_no_acceptance_means_no_protocol_or_execution(tmp_path):
    from flybrain.training.banc_frozen_evaluation import prepare_protocol

    out = tmp_path / 'assay'
    with pytest.raises(ValueError, match='acceptance'):
        prepare_protocol(out, acceptance=None)
    assert not out.exists()


def test_consumed_central_receipt_blocks_before_evidence_or_episodes(tmp_path, monkeypatch):
    from flybrain.training import banc_frozen_evaluation as runner

    receipt = tmp_path / 'central.json'
    receipt.write_text('{}')
    monkeypatch.setattr(runner, 'CENTRAL_RECEIPT', receipt)
    with pytest.raises(FileExistsError):
        runner.run_evaluation(tmp_path / 'nonexistent.json')


def test_real_projection_evaluation_uses_reset_fixed_core_and_exact_substeps():
    from flybrain.neural.banc_projection import ProjectionReadout
    from flybrain.training.banc_frozen_evaluation import evaluate_group
    from flybrain.training.evaluation import EvaluationConfig, locked_cases

    models = {n: ProjectionReadout(torch.zeros(20), torch.ones(20)) for n in ('trained', 'untrained')}
    before = {n: {k: v.clone() for k, v in m.state_dict().items()} for n, m in models.items()}
    rows, account = evaluate_group(locked_cases()['train'][:1], models, EvaluationConfig(horizon_ticks=2))
    assert [len(group) for group in rows.values()] == [1, 1, 1]
    assert account['world_ticks'] == 6
    assert account['neural_steps'] == 120
    assert account['effective_weights_unchanged']
    assert all(torch.equal(v, before[n][k]) for n, m in models.items() for k, v in m.state_dict().items())


def test_altered_candidate_rejected_before_any_receipt(tmp_path, monkeypatch):
    from flybrain.training import banc_frozen_evaluation as runner

    (tmp_path / 'protocol.json').write_text('{}')
    monkeypatch.setattr(runner, 'CANDIDATE', tmp_path)
    with pytest.raises(ValueError, match='candidate bytes'):
        runner.training_evidence()
    assert not (tmp_path / 'fresh-evaluation.execution.json').exists()


def test_protocol_keeps_exception_and_rejects_copied_destination_or_changed_source(tmp_path, monkeypatch):
    import hashlib
    from flybrain.training import banc_frozen_evaluation as runner

    # Exercise actual scanning on an isolated history, independent of whether
    # the real scientific assay has since been consumed. No scientific stepping.
    history = tmp_path / 'history'
    history.mkdir()
    scan = runner.historical_seeds
    monkeypatch.setattr(runner, 'historical_seeds', lambda *, exclude=None: scan(history, exclude=exclude))
    monkeypatch.setattr(runner, 'CENTRAL_RECEIPT', tmp_path / 'central.json')
    acceptance = {'rule': runner.ACCEPTANCE_RULE, 'covered_failure_retained': True,
                  'user_decision': 'TEST FIXTURE ONLY: approve synthetic preparation; no real assay authorized'}
    path = runner.prepare_protocol(tmp_path / 'assay', acceptance=acceptance)
    p, _ = runner.load_protocol(path)
    assert p['eligibility']['original_fit_training_gate_passed'] is False
    assert p['eligibility']['original_gates']['passed'] is True
    assert p['eligibility']['failed_gate']['observed'] > p['eligibility']['failed_gate']['ceiling']
    assert not (path.parent / 'execution.json').exists()
    copy = tmp_path / 'copy'
    copy.mkdir()
    for name in ('protocol.json', 'protocol.sha256'):
        (copy / name).write_bytes((path.parent / name).read_bytes())
    with pytest.raises(ValueError, match='destination'):
        runner.load_protocol(copy / 'protocol.json')
    import copy as copying
    for remove_group in (True, False):
        incomplete = copying.deepcopy(p)
        if remove_group:
            del incomplete['provenance']['source_code_sha256']
        else:
            del incomplete['provenance']['source_code_sha256'][str(runner.PROJECT_ROOT / 'src/flybrain/training/banc_frozen_evaluation.py')]
        path.write_text(json.dumps(incomplete))
        path.with_suffix('.sha256').write_text(hashlib.sha256(path.read_bytes()).hexdigest())
        with pytest.raises(ValueError, match='source/history'):
            runner.load_protocol(path)
    witness = tmp_path / 'source.py'
    witness.write_text('original')
    p['provenance']['source_code_sha256'][str(witness)] = hashlib.sha256(witness.read_bytes()).hexdigest()
    path.write_text(json.dumps(p))
    path.with_suffix('.sha256').write_text(hashlib.sha256(path.read_bytes()).hexdigest())
    witness.write_text('changed')
    with pytest.raises(ValueError, match='source/history'):
        runner.load_protocol(path)


def coordinator_fixture(tmp_path, monkeypatch):
    """Isolate slow episode execution to test orchestration with artificial cases."""
    from dataclasses import asdict
    from flybrain.training import banc_frozen_evaluation as runner
    from flybrain.training.evaluation import EvaluationConfig, locked_cases
    from flybrain.neural.banc_projection import ProjectionReadout

    models = {n: ProjectionReadout(torch.zeros(20), torch.ones(20)) for n in ('trained', 'untrained')}
    cases = [dict(asdict(locked_cases()['train'][0]), case_id=f'test-{i}', seed=900+i, split='validation') for i in range(16)]
    p = {'cases': {'validation': cases, 'final': [{'deliberately': 'invalid final; must not construct'}]},
         'config': asdict(EvaluationConfig()), 'experiment': 'unit_test', 'acceptance': {},
         'eligibility': {}, 'git_commit': None, 'objective': 'test', 'scope_limit': 'test'}
    monkeypatch.setattr(runner, 'load_protocol', lambda path: (p, 'a' * 64))
    monkeypatch.setattr(runner, 'training_evidence', lambda: ({}, 'a' * 64, models, {}))
    monkeypatch.setattr(runner, 'CENTRAL_RECEIPT', tmp_path / 'central.json')
    out = tmp_path / 'assay'
    out.mkdir()
    return runner, out / 'protocol.json'


def test_failed_validation_never_constructs_final_and_receipt_prevents_renamed_retry(tmp_path, monkeypatch):
    from dataclasses import replace
    from flybrain.training.evaluation import EpisodeResult

    runner, path = coordinator_fixture(tmp_path, monkeypatch)
    stored = json.loads((runner.CANDIDATE / 'results.json').read_text())
    template = EpisodeResult(**stored['episodes']['untrained'][0])
    # Numerical stepping has its own real test above. This fixture returns
    # artificial timeouts to exercise only the runner's stop/journal behavior.
    def timeout(case, config, factory):
        return replace(template, case_id=case.case_id, split=case.split, seed=case.seed,
                       ticks=0, elapsed_seconds=0)
    monkeypatch.setattr(runner, 'run_episode', timeout)
    result = runner.run_evaluation(path)
    assert result['actual_episodes'] == 48
    assert result['actual_world_ticks'] == result['actual_neural_steps'] == 0
    assert result['final_cases_evaluated'] is False
    assert result['fb10_gate_passed'] is False
    assert len((path.parent / 'episodes.jsonl').read_text().splitlines()) == 48
    assert not (path.parent / 'final.json').exists()
    with pytest.raises(FileExistsError):
        runner.run_evaluation(tmp_path / 'renamed' / 'protocol.json')


def test_abort_keeps_partial_neural_work_and_marks_unknown_world_ticks(tmp_path, monkeypatch):
    from flybrain.world.sensing import SensoryObservation

    runner, path = coordinator_fixture(tmp_path, monkeypatch)
    def interrupt(case, config, factory):
        # Use a real partial neural step; no fresh world or optimizer stepping.
        factory().network.advance(SensoryObservation(.1, .2), substeps=1)
        raise KeyboardInterrupt('controlled test interruption')
    monkeypatch.setattr(runner, 'run_episode', interrupt)
    with pytest.raises(KeyboardInterrupt):
        runner.run_evaluation(path)
    aborted = json.loads((path.parent / 'aborted.json').read_text())
    assert aborted['completed_episodes'] == aborted['completed_world_ticks'] == 0
    assert aborted['actual_neural_steps'] == aborted['in_progress_neural_steps'] == 1
    assert aborted['in_progress_world_ticks'] is None
    assert aborted['in_progress']['case']['case_id'] == 'test-0'
    assert (tmp_path / 'central.json').exists()


@pytest.mark.parametrize('failure_method', ['write', 'flush'])
def test_journal_failure_counts_finished_episode_neural_work_only_once(tmp_path, monkeypatch, failure_method):
    from dataclasses import replace
    from pathlib import Path
    from flybrain.training.evaluation import EpisodeResult
    from flybrain.world.sensing import SensoryObservation

    runner, path = coordinator_fixture(tmp_path, monkeypatch)
    stored = json.loads((runner.CANDIDATE / 'results.json').read_text())
    template = EpisodeResult(**stored['episodes']['untrained'][0])
    def one_interval(case, config, factory):
        factory().network.advance(SensoryObservation(.1, .2), substeps=20)
        return replace(template, case_id=case.case_id, split=case.split, seed=case.seed,
                       ticks=1, elapsed_seconds=.02)
    monkeypatch.setattr(runner, 'run_episode', one_interval)
    original_open = Path.open
    class FailingJournal:
        def __init__(self, stream):
            self.stream = stream
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.stream.close()
        def write(self, text):
            if failure_method == 'write':
                raise OSError('controlled journal failure')
            return self.stream.write(text)
        def flush(self):
            raise OSError('controlled journal failure')
    def opening(file, *args, **kwargs):
        stream = original_open(file, *args, **kwargs)
        return FailingJournal(stream) if file == path.parent / 'episodes.jsonl' else stream
    monkeypatch.setattr(Path, 'open', opening)
    with pytest.raises(OSError, match='journal failure'):
        runner.run_evaluation(path)
    aborted = json.loads((path.parent / 'aborted.json').read_text())
    assert aborted['completed_episodes'] == aborted['completed_world_ticks'] == 1
    assert aborted['completed_neural_steps'] == aborted['actual_neural_steps'] == 20
    assert aborted['in_progress'] is None
