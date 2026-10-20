"""Fresh-split safeguards and real short episode accounting for FB-10."""

import pytest
import torch


def history_at_original_lock():
    """Replay the history inventory frozen before this now-consumed assay.

    Later rollout diagnosis correctly makes five validation seeds historical.
    Preparation unit checks must use the recorded pre-assay environment rather
    than require those seeds to remain fresh forever. Numerical stepping and
    candidate/source checks stay real.
    """
    import json
    from pathlib import Path
    from flybrain.training.banc_memory_evaluation import DEFAULT_OUTPUT
    protocol=json.loads((DEFAULT_OUTPUT/'protocol.json').read_text(encoding='utf-8'))
    seeds=set()
    def visit(value):
        if isinstance(value,dict):
            for key,item in value.items():
                if key.endswith('seed') and type(item) is int:seeds.add(item)
                if key=='seeds' and isinstance(item,list):seeds.update(x for x in item if type(x) is int)
                visit(item)
        elif isinstance(value,list):
            for item in value:visit(item)
    for name in protocol['provenance']['previous_files_sha256']:
        path=Path(name)
        if path.suffix=='.json':visit(json.loads(path.read_text(encoding='utf-8-sig')))
    return seeds


def test_fresh_cases_are_reproducible_local_seeded_and_disjoint_from_history():
    import random, math
    from flybrain.training.banc_memory_evaluation import fresh_cases,historical_seeds
    state=random.getstate();cases=fresh_cases()
    assert random.getstate()==state and cases==fresh_cases()
    assert len(cases['validation'])==16 and len(cases['final'])==32
    seeds={c.seed for rows in cases.values() for c in rows}
    assert len(seeds)==48 and not seeds.intersection(history_at_original_lock())
    # Later diagnosis is historical evidence; the live guard now rejects reuse.
    assert {620001,620007,620010,620012,620013}.issubset(historical_seeds())
    for split,rows in cases.items():
        for c in rows:
            assert c.split==split and 60<=math.hypot(c.x-600,c.y-300)<=180
            assert 0<=c.heading<math.tau


def test_arrival_gate_requires_final_effect_size_and_both_control_comparisons():
    from flybrain.training.banc_memory_evaluation import assess_split
    outcomes={'trained':[True]*26+[False]*6,'untrained':[False]*32,'fixed':[False]*32}
    summaries={'trained':{'mean_progress_fraction':.7},'untrained':{'mean_progress_fraction':-.1},'fixed':{'mean_progress_fraction':-1}}
    assert assess_split('final',outcomes,summaries)['passed']
    outcomes['fixed']=[True]*26+[False]*6
    assert not assess_split('final',outcomes,summaries)['passed']


def test_validation_or_progress_failure_closes_gate_without_favorable_ties():
    from flybrain.training.banc_memory_evaluation import assess_split
    outcomes={'trained':[True]*12+[False]*4,'untrained':[False]*16,'fixed':[False]*16}
    summaries={'trained':{'mean_progress_fraction':.7},'untrained':{'mean_progress_fraction':-.1},'fixed':{'mean_progress_fraction':-1}}
    assert not assess_split('validation',outcomes,summaries)['passed']
    outcomes['trained']=[True]*13+[False]*3
    summaries['untrained']['mean_progress_fraction']=.7
    assert not assess_split('validation',outcomes,summaries)['passed']


def test_real_two_tick_evaluation_group_keeps_model_and_core_fixed():
    from flybrain.neural.banc_memory import MemoryBancReadout
    from flybrain.training.banc_memory_evaluation import evaluate_group
    from flybrain.training.evaluation import EvaluationConfig,locked_cases
    a=MemoryBancReadout(torch.zeros(16),torch.ones(16));b=MemoryBancReadout(torch.zeros(16),torch.ones(16))
    original={k:v.clone() for k,v in a.state_dict().items()}
    rows,account=evaluate_group(locked_cases()['train'][:1],{'trained':a,'untrained':b},EvaluationConfig(horizon_ticks=2))
    assert all(len(group)==1 for group in rows.values())
    assert account['episodes']==3 and account['world_ticks']==6 and account['neural_steps']==120
    assert account['effective_weights_unchanged']
    assert all(torch.equal(v,original[k]) for k,v in a.state_dict().items())


def test_preparation_requires_verified_passing_candidate_and_uses_zero_updates(tmp_path,monkeypatch):
    from flybrain.training import banc_memory_evaluation as runner
    monkeypatch.setattr(runner,'historical_seeds',lambda exclude=None:history_at_original_lock())
    from flybrain.training.banc_memory_evaluation import prepare_protocol,load_protocol
    path=prepare_protocol(tmp_path/'evaluation')
    p,_=load_protocol(path)
    assert p['budget']['optimizer_steps']==0 and p['budget']['maximum_world_ticks']==86400
    assert len(p['cases']['validation'])==16 and len(p['cases']['final'])==32
    assert not (path.parent/'execution.json').exists()
    with pytest.raises(FileExistsError):prepare_protocol(path.parent)


def test_copied_resealed_or_historical_evaluation_cases_are_rejected(tmp_path,monkeypatch):
    import hashlib,json
    from flybrain.training import banc_memory_evaluation as runner
    monkeypatch.setattr(runner,'historical_seeds',lambda exclude=None:history_at_original_lock())
    from flybrain.training.banc_memory_evaluation import prepare_protocol,load_protocol
    path=prepare_protocol(tmp_path/'evaluation');original=path.read_bytes()
    copied=tmp_path/'copy';copied.mkdir()
    for name in ('protocol.json','protocol.sha256'):(copied/name).write_bytes((path.parent/name).read_bytes())
    with pytest.raises(ValueError):load_protocol(copied/'protocol.json')
    p=json.loads(original);p['cases']['final'][0]['seed']=0
    path.write_text(json.dumps(p));path.with_suffix('.sha256').write_text(hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(ValueError):load_protocol(path)


def test_consumed_case_history_prevents_new_preparation_without_fixture(tmp_path):
    from flybrain.training.banc_memory_evaluation import prepare_protocol
    with pytest.raises(ValueError,match='fresh seeds overlap'):
        prepare_protocol(tmp_path/'historical-retry')


def test_evaluation_refuses_consumed_receipt_before_loading_or_stepping(tmp_path):
    from flybrain.training.banc_memory_evaluation import run_evaluation
    path=tmp_path/'protocol.json';path.write_text('{}')
    (tmp_path/'execution.json').write_text('{}')
    with pytest.raises(FileExistsError):run_evaluation(path)
