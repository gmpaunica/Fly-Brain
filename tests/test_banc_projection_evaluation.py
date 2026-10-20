"""Fresh identities and unchanged arrival/effect/progress gates."""

import pytest


def test_projection_fresh_cases_are_disjoint_reproducible_and_not_training():
    from flybrain.training.banc_projection_evaluation import fresh_cases,historical_seeds
    groups=fresh_cases();assert groups==fresh_cases()
    assert [len(groups[s]) for s in ('validation','final')]==[16,32]
    seeds={c.seed for group in groups.values() for c in group}
    assert len(seeds)==48 and not seeds.intersection(historical_seeds())
    assert all(c.split==s for s,group in groups.items() for c in group)


def test_new_projection_assay_still_requires_original_minimum_arrivals():
    from flybrain.training.banc_projection_evaluation import assess_split
    summaries={'trained':{'mean_progress_fraction':1},'untrained':{'mean_progress_fraction':0},'fixed':{'mean_progress_fraction':0}}
    for n,passed in ((12,False),(13,True)):
        outcomes={'trained':[True]*n+[False]*(16-n),'untrained':[False]*16,'fixed':[False]*16}
        assert assess_split('validation',outcomes,summaries)['passed']==passed
    outcomes={'trained':[True]*26+[False]*6,'untrained':[False]*32,'fixed':[False]*32}
    assert assess_split('final',outcomes,summaries)['passed']
    summaries['fixed']['mean_progress_fraction']=1
    assert not assess_split('final',outcomes,summaries)['passed']


def test_projection_evaluation_receipt_blocks_even_before_loading_protocol(tmp_path):
    from flybrain.training.banc_projection_evaluation import run_evaluation
    path=tmp_path/'protocol.json';path.write_text('{}')
    (path.parent/'execution.json').write_text('{}')
    with pytest.raises(FileExistsError):run_evaluation(path)
