from collections import Counter
import numpy as np
from flybrain.flight3d.experiment import placements, paired_p, accepted, action_errors


def test_predeclared_cases_are_balanced_disjoint_reproducible():
    groups=[placements(split) for split in ('train','validation','final')]
    assert [len(g) for g in groups]==[48,24,96]
    for group in groups:
        assert len(set(Counter((c.sector,c.height_group) for c in group).values()))==1
    assert len({c.seed for g in groups for c in g})==168
    assert groups[0]==placements('train')


def test_exact_paired_test_and_height_gate():
    assert paired_p([True]*8,[False]*8)==1/256
    assert paired_p([True]*8,[True]*8)==1
    rows=[{'success':True,'height_group':h} for h in ('lower','level','higher') for _ in range(32)]
    assert accepted(rows,82,26)
    for r in rows[:7]: r['success']=False
    assert not accepted(rows,82,26)


def test_each_channel_error_stays_separate():
    target=np.zeros((8,5)); prediction=target.copy(); prediction[:,2]=2
    assert action_errors(prediction,target)==[0.,0.,4.,0.]
