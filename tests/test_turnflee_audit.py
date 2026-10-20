"""Offline count reconstruction must catch forged rolling vectors and rates."""
import importlib
import numpy as np
import pytest
from flybrain.flight3d.anatomy import load_manifest


def test_offline_reconstruction_rejects_wrong_window_without_neural_steps():
    assert importlib.util.find_spec('flybrain.turnflee.audit'), 'independent disk audit required'
    m=importlib.import_module('flybrain.turnflee.audit'); manifest=load_manifest()
    wings=[i for i,n in enumerate(manifest['nodes']) if set(n['roles'])&{'wing_power','wing_steering'}]
    records=[]
    for interval in range(12):
        a=[0]*589; b=a.copy()
        if interval==0: a[wings[0]]=1
        vector=[0]*48
        if interval<10: vector[0]=1
        records.append(dict(actual_counts=a,reference_counts=b,wing_vector=vector,
                            interval=interval,end_seconds=(interval+1)*.02,
                            max_downstream_hz=0 if interval<9 else 5,finite=True))
    result=m.audit_counts(records,manifest)
    assert result['intervals']==12 and result['baseline_equal'] is False
    records[-1]['wing_vector'][0]=1
    with pytest.raises(ValueError,match='rolling'): m.audit_counts(records,manifest)
