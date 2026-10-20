"""Explicit rounding reconciliation; failed raw collection stays immutable."""

import pytest
import torch


def test_canonicalization_preserves_original_values_and_pn_values_without_steps(tmp_path):
    from flybrain.training.banc_projection_fit import prepare_data,load_data
    from flybrain.training.banc_coverage import load_training_data
    p=prepare_data(tmp_path/'data');data=load_data(p.parent)
    assert data.features.shape==(15864,20)
    old=load_training_data()
    assert torch.equal(data.features[:,:16],old.features) and torch.equal(data.targets,old.targets)
    import json
    r=json.loads((p.parent/'results.json').read_text())
    assert r['actual_neural_steps']==r['actual_world_ticks']==r['optimizer_steps']==0
    assert 0<r['maximum_original_slow_rounding_error_hz']<1e-12
    with pytest.raises(FileExistsError):prepare_data(p.parent)


def test_canonicalization_refuses_material_feature_change():
    from flybrain.training.banc_projection_fit import canonical_features
    old=torch.ones(1009,16,dtype=torch.float64);raw=torch.ones(1009,20,dtype=torch.float64)
    raw[0,8]+=1e-13
    result,error=canonical_features(raw,old)
    assert error<1e-12 and torch.equal(result[:,:16],old)
    raw[0,8]+=.001
    with pytest.raises(ValueError):canonical_features(raw,old)
    raw[:,:16]=old;raw[1008,8]+=1e-13
    with pytest.raises(ValueError):canonical_features(raw,old)


def test_projection_precision_fit_still_refuses_consumed_receipt(tmp_path):
    from flybrain.training.banc_projection_fit import run_training
    path=tmp_path/'protocol.json';path.write_text('{}');(path.parent/'execution.json').write_text('{}')
    with pytest.raises(FileExistsError):run_training(path)
