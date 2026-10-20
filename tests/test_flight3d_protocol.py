"""Frozen-case consumption, historical overlap and global one-use execution."""
import json
import pytest
from flybrain.flight3d import experiment


def test_runner_uses_frozen_case_coordinates_not_historical_generator():
    runner=object.__new__(experiment.Experiment)
    runner.protocol={'cases':{'train':[{'seed':123,'flower':[.1,.2,.15],'sector':2,'height_group':'higher'}]}}
    assert hasattr(runner,'cases'), 'runner must consume protocol cases'
    assert runner.cases('train')[0].flower==(.1,.2,.15)
    assert runner.cases('train')[0].seed==123


def test_historical_overlap_blocks_even_reserved_and_development_identities(tmp_path):
    from flybrain.flight3d.protocol import check_overlap
    (tmp_path/'old.json').write_text(json.dumps({'reserved_final_seeds':[982001]}))
    with pytest.raises(ValueError,match='overlap'):check_overlap(tmp_path,{982001})
    (tmp_path/'old.json').write_text(json.dumps({'cases':[{'seed':970001}]}))
    with pytest.raises(ValueError,match='overlap'):check_overlap(tmp_path,{970001})
    assert check_overlap(tmp_path,{980000})['collisions']==[]


def test_one_use_receipt_survives_directory_rename(tmp_path):
    from flybrain.flight3d.protocol import consume_authorization
    directory=tmp_path/'run';directory.mkdir()
    receipt=tmp_path/'authorization.json'
    consume_authorization(directory,'abc',receipt)
    renamed=directory.rename(tmp_path/'renamed')
    with pytest.raises(FileExistsError):consume_authorization(renamed,'abc',receipt)
