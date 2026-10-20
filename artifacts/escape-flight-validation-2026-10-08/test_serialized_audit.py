"""Saved-only representation regression; never instantiate a neural model."""
import importlib.util
from pathlib import Path
import pytest


def api():
    path=Path(__file__).with_name('audit_serialized.py')
    assert path.exists(),'additive sealed-record format adapter required'
    spec=importlib.util.spec_from_file_location('flight_serialized_audit',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_numpy_csv_vector_has_exact_same_values():
    assert api().csv_vector('[np.float64(0.3), np.float64(-0.0), np.float64(0.003)]')==[.3,-0.,.003]


@pytest.mark.parametrize('value', ['[np.float64(float("nan")),0,0]','[np.float64(__import__("os")),0,0]','[1,2]','[True,0,0]'])
def test_only_finite_literal_three_vectors_are_accepted(value):
    with pytest.raises(ValueError):api().csv_vector(value)


def test_numeric_change_is_rejected(tmp_path):
    import csv,gzip,json
    trace=tmp_path/'traces.gz';table=tmp_path/'table.csv'
    with gzip.open(trace,'wt') as f:f.write(json.dumps(dict(position=[.3,.3,.003]))+'\n')
    with table.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['position']);w.writeheader();w.writerow(dict(position='[np.float64(0.3), np.float64(0.31), np.float64(0.003)]'))
    with pytest.raises(ValueError,match='vector value mismatch'):api().normalize_table(trace,table,tmp_path/'normalized.csv')
