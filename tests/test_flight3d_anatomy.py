"""Exact graph identities and morphology alignment, independent of downloads."""

import numpy as np
import pytest

from flybrain.flight3d.anatomy import parse_swc, anatomical_positions, select_paths, validate_manifest


def test_swc_preserves_nm_geometry_and_parent_links():
    points, edges = parse_swc('# fixture\n1 1 1000 2000 3000 2 -1\n9 3 1100 2100 3200 1 1\n')
    assert points.tolist()==[[1000,2000,3000],[1100,2100,3200]]
    assert edges.tolist()==[[0,1]]
    with pytest.raises(ValueError): parse_swc('1 1 0 0 0 1 99\n')


def test_transform_is_global_not_per_subset():
    a=np.array([[1000,2000,3000],[1100,2200,3300]],float)
    assert np.array_equal(anatomical_positions(a)[1],anatomical_positions(a[1:])[0])
    # Native source nanometres to view millimetres; no graph layout.
    assert np.allclose(anatomical_positions(a)[1]-anatomical_positions(a)[0],(.0001,-.0002,.0003))


def test_path_selection_uses_only_source_directed_edges_and_stable_ties():
    edges=[('a','b',10),('b','d',10),('a','c',20),('c','d',20),('z','a',40)]
    paths=select_paths(edges,['a','x'],['d'],max_hops=3)
    assert paths['a']==['a','c','d'] and 'x' not in paths
    assert select_paths(edges,['d'],['a'])=={}


def test_manifest_rejects_missing_ids_edges_and_non_string_roots():
    nodes=[{'id':'1','roles':['vision']},{'id':'2','roles':['wing_power']}]
    good={'nodes':nodes,'edges':[{'pre':'1','post':'2','count':5}], 'schema':'flight3d-circuit-v1'}
    validate_manifest(good)
    with pytest.raises(ValueError): validate_manifest({**good,'edges':[{'pre':'9','post':'2','count':5}]})
    with pytest.raises(ValueError): validate_manifest({**good,'nodes':[{'id':1,'roles':[]}]})
