import numpy as np
import pytest
from flybrain.flight3d.assets import parse_mesh, simplify_skeleton


def test_source_mesh_indices_and_units():
    xyz=np.array([[1000,0,0],[0,2000,0],[0,0,3000]],dtype='<f4')
    raw=np.array([3],dtype='<u4').tobytes()+xyz.tobytes()+np.array([0,1,2],dtype='<u4').tobytes()
    points,faces=parse_mesh(raw)
    np.testing.assert_array_equal(points,xyz)
    assert faces.tolist()==[[0,1,2]]
    with pytest.raises(ValueError): parse_mesh(raw[:-1])


def test_lod_preserves_branch_endpoints_and_curved_geometry():
    xyz=np.array([[0,0,0],[1,0,0],[2,0,0],[2,1,0],[2,-1,0]],float)
    edges=np.array([[0,1],[1,2],[2,3],[2,4]])
    lines=simplify_skeleton(xyz,edges,.1)
    assert lines.shape==(3,2,3)
    assert any(np.array_equal(line,xyz[[0,2]]) for line in lines)
