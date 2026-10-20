"""Source coordinates and immutable morphology cache."""
import numpy as np


def test_source_cache_preserves_segments_and_returns_same_cached_arrays():
    from flybrain.flight3d import anatomical_display
    cache=anatomical_display.MorphologyCache()
    from flybrain.flight3d.anatomy import CACHE,load_manifest
    id=next(n['id'] for n in load_manifest()['nodes'] if (CACHE/'skeletons'/f"{n['id']}.npz").exists())
    first=cache.coarse(id)
    with np.load(CACHE/'skeletons'/f'{id}.npz') as z:
        np.testing.assert_array_equal(first,z['coarse'])
    assert cache.coarse(id) is first
    assert not first.flags.writeable
    full=cache.full(id)
    with np.load(CACHE/'skeletons'/f'{id}.npz') as z:
        np.testing.assert_array_equal(full,z['xyz'][z['edges']])


