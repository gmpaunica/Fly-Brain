"""Read-only reconciliation of registry, geometry and pinned source archives."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from .anatomy import CACHE, SOURCE, anatomical_positions, digest, load_manifest, parse_swc


def audit(source: Path) -> dict:
    """Compare every retained source edge and every available selected skeleton."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.feather as feather
    manifest=load_manifest(); ids=[n['id'] for n in manifest['nodes']]
    for filename,md5 in SOURCE.values():
        if digest(source/filename,'md5')!=md5: raise ValueError('source archive checksum mismatch')
    edges=feather.read_table(source/SOURCE['edges'][0])
    wanted=pa.array(ids)
    selected=edges.filter(pc.and_(pc.and_(pc.is_in(edges['pre'],value_set=wanted),
        pc.is_in(edges['post'],value_set=wanted)),pc.greater_equal(edges['count'],5)))
    expected={(r['pre'],r['post']):r['count'] for r in selected.select(['pre','post','count']).to_pylist()}
    actual={(r['pre'],r['post']):r['count'] for r in manifest['edges']}
    if expected!=actual: raise ValueError('retained edge direction/count mismatch')
    del edges,selected
    metadata=feather.read_table(source/SOURCE['metadata'][0],columns=['root_888','root_position_nm','cell_type'])
    source_nodes={r['root_888']:r for r in metadata.filter(pc.is_in(metadata['root_888'],value_set=wanted)).to_pylist()}
    checked=[]; missing=[]
    for node in manifest['nodes']:
        id=node['id']; source_node=source_nodes[id]
        position=[float(x) for x in source_node['root_position_nm'].split(',')] if source_node['root_position_nm'] else None
        if position!=node['position_nm'] or source_node['cell_type']!=node['metadata']['cell_type']:
            raise ValueError('source identity/position mismatch')
        raw_path=CACHE/'skeletons'/f'{id}.swc.gz'; data_path=raw_path.with_name(f'{id}.npz')
        receipt_path=raw_path.with_name(f'{id}.json')
        if not all(p.exists() for p in (raw_path,data_path,receipt_path)):
            missing.append(id); continue
        raw=gzip.decompress(raw_path.read_bytes()); receipt=json.loads(receipt_path.read_text())
        if hashlib.sha256(raw).hexdigest()!=receipt['sha256']: raise ValueError('source skeleton hash mismatch')
        xyz,parents=parse_swc(raw.decode())
        with np.load(data_path) as geometry:
            np.testing.assert_array_equal(geometry['xyz'],anatomical_positions(xyz).astype(np.float32))
            np.testing.assert_array_equal(geometry['edges'],parents)
        checked.append(id)
    result={'source_archives_verified':True,'source_cells_verified':len(ids),
            'source_directed_edges_verified':len(actual),'skeletons_exact_transform_verified':len(checked),
            'missing_skeletons':missing,'circuit_sha256':digest(CACHE/'circuit.json'),
            'geometry_transform':[1e-6,-1e-6,1e-6]}
    (CACHE/'audit.json').write_text(json.dumps(result,indent=2)); return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('source',type=Path)
    args=parser.parse_args(); print(json.dumps(audit(args.source)))
