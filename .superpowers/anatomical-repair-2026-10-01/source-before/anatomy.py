"""Source identities, graph audit and native BANC geometry; never simulated anatomy."""

from collections import defaultdict
import gzip
import hashlib
import heapq
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / 'data' / 'flight3d'
GCS = 'https://storage.googleapis.com/lee-lab_brain-and-nerve-cord-fly-connectome/'
SOURCE = {
    'metadata':('banc_888_meta.feather','6275eda42f98c49539d1ab513d979d09'),
    'edges':('banc_888_edgelist_simple_v2.feather','394406f8a9bdf093c895f95aff4f6c49')}


def digest(path: Path, algorithm: str='sha256') -> str:
    """Stream a file checksum without loading an archive into memory."""
    h=hashlib.new(algorithm)
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def anatomical_positions(points: np.ndarray) -> np.ndarray:
    """One global nm->mm transform: X right, native anterior -Y, Z retained.

    Source geometry is never independently centred or laid out per selection.
    Camera presets supply presentation orientation only.
    """
    return np.asarray(points,dtype=np.float64)*np.array((1e-6,-1e-6,1e-6))


def parse_swc(text: str) -> tuple[np.ndarray,np.ndarray]:
    """Read source XYZ nanometres and parent edges; reject missing/duplicate nodes."""
    records=[line.split() for line in text.splitlines() if line.strip() and not line.lstrip().startswith('#')]
    if not records or any(len(r)!=7 for r in records): raise ValueError('invalid SWC rows')
    ids={int(r[0]):i for i,r in enumerate(records)}
    if len(ids)!=len(records): raise ValueError('duplicate SWC point')
    xyz=np.array([[float(x) for x in r[2:5]] for r in records])
    if not np.isfinite(xyz).all(): raise ValueError('nonfinite skeleton')
    edges=[]
    for i,row in enumerate(records):
        parent=int(row[6])
        if parent!=-1:
            if parent not in ids or ids[parent]==i: raise ValueError('invalid SWC parent')
            edges.append((ids[parent],i))
    return xyz,np.array(edges,dtype=np.int32).reshape(-1,2)


def select_paths(edges, inputs, outputs, max_hops: int=6) -> dict[str,list[str]]:
    """Choose directed shortest corridors, preferring stronger edges then IDs.

    This is a structural selection rule, not evidence of a unique functional
    pathway. Reverse Dijkstra finds one route from each input to any output.
    """
    reverse=defaultdict(list)
    for pre,post,count in edges:
        if pre!=post: reverse[post].append((pre,int(count)))
    distances={str(node):(0,0.,(str(node),)) for node in outputs}
    queue=[(0,0.,(str(node),),str(node)) for node in sorted(outputs)]
    heapq.heapify(queue)
    while queue:
        hops,cost,path,node=heapq.heappop(queue)
        if distances.get(node)!=(hops,cost,path) or hops>=max_hops: continue
        for pre,count in reverse.get(node,()):
            proposal=(hops+1,cost+1/count,(pre,)+path)
            if pre not in distances or proposal<distances[pre]:
                distances[pre]=proposal
                heapq.heappush(queue,(*proposal,pre))
    return {node:list(distances[node][2]) for node in sorted(inputs) if node in distances}


def validate_manifest(manifest: dict) -> None:
    """Validate the source graph's identifier/direction contract."""
    if manifest.get('schema')!='flight3d-circuit-v1': raise ValueError('unknown circuit schema')
    nodes=manifest['nodes']; ids=[n['id'] for n in nodes]
    if any(type(i) is not str for i in ids) or len(set(ids))!=len(ids): raise ValueError('unique string IDs required')
    known=set(ids); seen=set()
    for e in manifest['edges']:
        pair=(e['pre'],e['post'])
        if not set(pair)<=known or pair in seen or type(e['count']) is not int or e['count']<5:
            raise ValueError('invalid source connection')
        seen.add(pair)


def load_manifest(path: Path | None=None) -> dict:
    """Load the frozen selected circuit, verifying its independent digest receipt."""
    path=path or CACHE/'circuit.json'
    manifest=json.loads(path.read_text(encoding='utf-8'))
    validate_manifest(manifest)
    receipt=json.loads(path.with_suffix('.sha256.json').read_text())
    if digest(path)!=receipt['sha256']: raise ValueError('circuit manifest changed')
    return manifest


def build_registry(source_dir: Path, destination: Path=CACHE) -> dict:
    """Audit pinned archive bytes; create source atlas and deterministic flight slice.

    All paths are selected before behavioral data collection. Retained edges
    are the entire induced source subgraph at count>=5, including self edges.
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.feather as feather

    destination.mkdir(parents=True,exist_ok=True)
    if (destination/'circuit.json').exists(): return load_manifest(destination/'circuit.json')
    for filename,md5 in SOURCE.values():
        if digest(source_dir/filename,'md5')!=md5: raise ValueError(f'source hash mismatch: {filename}')
    m=feather.read_table(source_dir/SOURCE['metadata'][0])
    cols=['root_888','cell_type','cell_class','super_class','side','proofread','region',
          'cell_function','cell_function_detailed','body_part_sensory','body_part_effector',
          'peripheral_target_type','root_position_nm','neurotransmitter_predicted','neurotransmitter_verified']
    raw=m.select(cols).to_pylist()
    rows={r['root_888']:r for r in raw}
    excluded={'glia','not_a_neuron','trachea','visceral_circulatory'}
    atlas=[]
    for r in raw:
        if r['super_class'] in excluded: continue
        pos=None
        if r['root_position_nm']:
            try: pos=[float(x) for x in r['root_position_nm'].split(',')]
            except (ValueError,AttributeError): pass
        atlas.append({'id':r['root_888'],'type':r['cell_type'],'class':r['super_class'],
                      'region':r['region'],'position_nm':pos,'side':r['side']})
    with gzip.open(destination/'atlas.json.gz','wt',encoding='utf-8') as stream:
        json.dump({'schema':'flight3d-atlas-v1','source':'BANC v888 Dataverse 3.0',
                   'coordinate_units':'nm','records':atlas,
                   'missing_positions':sum(r['position_nm'] is None for r in atlas),
                   'untyped_records':sum(r['class'] is None for r in atlas)},stream)
    roles=defaultdict(set)
    def add(predicate,role,limit_side=None):
        selected=[r for r in raw if r['proofread']=='TRUE' and predicate(r)]
        selected.sort(key=lambda r:r['root_888'])
        if limit_side is not None:
            selected=[r for side in ('left','right') for r in [v for v in selected if v['side']==side][:limit_side]]
        for r in selected: roles[r['root_888']].add(role)
    add(lambda r:r['cell_type']=='ORN_VA2','odor')
    add(lambda r:r['cell_type']=='VA2_adPN','odor_projection')
    add(lambda r:r['cell_type']=='LPLC4','vision')
    add(lambda r:r['cell_type'] in ('HSN','HSE','HSS') or str(r['cell_type']) in tuple(f'VS{i}' for i in range(1,9)),'optic_flow')
    add(lambda r:r['body_part_sensory']=='haltere' and r['cell_class']=='campaniform_sensillum_neuron','motion',8)
    add(lambda r:r['body_part_sensory']=='front_leg' and r['cell_function']=='gustatory' and 'sugar' in str(r['cell_function_detailed']),'taste',4)
    add(lambda r:r['body_part_sensory']=='front_leg' and r['cell_function']=='tactile','touch',4)
    add(lambda r:str(r['cell_type']).startswith('DNg02'),'flight_power_dn')
    add(lambda r:r['cell_type']=='DNa04','flight_steering_dn')
    add(lambda r:r['cell_type'] in ('DNp07','DNp10'),'landing_dn')
    add(lambda r:r['cell_function']=='wing_power','wing_power')
    add(lambda r:r['cell_function']=='wing_steering','wing_steering')
    add(lambda r:r['cell_function']=='wing_tension','wing_tension')
    add(lambda r:r['body_part_effector']=='front_leg' and str(r['cell_function_detailed']).startswith('extend_'),'landing_motor')
    proofread=[r['root_888'] for r in raw if r['proofread']=='TRUE' and r['super_class'] not in excluded]
    e=feather.read_table(source_dir/SOURCE['edges'][0])
    strong=e.filter(pc.greater_equal(e['count'],5))
    eligible=strong.filter(pc.and_(pc.is_in(strong['pre'],value_set=pa.array(proofread)),
                                   pc.is_in(strong['post'],value_set=pa.array(proofread))))
    edge_rows=eligible.select(['pre','post','count']).to_pylist()
    triples=[(e['pre'],e['post'],e['count']) for e in edge_rows]
    del edge_rows
    sensory={i for i,rs in roles.items() if rs & {'odor','vision','optic_flow','motion','taste','touch'}}
    routes={}; missing={}
    for group in ('flight_power_dn','flight_steering_dn','landing_dn'):
        target={i for i,rs in roles.items() if group in rs}
        route=select_paths(triples,sensory,target)
        routes[group]=route; missing[group]=sorted(sensory-set(route))
    for dn_group,motor_group in (('flight_power_dn','wing_power'),('flight_steering_dn','wing_steering'),('landing_dn','landing_motor')):
        starts={i for i,rs in roles.items() if dn_group in rs}
        ends={i for i,rs in roles.items() if motor_group in rs}
        # Reverse the graph to obtain one authentic forward route to EACH effector.
        backward=select_paths([(b,a,c) for a,b,c in triples],ends,starts)
        routes[motor_group]={i:list(reversed(path)) for i,path in backward.items()}
        missing[motor_group]=sorted(ends-set(backward))
    for mapping in routes.values():
        for path in mapping.values():
            for node in path:
                if node not in roles: roles[node].add('structural_corridor')
    selected=sorted(roles)
    retained=strong.filter(pc.and_(pc.is_in(strong['pre'],value_set=pa.array(selected)),
                                  pc.is_in(strong['post'],value_set=pa.array(selected))))
    nodes=[]
    for node in selected:
        row=rows[node]
        meta={k:v for k,v in row.items() if k not in ('root_888','root_position_nm')}
        pos=[float(x) for x in row['root_position_nm'].split(',')] if row['root_position_nm'] else None
        nodes.append({'id':node,'roles':sorted(roles[node]),'position_nm':pos,'metadata':meta,
                      'evidence':('source_annotation_and_literature' if roles[node]&{'flight_power_dn','landing_dn','flight_steering_dn'}
                                  else 'structural_corridor_only' if roles[node]=={'structural_corridor'} else 'source_annotation'),
                      'physiology':'assumed LIF; unsigned count scaling; sensory tuning/actuators engineered'})
    result={'schema':'flight3d-circuit-v1','source':{'doi':'10.7910/DVN/7WTH1N','version':'3.0','materialization':888,
        'license':'CC BY 4.0','files':SOURCE},'selection':'proofread; count>=5; min-hop/reciprocal-count corridors <=6 hops; ID ties',
        'nodes':nodes,'edges':sorted(retained.select(['pre','post','count','norm']).to_pylist(),key=lambda x:(x['pre'],x['post'])),
        'paths':routes,'missing_paths':missing,
        'limitations':['Structural reachability does not establish functional mediation.',
            'Predicted transmitters are retained, not treated as validated receptor signs.',
            'Visual receptive fields and sensor gains are model assumptions, not inferred from soma position.'],
        'references':{'flight':'https://doi.org/10.1016/j.cub.2022.01.008',
                      'landing':'https://doi.org/10.1038/s41593-019-0413-4',
                      'BANC':'https://doi.org/10.1038/s41586-026-10735-w'}}
    validate_manifest(result)
    out=destination/'circuit.json'
    out.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    out.with_suffix('.sha256.json').write_text(json.dumps({'sha256':digest(out)}))
    return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    args=parser.parse_args()
    result=build_registry(args.source)
    print(json.dumps({'neurons':len(result['nodes']),'edges':len(result['edges']),
                      'missing_paths':{k:len(v) for k,v in result['missing_paths'].items()},'sha256':digest(CACHE/'circuit.json')}))
