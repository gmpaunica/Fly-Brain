"""Cache authentic BANC meshes/skeletons with object-generation and hash receipts.

Coarse line segments simplify only degree-two chains. Branches and endpoints
retain their source coordinates. The full SWC is retained for detailed selection.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip
import hashlib
import json
from pathlib import Path
import time
import urllib.request
import numpy as np

from .anatomy import CACHE, GCS, anatomical_positions, load_manifest, parse_swc


def parse_mesh(raw: bytes) -> tuple[np.ndarray,np.ndarray]:
    """Neuroglancer legacy mesh: vertex count, XYZ float32 nm, triangle uint32."""
    if len(raw)<4: raise ValueError('truncated mesh')
    n=int(np.frombuffer(raw[:4],dtype='<u4')[0]); split=4+12*n
    if n<1 or len(raw)<split or (len(raw)-split)%12: raise ValueError('invalid mesh size')
    xyz=np.frombuffer(raw[4:split],dtype='<f4').reshape(-1,3).copy()
    faces=np.frombuffer(raw[split:],dtype='<u4').reshape(-1,3).copy()
    if not np.isfinite(xyz).all() or (len(faces) and faces.max()>=n): raise ValueError('invalid mesh geometry')
    return xyz,faces


def simplify_skeleton(xyz: np.ndarray, edges: np.ndarray, tolerance: float) -> np.ndarray:
    """Reduce degree-two chains with Ramer-Douglas-Peucker in supplied units."""
    adjacency=[[] for _ in xyz]
    for a,b in edges: adjacency[a].append(b); adjacency[b].append(a)
    seen=set(); lines=[]
    for start,neighbors in enumerate(adjacency):
        if len(neighbors)==2: continue
        for nxt in neighbors:
            if (start,nxt) in seen: continue
            chain=[start]; prev=start; node=nxt
            while True:
                seen.add((prev,node)); seen.add((node,prev)); chain.append(node)
                if len(adjacency[node])!=2: break
                following=next(v for v in adjacency[node] if v!=prev)
                prev,node=node,following
            points=xyz[chain]; pending=[(0,len(chain)-1)]
            while pending:
                a,b=pending.pop(); delta=points[b]-points[a]
                if b-a>1:
                    mids=points[a+1:b]; denom=float(delta@delta)
                    t=np.clip(((mids-points[a])@delta)/max(denom,1e-30),0,1)
                    distances=np.linalg.norm(mids-points[a]-t[:,None]*delta,axis=1)
                    i=int(distances.argmax())
                    if distances[i]>tolerance:
                        pivot=a+1+i; pending.extend(((a,pivot),(pivot,b))); continue
                lines.append((points[a],points[b]))
    return np.asarray(lines,dtype=np.float32).reshape(-1,2,3)


def fetch_object(key: str) -> tuple[bytes,dict]:
    """Download public source bytes; pin the exact received object generation."""
    url=GCS+key
    with urllib.request.urlopen(url,timeout=60) as response:
        raw=response.read(); headers=response.headers
    return raw,{'url':url,'generation':headers.get('x-goog-generation'),
        'etag':headers.get('ETag'),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}


def fetch_skeleton(neuron_id: str, directory: Path=CACHE/'skeletons') -> dict:
    """Fetch one native BANC-space morphology, or report missing coverage."""
    if not neuron_id.isdecimal(): raise ValueError('numeric string source ID required')
    directory.mkdir(parents=True,exist_ok=True)
    receipt=directory/f'{neuron_id}.json'; geometry=directory/f'{neuron_id}.npz'
    if receipt.exists() and geometry.exists(): return json.loads(receipt.read_text())
    failures=[]
    for suffix in ('skeleton','l2'):
        try:
            raw,record=fetch_object(f'compiled_data/banc_888/banc_banc_space_swc/{neuron_id}_{suffix}.swc')
            xyz,edges=parse_swc(raw.decode()); points=anatomical_positions(xyz)
            np.savez_compressed(geometry,xyz=points.astype(np.float32),edges=edges,
                                coarse=simplify_skeleton(points,edges,.004))
            with gzip.open(directory/f'{neuron_id}.swc.gz','wb') as f: f.write(raw)
            record.update(id=neuron_id,status='available',points=len(xyz),segments=len(edges),units='source nm; cached mm')
            receipt.write_text(json.dumps(record,indent=2)); return record
        except Exception as exc: failures.append(f'{suffix}: {exc}')
    return {'id':neuron_id,'status':'missing','errors':failures}


def fetch_regions(directory: Path=CACHE/'regions') -> list[dict]:
    """Registered source brain and nerve-cord surfaces, same transform as SWCs."""
    directory.mkdir(parents=True,exist_ok=True); records=[]
    for region,label in (('3','brain'),('4','nerve_cord')):
        receipt=directory/f'{label}.json'; geometry=directory/f'{label}.npz'
        if receipt.exists() and geometry.exists(): records.append(json.loads(receipt.read_text())); continue
        manifest,mr=fetch_object(f'region_outlines/meshes/{region}:0')
        vertices=[]; triangles=[]; receipts=[]; offset=0
        for fragment in json.loads(manifest)['fragments']:
            raw,record=fetch_object('region_outlines/meshes/'+fragment)
            xyz,faces=parse_mesh(raw); vertices.append(xyz); triangles.append(faces+offset); offset+=len(xyz); receipts.append(record)
        np.savez_compressed(geometry,xyz=anatomical_positions(np.concatenate(vertices)).astype(np.float32),faces=np.concatenate(triangles))
        record={'region':label,'manifest':mr,'fragments':receipts,'source_units':'nm','transform':[1e-6,-1e-6,1e-6]}
        receipt.write_text(json.dumps(record,indent=2)); records.append(record)
    return records


def prepare(workers: int=6) -> dict:
    """Download selected morphology concurrently; every omission is recorded."""
    manifest=load_manifest(); started=time.monotonic(); regions=fetch_regions(); records=[]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs=[pool.submit(fetch_skeleton,n['id']) for n in manifest['nodes']]
        for job in as_completed(jobs):
            records.append(job.result())
            if len(records)%25==0: print(f'Anatomy {len(records)}/{len(jobs)}',flush=True)
    result={'source':'BANC v888','regions':regions,'skeletons':sorted(records,key=lambda r:r['id']),
            'available':sum(r['status']=='available' for r in records),'requested':len(records),
            'elapsed_seconds':time.monotonic()-started}
    (CACHE/'coverage.json').write_text(json.dumps(result,indent=2)); return result


if __name__=='__main__':
    result=prepare(); print(json.dumps({k:result[k] for k in ('available','requested','elapsed_seconds')}))
