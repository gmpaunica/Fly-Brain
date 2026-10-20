"""Deterministic source grouping and passive, simulation-time transmission traces."""
from collections import defaultdict
from dataclasses import dataclass
import math
import numpy as np
from .anatomy import anatomical_positions


@dataclass(frozen=True)
class NeuronGroup:
    """Source members remain traceable; missing positions are never invented."""
    members: tuple[str,...]
    centroid: tuple[float,float,float] | None
    side: str
    region: str
    roles: tuple[str,...]


@dataclass(frozen=True)
class ConnectionBundle:
    pre: int
    post: int
    edge_indices: tuple[int,...]
    synapses: int


@dataclass(frozen=True)
class GroupedNetwork:
    groups: tuple[NeuronGroup,...]
    bundles: tuple[ConnectionBundle,...]

    def aggregate(self, totals) -> np.ndarray:
        """Sum all source-edge contributions, including within-group edges."""
        values=np.asarray(totals)
        return np.array([values[list(b.edge_indices)].sum() for b in self.bundles])


def group_network(manifest: dict, count: int=96) -> GroupedNetwork:
    """Median spatial splits inside source hemisphere/region partitions.

    Missing-position cells have explicit unpositioned groups. No arbitrary
    visual position is assigned to them. IDs break all spatial ties.
    """
    partitions=defaultdict(list)
    for n in sorted(manifest['nodes'],key=lambda n:n['id']):
        partitions[(n['metadata'].get('side','unknown'),n['metadata'].get('region','unknown'),
                    n.get('position_nm') is None)].append(n)
    clusters=list(partitions.values())
    while len(clusters)<min(count,len(manifest['nodes'])):
        candidates=[(len(c),i) for i,c in enumerate(clusters) if len(c)>1 and c[0].get('position_nm') is not None]
        if not candidates: break
        _,index=max(candidates,key=lambda v:(v[0],-v[1])); nodes=clusters.pop(index)
        points=np.array([n['position_nm'] for n in nodes]); axis=int(np.ptp(points,axis=0).argmax())
        nodes=sorted(nodes,key=lambda n:(n['position_nm'][axis],n['id']))
        middle=len(nodes)//2; clusters[index:index]=[nodes[:middle],nodes[middle:]]
    groups=[]; membership={}
    for i,nodes in enumerate(clusters):
        centroid=None if nodes[0].get('position_nm') is None else tuple(anatomical_positions(np.mean([n['position_nm'] for n in nodes],axis=0)))
        group=NeuronGroup(tuple(n['id'] for n in nodes),centroid,nodes[0]['metadata'].get('side','unknown'),
            nodes[0]['metadata'].get('region','unknown'),tuple(sorted({r for n in nodes for r in n['roles']})))
        groups.append(group); membership.update({id:i for id in group.members})
    bundled=defaultdict(list)
    for i,e in enumerate(manifest['edges']): bundled[(membership[e['pre']],membership[e['post']])].append(i)
    bundles=tuple(ConnectionBundle(a,b,tuple(indices),sum(manifest['edges'][i]['count'] for i in indices))
        for (a,b),indices in sorted(bundled.items()))
    return GroupedNetwork(tuple(groups),bundles)


class TransmissionTrace:
    """300 ms exponential mean of delivered current/second, fixed physical units.

    Update once per simulation interval, independent of rendering. Equal times
    are ignored; backwards time clears history. No per-frame normalization.
    """
    def __init__(self,count: int):
        self.values=np.zeros(count); self.time=0.

    def update(self,time: float,totals) -> None:
        if time<self.time:
            self.values[:]=0.; self.time=time; return
        if time==self.time: return
        values=np.asarray(totals,dtype=float)
        if values.shape!=self.values.shape or not np.isfinite(values).all() or (values<0).any():
            raise ValueError('one nonnegative finite total per edge required')
        dt=time-self.time; alpha=-math.expm1(-dt/.3)
        self.values+=(values/dt-self.values)*alpha
        self.time=time
