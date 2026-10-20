"""Validated, unsigned BANC v888 subset to directed PyTorch weights."""

from dataclasses import dataclass, replace
import hashlib
from importlib.resources import files
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import torch
from torch import Tensor


BANC_SUBSET_SHA256 = '76c254e22741230b3d97aeb21c9938405b2e98c07ca29b8cde9331f3c4e6d0ae'
BANC_TRANSFORM_VERSION = 'nonself-selected-input-count-normalized-v1'


_SOURCE_FILES = {
    'metadata': ('banc_888_meta.feather', 14033740, 57550610,
                 '6275eda42f98c49539d1ab513d979d09'),
    'connectivity': ('banc_888_edgelist_simple_v2.feather', 13992792,
                     305250378, '394406f8a9bdf093c895f95aff4f6c49'),
}
_LEFT_PN = '720575941653217813'
_RIGHT_PN = '720575941497888384'
_LEFT_DNA02 = '720575941510475536'
_RIGHT_DNA02 = '720575941456897005'
_PATHS = (
    (_LEFT_PN, '720575941539370133', '720575941594458987',
     '720575941500284745', _LEFT_DNA02),
    (_RIGHT_PN, '720575941504828722', '720575941645545928', _RIGHT_DNA02),
)
_MOTORS = {
    _LEFT_DNA02: ('720575941531923147', '720575941519664856',
                  '720575941538992021'),
    _RIGHT_DNA02: ('720575941531180261', '720575941568355066',
                   '720575941574355016'),
}


@dataclass(frozen=True)
class BancNode:
    """Original materialization ID and retained source annotations."""

    root_id: str
    role: str
    metadata: Mapping[str, object]


@dataclass(frozen=True)
class BancEdge:
    """Source-directed pre to post connection before synthetic scaling."""

    pre: str
    post: str
    count: int
    norm: float
    post_count: int
    pre_count: int


@dataclass(frozen=True)
class BancTopology:
    """Pinned node order, provenance and copied W[post, pre] matrix."""

    source: Mapping[str, object]
    nodes: tuple[BancNode, ...]
    edges: tuple[BancEdge, ...]
    root_ids: tuple[str, ...]
    groups: tuple[str, ...]
    _weights: Tensor

    @property
    def labels(self) -> tuple[str, ...]:
        """Stable original IDs for snapshots and provenance."""
        return self.root_ids

    @property
    def weights(self) -> Tensor:
        """A detached copy of voltage jumps indexed by destination, source."""
        return self._weights.clone()

    def index(self, root_id: str) -> int:
        """Return the tensor index of an original v888 root ID."""
        try:
            return self.root_ids.index(root_id)
        except ValueError as exc:
            raise ValueError(f'unknown BANC root ID: {root_id}') from exc

    def indices(self, group: str) -> tuple[int, ...]:
        """Return ordered indices for an anatomical role."""
        found = tuple(i for i, role in enumerate(self.groups) if role == group)
        if not found:
            raise ValueError(f'unknown BANC role: {group}')
        return found


def build_banc_topology(
    nodes: Sequence[Mapping[str, object]],
    edges: Sequence[Mapping[str, object]],
    *,
    connectome_gain: float,
    dtype: torch.dtype = torch.float32,
    source: Mapping[str, object] | None = None,
) -> BancTopology:
    """Validate source rows and apply an explicit synthetic global gain.

    `norm` remains the source fraction of *all* postsynaptic input, including
    cells outside this selected subset. No functional sign is inferred.
    """
    if (isinstance(connectome_gain, bool)
            or not isinstance(connectome_gain, (int, float))
            or not math.isfinite(connectome_gain) or connectome_gain <= 0):
        raise ValueError('connectome_gain must be finite and positive')
    if dtype not in (torch.float32, torch.float64):
        raise ValueError('only float32 and float64 are supported')
    if not nodes:
        raise ValueError('BANC subset must have nodes')
    parsed_nodes: list[BancNode] = []
    seen_ids: set[str] = set()
    for row in nodes:
        if not isinstance(row, Mapping):
            raise ValueError('BANC node must be an object')
        root_id = row.get('root_888')
        role = row.get('role')
        if (not isinstance(root_id, str) or not root_id.isdecimal()
                or not isinstance(role, str) or not role
                or row.get('side') not in ('left', 'right')
                or not isinstance(row.get('cell_type'), str)
                or row.get('proofread') != 'TRUE'):
            raise ValueError('invalid BANC node metadata')
        if root_id in seen_ids:
            raise ValueError(f'duplicate BANC node: {root_id}')
        seen_ids.add(root_id)
        parsed_nodes.append(BancNode(root_id, role, MappingProxyType(dict(row))))
    root_ids = tuple(node.root_id for node in parsed_nodes)
    index = {root_id: i for i, root_id in enumerate(root_ids)}
    weights = torch.zeros((len(root_ids), len(root_ids)), dtype=dtype, device='cpu')
    parsed_edges: list[BancEdge] = []
    seen_pairs: set[tuple[str, str]] = set()
    source_totals: dict[tuple[str, str], int] = {}
    for row in edges:
        if not isinstance(row, Mapping):
            raise ValueError('BANC edge must be an object')
        pre, post = row.get('pre'), row.get('post')
        count, norm = row.get('count'), row.get('norm')
        post_count, pre_count = row.get('post_count'), row.get('pre_count')
        if not isinstance(pre, str) or not isinstance(post, str):
            raise ValueError('BANC edge IDs must remain strings')
        if pre not in index or post not in index:
            raise ValueError(f'BANC edge has missing endpoint: {pre} -> {post}')
        if (type(count) is not int or count < 5
                or type(post_count) is not int or post_count < count
                or type(pre_count) is not int or pre_count < count
                or isinstance(norm, bool) or not isinstance(norm, (int, float))
                or not math.isfinite(norm) or norm <= 0
                or not math.isclose(norm, count / post_count,
                                    rel_tol=0, abs_tol=1e-6)):
            raise ValueError(f'invalid BANC edge: {pre} -> {post}')
        for key, total in ((('pre', pre), pre_count), (('post', post), post_count)):
            if key in source_totals and source_totals[key] != total:
                raise ValueError('inconsistent BANC source endpoint totals')
            source_totals[key] = total
        pair = (pre, post)
        if pair in seen_pairs:
            raise ValueError(f'duplicate BANC edge: {pre} -> {post}')
        seen_pairs.add(pair)
        weight = connectome_gain * norm
        if not math.isfinite(weight):
            raise ValueError('scaled BANC weight overflowed')
        weights[index[post], index[pre]] = weight
        parsed_edges.append(BancEdge(pre, post, count, float(norm),
                                     post_count, pre_count))
    if not torch.isfinite(weights).all().item():
        raise ValueError('scaled BANC weights overflowed tensor dtype')
    return BancTopology(MappingProxyType(dict(source or {})), tuple(parsed_nodes),
                        tuple(parsed_edges), root_ids,
                        tuple(node.role for node in parsed_nodes), weights)


def load_banc_subset(*, connectome_gain: float,
                     path: Path | None = None,
                     dtype: torch.dtype = torch.float32) -> BancTopology:
    """Load the checked-in, source-audited BANC v888 VA2 to DNa02 subset."""
    if path is None:
        artifact = files('flybrain.data').joinpath('banc_v888_va2_dna02.json').read_bytes()
    else:
        artifact = Path(path).read_bytes()
    if hashlib.sha256(artifact).hexdigest() != BANC_SUBSET_SHA256:
        raise ValueError('BANC subset SHA-256 differs from the pinned source artifact')
    payload = json.loads(artifact)
    if (not isinstance(payload, dict)
            or not isinstance(payload.get('source'), dict)
            or not isinstance(payload.get('nodes'), list)
            or not isinstance(payload.get('edges'), list)):
        raise ValueError('invalid BANC subset document')
    source = payload['source']
    if (source.get('dataset_doi') != '10.7910/DVN/7WTH1N'
            or source.get('release') != '3.0'
            or source.get('materialization') != '888'
            or source.get('license') != 'CC BY 4.0'):
        raise ValueError('BANC source version/license mismatch')
    for key, (filename, file_id, byte_size, md5) in _SOURCE_FILES.items():
        if source.get(key) != {'filename': filename, 'dataverse_file_id': file_id,
                               'bytes': byte_size, 'md5': md5}:
            raise ValueError(f'BANC {key} provenance mismatch')
    graph = build_banc_topology(payload['nodes'], payload['edges'],
                                connectome_gain=connectome_gain,
                                dtype=dtype, source=source)
    if (len(graph.nodes), len(graph.edges),
            sum(edge.count for edge in graph.edges),
            sum(edge.pre == edge.post for edge in graph.edges)) != (83, 176, 6062, 31):
        raise ValueError('BANC subset audit totals mismatch')
    if graph.root_ids != tuple(sorted(graph.root_ids)):
        raise ValueError('BANC node order is not deterministic')
    pairs = {(edge.pre, edge.post) for edge in graph.edges}
    if any((pre, post) not in pairs for path_ids in _PATHS
           for pre, post in zip(path_ids, path_ids[1:])):
        raise ValueError('BANC corridor direction/endpoint mismatch')
    if any((pre, post) not in pairs for pre, posts in _MOTORS.items()
           for post in posts):
        raise ValueError('BANC motor direction/endpoint mismatch')
    for side, pn in (('left', _LEFT_PN), ('right', _RIGHT_PN)):
        sensory = graph.indices(f'sensory_{side}')
        if len(sensory) != (31 if side == 'left' else 37):
            raise ValueError('BANC sensory pool size mismatch')
        if any((graph.root_ids[i], pn) not in pairs for i in sensory):
            raise ValueError('BANC sensory input direction mismatch')
    return graph


def normalize_banc_topology(source: BancTopology, synaptic_budget: float) -> BancTopology:
    """Build an engineered nonself matrix without changing source rows.

    Each nonempty destination row sums to the explicit voltage-jump budget:
    W[post,pre] = budget * count / sum(selected nonself incoming counts).
    This compensates subset truncation; it is not a physiological weight.
    Source self-edges remain in the caller's original topology.
    """
    if (isinstance(synaptic_budget, bool) or not isinstance(synaptic_budget, (int, float))
            or not math.isfinite(synaptic_budget) or synaptic_budget <= 0):
        raise ValueError('synaptic_budget must be finite and positive')
    weights = torch.zeros_like(source.weights)
    edges = tuple(edge for edge in source.edges if edge.pre != edge.post)
    indices = {root_id: i for i, root_id in enumerate(source.root_ids)}
    incoming: dict[str, int] = {}
    for edge in edges:
        incoming[edge.post] = incoming.get(edge.post, 0) + edge.count
    for edge in edges:
        weights[indices[edge.post], indices[edge.pre]] = synaptic_budget * edge.count / incoming[edge.post]
    if not torch.isfinite(weights).all().item():
        raise ValueError('normalized BANC weight overflowed tensor dtype')
    return replace(source, edges=edges, _weights=weights)
