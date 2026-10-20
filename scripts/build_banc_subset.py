"""Regenerate the small FB-09 package artifact from pinned BANC v888 files.

Usage: python scripts/build_banc_subset.py SOURCE_DIRECTORY OUTPUT_JSON
Requires optional pyarrow; PyTorch and the live application do not.
"""

import hashlib
import json
import math
from pathlib import Path
import sys

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather


SOURCE = {
    'dataset_doi': '10.7910/DVN/7WTH1N',
    'release': '3.0',
    'materialization': '888',
    'license': 'CC BY 4.0',
    'metadata': {'filename': 'banc_888_meta.feather', 'dataverse_file_id': 14033740,
                 'bytes': 57550610, 'md5': '6275eda42f98c49539d1ab513d979d09'},
    'connectivity': {'filename': 'banc_888_edgelist_simple_v2.feather',
                     'dataverse_file_id': 13992792, 'bytes': 305250378,
                     'md5': '394406f8a9bdf093c895f95aff4f6c49'},
}
PN = {'left': '720575941653217813', 'right': '720575941497888384'}
DNA02 = {'left': '720575941510475536', 'right': '720575941456897005'}
EXPECTED_PATHS = {
    'left': (PN['left'], '720575941539370133', '720575941594458987',
             '720575941500284745', DNA02['left']),
    'right': (PN['right'], '720575941504828722',
              '720575941645545928', DNA02['right']),
}
EXPECTED_MOTORS = {
    'left': ('720575941531923147', '720575941519664856',
             '720575941538992021'),
    'right': ('720575941531180261', '720575941568355066',
              '720575941574355016'),
}
META_COLUMNS = ('root_888', 'proofread', 'status', 'side', 'super_class',
                'cell_class', 'cell_sub_class', 'cell_type',
                'body_part_effector', 'peripheral_target_type',
                'cell_function', 'cell_function_detailed',
                'neurotransmitter_predicted', 'neurotransmitter_score',
                'neurotransmitter_verified')


def verify_archive_file(directory: Path, key: str) -> Path:
    """Reject a moving or incomplete source file before parsing it."""
    record = SOURCE[key]
    path = directory / record['filename']
    if path.stat().st_size != record['bytes']:
        raise ValueError(f'{key} size does not match pinned release')
    digest = hashlib.md5(usedforsecurity=False)
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != record['md5']:
        raise ValueError(f'{key} MD5 does not match pinned release')
    return path


def arrow_members(table: pa.Table, column: str, values: set[str]) -> pa.Table:
    """Filter a Feather table by opaque string root IDs."""
    return table.filter(pc.is_in(table[column], value_set=pa.array(sorted(values))))


def build(source_dir: Path) -> dict[str, object]:
    """Select the audited VA2 to DNa02 to leg motor corridor."""
    metadata_path = verify_archive_file(source_dir, 'metadata')
    edge_path = verify_archive_file(source_dir, 'connectivity')
    metadata = feather.read_table(metadata_path, columns=META_COLUMNS)
    edges = feather.read_table(edge_path)

    # Keep annotation rows narrow; edge-to-PN queries stay columnar.
    orn_rows = metadata.filter(pc.and_(
        pc.equal(metadata['proofread'], 'TRUE'),
        pc.equal(metadata['cell_type'], 'ORN_VA2'))).to_pylist()
    orn_by_side = {
        side: {row['root_888'] for row in orn_rows if row['side'] == side}
        for side in PN
    }
    selected: set[str] = set(PN.values()) | set(DNA02.values())
    roles: dict[str, str] = {}
    for side, pn in PN.items():
        incoming = edges.filter(pc.and_(pc.equal(edges['post'], pn),
                                        pc.greater_equal(edges['count'], 10)))
        qualified = {row['pre'] for row in incoming.select(['pre']).to_pylist()}
        sensory = orn_by_side[side] & qualified
        if len(sensory) != (31 if side == 'left' else 37):
            raise ValueError(f'unexpected {side} VA2 ORN selection')
        selected.update(sensory)
        roles.update({root_id: f'sensory_{side}' for root_id in sensory})
        roles[pn] = f'projection_{side}'
        roles[DNA02[side]] = f'steering_{side}'

    # These are explicitly curated corridor IDs from the feasibility audit.
    # They are not the output of a global strongest-first BFS on all BANC cells.
    route_sources = {root_id for path in EXPECTED_PATHS.values()
                     for root_id in path[:-1]} | set(DNA02.values())
    strong = arrow_members(edges.filter(pc.greater_equal(edges['count'], 10)),
                           'pre', route_sources).select(['pre', 'post', 'count'])
    adjacency: dict[str, list[tuple[str, int]]] = {}
    for row in strong.to_pylist():
        adjacency.setdefault(row['pre'], []).append((row['post'], row['count']))
    for side, path in EXPECTED_PATHS.items():
        if any(post not in {target for target, _count in adjacency.get(pre, ())}
               for pre, post in zip(path, path[1:])):
            raise ValueError(f'{side} pinned corridor edge is absent')
        for root_id in path[1:-1]:
            roles[root_id] = f'corridor_{side}'
            selected.add(root_id)

    motor_meta = metadata.filter(pc.and_(
        pc.equal(metadata['proofread'], 'TRUE'),
        pc.equal(metadata['cell_function_detailed'], 'move_coxa_anterior')))
    motor_lookup = {row['root_888']: row for row in motor_meta.to_pylist()
                    if row['cell_function'] == 'leg_motor'}
    for side, dna in DNA02.items():
        candidates = [(post, count) for post, count in adjacency[dna]
                      if post in motor_lookup and motor_lookup[post]['side'] == side]
        for leg, expected in zip(('front_leg', 'middle_leg', 'hind_leg'),
                                 EXPECTED_MOTORS[side]):
            ranked = sorted((post, count) for post, count in candidates
                            if motor_lookup[post]['body_part_effector'] == leg)
            ranked.sort(key=lambda item: (-item[1], item[0]))
            if not ranked or ranked[0][0] != expected:
                raise ValueError(f'{side} {leg} motor selection changed')
            selected.add(expected)
            roles[expected] = f'motor_{side}'

    selected_meta = arrow_members(metadata, 'root_888', selected).to_pylist()
    if len(selected_meta) != len(selected) or len(selected) != 83:
        raise ValueError('missing or duplicated selected metadata')
    node_rows = []
    for row in selected_meta:
        clean = {key: (None if isinstance(value, float) and math.isnan(value)
                       else value) for key, value in row.items()}
        clean['role'] = roles[row['root_888']]
        node_rows.append(clean)
    node_rows.sort(key=lambda row: row['root_888'])

    retained = edges.filter(pc.greater_equal(edges['count'], 5))
    retained = arrow_members(arrow_members(retained, 'pre', selected),
                             'post', selected)
    edge_rows = retained.select(['pre', 'post', 'count', 'norm',
                                 'post_count', 'pre_count']).to_pylist()
    edge_rows.sort(key=lambda row: (row['pre'], row['post']))
    if (len(node_rows), len(edge_rows), sum(row['count'] for row in edge_rows),
            sum(row['pre'] == row['post'] for row in edge_rows)) != (83, 176, 6062, 31):
        raise ValueError('source subset totals changed')
    return {'source': SOURCE, 'nodes': node_rows, 'edges': edge_rows}


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit('usage: build_banc_subset.py SOURCE_DIRECTORY OUTPUT_JSON')
    payload = build(Path(sys.argv[1]))
    output = Path(sys.argv[2])
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False,
                                 allow_nan=False) + '\n', encoding='utf-8')
    print(f'Wrote {len(payload["nodes"])} nodes and '
          f'{len(payload["edges"])} edges to {output}')


if __name__ == '__main__':
    main()
