"""Exact paired and heading-sector gates for a locked fresh food assay."""

from flybrain.training.food_fresh import final_gates, paired_exact_p, validation_gate


def test_exact_one_sided_p_uses_discordant_pairs():
    trained = [True] * 10 + [False] * 2
    control = [False] * 10 + [True] * 2
    assert paired_exact_p(trained, control) == 79 / 4096
    assert paired_exact_p([True] * 8, [False] * 8) == 1 / 256


def test_final_requires_each_heading_sector_and_both_paired_controls():
    rows = []
    for sector in range(8):
        for index in range(8):
            rows.append({'heading_sector': sector, 'success': index < 7})
    controls = [{'heading_sector': r['heading_sector'], 'success': False} for r in rows]
    result = final_gates(rows, controls, controls)
    assert not result['passed']  # 56/64 misses 58/64 despite every sector passing.
    rows[7]['success'] = rows[15]['success'] = True
    assert final_gates(rows, controls, controls)['passed']
    assert validation_gate([{'success': i < 14} for i in range(16)])
    assert not validation_gate([{'success': i < 13} for i in range(16)])
