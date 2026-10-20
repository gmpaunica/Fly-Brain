"""Saved trajectories only; no neural simulation or policy calls."""
import gzip
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

art = Path('artifacts/escape-repair-2026-10-08')
run = Path('experiments/escape-repair-2026-10-08/run')
indices = (0, 4, 24, 26, 28)
titles = ('0 deg resting', '0 deg cruising', '180 deg resting', '180 deg hovering', '180 deg cruising')
new = {}
for index in indices:
    with gzip.open(run / 'traces' / f'{index:03}.jsonl.gz', 'rt') as stream:
        new[index] = [json.loads(line) for line in stream]
seeds = {rows[0]['seed']: index for index, rows in new.items()}
old = {index: [] for index in indices}
with gzip.open('experiments/escape-flight-validation-2026-10-08/run/traces.jsonl.gz', 'rt') as stream:
    for line in stream:
        row = json.loads(line)
        if row['seed'] in seeds and row['intervention'] == 'intact':
            old[seeds[row['seed']]].append(row)
fig, axes = plt.subplots(3, 5, figsize=(19, 10), layout='constrained')
for col, (index, title) in enumerate(zip(indices, titles)):
    for label, groups, color in (('Before', old, '#bb4e46'), ('Repaired', new, '#237fb5')):
        rows = groups[index]
        p = np.array([r['position'] for r in rows]) * 1000
        t = np.array([r['end_seconds'] for r in rows])
        speed = np.linalg.norm([r['velocity'][:2] for r in rows], axis=1) * 1000
        axes[0, col].plot(p[:, 0], p[:, 1], color=color, label=label)
        axes[0, col].scatter(*p[-1, :2], marker='x', color=color)
        axes[1, col].plot(t, [r['threat']['size_degrees'] for r in rows], color=color)
        axes[2, col].plot(t, speed, color=color)
        for r in rows:
            if r['recovered']:
                for ax in axes[1:, col]:
                    ax.axvline(r['end_seconds'], color=color, ls=':', alpha=.7)
                break
    predator = np.array(new[index][0]['predator_position']) * 1000
    axes[0, col].scatter(*predator[:2], marker='^', color='black', label='Predator')
    axes[0, col].set(xlim=(0, 600), ylim=(0, 600), xlabel='X (mm)', ylabel='Y (mm)', title=title)
    axes[0, col].set_aspect('equal')
    axes[1, col].axhline(10, color='grey', ls='--')
    axes[1, col].set(ylabel='Apparent size (deg)', xlabel='Time (s)')
    axes[2, col].axhline(20, color='grey', ls='--')
    axes[2, col].set(ylabel='Horizontal speed (mm/s)', xlabel='Time (s)')
    for ax in axes[:, col]:
        ax.grid(alpha=.2)
axes[0, 0].legend(fontsize=8)
fig.suptitle('Five historical physical escape failures: before and after repair\n'
             'Dotted lines: measured recovery. Dashed lines: weak size / braking-speed thresholds.', fontsize=16)
fig.savefig(art / 'five-failures-repaired.png', dpi=140)
plt.close(fig)
print('Saved five-case before/after plot; zero new neural steps.')
