"""Plot the closed FB-09 contrast experiment from saved JSON, never simulate.

Usage: python scripts/plot_banc_recovery.py [RESULTS_JSON] [OUTPUT_PNG]
An existing output is protected; select an unused path for another rendering.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np


def plot(results_path: Path, output_path: Path) -> None:
    """Render recorded directional shifts and their predeclared thresholds."""
    if output_path.exists():
        raise FileExistsError(f'Choose an unused plot destination: {output_path}')
    payload = json.loads(results_path.read_text(encoding='utf-8'))
    candidates = payload['candidates']
    fig, axes = plt.subplots(1, len(candidates), figsize=(12, 4.8), sharey=True)
    colors = ('#2166ac', '#d66018')
    for axis, candidate in zip(np.atleast_1d(axes), candidates):
        rows = candidate['contrasts']
        centers = sorted(rows, key=float)
        locations = np.arange(len(centers), dtype=float)
        for offset, direction, color in zip((-.18, .18), ('right', 'left'), colors):
            shifts = [rows[center][f'{direction}_shift_hz'] for center in centers]
            thresholds = [rows[center][f'{direction}_minimum_hz'] for center in centers]
            axis.bar(locations + offset, shifts, width=.32, color=color, zorder=3)
            axis.plot(locations + offset, thresholds, linestyle='none', marker='_',
                      markersize=19, markeredgewidth=2.1, color='#111827', zorder=4)
        axis.axhline(0, color='#475569', linewidth=.9)
        axis.set_xticks(locations, centers)
        axis.set_xlabel('Center odor concentration (arbitrary units)')
        axis.set_title(f"Row budget {candidate['config']['synaptic_budget']:g}")
        axis.spines[['top', 'right']].set_visible(False)
        axis.grid(axis='y', color='#e2e8f0', zorder=0)
        axis.set_axisbelow(True)
    np.atleast_1d(axes)[0].set_ylabel('Directional response shift (Hz)')
    fig.suptitle('FB-09: directional contrast measurements', fontsize=16, x=.055, ha='left', y=.99)
    fig.text(.055, .90,
             'DNa02 right-minus-left rate, relative to equal odor. Left-strong shifts use the opposite sign.\n'
             'Each bar must reach its black threshold; negative bars respond in the wrong direction.',
             fontsize=10, color='#334155')
    handles = [Patch(facecolor=colors[0], label='Right odor stronger'),
               Patch(facecolor=colors[1], label='Left odor stronger'),
               Line2D([], [], color='#111827', marker='_', markersize=18,
                      markeredgewidth=2, linestyle='none', label='Required minimum')]
    fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(.5, -.015))
    fig.subplots_adjust(left=.075, right=.99, bottom=.18, top=.79, wspace=.12)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    print(f'Plotted recorded results to {output_path}')


def main() -> None:
    """Choose evidence and an unused output path; no neural code is imported."""
    root = Path(__file__).resolve().parents[1]
    default_dir = root / 'experiments/fb09-dynamics-recovery'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', nargs='?', type=Path, default=default_dir / 'results.json')
    parser.add_argument('output', nargs='?', type=Path, default=default_dir / 'contrast_gates.png')
    args = parser.parse_args()
    plot(args.results, args.output)


if __name__ == '__main__':
    main()
