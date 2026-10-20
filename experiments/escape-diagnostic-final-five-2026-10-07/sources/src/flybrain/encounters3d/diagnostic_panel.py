"""Read-only pathway text built exclusively from immutable display publications."""
from .diagnostics import stage_indices


def pathway_text(frame, manifest: dict) -> str:
    """Summarize actual spikes, maximum incoming drive and threshold proximity.

    Maxima belong to individual cells; they are never presented as stage averages.
    First spike is episode time, including subinterval timing, not render time.
    """
    record = getattr(frame.snapshot, 'pathway', None)
    lines = ['Pathway diagnostics | completed 20 ms',
             'I = effective normalized increment / 1 ms']
    if record is None:
        lines += ['Visual neurons: waiting', 'DNp03 / DNp26: waiting',
                  'Wing-control neurons: waiting', 'No completed interval yet.']
        return '\n'.join(lines)
    for name, indices in stage_indices(manifest).items():
        spikes = sum(record.spike_counts[i] for i in indices)
        total = sum(frame.cumulative_counts[i] for i in indices)
        current = max((record.current_peak[i] for i in indices), default=0.)
        measure = record.peak_phase if name == 'Visual neurons' else record.peak_voltage
        peak = max((measure[i] for i in indices), default=0.)
        first = min((record.episode_first_spike_seconds[i] for i in indices
                     if record.episode_first_spike_seconds[i] is not None), default=None)
        timing = 'none' if first is None else f'{first:.3f}s'
        counter = 'phase' if name == 'Visual neurons' else 'V'
        first_line = f'  First spike {timing}'
        if name == 'Visual neurons':
            lost = sum(record.discarded_current[i] for i in indices)
            first_line += f' | I discarded sum {lost:.3f}'
        lines += [f'{name} ({len(indices)}) | spikes {spikes} / total {total}',
                  f'  I max {current:.3f} | {counter} peak {peak:.3f} / 1',
                  first_line]
    lines += ['Visual: rate-to-phase input; incoming I is discarded.',
              'No escape decoder; spikes do not command retreat.',
              'Direction/path diagnosis: saved investigation report.']
    return '\n'.join(lines)
