"""Capture dashboard/filter/inspection/help at two sizes with zero neural steps."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import torch

from flybrain.escape_edge.delivery import verified_candidate
from flybrain.escape_diagnostic_recovery.runner import verify_hashes
from flybrain.escape_ui.viewer import DemoViewer
from panda3d.core import Filename, PNMImage


def main() -> None:
    """Save actual frame-task renders and separately report unchanged state."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--size', type=int, nargs=2, metavar=('WIDTH', 'HEIGHT'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    candidate = verified_candidate()
    if candidate is None:
        raise SystemExit('Verified edge candidate unavailable')
    protected = json.loads(Path('artifacts/escape-edge-2026-10-11/protected.json').read_text())
    verify_hashes(protected)
    frames = []
    header_pixels = {}
    # Each size follows the actual launcher lifecycle with a fresh process and
    # graphics context; pixel checks complement image-tool preview inspection.
    if args.size is None:
        for width, height in ((1000, 700), (1440, 900)):
            subprocess.run([sys.executable, __file__, str(args.output),
                            '--size', str(width), str(height)], check=True)
            record = json.loads((args.output / f'capture-{width}.json').read_text())
            frames.extend(record['frames'])
            header_pixels.update(record['bright_header_pixels'])
    for size in (() if args.size is None else (tuple(args.size),)):
        app = DemoViewer(1110000, candidate, verified=True,
                         offscreen=True, window_size=size)
        try:
            app.paused = True
            app.simulation.pause()
            app.load_geometry(100000)
            app.fit_view()
            app.brain_motion.settle()
            app._brain_camera()
            before = app.session.snapshot
            for name in ('dashboard', 'filtered', 'inspect', 'help', 'empty'):
                if name == 'filtered':
                    app.show_only_group('vision')
                elif name == 'inspect':
                    owner = min(app.activity_index.members['vision'])
                    app.select(app.activity_index.ids[owner])
                elif name == 'help':
                    app.show_tab('help')
                elif name == 'empty':
                    app.hide_all_groups()
                    app.show_tab('groups')
                path = args.output / f'{name}-{size[0]}.png'
                if path.exists():
                    raise FileExistsError(path)
                app.capture_paused(path)
                image = PNMImage()
                assert image.read(Filename.fromOsSpecific(str(path)))
                bright = sum(min(image.getXel(x, y)) > .3
                    for x in range(round(size[0] * .45))
                    for y in range(round(size[1] * .12)))
                assert bright > 100, 'header labels missing from captured frame'
                header_pixels[path.name] = bright
                assert app.session.snapshot == before
                assert app.session.brain.steps == app.session.pair.reference.steps == 0
                assert app.session.world.time == 0
                frames.append(path.as_posix())
        finally:
            app.destroy()
    verify_hashes(protected)
    assert verified_candidate().identity == candidate.identity
    result = dict(status='captures-pass', simulation_seconds=0, neural_steps=0,
                  protected_files=len(protected), frames=frames,
                  bright_header_pixels=header_pixels,
                  visual_review='pending', native_input_review='not performed')
    name = 'capture.json' if args.size is None else f'capture-{args.size[0]}.json'
    with (args.output / name).open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
