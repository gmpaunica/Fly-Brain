"""Launch the polished demo with the original source-bound verified controller."""
import argparse
from pathlib import Path


def main() -> None:
    """Open the interactive demo or capture one paused, zero-step frame."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=1110000)
    parser.add_argument('--capture', type=Path, help='Save a paused zero-step PNG and exit')
    args = parser.parse_args()
    if args.seed < 0:
        parser.error('nonnegative seed required')
    from flybrain.escape_edge.delivery import verified_candidate
    candidate = verified_candidate()
    if candidate is None:
        raise SystemExit('Sustained edge escape has no passing delivery receipt.')
    import torch
    torch.set_num_threads(1)
    from .viewer import DemoViewer
    app = DemoViewer(args.seed, candidate, verified=True, offscreen=args.capture is not None)
    try:
        if args.capture:
            app.capture_paused(args.capture)
        else:
            app.run()
    finally:
        app.destroy()


if __name__ == '__main__':
    main()
