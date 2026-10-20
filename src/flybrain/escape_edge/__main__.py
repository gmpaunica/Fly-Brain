"""Launch only the verified edge controller; never start an experiment."""
import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,default=1110000)
    parser.add_argument('--headless',action='store_true')
    parser.add_argument('--steps',type=int,default=120)
    parser.add_argument('--record',type=Path)
    parser.add_argument('--capture',type=Path)
    args = parser.parse_args()
    if args.seed<0 or args.steps<0:
        parser.error('nonnegative seed and steps required')
    from .delivery import verified_candidate
    candidate = verified_candidate()
    if candidate is None:
        raise SystemExit('Sustained edge escape has no passing delivery receipt.')
    import torch
    torch.set_num_threads(1)
    if args.headless:
        from .session import EdgeSession
        session = EdgeSession(args.seed,candidate)
        for _ in range(args.steps):
            session.advance()
        if args.record:
            session.log.save(args.record)
        print(f'{session.snapshot.time:.2f}s | {session.snapshot.behavior_state}')
        return
    from .viewer import EdgeViewer
    app = EdgeViewer(args.seed,candidate,verified=True,offscreen=args.capture is not None)
    try:
        if args.capture:
            app.capture_paused(args.capture)
        else:
            app.run()
    finally:
        if args.record:
            app.session.log.save(args.record)
        app.destroy()


if __name__ == '__main__':
    main()
