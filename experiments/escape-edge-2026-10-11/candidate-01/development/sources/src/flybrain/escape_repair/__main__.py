"""Launch the verified repair; this command never runs an experiment."""
import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=1310100)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--steps', type=int, default=120)
    parser.add_argument('--record', type=Path)
    args = parser.parse_args()
    if args.seed < 0 or args.steps < 0:
        parser.error('seed and steps must be nonnegative')
    from .delivery import verified_candidate
    candidate = verified_candidate()
    if candidate is None:
        raise SystemExit('Escape repair has no passing delivery receipt.')
    import torch
    torch.set_num_threads(1)
    if args.headless:
        from .session import RepairSession
        session = RepairSession(args.seed, candidate)
        for _ in range(args.steps):
            session.advance()
        if args.record:
            session.log.save(args.record)
        print(f'{session.snapshot.time:.2f}s | {session.snapshot.behavior_state}')
        return
    from .viewer import RepairViewer
    app = RepairViewer(args.seed, candidate, verified=True)
    try:
        app.run()
    finally:
        if args.record:
            app.session.log.save(args.record)
        app.destroy()


if __name__ == '__main__':
    main()
