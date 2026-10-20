"""Launch additive idle encounter lab; never run training or scientific retries."""
import argparse
from pathlib import Path
import secrets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--steps', type=int, default=120)
    parser.add_argument('--capture', type=Path)
    parser.add_argument('--record', type=Path)
    parser.add_argument('--replay-log', type=Path)
    args = parser.parse_args()
    if args.steps < 0 or (args.seed is not None and args.seed < 0):
        parser.error('nonnegative steps/seed required')
    import torch
    torch.set_num_threads(1)
    from .session import RetreatSession
    seed = args.seed if args.seed is not None else 1000000+secrets.randbelow(1000000000)
    if args.headless:
        session = RetreatSession.from_log(args.replay_log) if args.replay_log else RetreatSession(seed)
        if not args.replay_log:
            for _ in range(args.steps): session.advance()
        if args.record: session.log.save(args.record)
        s = session.snapshot
        print(f'{s.behavior_state} | {s.object_kind} | seed {s.seed} | {s.time:.2f}s | crashed={s.crashed} | neural escape unavailable')
        return
    if args.replay_log:
        parser.error('--replay-log currently requires --headless; interactive Backspace replays the current log')
    from .viewer import RetreatViewer
    app = RetreatViewer(seed, offscreen=args.capture is not None)
    try:
        if args.capture:
            app.capture(args.capture)
        else:
            app.run()
    finally:
        if args.record: app.session.log.save(args.record)
        app.destroy()


if __name__ == '__main__':
    main()
