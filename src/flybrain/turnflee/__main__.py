"""Gate-aware launcher. No default activation or experiment from opening a view."""
import argparse
from pathlib import Path
import secrets


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int); parser.add_argument('--headless',action='store_true')
    parser.add_argument('--steps',type=int,default=120); parser.add_argument('--capture',type=Path)
    parser.add_argument('--record',type=Path); parser.add_argument('--replay-log',type=Path)
    args=parser.parse_args()
    if args.steps<0 or args.seed is not None and args.seed<0: parser.error('nonnegative steps/seed required')
    import torch
    torch.set_num_threads(1)
    from .session import TurnFleeSession, promoted_config
    from flybrain.retreat3d.session import RetreatSession
    config=promoted_config(); seed=args.seed if args.seed is not None else 1100000+secrets.randbelow(1000000000)
    if args.headless:
        s=TurnFleeSession.from_log(args.replay_log) if args.replay_log else TurnFleeSession(seed,config) if config else RetreatSession(seed)
        if not args.replay_log:
            for _ in range(args.steps): s.advance()
        if args.record: s.log.save(args.record)
        print(f'{s.snapshot.behavior_state} | {s.snapshot.time:.2f}s | escape supported={s.snapshot.escape_supported}')
        return
    if args.replay_log: parser.error('--replay-log requires --headless; Backspace replays current interactive log')
    if config:
        from .viewer import TurnFleeViewer
        app=TurnFleeViewer(seed,config,offscreen=args.capture is not None)
    else:
        from flybrain.retreat3d.viewer import RetreatViewer
        app=RetreatViewer(seed,offscreen=args.capture is not None)
    try:
        if args.capture: app.capture(args.capture)
        else: app.run()
    finally:
        if args.record: app.session.log.save(args.record)
        app.destroy()


if __name__=='__main__': main()
