"""Receipt-gated prototype launcher; never invokes an experiment."""
import argparse
from pathlib import Path
import secrets


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int);parser.add_argument('--headless',action='store_true')
    parser.add_argument('--steps',type=int,default=120);parser.add_argument('--record',type=Path)
    args=parser.parse_args()
    if args.steps<0 or args.seed is not None and args.seed<0:parser.error('nonnegative seed/steps required')
    from .activation import verified_candidate
    candidate=verified_candidate()
    if candidate is None:raise SystemExit('Physical escape prototype not activated; see the saved validation result.')
    import torch
    torch.set_num_threads(1)
    seed=args.seed if args.seed is not None else 1200000+secrets.randbelow(100000000)
    if args.headless:
        from flybrain.escape_recovery.session import RecoverySession
        session=RecoverySession(seed,candidate)
        for _ in range(args.steps):session.advance()
        if args.record:session.log.save(args.record)
        print(f'{session.snapshot.time:.2f}s | {session.snapshot.behavior_state}');return
    from .viewer import PrototypeViewer
    app=PrototypeViewer(seed,candidate)
    try:app.run()
    finally:
        if args.record:app.session.log.save(args.record)
        app.destroy()


if __name__=='__main__':main()
