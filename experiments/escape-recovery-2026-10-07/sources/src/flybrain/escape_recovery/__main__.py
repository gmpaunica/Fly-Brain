"""Verified new recovery launcher; opening the app never runs an experiment."""
import argparse
from pathlib import Path
import secrets


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seed',type=int);p.add_argument('--headless',action='store_true')
    p.add_argument('--steps',type=int,default=120);p.add_argument('--record',type=Path);p.add_argument('--replay-log',type=Path)
    args=p.parse_args()
    if args.steps<0 or args.seed is not None and args.seed<0:p.error("nonnegative steps/seed required")
    if args.replay_log and not args.headless:p.error("replay-log requires headless")
    import torch
    torch.set_num_threads(1)
    from .activation import verified_candidate
    from .session import RecoverySession
    c=verified_candidate()
    if c is None:raise SystemExit('Escape recovery has not passed/been activated. Historical launchers remain available.')
    seed=args.seed if args.seed is not None else 1200000+secrets.randbelow(100000000)
    if args.headless:
        s=RecoverySession.from_log(args.replay_log) if args.replay_log else RecoverySession(seed,c)
        if not args.replay_log:
            for _ in range(args.steps):s.advance()
        if args.record:s.log.save(args.record)
        print(f'{s.snapshot.time:.2f}s | {s.snapshot.behavior_state}');return
    from .viewer import RecoveryViewer
    app=RecoveryViewer(seed,c,verified=True)
    try:app.run()
    finally:
        if args.record:app.session.log.save(args.record)
        app.destroy()


if __name__=='__main__':main()
