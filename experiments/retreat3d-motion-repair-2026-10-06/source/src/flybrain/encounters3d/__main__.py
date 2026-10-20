"""Launch additive exclusive 3D trials; no training/checkpoint continuation."""
import argparse
from pathlib import Path
import secrets


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int)
    parser.add_argument('--kind',choices=('flower','predator'),default='flower')
    parser.add_argument('--capture',type=Path)
    parser.add_argument('--headless',action='store_true')
    parser.add_argument('--steps',type=int,default=120)
    args=parser.parse_args()
    seed=args.seed if args.seed is not None else 1000000+secrets.randbelow(1000000000)
    import torch
    torch.set_num_threads(1)
    if args.headless:
        if args.steps<0: parser.error('--steps must be nonnegative')
        from .scene import make_scene
        from .session import EncounterSession
        session=EncounterSession(make_scene(seed,args.kind))
        for _ in range(args.steps): session.advance()
        print(f'{session.mode} | {args.kind} | seed {seed} | {session.snapshot.time:.2f}s')
        return
    from .viewer import EncounterViewer
    app=EncounterViewer(seed,args.kind,offscreen=args.capture is not None)
    if args.capture:
        try: app.capture(args.capture)
        finally: app.destroy()
    else:
        app.run()


if __name__=='__main__': main()
