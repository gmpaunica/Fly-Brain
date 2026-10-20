"""Launch the isolated 3D flight lab without changing the established 2D app."""
import argparse
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,help='reproduce a flower sequence; otherwise choose a fresh launch seed')
    parser.add_argument('--control',choices=('teacher','manual','fixed','untrained','trained'),default='teacher')
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--capture',type=Path,help='save offscreen front/side/top and project-only frames')
    args=parser.parse_args()
    import secrets
    seed=args.seed if args.seed is not None else 1000000+secrets.randbelow(1000000000)
    import torch
    torch.set_num_threads(1)
    from .network_viewer import FlightViewer
    app=FlightViewer(seed,args.control,args.checkpoint,offscreen=args.capture is not None)
    if args.capture:
        try: app.capture(args.capture)
        finally: app.destroy()
    else: app.run()


if __name__=='__main__': main()
