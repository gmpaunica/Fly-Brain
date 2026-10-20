"""Verified corrected-run launcher; never starts a scientific experiment."""
from .execution import RUN,ART
from . import activation

if __name__=='__main__':
    activation.RUN=RUN;activation.MARK=ART/'activation.json'
    from .__main__ import main
    main()
