"""Reuse the tested one-child supervisor; interruption never permits restart."""
import sys
from flybrain.escape_flight_validation.worker import supervise
from flybrain.escape_repair.validation import RUN, ART

raise SystemExit(supervise([sys.executable, '-u', '-m', 'flybrain.escape_repair.validation'],
                          RUN, ART / 'experiment.log', ART / 'worker.json'))
