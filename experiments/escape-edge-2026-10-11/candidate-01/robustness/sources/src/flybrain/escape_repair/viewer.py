"""Existing live brain and encounter controls with the repaired session."""
import torch  # Windows: initialize the neural DLLs before Panda3D's DLLs.
from panda3d.core import WindowProperties
from flybrain.escape_recovery.viewer import RecoveryViewer
from flybrain.flight3d.activity import SpikeRecorder
from flybrain.retreat3d.pump import InteractionPump
from .session import RepairSession


class RepairViewer(RecoveryViewer):
    """Render actual activity; drawing never advances physics or neural state."""
    def __init__(self, seed, candidate, verified=False, **kwargs):
        self.repair_verified = verified
        super().__init__(seed, candidate, verified=False, **kwargs)
        self.simulation.close()
        self.session = RepairSession(seed, candidate)
        self.session.verified_default = verified
        self.session.snapshot = self.session._snapshot()
        self.simulation = InteractionPump(self.session, SpikeRecorder(self.session.snapshot))
        self._object_token = self._activity_token = self._body_token = None
        self._ui()
        self._flower()
        self.draw_frame(self.simulation.display_frame)
        if not self.offscreen:
            props = WindowProperties()
            props.setTitle('FlyBrain | Escape repair')
            self.win.requestProperties(props)

    def _ui(self):
        super()._ui()
        self.help.setText(self.help.getText().replace(
            'Escape unavailable: directional wing-spike gate has not passed.',
            'Escape: spike-selected turn, flee, visual clearance, braking and bounded recovery.'))

    def draw_frame(self, frame):
        super().draw_frame(frame)
        if not isinstance(getattr(self, 'session', None), RepairSession):
            return
        s = frame.snapshot
        self.title.setText('F L Y B R A I N   /   ESCAPE REPAIR')
        status = '48/48 measured escapes passed' if self.repair_verified else 'repair validation pending'
        self.hud.setText(f'{s.behavior_state.upper()} | {s.object_kind} | trial {s.seed} | {s.time:.2f}s'
                         + (' | PAUSED' if self.paused else '') + (' | CRASH' if s.crashed else '')
                         + f'\n{status} | size {s.angular_size_degrees:.1f} deg | power {s.command.power:.2f}')
        self.pathway_label.setText(status + '\nHistorical direction: 94/96; two known 86.25-degree failures.\n'
                                   + self.pathway_label.getText())
