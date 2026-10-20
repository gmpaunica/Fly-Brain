"""Retained anatomy/controls with explicit temporal evidence display."""
from panda3d.core import WindowProperties
from flybrain.turnflee.viewer import TurnFleeViewer
from flybrain.retreat3d.pump import InteractionPump
from flybrain.flight3d.activity import SpikeRecorder
from .session import RecoverySession


class RecoveryViewer(TurnFleeViewer):
    """New launcher only; no historical app/default promotion."""
    def __init__(self,seed,candidate,verified=False,**kwargs):
        super().__init__(seed,candidate.response,**kwargs);self.simulation.close()
        self.session=RecoverySession(seed,candidate);self.session.verified_default=verified
        self.session.snapshot=self.session._snapshot()
        self.simulation=InteractionPump(self.session,SpikeRecorder(self.session.snapshot))
        self._object_token=None;self._activity_token=None;self._body_token=None
        self._ui();self._flower();self.draw_frame(self.simulation.display_frame)
        if not self.offscreen:
            props=WindowProperties();props.setTitle('FlyBrain | Verified temporal escape recovery' if verified else 'FlyBrain | Experimental temporal escape recovery')
            self.win.requestProperties(props)

    def draw_frame(self,frame):
        super().draw_frame(frame)
        if not isinstance(getattr(self,'session',None),RecoverySession):return
        d=self.session.readout.last;s=frame.snapshot
        self.title.setText('F L Y B R A I N   /   TEMPORAL ESCAPE')
        self.pathway_label.setText('Actual spikes color anatomy | Cumulative paired steering spikes\n'
            +f'L {d.left:+d} / R {d.right:+d} | agreement {d.streak}/2 | side {s.selected_side or "pending"}\n'
            +'Exact ties defer for neural evidence | no random side\n'
            +f'Escape {s.escape_state} | {"verified" if s.escape_supported else "experimental"}\n'
            +self.pathway_label.getText())
        self.activity_label.setText(f'Temporal paired spikes | {s.escape_state} | {s.selected_side or "pending"}\n'
            +f'L {d.left:+d} / R {d.right:+d} | actual firing shown in anatomy')
