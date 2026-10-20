"""Passive response diagnostics on the retained actual-spike anatomy view."""
from panda3d.core import WindowProperties
from flybrain.retreat3d.viewer import RetreatViewer
from flybrain.retreat3d.pump import InteractionPump
from flybrain.flight3d.activity import SpikeRecorder
from flybrain.encounters3d.diagnostic_panel import pathway_text
from .session import TurnFleeSession
from .neural import ResponseConfig


def response_text(snapshot) -> str:
    """Format one immutable publication; no stepping or neural state access."""
    config=snapshot.response_config
    return (f'Actual spikes color anatomy | Response gains {config.gains} | {config.identity[:12]}\n'
            f'Wing scores L {snapshot.wing_scores[0]:+.0f} / R {snapshot.wing_scores[1]:+.0f} | '
            f'steering increment norm {snapshot.response_strength:.2f}\n'
            f'Escape {snapshot.escape_state} | side {snapshot.selected_side or "none"} | '
            f'home guard {snapshot.home_guard}\n'
            f'Exact bilateral tie: engineering symmetry breaking ({snapshot.symmetry_breaking})\n'
            f'Verification: {"promoted" if snapshot.escape_supported else "experimental; default disabled"} | '
            f'{config.verification_receipt}')


class TurnFleeViewer(RetreatViewer):
    """Construct only with an explicitly verified/promoted or development config."""
    def __init__(self, seed: int, config: ResponseConfig, **kwargs):
        super().__init__(seed,**kwargs); self.simulation.close()
        self.session=TurnFleeSession(seed,config)
        self.simulation=InteractionPump(self.session,SpikeRecorder(self.session.snapshot))
        self._object_token=None; self._activity_token=None; self._body_token=None
        self._ui(); self._flower(); self.draw_frame(self.simulation.display_frame)
        if not self.offscreen:
            props=WindowProperties(); props.setTitle('FlyBrain | Turn-and-flee | engineered gains and flight')
            self.win.requestProperties(props)

    def _ui(self) -> None:
        super()._ui()
        if isinstance(getattr(self,'session',None),TurnFleeSession):
            self.title.setText('F L Y B R A I N   /   TURN & FLEE')
            self.help.setText(self.help.getText().replace(
                'Escape unavailable: directional wing-spike gate has not passed.',
                'Escape: actual-minus-reference i1/b3/b1 spikes plus strong threat; 90 degree turn then 12 cm/s.').replace(
                'Baseline subtraction is a diagnostic/control calculation; anatomy shows actual spikes.',
                '200 ms rolling spike subtraction; anatomy shows actual spikes. Gains and flight servo are engineering assumptions.'))

    def draw_frame(self, frame) -> None:
        super().draw_frame(frame)
        if not isinstance(getattr(self,'session',None),TurnFleeSession): return
        s=frame.snapshot
        self.pathway_label.setText(response_text(s)+'\n\n'+pathway_text(frame,self.manifest).replace(
            'No escape decoder; spikes do not command retreat.',
            'Spike difference selects turn side; voltage is diagnostic only.')+
            f'\nNeural times relative to spawn at {s.neural_origin_seconds:.2f}s.')
        self.hud.setText(f'{s.behavior_state.upper()} | {s.object_kind} | trial {s.seed} | {s.time:.2f}s'
            +(' | PAUSED' if self.paused else '')+(' | CRASH' if s.crashed else '')+'\n'
            +f'{s.selected_side or "no turn"} | strength {s.response_strength:.2f} | size {s.angular_size_degrees:.1f} deg | power {s.command.power:.2f}')
        self.activity_label.setText(f'Actual spikes | Response gains {s.response_config.gains} | '
            f'{"verified" if s.escape_supported else "experimental; default disabled"}\n'
            f'{s.escape_state} / {s.selected_side or "none"} | '
            f'L {s.wing_scores[0]:+.0f} / R {s.wing_scores[1]:+.0f} | strength {s.response_strength:.2f}')
