"""Existing live brain/input UI displaying the new sustained escape session."""
import torch  # Initialize Windows neural DLLs before Panda3D.
import numpy as np
from pathlib import Path
from panda3d.core import WindowProperties
from flybrain.escape_recovery.viewer import RecoveryViewer
from flybrain.flight3d.activity import SpikeRecorder
from flybrain.retreat3d.pump import InteractionPump
from .session import EdgeSession


def review_predator_position(position: tuple[float,float,float]) -> tuple[float,float,float]:
    """Viewer-only encounter placement, including body poses near an edge."""
    from flybrain.encounters3d.scene import PREDATOR_RADIUS
    x,y,z = position
    y += .075 if y <= .3 else -.075
    return tuple(float(v) for v in np.clip((x,y,max(.021,z)),PREDATOR_RADIUS,
                                           np.array((.6,.6,.3))-PREDATOR_RADIUS))


class EdgeViewer(RecoveryViewer):
    """Drawing reads immutable frames; only the pump advances simulation."""

    def __init__(self,seed,candidate,verified=False,**kwargs):
        self.edge_verified = verified
        super().__init__(seed,candidate,verified=False,**kwargs)
        self.simulation.close()
        self.session = EdgeSession(seed,candidate)
        self.session.verified_default = verified
        self.session.snapshot = self.session._snapshot()
        self.simulation = InteractionPump(self.session,SpikeRecorder(self.session.snapshot))
        self._object_token = self._activity_token = self._body_token = None
        self._ui()
        self._flower()
        self.draw_frame(self.simulation.display_frame)
        if not self.offscreen:
            props = WindowProperties()
            props.setTitle('FlyBrain | Sustained edge escape')
            self.win.requestProperties(props)

    def _ui(self):
        super()._ui()
        self.help.setText(self.help.getText().replace(
            'Escape: actual-minus-reference i1/b3/b1 spikes plus strong threat; 90 degree turn then 12 cm/s.',
            'Predator: turn, keep flying to the map edge, then hover. Shrinking threat does not stop flight.').replace(
            '200 ms rolling spike subtraction; anatomy shows actual spikes.',
            'Neural activity selects the turn; anatomy shows actual spikes.'))

    def draw_frame(self,frame):
        super().draw_frame(frame)
        if not isinstance(getattr(self,'session',None),EdgeSession):
            return
        s = frame.snapshot
        self.title.setText('F L Y B R A I N   /   SUSTAINED EDGE ESCAPE')
        status = 'Edge flight verified on 48 intact cases' if self.edge_verified else 'Edge flight validation pending'
        self.hud.setText(f'{s.behavior_state.upper()} | {s.object_kind} | trial {s.seed} | {s.time:.2f}s'
            + (' | PAUSED' if self.paused else '') + (' | CRASH' if s.crashed else '')
            + f'\n{status} | power {s.command.power:.2f}')
        from flybrain.encounters3d.diagnostic_panel import pathway_text
        d = self.session.readout.last
        self.pathway_label.setText(status+'\nHistorical direction: 94/96; two known 86.25-degree failures.\n'
            + f'Escape {s.escape_state} | side {s.selected_side or "pending"} | spikes L {d.left:+d} / R {d.right:+d}\n'
            + 'Brain triggers turn; flight to edge uses an engineered controller.\n'
            + pathway_text(frame,self.manifest).replace('No escape decoder; spikes do not command retreat.',
                'Cumulative paired wing spikes select turn side.'))

    def spawn(self,kind):
        """Make the review encounter visible during flight/edge hover."""
        if kind=='predator':
            self.release_input()
            # Viewer/evaluator geometry only. Session control receives senses.
            def spawn_at_visible_range():
                pos = self.session.world.position
                self.session.spawn(kind,review_predator_position(tuple(pos)))
            self.simulation.reset(spawn_at_visible_range)
            self._activity_token = self._body_token = None
            self._flower()
            self.draw_frame(self.simulation.display_frame)
        else:
            super().spawn(kind)

    def capture_paused(self,path: Path) -> None:
        """Render with normal frame tasks while physics and neural state stay paused."""
        from panda3d.core import PNMImage, Filename
        self.paused = True
        self.simulation.pause()
        before = self.session.snapshot
        for _ in range(20):
            self.taskMgr.step()
            self.graphicsEngine.syncFrame()
        if self.session.snapshot != before:
            raise RuntimeError('rendering advanced simulation state')
        image = PNMImage()
        self.win.getScreenshot(image)
        if not image.write(Filename.fromOsSpecific(str(path))):
            raise RuntimeError('frame write failed')
