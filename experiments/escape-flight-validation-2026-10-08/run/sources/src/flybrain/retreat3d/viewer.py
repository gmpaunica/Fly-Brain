"""Add idle/continuous encounters to retained anatomy, cameras and diagnostics."""
from pathlib import Path
import numpy as np
import torch  # Retain Windows DLL initialization order.
from panda3d.core import ClockObject, PNMImage, WindowProperties
from flybrain.encounters3d.viewer import EncounterViewer
from flybrain.encounters3d.diagnostic_panel import pathway_text
from flybrain.encounters3d.scene import PREDATOR_RADIUS, PREDATOR_RGB
from flybrain.flight3d.activity import SpikeRecorder
from flybrain.flight3d.anatomical_viewer import FlightViewer
from flybrain.flight3d.viewer import ellipsoid
from .world import PredatorInput
from .session import RetreatSession
from .pump import InteractionPump


class RetreatViewer(EncounterViewer):
    """Only actual brain spikes color anatomy. Reference subtraction is labeled."""
    def __init__(self, seed: int = 1000000, **kwargs):
        super().__init__(seed, 'flower', **kwargs)
        self.simulation.close()
        self.session = RetreatSession(seed)
        self.simulation = InteractionPump(self.session, SpikeRecorder(self.session.snapshot))
        self._object_token = None; self._activity_token = None; self._body_token = None
        self._flower(); self.draw_frame(self.simulation.display_frame)
        if not self.offscreen:
            props = WindowProperties(); props.setTitle('FlyBrain | Idle encounter lab | neural escape unavailable')
            self.win.requestProperties(props)

    def _ui(self) -> None:
        super()._ui()
        self.title.setText('F L Y B R A I N   /   IDLE ENCOUNTER LAB')
        for node in self.ui_nodes:
            if hasattr(node, 'getText') and node.getText() == '01   EXCLUSIVE OBJECT TRIAL':
                node.setText('01   CONTINUOUS FLIGHT & REST')
        self.help.setText('Engineered idle: takeoff, 6 cm/s cruise for seeded 3–6 s, safe landing, 2–5 s rest.\n'
            'Spawn flower / predator replaces the one object while flight continues.\n'
            'Predator: W/S arena +Y/-Y, A/D -X/+X, Q/E lower/raise; 8 cm/s maximum.\n'
            'Pause or focus loss releases held movement keys.\n'
            'R: fresh no-object trial. Backspace: replay recorded interaction to endpoint and pause.\n'
            'Flower: retained sensory teacher after takeoff.\n'
            'Escape unavailable: directional wing-spike gate has not passed.\n'
            'Baseline subtraction is a diagnostic/control calculation; anatomy shows actual spikes.\n'
            'Flight and stopping defaults are engineering assumptions, not measured fly behavior.\n\n'
            'Retained cameras, anatomy settings, classes, selection, pause and pathway diagnostics.')
        self.panels['help']['canvasSize'] = (0, .9*self.getAspectRatio(),
            min(-.5, float(self.help.getTightBounds()[0].z)-.04), 0)

    def _flower(self, snapshot=None) -> None:
        if not isinstance(self.session, RetreatSession):
            super()._flower(); return
        self.flower.getChildren().detach(); self._object_token = None
        s = snapshot or self.simulation.display_frame.snapshot
        if s.object_kind == 'flower':
            FlightViewer._flower(self)
        elif s.object_kind == 'predator':
            node = ellipsoid(self.flower, (0., 0., 0.), (PREDATOR_RADIUS,)*3,
                             (*PREDATOR_RGB, 1), 'predator-proxy')
            node.setPos(*s.object_position)
            node.setColor(*PREDATOR_RGB, 1)
        self._object_token = (s.object_kind, s.object_position)

    def draw_frame(self, frame) -> None:
        if not isinstance(self.session, RetreatSession):
            super().draw_frame(frame); return
        FlightViewer.draw_frame(self, frame)
        s = frame.snapshot
        token = (s.object_kind, s.object_position)
        if getattr(self, '_object_token', None) != token:
            if s.object_kind == 'predator' and getattr(self, '_object_token', (None,))[0] == 'predator':
                self.flower.find('**/predator-proxy').setPos(*s.object_position)
                self._object_token = token
            else:
                self._flower(s)
        diagnostic = pathway_text(frame, self.manifest).replace(
            'No escape decoder; spikes do not command retreat.',
            'Frozen decoder: diagnostic only; retreat gate failed.')
        diagnostic += f'\nNeural times relative to spawn at {s.neural_origin_seconds:.2f}s.'
        self.pathway_label.setText(diagnostic)
        self.hud.setText(f'{s.behavior_state.upper()} | {s.object_kind} | trial {s.seed} | {s.time:.2f}s'
            + (' | PAUSED' if self.paused else '') + (' | CRASH' if s.crashed else '') + '\n'
            f'Neural escape unavailable | size {s.angular_size_degrees:.1f}° | '
            f'growth +{s.angular_growth_degrees_s:.1f}°/s | power {s.command.power:.2f}')
        self.activity_label.setText('ACTUAL recorded brain | baseline subtraction: diagnostic only\n'
            f'Wing norm {np.linalg.norm(s.wing_increment):.2f} | cosine {s.decoder_similarity:.2f} | '
            f'margin {s.decoder_margin:.2f} | confident {s.decoder_confident}')

    def predator_input(self) -> PredatorInput:
        return PredatorInput.from_keys(self.keys)

    def release_input(self) -> None:
        """Drop worker credit and held input at a barrier, without altering pause."""
        self.keys.clear(); self.cancel_gesture()
        if isinstance(getattr(self, 'simulation', None), InteractionPump):
            paused = self.paused
            self.simulation.pause()
            with self.simulation._condition:
                self.simulation._credit = 0.; self.simulation._command = PredatorInput()
            if not paused:
                self.simulation.resume()

    def focus_changed(self, foreground: bool) -> None:
        super().focus_changed(foreground)
        if not foreground:
            self.release_input()

    def toggle_pause(self) -> None:
        self.release_input()
        super().toggle_pause()

    def spawn(self, kind: str) -> None:
        self.release_input()
        self.simulation.reset(lambda: self.session.spawn(kind))
        self._activity_token = None; self._body_token = None
        self._flower(); self.draw_frame(self.simulation.display_frame)

    def replay(self) -> None:
        if not isinstance(self.session, RetreatSession):
            super().replay(); return
        self.release_input(); self.paused = True
        self.simulation.replay_interaction()
        self._activity_token = None; self._body_token = None
        self._flower(); self.draw_frame(self.simulation.display_frame)

    def new_trial(self) -> None:
        self.release_input(); super().new_trial()

    def tick(self, task):
        if self.ui_size != (self.win.getXSize(), self.win.getYSize()):
            self._ui(); self._resize()
        dt = min(ClockObject.getGlobalClock().getDt(), .1)
        if self.drag:
            if self.mouseWatcherNode and self.mouseWatcherNode.hasMouse():
                mouse = self.mouseWatcherNode.getMouse(); self.move_gesture(mouse.x, mouse.y)
            else:
                self.cancel_gesture()
        self.brain_motion.advance(dt); self.world_motion.advance(dt)
        self.world_distance = self.world_motion.distance; self._brain_camera()
        if self.paused:
            self.simulation.pause()
        else:
            self.simulation.resume(); self.simulation.request(dt, self.predator_input())
        self.load_geometry(); self.draw_frame(self.simulation.display_frame)
        return task.cont

    def capture(self, directory: Path) -> None:
        """Saved-frame review is separate from a native interactive review."""
        directory.mkdir(parents=True, exist_ok=True)
        self.paused = True; self.simulation.pause(); self.load_geometry(100000)
        def save(name):
            self.draw_frame(self.simulation.display_frame)
            for _ in range(3):
                self.graphicsEngine.renderFrame(); self.graphicsEngine.syncFrame()
            picture = PNMImage(); self.win.getScreenshot(picture)
            picture.removeAlpha(); picture.write(str(directory/name))
        save('initial-grounded.png')
        self.simulation.advance_fixed(140); save('idle-cruise.png')
        self.spawn('predator'); self.simulation.advance_fixed(20, PredatorInput.from_keys({'a', 'e'}))
        save('moving-predator.png')
        self.session.log.save(directory/'actions.json')
        self.replay(); save('replay-endpoint.png')
