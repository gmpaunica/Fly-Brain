"""Add exclusive spawn controls to the retained anatomical 3D viewer."""
from pathlib import Path
import torch  # Windows native DLL ordering, as in the retained viewer.
from direct.gui.DirectGui import DirectButton
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import PNMImage, WindowProperties, TextNode
from flybrain.flight3d.activity import SpikeRecorder
from flybrain.flight3d.anatomical_viewer import FlightViewer
from flybrain.flight3d.simulation_pump import SimulationPump
from flybrain.flight3d.viewer import ellipsoid
from .scene import PREDATOR_RADIUS, PREDATOR_RGB, make_scene
from .session import EncounterSession
from .diagnostic_panel import pathway_text


class EncounterViewer(FlightViewer):
    """Use original anatomy/cameras/display history; install an additive session.

    No historical viewer factory or neural method is monkeypatched. Its initial
    teacher session is closed before installing the exclusive encounter session.
    The source manifest and geometry ordering are identical.
    """
    def __init__(self, seed: int=980000, kind: str='flower', **kwargs):
        super().__init__(seed,'teacher',**kwargs)
        self.simulation.close()
        self.session=EncounterSession(make_scene(seed,kind))
        self.simulation=SimulationPump(self.session,SpikeRecorder(self.session.snapshot))
        self._activity_token=None; self._body_token=None
        self.all_activity()  # Encounter launch shows the circuit; later choices persist.
        self._flower(); self.draw_frame(self.simulation.display_frame)
        if not self.offscreen:
            properties=WindowProperties()
            properties.setTitle('FlyBrain | Pathway diagnostics | avoidance unavailable')
            self.win.requestProperties(properties)

    def _ui(self) -> None:
        super()._ui()
        self.title.setText('F L Y B R A I N   /   ENCOUNTER LAB')
        for node in self.ui_nodes:
            if hasattr(node,'getText') and node.getText()=='01   FLOWER LANDING':
                node.setText('01   EXCLUSIVE OBJECT TRIAL')
        # Replace the bottom toolbar in the subclass, retaining all other UI.
        for node in tuple(self.ui_nodes):
            if isinstance(node,DirectButton) and abs(node.getZ()+.92)<1e-5:
                node.destroy(); self.ui_nodes.remove(node)
        def button(value: str, x: float, command):
            node=DirectButton(text=value,pos=(x*self.getAspectRatio(),0,-.92),
                scale=min(.023,self.getAspectRatio()*.015),command=command,
                frameColor=(.11,.09,.20,1),text_fg=(.93,.91,.99,1),pad=(.65,.45),relief=1)
            self.ui_nodes.append(node)
            return node
        button('Pause / Space',-.84,self.toggle_pause)
        self.spawn_flower_button=button('Spawn flower',-.52,lambda:self.spawn('flower'))
        self.spawn_predator_button=button('Spawn predator',-.20,lambda:self.spawn('predator'))
        button('Replay / Backspace',.20,self.replay)
        button('Neurons',.56,self.list_groups)
        button('Exit / Esc',.83,self.userExit)
        helptext=self.help.getText().replace(
            'Manual: Up/Down power; W/S pitch; A/D roll; Q/E yaw; L legs.\n\n','')
        self.help.setText(helptext+'\n\nSpawn: one fresh object only. R: new selected object; Backspace: exact replay.\n'
            'Flower: unchanged teacher-controlled landing.\n'
            'Cyan sphere: stationary predator proxy; RGB recognition is engineered.\n'
            'Escape gate failed: local hover only; no escape decoder or training.\n'
            'DNp03 signal is recorded spiking, not a claim of subjective fear.')
        self.panels['help']['canvasSize']=(0,.9*self.getAspectRatio(),
                                          min(-.4,float(self.help.getTightBounds()[0].z)-.04),0)
        # Add a diagnostics section ahead of the retained legend and controls.
        canvas=self.panels['help'].getCanvas()
        for child in canvas.getChildren(): child.setZ(child.getZ()-.47)
        self.pathway_label=OnscreenText(parent=canvas,text='Pathway diagnostics',
            pos=(.018,-.030),align=TextNode.ALeft,scale=.023,
            fg=(.88,.86,.97,1),mayChange=True,wordwrap=max(30,self.getAspectRatio()*38))
        self.ui_nodes.append(self.pathway_label)
        self.panels['help']['canvasSize']=(0,.9*self.getAspectRatio(),
            min(-.5,float(self.help.getTightBounds()[0].z)-.04),0)

    def _flower(self) -> None:
        if not isinstance(self.session,EncounterSession) or self.session.scene.kind=='flower':
            super()._flower()
            return
        self.flower.getChildren().detach()
        predator=ellipsoid(self.flower,self.session.scene.position,(PREDATOR_RADIUS,)*3,
                           (*PREDATOR_RGB,1),'predator-proxy')
        # Use explicit surface color, matching the unshaded simulated eyes.
        # On this WGL path the historical packed vertex colors swapped R/B.
        # A flat ColorAttrib avoids that path without changing old renderers.
        predator.setColor(*PREDATOR_RGB,1)

    def spawn(self, kind: str) -> None:
        """Barrier-reset trial and display history while retaining preferences."""
        self.simulation.reset(lambda:self.session.spawn(kind))
        self._flower(); self.accumulator=0.
        self._activity_token=None; self._body_token=None
        self.draw_frame(self.simulation.display_frame)

    def draw_frame(self, frame) -> None:
        super().draw_frame(frame)
        s=frame.snapshot
        if not hasattr(s,'object_kind'): return  # original base initialization
        self.pathway_label.setText(pathway_text(frame,self.manifest))
        status='Teacher-controlled flower' if s.object_kind=='flower' else 'Hover stabilizer | Avoidance unavailable'
        self.hud.setText(f'{status}\nTrial {s.seed} | {s.time:.2f}s | power {s.command.power:.2f}'
                         +(' | PAUSED' if self.paused else '')+(' | IMPACT' if s.crashed else '')
                         +(f' | food dwell {s.dwell:.2f}s' if s.object_kind=='flower' else ''))
        if s.object_kind=='flower':
            self.activity_label.setText('OBSERVATIONAL recorded activity\nTeacher supplies flight commands')
        else:
            dn=[i for i,n in enumerate(self.manifest['nodes']) if n['metadata'].get('cell_type')=='DNp03']
            text=' / '.join(f"{self.manifest['nodes'][i]['metadata']['side'][0].upper()} {frame.firing_rates_hz[i]:.1f}" for i in dn)
            motor=sum(s.spike_counts[i] for i,n in enumerate(self.manifest['nodes'])
                      if set(n['roles']) & {'wing_power','wing_steering'})
            self.activity_label.setText(f'DNp03 recorded: {text} Hz\nWing endpoints: {motor} spikes this interval')

    def replay(self) -> None:
        super().replay()
        self.draw_frame(self.simulation.display_frame)

    def new_trial(self) -> None:
        super().new_trial()
        self.draw_frame(self.simulation.display_frame)

    def capture(self, directory: Path) -> None:
        """Save labeled encounter frames; capture is separate from live review."""
        directory.mkdir(parents=True,exist_ok=True)
        self.paused=True; self.simulation.pause(); self.load_geometry(100000)
        self.all_activity()
        def save(name: str) -> None:
            self.draw_frame(self.simulation.display_frame)
            for _ in range(3): self.graphicsEngine.renderFrame(); self.graphicsEngine.syncFrame()
            picture=PNMImage(); self.win.getScreenshot(picture)
            picture.removeAlpha(); picture.write(str(directory/name))
        save('initial.png')
        self.simulation.advance_fixed(50); save(f'{self.session.scene.kind}-1s.png')
        self.spawn('predator' if self.session.scene.kind=='flower' else 'flower')
        save('other-reset.png')
        self.simulation.advance_fixed(50); save(f'{self.session.scene.kind}-1s.png')
