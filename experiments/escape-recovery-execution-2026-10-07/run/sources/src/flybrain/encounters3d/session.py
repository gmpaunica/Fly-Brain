"""Exclusive scene/controller lifecycle over unchanged flight step scheduling."""
from flybrain.flight3d.anatomy import load_manifest
from dataclasses import replace
from .diagnostics import DiagnosticFlightBrain, DiagnosticEscapeBrain
from flybrain.flight3d.teacher import FlightTeacher
from flybrain.flight3d.session import FlightSession
from flybrain.flight3d.network import TransmissionTrace
from .controller import PredatorController
from .scene import SceneSpec, EncounterWorld, make_scene


class EncounterSession(FlightSession):
    """Controllers receive only observations; scene selection belongs here."""
    def __init__(self, scene: SceneSpec):
        self.manifest=load_manifest()
        self._install(scene)

    def _install(self, scene: SceneSpec) -> None:
        self.scene=scene; self.world=EncounterWorld(scene)
        if scene.kind=='flower':
            self.mode='teacher'; self.controller=FlightTeacher()
            self.brain=DiagnosticFlightBrain(self.manifest)
        else:
            self.mode='circuit probe / avoidance unavailable'
            self.controller=PredatorController(self.manifest)
            self.controller.brain=DiagnosticEscapeBrain(self.manifest)
            self.brain=self.controller.brain
        self.transmission_trace=TransmissionTrace(len(self.brain.edge_order))
        self.snapshot=self._snapshot()

    def _snapshot(self):
        """Publish diagnostics with the same completed body/spike interval."""
        return replace(super()._snapshot(), pathway=self.brain.diagnostic)

    def spawn(self, kind: str) -> None:
        """Reset all trial/controller/neural state with the next seed and one object."""
        self._install(make_scene(self.scene.seed+1,kind))

    def new_trial(self) -> None:
        self.spawn(self.scene.kind)
