"""Sensory-only control, actual/reference neural stepping and recorded inputs."""
from dataclasses import replace
from pathlib import Path
from flybrain.flight3d.anatomy import load_manifest
from flybrain.flight3d.network import TransmissionTrace
from flybrain.encounters3d.diagnostics import DiagnosticFlightBrain
from .controller import BehaviorController
from .interaction import InteractionLog, SpawnEvent
from .neural import BrainPair, MotionVisionBrain, DecodeResult
from .threat import ThreatMonitor
from .world import RetreatWorld, PredatorInput
from .decoder_record import diagnostic_decoder


class RetreatSession:
    """World owns scene geometry; control sees only FlightObservation and events.

    No decoder is promoted by calibration alone. This additive partial lab
    reports unavailable escape until the separately recorded scientific gates
    pass. Historical flower teacher and launchers remain intact.
    """
    def __init__(self, seed: int):
        self.manifest = load_manifest(); self.seed = seed
        self.decoder = diagnostic_decoder(self.manifest)
        self.log = InteractionLog(seed); self._recording = True
        self._install()

    def _install(self) -> None:
        self.world = RetreatWorld(self.seed); self.controller = BehaviorController(self.seed)
        self.pair = BrainPair(self.manifest); self.brain = self.pair.actual
        self.transmission_trace = TransmissionTrace(len(self.brain.edge_order))
        self.threat = ThreatMonitor(); self.decoded = DecodeResult()
        self.neural_origin = 0.; self.step_index = 0
        self.mode = 'engineered idle | neural escape unavailable'
        self.snapshot = self._snapshot()

    def _snapshot(self):
        value = self.threat.value
        return replace(self.world.snapshot(), mode=self.mode,
            neuron_ids=self.brain.ids, spike_counts=tuple(int(v) for v in self.brain.last_counts),
            edge_order=self.brain.edge_order, edge_transmissions=self.brain.edge_transmissions,
            edge_activity=tuple(float(v) for v in self.transmission_trace.values),
            pathway=self.brain.diagnostic, behavior_state=self.controller.state,
            decoder_confident=self.decoded.confident, decoder_norm=self.decoded.norm,
            decoder_similarity=self.decoded.similarity, decoder_margin=self.decoded.margin,
            wing_increment=tuple(int(v) for v in self.pair.last_vector),
            angular_size_degrees=value.size_degrees,
            angular_growth_degrees_s=value.positive_growth_degrees_s,
            neural_origin_seconds=self.neural_origin, escape_supported=False)

    def spawn(self, kind: str, position: tuple | None = None) -> None:
        """Barrier operation: retain body/IMU, clear encounter/neural histories."""
        self.world.spawn(kind, position)
        self.controller.encounter(kind, self.world.observe())
        if kind == 'flower':
            self.pair.actual = DiagnosticFlightBrain(self.manifest)
        else:
            self.pair.actual = MotionVisionBrain(self.manifest)
        self.pair.reset(); self.brain = self.pair.actual
        self.transmission_trace = TransmissionTrace(len(self.brain.edge_order))
        self.threat.reset(); self.decoded = DecodeResult()
        self.neural_origin = self.world.time
        self.mode = 'retained flower teacher' if kind == 'flower' else 'engineered idle | neural escape unavailable'
        if self._recording:
            self.log.spawns.append(SpawnEvent(self.step_index, kind, self.world.scene.position))
        self.snapshot = self._snapshot()

    def advance(self, predator: PredatorInput | None = None):
        """Apply one recorded environmental input, then four fixed physics steps."""
        applied = predator or PredatorInput()
        if not isinstance(applied, PredatorInput):
            raise ValueError('environment-only PredatorInput required')
        observation = self.world.observe()
        window = self.pair.advance(observation)
        if window is not None and self.world.scene is not None and self.world.scene.kind == 'predator':
            self.decoded = self.decoder.decode(window) if self.decoder is not None else DecodeResult()
        self.threat.update(observation.eyes)
        command = self.controller.command(observation)
        self.world.step(command, applied)
        self.transmission_trace.update(self.world.time, self.brain.edge_transmissions)
        if self._recording:
            self.log.steps.append(applied)
        self.step_index += 1
        self.snapshot = self._snapshot()
        return self.snapshot

    def begin_replay(self) -> None:
        """Retain the immutable recorded inputs while resetting all mutable state."""
        self._recording = False; self._install()

    def replay(self) -> None:
        """Reconstruct recorded endpoint, including same-step terminal spawns."""
        self.begin_replay()
        try:
            event = 0
            for step in range(len(self.log.steps)+1):
                while event < len(self.log.spawns) and self.log.spawns[event].step == step:
                    value = self.log.spawns[event]; self.spawn(value.kind, value.position); event += 1
                if step < len(self.log.steps):
                    self.advance(self.log.steps[step])
        finally:
            self._recording = True

    def new_trial(self) -> None:
        self.seed += 1; self.log = InteractionLog(self.seed); self._recording = True
        self._install()

    @classmethod
    def from_log(cls, path: Path) -> 'RetreatSession':
        log = InteractionLog.load(path); session = cls(log.seed)
        session.log = log; session.replay()
        return session
