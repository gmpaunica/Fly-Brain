"""Sensory-only neural decision boundary for embodied control."""

from dataclasses import dataclass
from typing import Protocol

from flybrain.world.fly import MotorCommand
from flybrain.world.sensing import SensoryObservation

from .lif import LIFPopulation
from .network import NeuralActivity


class SpikingNetwork(Protocol):
    """The state/step contract required for an embodied neural decision."""

    population: LIFPopulation

    def advance(self, observation: SensoryObservation,
                substeps: int = 20) -> NeuralActivity: ...

    def reset(self) -> None: ...


NEURAL_SUBSTEPS_PER_WORLD_TICK = 20


class MotorReadout(Protocol):
    """Decoder boundary shared by fixed and learned motor readouts."""

    def decode(self, activity: NeuralActivity) -> MotorCommand: ...

    def reset(self) -> None: ...


@dataclass(frozen=True)
class NeuralDecision:
    """Complete interval activity and its decoded motor command."""

    activity: NeuralActivity
    command: MotorCommand


class NeuralController:
    """Advance the fixed network using sensory values, never world state."""

    def __init__(self, network: SpikingNetwork, decoder: MotorReadout) -> None:
        self.network = network
        self.decoder = decoder
        self._motor_enabled = True

    @property
    def neural_interval_seconds(self) -> float:
        """Elapsed neural simulation time represented by one decision."""
        return (NEURAL_SUBSTEPS_PER_WORLD_TICK
                * self.network.population.config.dt)

    def set_motor_enabled(self, enabled: bool) -> None:
        """Gate physical output while allowing neural/decoder state to advance."""
        if type(enabled) is not bool:
            raise ValueError('enabled must be a bool')
        self._motor_enabled = enabled

    def decide(self, observation: SensoryObservation) -> NeuralDecision:
        """Advance exactly one world tick of neural time from sensory input."""
        activity = self.network.advance(
            observation,
            substeps=NEURAL_SUBSTEPS_PER_WORLD_TICK,
        )
        decoded = self.decoder.decode(activity)
        command = decoded if self._motor_enabled else MotorCommand()
        return NeuralDecision(activity, command)

    def reset(self) -> None:
        """Clear neural and decoder episode state, retaining the output gate."""
        self.network.reset()
        self.decoder.reset()
