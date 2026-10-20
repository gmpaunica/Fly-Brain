"""Sourced BANC topology running through the existing synthetic LIF rules."""

from dataclasses import asdict, dataclass
import hashlib
from importlib.resources import files
import json
import math
from pathlib import Path

import torch
from torch import Tensor

from flybrain.data.banc import (BANC_SUBSET_SHA256, BANC_TRANSFORM_VERSION,
                                BancTopology, normalize_banc_topology)

from .lif import LIFConfig, LIFPopulation
from .network import NeuralActivity, NeuralSnapshot
from .sensory import OdorObservation, SensoryEncoder


@dataclass(frozen=True)
class BancDynamicsConfig:
    """Explicit engineered mapping; no synaptic budget is assumed verified.

    The two scale endpoints and rational current gain are fixed by the
    recovery protocol, not biological measurements or learned parameters.
    """

    synaptic_budget: float
    encoder_min_scale: float = 0.005
    encoder_max_scale: float = 0.05
    projection_filter_tau: float = 0.0
    sensory_mode: str = 'current'
    projection_dynamics: str = 'hard_reset'

    def __post_init__(self) -> None:
        if (isinstance(self.synaptic_budget, bool)
                or not isinstance(self.synaptic_budget, (int, float))
                or not math.isfinite(self.synaptic_budget) or self.synaptic_budget <= 0):
            raise ValueError('synaptic_budget must be finite and positive')
        if self.encoder_min_scale != .005 or self.encoder_max_scale not in (.05, .375):
            raise ValueError('BANC encoder must use a declared sensitivity range')
        if self.encoder_max_scale == .375 and self.synaptic_budget != 3:
            raise ValueError('wide-range BANC encoder has exactly one declared budget: 3')
        if self.projection_filter_tau not in (0., .005) or isinstance(self.projection_filter_tau, bool):
            raise ValueError('BANC projection filter must be disabled or the declared 5 ms')
        if self.projection_filter_tau and (self.synaptic_budget != 3 or self.encoder_max_scale != .375):
            raise ValueError('PN filtering uses only the declared wide-range budget-3 candidate')
        if self.sensory_mode not in ('current', 'phase_rate'):
            raise ValueError('unknown BANC sensory mode')
        if self.projection_dynamics not in ('hard_reset', 'charge_preserving'):
            raise ValueError('unknown BANC projection dynamics')
        if self.projection_dynamics == 'charge_preserving' and self.sensory_mode != 'phase_rate':
            raise ValueError('charge-preserving PNs require the declared phase-rate candidate')
        if self.sensory_mode == 'phase_rate' and (
                self.synaptic_budget != 3 or self.projection_filter_tau != 0 or self.encoder_max_scale != .05):
            raise ValueError('phase-rate encoding has one declared unfiltered budget-3 configuration')


def load_verified_banc_config(path: Path | None = None) -> BancDynamicsConfig:
    """Bind a receipt to successful sidecar evidence and the installed source.

    The receipt, original results and original protocol must be bundled
    together. These consistency checks prevent stale/failed evidence from
    enabling an app default; they are not signatures against malicious edits.
    """
    directory = Path(path).parent if path is not None else files('flybrain.data')
    text = (Path(path).read_text(encoding='utf-8') if path is not None else
            directory.joinpath('banc_dynamics_config.json').read_text(encoding='utf-8'))
    record = json.loads(text)
    if (not isinstance(record, dict) or record.get('schema_version') != 1
            or record.get('subset_sha256') != BANC_SUBSET_SHA256
            or record.get('transform_version') != BANC_TRANSFORM_VERSION
            or record.get('dynamics_gate_passed') is not True
            or record.get('causal_gate_passed') is not True
            or not isinstance(record.get('probe_results_sha256'), str)
            or len(record['probe_results_sha256']) != 64
            or any(ch not in '0123456789abcdef' for ch in record['probe_results_sha256'])
            or not isinstance(record.get('config'), dict)):
        raise ValueError('BANC dynamics receipt is not verified')
    try:
        config = BancDynamicsConfig(**record['config'])
    except (TypeError, ValueError) as exc:
        raise ValueError('BANC dynamics receipt has invalid configuration') from exc
    if config.synaptic_budget not in (1.25, 2.0, 3.0):
        raise ValueError('BANC dynamics receipt has an undeclared synaptic budget')
    try:
        result_bytes = directory.joinpath('banc_dynamics_results.json').read_bytes()
        protocol_bytes = directory.joinpath('banc_dynamics_protocol.json').read_bytes()
        result = json.loads(result_bytes)
        protocol = json.loads(protocol_bytes)
        if (not isinstance(result, dict) or not isinstance(protocol, dict)
                or hashlib.sha256(result_bytes).hexdigest() != record['probe_results_sha256']
                or hashlib.sha256(protocol_bytes).hexdigest() != result.get('protocol_sha256')
                or any(result.get(key) is not True for key in
                       ('passed', 'dynamics_gate_passed', 'causal_gate_passed',
                        'previous_files_preserved', 'source_code_unchanged'))
                or result.get('selected_config') != asdict(config)
                or asdict(config) not in protocol.get('candidate_configs', [])
                or result.get('optimizer_steps') != 0
                or any(data.get('subset_sha256') != BANC_SUBSET_SHA256
                       or data.get('transform_version') != BANC_TRANSFORM_VERSION
                       for data in (result, protocol))):
            raise ValueError('BANC dynamics evidence does not match a successful receipt')
        expected_paths = {
            'src/flybrain/data/banc.py', 'src/flybrain/neural/banc_network.py',
            'src/flybrain/neural/banc_motor.py', 'src/flybrain/neural/lif.py',
            'src/flybrain/training/banc_recovery.py',
            'src/flybrain/neural/banc_synapses.py',
        }
        hashes = protocol.get('source_code_sha256')
        if not isinstance(hashes, dict) or set(hashes) != expected_paths:
            raise ValueError('BANC dynamics evidence lacks the required source hashes')
        for relative, digest in hashes.items():
            source = files('flybrain').joinpath(relative.removeprefix('src/flybrain/'))
            if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
                raise ValueError('BANC dynamics evidence is stale for the installed source')
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError('BANC dynamics evidence is missing or unreadable') from exc
    return config


class BancNetwork:
    """Inference-only BANC graph with bilateral odor current and no tonic drive."""

    def __init__(self, topology: BancTopology,
                 *, encoder_scale: float = 0.025,
                 dynamics_config: BancDynamicsConfig | None = None) -> None:
        if dynamics_config is not None and not isinstance(dynamics_config, BancDynamicsConfig):
            raise ValueError('dynamics_config must be a BancDynamicsConfig')
        self.source_topology = topology
        self.dynamics_config = dynamics_config
        self.topology = (normalize_banc_topology(topology, dynamics_config.synaptic_budget)
                         if dynamics_config is not None else topology)
        self.population = self.population_for_weights(self.topology.weights)
        self._encoder = SensoryEncoder(topology, encoder_scale)
        self._scales: dict[str, tuple[tuple[int, ...], Tensor]] = {}
        self._uses_phase_rate = dynamics_config is not None and dynamics_config.sensory_mode == 'phase_rate'
        if dynamics_config is not None:
            for side in ('left', 'right'):
                indices = tuple(sorted(topology.indices(f'sensory_{side}'),
                                       key=lambda i: topology.root_ids[i]))
                scales = torch.logspace(math.log10(dynamics_config.encoder_min_scale),
                                        math.log10(dynamics_config.encoder_max_scale),
                                        len(indices), dtype=self.population.dtype, device='cpu')
                self._scales[side] = indices, scales
        self._initial_phases = torch.zeros(self.population.size, dtype=torch.float64, device='cpu')
        if self._uses_phase_rate:
            for indices, _ in self._scales.values():
                self._initial_phases[list(indices)] = (
                    torch.arange(len(indices), dtype=torch.float64, device='cpu') + .5) / len(indices)
        self._phases = self._initial_phases.clone()

    def population_for_weights(self, weights: Tensor) -> LIFPopulation:
        """Build fresh configured state, retaining PN timing during causal cuts."""
        if self.dynamics_config is not None and self.dynamics_config.projection_dynamics == 'charge_preserving':
            from .banc_synapses import ProjectionChargePopulation
            indices = self.topology.indices('projection_left') + self.topology.indices('projection_right')
            return ProjectionChargePopulation(weights, indices)
        if self.dynamics_config is not None and self.dynamics_config.projection_filter_tau:
            from .banc_synapses import ProjectionFilteredPopulation
            indices = self.topology.indices('projection_left') + self.topology.indices('projection_right')
            return ProjectionFilteredPopulation(weights, indices, LIFConfig())
        return LIFPopulation(weights, LIFConfig())

    def encode(self, observation: OdorObservation) -> Tensor:
        """Preview next ORN current without advancing phases or neural time."""
        if self._uses_phase_rate:
            return self._rate_drive(self._sensory_rates(observation))[0]
        if self.dynamics_config is None:
            return self._encoder.encode(observation)
        values = observation.odor_left, observation.odor_right
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError('concentrations must be finite and nonnegative')
        current = torch.zeros(self.population.size, dtype=self.population.dtype, device='cpu')
        for side, concentration in zip(('left', 'right'), values):
            indices, scales = self._scales[side]
            # Python float division avoids converting a huge finite odor to
            # float32 infinity before evaluating the bounded rational map.
            encoded = [2.5 * (concentration / (concentration + float(scale)))
                       for scale in scales]
            current[list(indices)] = torch.tensor(encoded, dtype=self.population.dtype, device='cpu')
        return current

    def _sensory_rates(self, observation: OdorObservation) -> Tensor:
        values = observation.odor_left, observation.odor_right
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError('concentrations must be finite and nonnegative')
        rates = torch.zeros(self.population.size, dtype=torch.float64, device='cpu')
        for side, value in zip(('left', 'right'), values):
            rates[list(self._scales[side][0])] = min(96., 176. * math.sqrt(value))
        return rates

    def _rate_drive(self, rates: Tensor) -> tuple[Tensor, Tensor]:
        phases = self._phases + rates * self.population.config.dt
        scheduled = phases >= 1.
        phases = phases - scheduled.to(torch.float64)
        # External pulses guarantee scheduled ORN spikes. The retained ORN->ORN
        # edge may also trigger extra spikes; do not erase that source wiring.
        fraction = -math.expm1(-self.population.config.dt / self.population.config.tau)
        amplitude = 1.01 * (self.population.config.threshold - self.population.config.rest) / fraction
        return scheduled.to(self.population.dtype) * amplitude, phases

    def reset(self) -> None:
        """Clear all episode voltages, delayed spikes and neural time."""
        self.population.reset()
        self._phases = self._initial_phases.clone()

    def snapshot(self) -> NeuralSnapshot:
        """Return a detached copy of current neural state."""
        return NeuralSnapshot(self.population.step_count,
                              self.population.voltage, self.population.spikes)

    def advance(self, observation: OdorObservation,
                substeps: int = 20) -> NeuralActivity:
        """Hold odor fixed; advance actual neurons and optional sensory phases."""
        if type(substeps) is not int or substeps <= 0:
            raise ValueError('substeps must be a positive integer')
        current = self.encode(observation)
        rates = self._sensory_rates(observation) if self._uses_phase_rate else None
        start = self.population.step_count
        counts = torch.zeros(self.population.size, dtype=torch.int64, device='cpu')
        for _ in range(substeps):
            if rates is None:
                counts += self.population.step(current).to(torch.int64)
            else:
                drive, next_phases = self._rate_drive(rates)
                counts += self.population.step(drive).to(torch.int64)
                self._phases = next_phases
        end = self.population.step_count
        return NeuralActivity(start, end, (end - start) * self.population.config.dt,
                              counts, self.snapshot())
