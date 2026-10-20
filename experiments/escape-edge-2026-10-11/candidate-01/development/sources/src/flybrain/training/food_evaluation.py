"""Matched 20-second contact, braking and dwell scoring for neural policies."""

from dataclasses import asdict
import math
from collections.abc import Sequence

from flybrain.behavior import FoodVisit
from flybrain.foraging import ForagingSimulation
from flybrain.neural.controller import NeuralController
from flybrain.simulation import FixedStepSimulation
from flybrain.world.arena import Arena
from flybrain.world.fly import Fly, MotorCommand, MAX_FORWARD_SPEED
from flybrain.world.odor import OdorSource

from .evaluation import EpisodeCase
from .food_preflight import HORIZON_TICKS, SOURCE_X, SOURCE_Y


def score_trace(trace: Sequence[tuple[bool, MotorCommand, float, float]]) -> dict:
    """Score every tick; geometry appears only in this evaluation function."""
    if len(trace) != HORIZON_TICKS:
        raise ValueError('complete 20-second trace required')
    visit = FoodVisit()
    first_arrival = first_contact = first_stop = None
    departures = 0
    previous_contact = False
    approach_speeds: list[float] = []
    minimum_distance = float('inf')
    for tick, (contact, command, actual_speed, distance) in enumerate(trace, 1):
        visit.observe(tick, contact, command, actual_speed)
        minimum_distance = min(minimum_distance, distance)
        if first_arrival is None and distance <= 20:
            first_arrival = tick
        if first_contact is None:
            approach_speeds.append(command.forward * MAX_FORWARD_SPEED)
            if contact:
                first_contact = tick
        if (first_contact is not None and first_stop is None
                and command == MotorCommand() and actual_speed <= 1):
            first_stop = tick
        if previous_contact and not contact:
            departures += 1
        previous_contact = contact
    approach = (sum(approach_speeds[-25:]) / min(25, len(approach_speeds))
                if approach_speeds else None)
    latency = ((first_stop - first_contact) * .02
               if first_contact is not None and first_stop is not None else None)
    success = (first_contact is not None and first_contact <= 850
               and approach is not None and approach <= 30
               and latency is not None and latency <= .2
               and visit.contact and visit.settled and visit.hold_seconds >= 2
               and departures == 0)
    return {'success': success,
            'first_arrival_seconds': first_arrival * .02 if first_arrival else None,
            'first_contact_seconds': first_contact * .02 if first_contact else None,
            'approach_speed': approach,
            'stop_latency_seconds': latency,
            'overshoot_units': max(0., 16. - minimum_distance),
            'departures': departures,
            'stationary_dwell_seconds': visit.hold_seconds,
            'final_contact': visit.contact,
            'final_distance': trace[-1][3],
            'final_commanded_speed': visit.commanded_speed,
            'final_actual_speed': visit.displacement_speed}


def run_controller_case(case: EpisodeCase, controller: NeuralController) -> dict:
    """Run a fresh controller over the exact recorded pose for 1,000 ticks."""
    source = OdorSource(SOURCE_X, SOURCE_Y)
    world = FixedStepSimulation(Arena(), Fly(case.x, case.y, case.heading),
                                odor_source=source)
    loop = ForagingSimulation(world, controller)
    trace = []
    for _ in range(HORIZON_TICKS):
        snapshot = loop.step()
        distance = math.hypot(world.fly.x - SOURCE_X, world.fly.y - SOURCE_Y)
        trace.append((loop.visit.contact, snapshot.command,
                      loop.visit.displacement_speed, distance))
    return {**asdict(case), 'heading_sector': int(case.heading / (math.tau / 8)),
            **score_trace(trace), 'world_ticks': world.tick_count,
            'neural_steps': controller.network.population.step_count}
