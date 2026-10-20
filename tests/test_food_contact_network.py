"""Candidate odor and engineered contact pathway checks."""

import torch

from flybrain.neural.food_contact import FoodContactNetwork
from flybrain.world.food import FoodObservation


def test_candidate_retains_source_graph_and_distinguishes_odor_in_spikes():
    brain = FoodContactNetwork()
    assert brain.population.size == 85
    assert len(brain.source_topology.root_ids) == 83
    assert brain.topology.root_ids[:83] == brain.source_topology.root_ids
    assert torch.equal(brain.population.weights[:83, :83],
                       brain.reference_weights)
    assert brain.odor_rate(.01) == 96 * .01 / (.01 + .1)
    assert brain.odor_rate(2.5) == 96 * 2.5 / (2.5 + .1)
    left = brain.topology.indices('sensory_left')
    right = brain.topology.indices('sensory_right')
    weak = brain.advance(FoodObservation(.01, .02, False), 1000)
    weak_left = int(weak.spike_counts[list(left)].sum())
    assert int(weak.spike_counts[list(right)].sum()) > weak_left
    brain.reset()
    strong = brain.advance(FoodObservation(.5, 1.0, False), 1000)
    strong_left = int(strong.spike_counts[list(left)].sum())
    assert strong_left > weak_left > 0
    assert int(strong.spike_counts[list(right)].sum()) > strong_left
    assert int(strong.spike_counts[brain.CONTACT]) == 0


def test_contact_spikes_propagate_one_substep_and_reset():
    brain = FoodContactNetwork()
    observation = FoodObservation(0, 0, True)
    activity = brain.advance(observation, 200)
    assert int(activity.spike_counts[brain.CONTACT]) >= 19
    assert int(activity.spike_counts[brain.RELAY]) >= 18
    assert brain.population.weights[brain.RELAY, brain.CONTACT] > 1
    assert int(activity.spike_counts[:83].sum()) == 0
    brain.reset()
    assert brain.population.step_count == 0
    assert not brain.population.spikes.any()
    assert not brain.population.voltage.any()
    quiet = brain.advance(FoodObservation(0, 0, False), 200)
    assert int(quiet.spike_counts[brain.CONTACT:].sum()) == 0
