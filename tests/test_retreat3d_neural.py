"""Sensory-only candidate, frozen count decoding and numerical parity."""
import importlib
import math
import numpy as np
import pytest
import torch
from flybrain.flight3d.anatomy import load_manifest
from flybrain.flight3d.contracts import FlightObservation
from flybrain.flight3d.neural import FlightBrain
from flybrain.encounters3d.neural import EscapeBrain
from flybrain.encounters3d.scene import encounter_eyes, SceneSpec


def api():
    assert importlib.util.find_spec('flybrain.retreat3d') is not None, 'additive retreat package required'
    return importlib.import_module('flybrain.retreat3d.neural')


def obs(eyes=None, angular=(.2, -.3, .4), force=(1., 2., 9.81)):
    if eyes is None:
        eyes = np.zeros((2, 24, 48, 3), dtype=np.float32)
    return FlightObservation(0., 0., eyes, angular, force, False, False)


def test_motion_formula_and_unchanged_visual_encoding():
    module = api(); manifest = load_manifest()
    candidate = module.MotionVisionBrain(manifest)
    original = EscapeBrain(manifest); motion = FlightBrain(manifest)
    for radius in (.17, .08, .06):
        eyes = encounter_eyes(np.array((.3, .3, .1)), np.zeros(3),
            SceneSpec(2, 'predator', (.3, .3+radius, .1)))
        observation = obs(eyes)
        rates = candidate.sensory_rates(observation)
        np.testing.assert_array_equal(rates[candidate.roles['vision']],
                                      original.sensory_rates(observation)[candidate.roles['vision']])
        np.testing.assert_array_equal(rates[candidate.roles['motion']],
                                      motion.sensory_rates(observation)[candidate.roles['motion']])
        allowed = set(candidate.roles['vision']) | set(candidate.roles['motion'])
        assert all(rates[i] == 0 for i in range(len(rates)) if i not in allowed)
    assert len(candidate.ids) == 589 and len(module.wing_indices(manifest)) == 48
    for field in ('pre', 'post', 'weights', 'inputs'):
        assert torch.equal(getattr(candidate, field), getattr(original, field))


def test_candidate_matches_original_dynamics_and_diagnostic_counts():
    module = api(); m = load_manifest()
    a, b = module.MotionVisionBrain(m), FlightBrain(m)
    rates = np.random.default_rng(19).uniform(0, 240, 589)
    for steps in (1, 20, 57):
        np.testing.assert_array_equal(a.advance_rates(rates, steps), b.advance_rates(rates, steps))
        for field in ('voltage', 'previous', 'phase', 'history'):
            assert torch.equal(getattr(a, field), getattr(b, field))
        assert a.edge_transmissions == b.edge_transmissions
        assert a.diagnostic.spike_counts == tuple(a.last_counts)


def test_motion_only_pair_is_identical_and_window_reset():
    module = api(); pair = module.BrainPair(load_manifest())
    for _ in range(30):
        window = pair.advance(obs())
        assert torch.equal(pair.actual.voltage, pair.reference.voltage)
        assert np.array_equal(pair.actual.last_counts, pair.reference.last_counts)
        if window is not None:
            np.testing.assert_array_equal(window, np.zeros(48))
    assert pair.actual.steps == pair.reference.steps == 600
    pair.reset()
    assert pair.actual.steps == pair.reference.steps == 0
    assert pair.intervals == 0 and pair.actual.diagnostic is None


def test_decoder_signed_counts_confidence_and_frozen_roundtrip(tmp_path):
    module = api()
    templates = np.eye(8, 48); ids = tuple(str(i) for i in range(48))
    d = module.FrozenDecoder(templates, ids)
    assert d.decode(templates[3]).heading_radians == pytest.approx(3*math.pi/4)
    assert d.decode(np.zeros(48)).confident is False
    assert d.decode(templates[0]*.9).confident is False
    assert d.decode(templates[0]+templates[1]).confident is False
    assert d.decode(-templates[0]).confident is False
    with pytest.raises(ValueError):
        d.templates[0, 0] = 10
    d.save(tmp_path/'templates.json')
    restored = module.FrozenDecoder.load(tmp_path/'templates.json', ids)
    np.testing.assert_array_equal(d.templates, restored.templates)
    with pytest.raises(ValueError):
        module.FrozenDecoder.load(tmp_path/'templates.json', tuple(reversed(ids)))
