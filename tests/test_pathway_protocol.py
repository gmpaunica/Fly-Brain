"""Protocol tests do not consume or repeat the bounded scientific run."""
import numpy as np
from flybrain.encounters3d.investigate import cases, presentation_images, interventions
from flybrain.encounters3d.neural import threat_contrast


def test_exact_budget_and_paired_growth_occlusion_reappearance():
    presentations = cases()
    assert len(presentations) == 24 and len(interventions()) == 5
    assert len({p['label'] for p in presentations}) == 24
    for case in presentations:
        images = presentation_images(case)
        assert len(images) == 50 and all(x.shape == (2, 24, 48, 3) for x in images)
        if case['kind'] == 'sequence':
            signals = [threat_contrast(x).sum() for x in images]
            assert signals[19] > signals[0] > 0
            assert not any(signals[20:30])
            assert signals[30] > 0
            np.testing.assert_array_equal(images[30], images[49])
        else:
            assert all(np.array_equal(images[0], x) for x in images)


def test_flower_and_background_are_rejected_by_measured_brain():
    from flybrain.encounters3d.diagnostics import DiagnosticEscapeBrain
    from flybrain.encounters3d.scene import EncounterWorld, SceneSpec
    from flybrain.flight3d.anatomy import load_manifest
    from flybrain.flight3d.contracts import FlightObservation
    brain = DiagnosticEscapeBrain(load_manifest())
    images = [EncounterWorld(SceneSpec(3, 'flower', (.3, .4, .1))).observe().eyes,
              np.full((2, 24, 48, 3), (.035, .055, .09))]
    for eyes in images:
        brain.reset()
        obs = FlightObservation(0, 0, eyes, (0, 0, 0), (0, 0, 9.81), False, False)
        for _ in range(3):
            assert not brain.advance(obs).any()
            assert not any(brain.diagnostic.current_sum)
