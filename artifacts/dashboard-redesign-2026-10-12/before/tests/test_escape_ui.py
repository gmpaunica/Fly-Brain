"""Presentation controls exercise the real frozen session and Panda3D UI."""
import importlib

import numpy as np
import pytest
import torch


def viewer_api():
    name = 'flybrain.escape_ui.viewer'
    assert importlib.util.find_spec('flybrain.escape_ui'), 'polished presentation package required'
    return importlib.import_module(name)


@pytest.mark.parametrize('size', [(1000, 700), (1440, 900)])
def test_controls_resize_diagnostics_and_capture_are_passive(size, tmp_path):
    from flybrain.escape_edge.delivery import verified_candidate
    from panda3d.core import Filename, PNMImage
    candidate = verified_candidate()
    assert candidate is not None
    torch.set_num_threads(1)
    app = viewer_api().DemoViewer(1110000, candidate, verified=True,
                                  offscreen=True, window_size=size)
    try:
        app.paused = True
        app.simulation.pause()
        app.draw_frame(app.simulation.display_frame)
        assert app.pause_button['text'] == 'Resume / Space'
        assert not app.diagnostics_open
        assert 'Spawn a flower' in app.help.getText()
        for button in app.action_buttons.values():
            bounds = button['frameSize']
            assert (bounds[3] - bounds[2]) * size[1] / 2 >= 38
            left, right = bounds[:2]
            x = button.getX() / app.getAspectRatio()
            assert -1 <= x + left / app.getAspectRatio()
            assert x + right / app.getAspectRatio() <= 1
            assert app.layout.route(x, button.getZ()) not in ('world', 'brain')
        # A large hit target is insufficient: primary actions need visible fill.
        app.capture_paused(tmp_path / 'actions.png')
        picture = PNMImage()
        assert picture.read(Filename.fromOsSpecific(str(tmp_path / 'actions.png')))
        flower = app.spawn_flower_button
        left = flower['frameSize'][0]
        px = round((flower.getX() / app.getAspectRatio() + 1) * size[0] / 2
                   + left / app.getAspectRatio() * size[0] / 2 + 12)
        py = round((1 - flower.getZ()) * size[1] / 2 - 12)
        assert picture.getXel(px, py)[1] > .15, 'flower button must have a visible green background'
        # Moving OnscreenText needs its text position, not an extra NodePath offset.
        assert app.legend_title.getTightBounds()[1].z > -.35
        before = app.session.snapshot
        voltage = app.session.brain.voltage.clone()
        camera = np.array(app.brain_camera.getMat())
        app.diagnostics_button['command']()
        assert app.diagnostics_open
        assert 'Historical direction' in app.pathway_label.getText()
        app._ui()  # The same rebuild used on a real window resize.
        app._resize()
        app.draw_frame(app.simulation.display_frame)
        assert app.diagnostics_open
        app.capture_paused(tmp_path / f'frame-{size[0]}.png')
        assert (tmp_path / f'frame-{size[0]}.png').stat().st_size > 10000
        assert before == app.session.snapshot
        assert torch.equal(voltage, app.session.brain.voltage)
        np.testing.assert_array_equal(camera, np.array(app.brain_camera.getMat()))
        app.diagnostics_button['command']()
        assert not app.diagnostics_open and app.pathway_label.isHidden()
        # UI callbacks and keyboard use the same existing pause implementation.
        app.messenger.send('space')
        assert not app.paused and app.pause_button['text'] == 'Pause / Space'
        app.pause_button['command']()
        assert app.paused and app.pause_button['text'] == 'Resume / Space'
        # Available source controls still work with the new presentation.
        app.settings_button['command']()
        assert app.settings_open and app.member_buttons
        assert min(button.getSx() * size[1] / 2 for button in app.member_buttons) >= 11
        app.list_groups()
        app.select(app.manifest['nodes'][0]['id'])
        selected = app.selected
        classes = app.enabled_classes
        app._ui()
        app.draw_frame(app.simulation.display_frame)
        assert app.selected == selected and app.enabled_classes == classes
        # Both source-bound scientific delivery and replay hashes stay valid.
        assert verified_candidate().identity == candidate.identity
    finally:
        app.destroy()


def test_spawn_replay_and_reset_use_existing_session_contract():
    from flybrain.escape_edge.delivery import verified_candidate
    from flybrain.retreat3d.world import PredatorInput
    torch.set_num_threads(1)
    app = viewer_api().DemoViewer(1110000, verified_candidate(), verified=True,
                                  offscreen=True, window_size=(1000, 700))
    try:
        app.paused = True
        app.simulation.pause()
        app.keys.add('w')
        app.spawn_flower_button['command']()
        assert app.session.snapshot.object_kind == 'flower' and not app.keys
        assert app.paused
        app.spawn_predator_button['command']()
        assert app.session.snapshot.object_kind == 'predator'
        assert app.flower.find('**/petal').isEmpty()
        assert not app.flower.find('**/predator-proxy').isEmpty()
        app.simulation.advance_fixed(3, PredatorInput.from_keys({'a'}))
        app.draw_frame(app.simulation.display_frame)
        endpoint = app.session.snapshot
        app.replay_button['command']()
        assert app.session.snapshot == endpoint and app.paused
        app.new_trial_button['command']()
        assert app.session.snapshot.object_kind == 'none'
        assert app.session.snapshot.time == 0 and app.session.snapshot.seed == endpoint.seed + 1
        assert app.paused
    finally:
        app.destroy()
