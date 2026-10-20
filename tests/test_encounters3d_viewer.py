"""Real Panda3D rendering/event tests; human live acceptance is separate."""
import numpy as np
import torch


def test_spawn_buttons_exclusivity_activity_reset_preferences_and_passive_draw(tmp_path):
    from flybrain.encounters3d.viewer import EncounterViewer
    from panda3d.core import PNMImage, Point2, Point3
    torch.set_num_threads(1)
    app=EncounterViewer(980000,offscreen=True,window_size=(1000,700))
    try:
        app.paused=True; app.simulation.pause()
        assert app.spawn_flower_button['text']=='Spawn flower'
        assert app.spawn_predator_button['text']=='Spawn predator'
        app.all_activity(); app.toggle_vnc(); app.toggle_settings()
        app.brain_motion.rotate_pixels(40,25); app.brain_motion.advance(.05); app._brain_camera()
        camera=np.array(app.brain_camera.getMat()); classes=app.enabled_classes
        app.spawn_predator_button['command']()
        assert app.session.scene.kind=='predator'
        assert app.flower.find('**/predator-proxy').isEmpty() is False
        assert app.flower.find('**/petal').isEmpty()
        assert app.paused and app.settings_open and app.vnc and app.enabled_classes==classes
        np.testing.assert_array_equal(np.array(app.brain_camera.getMat()),camera)
        app.simulation.advance_fixed(50); app.draw_frame(app.simulation.display_frame)
        assert 'Avoidance unavailable' in app.hud.getText()
        assert 'DNp03 recorded' in app.activity_label.getText()
        frame=app.simulation.display_frame; voltage=app.session.brain.voltage.clone()
        for _ in range(4):
            app.draw_frame(frame); app.graphicsEngine.renderFrame()
        # Verify the actual framebuffer matches the eye encoder's cyan surface.
        projected=Point2()
        point=app.cam.getRelativePoint(app.world_scene,Point3(*app.session.scene.position))
        assert app.camLens.project(point,projected)
        region=app.cam.node().getDisplayRegion(0)
        image=PNMImage(); app.win.getScreenshot(image)
        x=int((region.getLeft()+(projected.x+1)/2*(region.getRight()-region.getLeft()))*image.getXSize())
        y=int((1-region.getBottom()-(projected.y+1)/2*(region.getTop()-region.getBottom()))*image.getYSize())
        r,g,b=image.getXel(x,y)
        assert g>r+.2 and b>r+.2
        assert frame is app.simulation.display_frame
        assert torch.equal(voltage,app.session.brain.voltage)
        app.messenger.send('backspace')
        assert app.session.scene.seed==980001 and not any(app.simulation.display_frame.cumulative_counts)
        app.messenger.send('r')
        assert app.session.scene.seed==980002 and app.session.scene.kind=='predator'
        app.spawn_flower_button['command']()
        assert app.session.scene.kind=='flower' and app.session.mode=='teacher'
        assert app.flower.find('**/predator-proxy').isEmpty()
        assert not app.flower.find('**/petal').isEmpty()
        assert 'Teacher-controlled' in app.hud.getText()
        np.testing.assert_array_equal(np.array(app.brain_camera.getMat()),camera)
        picture=PNMImage(); app.win.getScreenshot(picture)
        assert picture.getXSize()==1000
    finally: app.destroy()
