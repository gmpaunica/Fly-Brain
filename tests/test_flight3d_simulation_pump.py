"""Background scheduling preserves serial fixed steps and reset barriers."""
import torch


def test_background_steps_match_serial_and_pause_reset_are_barriers():
    from flybrain.flight3d.simulation_pump import SimulationPump
    from flybrain.flight3d.session import FlightSession
    from flybrain.flight3d.world import make_trial
    torch.set_num_threads(1)
    serial=FlightSession(make_trial(970000),'teacher')
    background=FlightSession(make_trial(970000),'teacher')
    pump=SimulationPump(background)
    try:
        for _ in range(5):serial.advance()
        pump.request(.1);pump.wait_idle()
        assert background.snapshot==serial.snapshot
        assert torch.equal(background.brain.voltage,serial.brain.voltage)
        pump.pause();frozen=background.snapshot
        pump.request(.1);pump.wait_idle()
        assert background.snapshot==frozen
        pump.reset(background.replay)
        assert background.snapshot.time==0 and not any(background.snapshot.edge_activity)
        pump.resume();pump.request(.04);pump.wait_idle()
        serial.replay()
        for _ in range(2):serial.advance()
        assert background.snapshot==serial.snapshot
    finally:pump.close()
