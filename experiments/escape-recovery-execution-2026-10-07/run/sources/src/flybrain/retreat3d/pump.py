"""Add interaction replay at the original fixed-step worker publication barrier."""
from flybrain.flight3d.simulation_pump import SimulationPump


class InteractionPump(SimulationPump):
    """Reuse original scheduling; replay reconstructs every recorded display frame."""
    def replay_interaction(self) -> None:
        """Freeze worker, replay applied inputs to endpoint, publish and stay paused."""
        with self._condition:
            self._paused = True; self._condition.wait_for(lambda: not self._busy)
            if self._error:
                raise RuntimeError('simulation worker failed') from self._error
            if self._closed:
                raise RuntimeError('simulation worker closed')
            self._credit = 0.; self._command = None
            session = self.session; session.begin_replay()
            def publish_reset():
                if self._recorder is not None:
                    frame = self._recorder.reset(session.snapshot)
                    with self._publication_lock:
                        self._display_frame = frame
            publish_reset()
            try:
                event = 0
                for step in range(len(session.log.steps)+1):
                    while event < len(session.log.spawns) and session.log.spawns[event].step == step:
                        value = session.log.spawns[event]
                        session.spawn(value.kind, value.position); event += 1; publish_reset()
                    if step < len(session.log.steps):
                        self._advance_observed(session.log.steps[step])
            finally:
                session._recording = True
                self._condition.notify_all()
