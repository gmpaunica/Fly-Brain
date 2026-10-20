"""Serial fixed-step computation independent of display frames."""
import threading
from typing import TYPE_CHECKING
from .contracts import CONTROL_DT, FlightCommand, FlightSnapshot
from .activity import DisplayFrame, SpikeRecorder
if TYPE_CHECKING:
    from .session import FlightSession


class SimulationPump:
    """One worker owns pending steps; no physics or neural rule is changed.

    Limit pending wall-time credit to 100 ms instead of allowing an unbounded
    catch-up loop to freeze navigation. Slow machines run less simulation time
    per wall second. Every completed step still uses the original time units.
    Pause waits for an in-flight step; reset discards pending credit at a barrier.
    """
    def __init__(self,session: 'FlightSession',recorder: SpikeRecorder | None = None):
        self.session=session;self._condition=threading.Condition()
        self._recorder=recorder
        self._publication_lock=threading.Lock()
        self._display_frame=recorder.frame if recorder is not None else None
        self._credit=0.;self._command=None;self._busy=False;self._paused=False
        self._closed=False;self._error=None
        self._thread=threading.Thread(target=self._run,name='flybrain-fixed-steps',daemon=True)
        self._thread.start()

    @property
    def display_frame(self) -> DisplayFrame | None:
        """Read one complete snapshot/history pair, including after a failure."""
        with self._publication_lock:
            return self._display_frame

    def _advance_observed(self,command: FlightCommand | None = None) -> FlightSnapshot:
        snapshot=self.session.advance(command)
        if self._recorder is not None:
            frame=self._recorder.observe(snapshot)
            with self._publication_lock:self._display_frame=frame
        return snapshot

    def advance_fixed(self,steps: int,command: FlightCommand | None = None) -> FlightSnapshot:
        """Step deterministically at a pause barrier through the observation path.

        Leaves the worker paused; call resume explicitly for wall-time credit.
        Captures never mutate a session concurrently with a pending worker step.
        """
        if type(steps) is not int or steps<0:raise ValueError('nonnegative integer steps required')
        with self._condition:
            self._paused=True;self._condition.wait_for(lambda:not self._busy)
            if self._error:raise RuntimeError('simulation worker failed') from self._error
            if self._closed:raise RuntimeError('simulation worker closed')
            self._credit=0.
            for _ in range(steps):self._advance_observed(command)
            return self.session.snapshot

    def request(self,seconds: float,command=None) -> None:
        """Offer bounded wall-time credit without waiting for computation."""
        with self._condition:
            if self._error:raise RuntimeError('simulation worker failed') from self._error
            if self._paused or self._closed:return
            self._credit=min(.1,self._credit+max(0.,seconds));self._command=command
            self._condition.notify_all()

    def _run(self):
        while True:
            with self._condition:
                self._condition.wait_for(lambda:self._closed or
                    (not self._paused and self._credit+1e-12>=CONTROL_DT))
                if self._closed:return
                self._credit=max(0.,self._credit-CONTROL_DT);command=self._command;self._busy=True
            try:self._advance_observed(command)
            except BaseException as error:
                with self._condition:self._error=error;self._closed=True
            finally:
                with self._condition:self._busy=False;self._condition.notify_all()

    def wait_idle(self) -> None:
        """Wait for offered fixed steps; used by deterministic checks/capture."""
        with self._condition:
            self._condition.wait_for(lambda:not self._busy and
                (self._paused or self._closed or self._credit+1e-12<CONTROL_DT))
            if self._error:raise RuntimeError('simulation worker failed') from self._error

    def pause(self) -> None:
        """Freeze the published snapshot after the current indivisible step."""
        with self._condition:
            self._paused=True;self._condition.wait_for(lambda:not self._busy)

    def resume(self) -> None:
        """Resume pending fixed steps after pause."""
        with self._condition:self._paused=False;self._condition.notify_all()

    def reset(self,operation) -> None:
        """Apply replay/new-trial without concurrent session mutation."""
        with self._condition:
            paused=self._paused;self._paused=True
            self._condition.wait_for(lambda:not self._busy)
            if self._error:raise RuntimeError('simulation worker failed') from self._error
            try:
                operation()
                if self._recorder is not None:
                    frame=self._recorder.reset(self.session.snapshot)
                    with self._publication_lock:self._display_frame=frame
                self._credit=0.
            finally:
                self._paused=paused;self._condition.notify_all()

    def close(self) -> None:
        """Join the worker, allowing no session mutation after return."""
        with self._condition:self._closed=True;self._condition.notify_all()
        self._thread.join()
