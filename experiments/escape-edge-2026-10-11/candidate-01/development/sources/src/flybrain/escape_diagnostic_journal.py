"""Append-only Windows-safe reservation/heartbeat I/O; no neural computation."""
import json
import os
from pathlib import Path
from typing import TextIO


def append_durable(stream: TextIO, value: dict) -> None:
    """Write one finite JSON line and fsync before the caller may advance state."""
    stream.write(json.dumps(value,separators=(',',':'),allow_nan=False)+'\n')
    stream.flush(); os.fsync(stream.fileno())


class ReservationJournal:
    """Reserve paired 20 ms intervals without replacing a file readers hold.

    The manifest is fresh/immutable. One persistent append handle avoids delete
    sharing requirements. A failed durability operation poisons the instance:
    the simulation must terminate, never retry that reservation automatically.
    """
    def __init__(self, path: Path, limit: int):
        if limit<=0 or limit%40: raise ValueError('paired interval limit required')
        self.path=path; self.limit=limit; self.steps=0; self.failed=False
        with path.open('x',encoding='utf-8') as f:
            json.dump(dict(schema='append-only-reservations-v1',limit=limit,reserved_steps=0,
                           journal=path.with_suffix('.reservations.jsonl').name),f)
        self.journal_path=path.with_suffix('.reservations.jsonl')
        self.stream=self.journal_path.open('x',encoding='utf-8')

    def reserve_interval(self) -> None:
        """Commit the counter only after a durable reservation; no kernel calls."""
        if self.failed: raise RuntimeError('reservation journal failed; no retry')
        if self.steps+40>self.limit: raise ValueError('scientific budget exceeded')
        next_steps=self.steps+40
        try:
            append_durable(self.stream,dict(reserved_steps=next_steps,limit=self.limit))
        except BaseException:
            self.failed=True
            raise
        self.steps=next_steps

    def close(self) -> None:
        """Close the persistent handle; preserve manifest and every saved line."""
        self.stream.close()


class HeartbeatJournal:
    """Append worker observations; write terminal JSON once, never rename it."""
    def __init__(self, stem: Path):
        self.path=stem.with_suffix('.jsonl'); self.terminal=stem.with_suffix('.json')
        self.stream=self.path.open('x',encoding='utf-8'); self.failed=False

    def observe(self, value: dict) -> None:
        """One durable monitoring observation without changing simulation state."""
        if self.failed: raise RuntimeError('heartbeat journal failed; no retry')
        try: append_durable(self.stream,value)
        except BaseException:
            self.failed=True
            raise

    def finish(self, value: dict) -> None:
        """Append terminal observation, close handle and create immutable receipt."""
        self.observe(value); self.stream.close()
        with self.terminal.open('x',encoding='utf-8') as f: json.dump(value,f,indent=2,allow_nan=False)
