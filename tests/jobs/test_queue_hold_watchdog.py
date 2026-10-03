"""Tests for the queue-hold watchdog.

Each test names the mutation it would catch. The failure being guarded is a queue left
SIGSTOPped forever, so every path must end in a resume.
"""

from __future__ import annotations

import signal
import subprocess

import pytest
import queue_hold_watchdog as mod


class Box:
    """Scripted world: the holder unit is active for ``active_polls`` checks."""

    def __init__(self) -> None:
        self.active_polls = 3
        self.stuck_seq: list[bool] = []
        self.stopped: list[str] = []
        self.resumed: list[int] = []
        self.sleeps = 0

    def unit_active(self, _unit: str) -> bool:
        if self.stopped:
            return False
        self.active_polls -= 1
        return self.active_polls >= 0

    def stuck(self, _cmd: str) -> bool:
        return self.stuck_seq.pop(0) if self.stuck_seq else False

    def resume(self, pid: int) -> bool:
        self.resumed.append(pid)
        return True

    def tick(self, _seconds: float) -> None:
        self.sleeps += 1


@pytest.fixture
def box(monkeypatch: pytest.MonkeyPatch) -> Box:
    b = Box()
    monkeypatch.setattr(mod, "unit_active", b.unit_active)
    monkeypatch.setattr(mod, "stuck", b.stuck)
    monkeypatch.setattr(mod, "stop_unit", b.stopped.append)
    monkeypatch.setattr(mod, "resume", b.resume)
    return b


def run(box: Box, stuck_if: str | None = None, strikes: int = 2) -> int:
    return mod.watch("holder", 4242, stuck_if, strikes, 0.0, sleep=box.tick)


def test_resumes_when_the_holder_ends(box: Box) -> None:
    """MUTATION: returning without a resume once the unit goes inactive.

    A hard-killed holder looks exactly like this from outside: the unit just ends.
    """
    assert run(box) == 0
    assert box.resumed == [4242]
    assert box.stopped == []


def test_resumes_when_the_holder_is_already_gone(box: Box) -> None:
    """MUTATION: treating 'never saw it active' as 'nothing to do'.

    Armed late, after the holder died, the queue must still be released.
    """
    box.active_polls = 0
    run(box)
    assert box.resumed == [4242]
    assert box.sleeps == 0


def test_a_stuck_holder_is_stopped_then_the_queue_resumed(box: Box) -> None:
    """MUTATION: ignoring --stuck-if, or stopping without resuming.

    A holder waiting on an input that will never come never ends on its own.
    """
    box.active_polls = 100
    box.stuck_seq = [True, True]
    run(box, stuck_if="true", strikes=2)
    assert box.stopped == ["holder"]
    assert box.resumed == [4242]


def test_strikes_must_be_consecutive(box: Box) -> None:
    """MUTATION: counting stuck polls cumulatively instead of consecutively.

    One transient positive (an input appearing a moment after its producer exits)
    must not kill a holder that is about to proceed.
    """
    box.active_polls = 5
    box.stuck_seq = [True, False, True, False]
    run(box, stuck_if="x", strikes=2)
    assert box.stopped == []
    assert box.resumed == [4242]


def test_no_stuck_condition_never_stops_the_holder(box: Box) -> None:
    """MUTATION: stopping the holder when --stuck-if is absent."""
    box.active_polls = 5
    box.stuck_seq = [True] * 10
    run(box, stuck_if=None, strikes=1)
    assert box.stopped == []


def test_resume_really_continues_a_stopped_process() -> None:
    """MUTATION: sending the wrong signal (or none) in ``resume``."""
    p = subprocess.Popen(["sleep", "30"])
    try:
        p.send_signal(signal.SIGSTOP)
        assert _wait_state(p.pid, {"T"})
        assert mod.resume(p.pid) is True
        assert _wait_state(p.pid, {"S", "R"})
    finally:
        p.kill()
        p.wait()


def test_resume_of_a_vanished_queue_is_not_an_error() -> None:
    """MUTATION: raising when the queue already exited (the watchdog would die noisily)."""
    p = subprocess.Popen(["true"])
    p.wait()
    assert mod.resume(p.pid) is False


def _wait_state(pid: int, want: set[str], tries: int = 50) -> bool:
    """Poll /proc briefly: a signal's state change is asynchronous."""
    import time

    for _ in range(tries):
        with open(f"/proc/{pid}/stat") as f:
            if f.read().rsplit(")", 1)[1].split()[0] in want:
                return True
        time.sleep(0.02)
    return False
