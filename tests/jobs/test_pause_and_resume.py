"""Tests for the pause-and-resume driver.

Each test names the mutation it would catch. The failures being guarded: A stopped before
a fresh checkpoint (work lost), A frozen or left stopped forever, A resumed when it had
already finished, or B started on a GPU A still holds.
"""

from __future__ import annotations

from pathlib import Path

import pause_and_resume as mod
import pytest


class Box:
    """Scripted world: units, a checkpoint whose mtime changes on given polls, a clock."""

    def __init__(self) -> None:
        self.active = {"A": True, "B": False}
        self.a_ends_after_polls: int | None = None
        self.b_active_polls = 3
        self.mtimes: list[int | None] = [100]
        self.events: list[tuple[str, str]] = []
        self.start_ok = {"B": True, "A-resume": True}
        self.now = 0.0
        self.stopped_at: float | None = None
        self.polls = 0

    def unit_active(self, unit: str) -> bool:
        if unit == "A":
            self.polls += 1
            if (
                self.a_ends_after_polls is not None
                and self.polls > self.a_ends_after_polls
            ):
                self.active["A"] = False
            return self.active["A"]
        if not self.active["B"]:
            return False
        self.b_active_polls -= 1
        if self.b_active_polls < 0:
            self.active["B"] = False
        return self.active["B"]

    def mtime_ns(self, _path: Path) -> int | None:
        return self.mtimes.pop(0) if len(self.mtimes) > 1 else self.mtimes[0]

    def stop_unit(self, unit: str) -> None:
        self.events.append(("stop", unit))
        self.stopped_at = self.now
        self.active[unit] = False

    def start_unit(self, unit: str, command: str, _workdir: Path) -> bool:
        self.events.append(("start", unit))
        if unit == "B" and self.start_ok["B"]:
            assert not self.active["A"], "B started while A still held the GPU"
            self.active["B"] = True
        return self.start_ok[unit]

    def sleep(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def box(monkeypatch: pytest.MonkeyPatch) -> Box:
    b = Box()
    monkeypatch.setattr(mod, "unit_active", b.unit_active)
    monkeypatch.setattr(mod, "mtime_ns", b.mtime_ns)
    monkeypatch.setattr(mod, "stop_unit", b.stop_unit)
    monkeypatch.setattr(mod, "start_unit", b.start_unit)
    return b


def run(box: Box, max_wait: float = 1000.0) -> int:
    return mod.pause_and_resume(
        "A",
        Path("last.ckpt"),
        "B",
        "train b",
        "A-resume",
        "resume a",
        Path("."),
        settle=10.0,
        interval=5.0,
        max_wait=max_wait,
        sleep=box.sleep,
        clock=lambda: box.now,
    )


def test_stops_only_after_a_fresh_checkpoint_then_resumes(box: Box) -> None:
    """MUTATION: stopping A at arming, before its checkpoint is rewritten."""
    box.mtimes = [100, 100, 100, 200, 200]
    assert run(box) == 0
    assert box.events == [("stop", "A"), ("start", "B"), ("start", "A-resume")]
    assert box.stopped_at == 2 * 5.0 + 10.0  # two unchanged polls, then the settle


def test_a_checkpoint_still_being_written_is_waited_out(box: Box) -> None:
    """MUTATION: stopping A on the first changed mtime without checking it settled.

    Lightning writes a ~170 MB file; stopping mid-write leaves a truncated last.ckpt.
    """
    box.mtimes = [100, 200, 300, 300, 300]
    run(box)
    assert box.events[0] == ("stop", "A")
    assert (
        box.stopped_at == 2 * 10.0
    )  # one settle broken by the second write, one clean


def test_the_resume_starts_only_after_b_has_ended(box: Box) -> None:
    """MUTATION: starting A's resume as soon as B starts (two jobs on one GPU)."""
    box.mtimes = [100, 200, 200]
    box.b_active_polls = 4
    run(box)
    assert box.events.index(("start", "A-resume")) > box.events.index(("start", "B"))
    assert not box.active["B"]


def test_a_failed_b_still_resumes_a(box: Box) -> None:
    """MUTATION: resuming only when B succeeded -- A would stay stopped for good."""
    box.mtimes = [100, 200, 200]
    box.start_ok["B"] = False
    assert run(box) == 0
    assert ("start", "A-resume") in box.events


def test_an_inactive_a_is_neither_stopped_nor_resumed(box: Box) -> None:
    """MUTATION: resuming a job that had already finished (it would train past its end)."""
    box.active["A"] = False
    assert run(box) == 0
    assert box.events == [("start", "B")]


def test_an_a_that_ends_while_waiting_is_not_resumed(box: Box) -> None:
    """MUTATION: treating 'A ended on its own' like 'A was paused'."""
    box.mtimes = [100]
    box.a_ends_after_polls = 3
    assert run(box) == 0
    assert box.events == [("start", "B")]


def test_no_rewrite_within_max_wait_touches_nothing(box: Box) -> None:
    """MUTATION: stopping A, or starting B, after the wait ran out (A may be hung)."""
    box.mtimes = [100]
    assert run(box, max_wait=20.0) == mod.EXIT_CHECKPOINT_TIMEOUT
    assert box.events == []


def test_a_resume_systemd_refuses_is_reported_as_failure(box: Box) -> None:
    """MUTATION: exiting 0 when A was stopped and could not be restarted."""
    box.mtimes = [100, 200, 200]
    box.start_ok["A-resume"] = False
    assert run(box) == 1


def test_start_unit_runs_the_command_as_its_own_user_unit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MUTATION: running B inside this watcher's process (it would die with it)."""
    seen: list[list[str]] = []

    class Done:
        returncode = 0

    monkeypatch.setattr(
        mod.subprocess, "run", lambda argv, check: seen.append(argv) or Done()
    )
    assert mod.start_unit("B", "echo hi", Path("/w")) is True
    assert seen == [
        [
            "systemd-run",
            "--user",
            "--unit=B",
            "--working-directory=/w",
            "bash",
            "-c",
            "echo hi",
        ]
    ]


def test_mtime_of_a_missing_checkpoint_is_none(tmp_path: Path) -> None:
    """MUTATION: raising on a checkpoint not yet written (A in its first epoch)."""
    assert mod.mtime_ns(tmp_path / "last.ckpt") is None
    (tmp_path / "last.ckpt").write_bytes(b"x")
    assert isinstance(mod.mtime_ns(tmp_path / "last.ckpt"), int)
