"""Tests for the resume-after-kill supervisor.

Each test names the mutation it would catch. The failures being guarded: a finished job
resumed again, a killed job never resumed, a resume started into the same squeeze that
killed it, a deterministic failure resumed forever, and a watched job judged before it
has ended.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import resume_after_kill as mod

#: Polls for which the fake pid stays alive.
PID_ALIVE_POLLS = 2
#: ready-if answers "not yet" twice, then "ready": three checks before the resume.
READY_CHECKS_BEFORE_RESUME = 3
MAX_RESUMES = 2
INTERVAL_S = 10.0
#: More shell checks than any test needs: a runaway loop fails instead of hanging.
CALL_BUDGET = 100


class Box:
    """Scripted world: shell checks answer from queues, a fake pid, a clock."""

    def __init__(self) -> None:
        self.answers: dict[str, list[int]] = {}
        self.calls: list[str] = []
        self.now = 0.0
        self.pid_polls = 0

    def script(self, command: str, *codes: int) -> None:
        self.answers[command] = list(codes)

    def shell(self, command: str, _workdir: Path) -> int:
        self.calls.append(command)
        if len(self.calls) > CALL_BUDGET:
            msg = "runaway loop: shell called past the budget"
            raise AssertionError(msg)
        codes = self.answers.get(command, [0])
        return codes.pop(0) if len(codes) > 1 else codes[0]

    def pid_alive(self, _pid: int) -> bool:
        self.pid_polls += 1
        return self.pid_polls <= PID_ALIVE_POLLS

    def sleep(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def box(monkeypatch: pytest.MonkeyPatch) -> Box:
    b = Box()
    monkeypatch.setattr(mod, "shell", b.shell)
    monkeypatch.setattr(mod, "pid_alive", b.pid_alive)
    return b


def run(box: Box, **kw: object) -> mod.Exit:
    args: dict[str, object] = {
        "resume_command": "resume",
        "done_if": "done?",
        "ready_if": "ready?",
        "workdir": Path("."),
        "command": "job",
        "interval": INTERVAL_S,
        "max_wait": 100.0,
        "sleep": box.sleep,
        "clock": lambda: box.now,
    }
    args.update(kw)
    return mod.supervise(**args)  # type: ignore[arg-type]


def test_finished_job_is_not_resumed(box: Box) -> None:
    # Mutation caught: resuming without checking done-if first.
    box.script("done?", 0)
    assert run(box) is mod.Exit.DONE
    assert "resume" not in box.calls


def test_killed_job_is_resumed_once_the_box_is_ready(box: Box) -> None:
    # Mutation caught: resuming before ready-if succeeds (straight back into the squeeze).
    box.script("done?", 1, 0)
    box.script("ready?", 1, 1, 0)
    assert run(box) is mod.Exit.DONE
    resume_at = box.calls.index("resume")
    assert box.calls[:resume_at].count("ready?") == READY_CHECKS_BEFORE_RESUME
    assert box.calls.count("resume") == 1


def test_give_up_check_stops_a_failure_a_resume_cannot_fix(box: Box) -> None:
    # Mutation caught: ignoring give-up-if, so a traceback is resumed max_resumes times.
    box.script("done?", 1)
    box.script("traceback?", 0)
    assert run(box, give_up_if="traceback?") is mod.Exit.GAVE_UP
    assert "resume" not in box.calls


def test_resumes_are_bounded(box: Box) -> None:
    # Mutation caught: an unbounded resume loop on a job that keeps dying.
    box.script("done?", 1)
    assert run(box, max_resumes=MAX_RESUMES) is mod.Exit.OUT_OF_RESUMES
    assert box.calls.count("resume") == MAX_RESUMES


def test_box_never_ready_gives_up_without_resuming(box: Box) -> None:
    # Mutation caught: resuming after max_wait expires anyway.
    box.script("done?", 1)
    box.script("ready?", 1)
    assert run(box) is mod.Exit.NEVER_READY
    assert "resume" not in box.calls


def test_watched_job_is_judged_only_after_it_ends(box: Box) -> None:
    # Mutation caught: checking done-if while the watched pid is still alive.
    box.script("done?", 0)
    assert run(box, command=None, watch_pid=123) is mod.Exit.DONE
    assert box.pid_polls == PID_ALIVE_POLLS + 1
    assert box.now == PID_ALIVE_POLLS * INTERVAL_S
    assert "job" not in box.calls
