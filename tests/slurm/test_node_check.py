"""Plan B2: the part-time node health check, against a fake slurm (no live slurm is touched)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import gpu_jobs
import node_check
import pytest

NODE = "laptop"
URGENT_NAME = "p:resume:urgent"
OURS = f"{node_check.REASON_PREFIX}low"
RESUME_CALL = ["scontrol", "update", f"NodeName={NODE}", "State=RESUME"]


def done(stdout: str = "", code: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess([], code, stdout, "boom" if code else "")


def job(job_id: str, started: int = 0, *, name: str = "p:resume:j", partition: str = "cpu", state: str = "RUNNING",
        node: str = NODE) -> dict:
    return {
        "job_id": job_id, "job_state": [state], "partition": partition, "name": name,
        "nodes": node if state == "RUNNING" else "", "start_time": {"set": True, "number": started},
    }


class FakeSlurm:
    """Answers the two reads the check makes, applies drain/resume/requeue, and records every other call."""

    def __init__(self, jobs: list[dict] | None = None, *, drained: str | None = None, extra: str = "") -> None:
        self.jobs, self.drained, self.extra, self.calls, self.fail = jobs or [], drained, extra, [], False

    def __call__(self, command: list[str]) -> subprocess.CompletedProcess:
        if self.fail:
            raise subprocess.TimeoutExpired(command, 10)
        if command[:3] == ["scontrol", "show", "node"]:
            state = ["IDLE"] + (["DRAIN"] if self.drained is not None else [])
            node = {"name": NODE, "state": state, "reason": self.drained or "", "extra": self.extra}
            return done(json.dumps({"nodes": [node]}))
        if command[0] == "squeue":
            return done(json.dumps({"jobs": self.jobs}))
        self.calls.append(command)
        if command[:2] == ["scontrol", "requeue"]:
            for entry in self.jobs:
                if entry["job_id"] == command[2]:
                    entry.update(job_state=["PENDING"], nodes="")
        if command[:2] == ["scontrol", "update"] and "State=DRAIN" in command:
            self.drained = command[-1].removeprefix("Reason=")
        if command[:2] == ["scontrol", "update"] and "State=RESUME" in command:
            self.drained = None
        return done()

    def updates(self) -> list[list[str]]:
        return [c for c in self.calls if c[:2] == ["scontrol", "update"]]


@pytest.fixture
def box(tmp_path: Path):
    """Settings on disk plus helpers to set the free memory and the power supplies."""
    meminfo, power = tmp_path / "meminfo", tmp_path / "power"
    power.mkdir()
    settings = {
        "node": NODE, "close_below_mb": 6000, "push_back_below_mb": 2000, "mains_only": None,
        "meminfo_path": str(meminfo), "power_supply_dir": str(power), "state_path": str(tmp_path / "state.json"),
        "log_path": str(tmp_path / "check.log"),
    }
    path = tmp_path / "node_check.json"
    path.write_text(json.dumps(settings))
    events: list[tuple] = []

    class Box:
        settings_path, log = path, Path(settings["log_path"])
        pushes = events

        @staticmethod
        def free(mb: int) -> None:  # MemAvailable is in kB; the check reads decimal MB
            meminfo.write_text(f"MemTotal: 1 kB\nMemAvailable: {-(-mb * 1_000_000 // 1024)} kB\n")

        @staticmethod
        def supply(name: str, kind: str, online: str = "1") -> None:
            (power / name).mkdir()
            (power / name / "type").write_text(kind + "\n")
            (power / name / "online").write_text(online + "\n")

        @staticmethod
        def run(slurm: FakeSlurm) -> list[str]:
            return node_check.check(
                path, slurm, lambda *args, **fields: events.append((*args, fields)), hostname=NODE + ".lan"
            )

    Box.free(20000)
    return Box


def test_push_back_takes_the_latest_started_resume_job_only_and_excludes_at_the_second(box):
    box.free(1000)
    slurm = FakeSlurm(
        [job("1", 10), job("2", 50), job("3", 90, name="p:restart:r"), job("4", 95, partition="urgent"),
         job("5", 99, partition="service")],
        drained="gpujobs-admin: held by the owner",  # foreign, so this run changes nothing but the push
    )
    decisions = box.run(slurm)
    assert [c[:3] for c in slurm.calls] == [["scontrol", "requeue", "2"]]  # one job; not restart/urgent/service
    assert box.pushes == [("2", gpu_jobs.EventAction.PUSH_BACK, {"node": NODE, "free_mb": 1000})]
    assert [d.split(" (")[0] for d in decisions] == ["push back 2", "would close"]  # and the foreign drain left alone
    slurm.jobs[1].update(job_state=["RUNNING"], nodes=NODE)  # it started here again
    box.run(slurm)
    assert slurm.calls[-1] == ["scontrol", "update", "JobId=2", f"ExcNodeList={NODE}"]
    assert slurm.updates() == [slurm.calls[-1]]  # barred only at the second push


def test_close_for_memory_or_battery_and_only_when_asked(box):
    quiet = FakeSlurm()  # plenty of memory and no power supply at all: logged as allowed, nothing is changed
    assert box.run(quiet) == [] and quiet.calls == []
    assert "free=20000 MB allow" in box.log.read_text()
    box.free(5000)
    slurm = FakeSlurm()
    box.run(slurm)
    drain = ["scontrol", "update", f"NodeName={NODE}", "State=DRAIN", "Reason=gpujobs-check: 5.0 GB free"]
    assert slurm.updates() == [drain]
    box.free(20000)  # plenty of memory but a battery and no mains: closed by default
    box.supply("BAT0", "Battery")
    box.supply("AC", "Mains", "0")
    battery = FakeSlurm()
    box.run(battery)
    assert "on battery" in battery.drained
    settings = json.loads(box.settings_path.read_text())
    box.settings_path.write_text(json.dumps({**settings, "mains_only": False}))
    allowed = FakeSlurm()
    box.run(allowed)
    assert allowed.calls == []


def test_urgent_waiting_never_closes_and_reopens_ours(box):
    box.free(1000)
    urgent = job("9", partition="urgent", state="PENDING", name=URGENT_NAME)
    slurm = FakeSlurm([urgent, job("1", 5)])
    box.run(slurm)
    assert slurm.drained is None  # below both thresholds, yet not closed; only the push happens
    assert [c[:2] for c in slurm.calls] == [["scontrol", "requeue"]]
    ours = FakeSlurm([urgent], drained=OURS)
    box.run(ours)
    assert ours.updates() == [RESUME_CALL]


def test_reopen_after_three_passing_runs_not_two_and_a_foreign_drain_never(box):
    slurm = FakeSlurm(drained=OURS)
    box.run(slurm)
    box.run(slurm)
    assert slurm.updates() == []
    box.run(slurm)
    assert slurm.updates() == [RESUME_CALL]
    box.free(1000)  # a failing run resets the count
    again = FakeSlurm(drained=OURS)
    box.run(again)
    box.free(20000)
    box.run(again)
    box.run(again)
    assert again.updates() == []
    foreign = FakeSlurm(drained="gpujobs-admin: closed from the page")  # never reopened, however many runs pass
    for _ in range(node_check.REOPEN_AFTER + 1):
        box.run(foreign)
    box.free(1000)  # and never changed while memory is low either
    box.run(foreign)
    assert foreign.updates() == [] and foreign.drained == "gpujobs-admin: closed from the page"


def test_extra_overrides_valid_keys_only_and_the_settings_may_be_keyed_by_node(box):
    box.free(5000)
    assert box.run(FakeSlurm(extra='{"close_below_mb": 4000}')) == []  # 5000 is above the overridden 4000
    invalid = FakeSlurm(extra='{"close_below_mb": "4000"}')
    assert "Extra ignored" in box.run(invalid)[0]
    assert invalid.drained  # the file's 6000 applies
    unknown = FakeSlurm(extra='{"close_below_mb": 4000, "meminfo_path": "/x"}')
    assert "Extra ignored" in box.run(unknown)[0]
    settings = json.loads(box.settings_path.read_text())
    box.settings_path.write_text(json.dumps({"other": {**settings, "close_below_mb": 1}, NODE: settings}))
    assert box.run(FakeSlurm())  # this node's entry (6000) closes it, not "other"'s


def test_any_failure_fails_closed_and_leaves_a_foreign_drain(box):
    box.settings_path.write_text("not json")
    broken = FakeSlurm()
    with pytest.raises(ValueError, match="Expecting value"):
        box.run(broken)
    node_check.fail_closed(ValueError("x"), broken, NODE, box.log)
    assert broken.drained == f"{node_check.REASON_PREFIX}check failed: ValueError"
    foreign = FakeSlurm(drained="someone else")
    node_check.fail_closed(ValueError("x"), foreign, NODE, box.log)
    assert foreign.drained == "someone else"
    assert box.log.read_text().count("check failed") == len(["broken", "foreign"])
    timeout = FakeSlurm()
    timeout.fail = True
    node_check.fail_closed(OSError("y"), timeout, NODE, box.log)  # even the drain times out: logged, no raise
    assert "could not drain" in box.log.read_text()
