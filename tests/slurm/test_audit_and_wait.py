"""audit.py attributes GPU processes; wait.py follows a chain."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import audit
import pytest
import wait


def fake_proc(root: Path, pid: int, cgroup: str, parent: int, command: str = "python x") -> None:
    d = root / str(pid)
    d.mkdir(parents=True)
    (d / "cgroup").write_text(f"0::{cgroup}\n")
    (d / "stat").write_text(f"{pid} (python) S {parent} 0 0\n")
    (d / "cmdline").write_bytes(command.replace(" ", "\0").encode())


def test_attribution_by_job_cgroup_service_cgroup_and_ancestry(tmp_path: Path) -> None:
    proc, leases = tmp_path / "proc", tmp_path / "leases"
    leases.mkdir()
    owner = os.getpid()  # a live pid standing for a service's lease owner
    (leases / "service-a.json").write_text(json.dumps({"pid": owner, "project": "service-a"}))
    fake_proc(proc, owner, "/user.slice/service-a-models.service", 1)
    fake_proc(proc, 101, "/system.slice/gpu-node-1_slurmstepd.scope/job_5/step_batch", 1)
    fake_proc(proc, 108, "/system.slice/slurmstepd.scope/s00000000000RN/step_batch/user/task_0", 1)  # 26.05: the SLUID
    fake_proc(proc, 102, "/user.slice/service-a-models.service", 1)  # same unit as the lease owner
    fake_proc(proc, 103, "/user.slice/other.scope", owner)  # a child of the lease owner
    fake_proc(proc, 104, "/user.slice/proj-a-patch4.service", 1, "python train patch4\x1b[2J")
    # containers: one labelled for the leased project, one for a project with no lease, one unlabelled
    labelled, other, bare = ("a" * 64), ("b" * 64), ("c" * 64)
    fake_proc(proc, 105, f"/system.slice/docker-{labelled}.scope", 2)
    fake_proc(proc, 106, f"/system.slice/docker-{other}.scope", 2)
    fake_proc(proc, 107, f"/system.slice/docker-{bare}.scope", 2)
    labels = {labelled: "service-a", other: "service-b"}
    usage = {101: 100.0, 102: 200.0, 103: 300.0, 104: 400.0, 105: 1.0, 106: 2.0, 107: 3.0, 108: 5.0}
    found = audit.unregistered(usage, leases, proc, lambda cgroup, _label: next((v for k, v in labels.items() if k in cgroup), ""))
    assert [(pid, mib) for pid, mib, _ in found] == [(104, 400.0), (106, 2.0), (107, 3.0)]
    assert "patch4" in found[0][2] and found[0][2].isprintable()  # any user's command line reaches admins' terminals


def test_chain_follows_dependents_transitively(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sidecars = {"1": {}, "2": {"after": ["1"]}, "3": {"after": ["2"]}, "4": {"after": ["9"]}}
    assert wait.chain("1", sidecars) == ["1", "2", "3"]
    monkeypatch.setattr(wait.gpu_jobs, "STATE_DIR", tmp_path)
    (tmp_path / "jobs").mkdir()
    for name in ("5", "\x1b]0;forged\x07"):  # anyone may create a file in the sticky jobs directory
        (tmp_path / "jobs" / f"{name}.json").write_text(json.dumps({"after": ["1"]}))
    assert wait.chain("1", wait.all_sidecars()) == ["1", "5"]  # so a name that is no job id never reaches the terminal
    assert wait.chain("1", {"1": {}, "2": {"after": "1"}, "3": {"after": 1}}) == ["1"]  # only a list names dependencies


def test_wait_repeats_the_last_runs_usage_report_from_the_jobs_own_record_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(wait.gpu_jobs, "STATE_DIR", tmp_path)
    (tmp_path / "jobs").mkdir()
    me = wait.gpu_jobs.current_user()
    report = ["[job] used: memory peak 9.4 GB of 42.9 GB asked (22%)", "[job] for the same work next time: --mem 10G"]

    def record(lines: object) -> None:
        legs = [{"usage": {"lines": ["[job] used: an earlier run"]}}, {"usage": {"lines": lines}}]
        (tmp_path / "jobs" / "5.json").write_text(json.dumps({"legs": legs}))

    record(report)
    assert wait.usage_lines("5", me) == report  # the LAST run's
    assert wait.usage_lines("5", "") == []  # accounting names no user: the owner check would be skipped, so nothing
    assert wait.usage_lines("5", "somebody-else") == []  # a file the job's own user did not write
    assert wait.usage_lines("6", me) == []  # no record
    for junk in (["[job] a", "[job] b", "[job] c"], ["not ours"], [5], "[job] x"):
        record(junk)
        assert wait.usage_lines("5", me) == []
    record(["[job] \x1b]0;forged\x07" + "x" * 400])
    assert wait.usage_lines("5", me)[0].isprintable() and len(wait.usage_lines("5", me)[0]) == wait.LINE_MAX
    # an idle finding the agent recorded on a waited-on job is printed once while waiting (D41); one written by any other
    # user (the events folder is open to all) is a forgery and is not
    (tmp_path / "events").mkdir()
    wait.gpu_jobs.write_event("5", wait.gpu_jobs.EventAction.IDLE, role="holding", other_job="7", message="holds 8 cores\x1b[2J")
    wait.gpu_jobs.write_event("5", wait.gpu_jobs.EventAction.MEMORY_STOP, message="stopped for memory")  # D44, the same rule
    assert wait.new_notices(["5"], set()) == []  # written by this test's user, not the agent's
    monkeypatch.setattr(wait.gpu_jobs, "SLURM_USER", me)  # now this user stands for the agent
    seen: set[tuple[str, int]] = set()
    assert wait.new_notices(["5"], seen) == ["[wait] #5 holding: holds 8 cores?[2J", "[wait] #5 memory stop: stopped for memory"]
    assert wait.new_notices(["5"], seen) == []  # once
    # the command itself prints them under the job's final state
    record(report)
    monkeypatch.setattr(wait.gpu_jobs, "read_queue", list)  # nothing unfinished
    monkeypatch.setattr(wait.gpu_jobs, "completion_records", lambda _jobs: {"5": {"JobState": "COMPLETED", "UserId": me}})
    monkeypatch.setattr(sys, "argv", ["wait.py", "5"])
    assert wait.main() == 0
    assert capsys.readouterr().out.splitlines() == ["[wait] #5 holding: holds 8 cores?[2J", "[wait] #5 memory stop: stopped for memory",
                                                    "[wait] 5: COMPLETED", *report]
