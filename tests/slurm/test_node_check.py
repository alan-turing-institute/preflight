"""Plan B2: the part-time node health check, against a fake slurm (no live slurm is touched)."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import gpu_jobs
import node_check
import pytest

NODE = "laptop"
URGENT_NAME = "p:resume:urgent"
OURS = f"{node_check.REASON_PREFIX}low"
JOB_UID = 1234
HOSTLISTS = {"n[1,3],x": "n1\nn3\nx", f"n[1,3],{NODE}": f"n1\nn3\n{NODE}", NODE: NODE, "n9": "n9", f"{NODE},n9": f"{NODE}\nn9"}  # `scontrol show hostnames`, expanded
RESUME_CALL = ["scontrol", "update", f"NodeName={NODE}", "State=RESUME"]


def done(stdout: str = "", code: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess([], code, stdout, "boom" if code else "")


def job(job_id: str, started: int = 0, *, name: str = "p:resume:j", partition: str = "cpu", state: str = "RUNNING",
        node: str = NODE, batch: bool = True, requeue: bool = True, excluded: str = "") -> dict:
    return {
        "job_id": job_id, "job_state": [state], "partition": partition, "name": name,
        "nodes": node if state == "RUNNING" else "", "start_time": {"set": True, "number": started},
        "batch_flag": batch, "requeue": requeue, "excluded_nodes": excluded, "user_id": JOB_UID,
    }


class FakeSlurm:
    """Answers the two reads the check makes, applies drain/resume/requeue, and records every other call."""

    def __init__(self, jobs: list[dict] | None = None, *, drained: str | None = None, extra: str = "",
                 requeue_fails: bool = False) -> None:
        self.jobs, self.drained, self.extra, self.calls, self.fail = jobs or [], drained, extra, [], False
        self.requeue_fails, self.comment = requeue_fails, ""
        self.show_state, self.show_uid, self.exc = "PENDING", JOB_UID, "(null)"  # what `scontrol show job` says of any job
        self.asked, self.partitions = "cpu", {"cpu": f"{NODE},n9", "part_time": NODE}  # a queued job's partitions; each one's nodes

    def __call__(self, command: list[str]) -> subprocess.CompletedProcess:
        if self.fail:
            raise subprocess.TimeoutExpired(command, 10)
        if command[:3] == ["scontrol", "show", "node"]:
            state = ["IDLE"] + (["DRAIN"] if self.drained is not None else [])
            node = {"name": NODE, "state": state, "reason": self.drained or "", "extra": self.extra,
                    "comment": self.comment}
            return done(json.dumps({"nodes": [node]}))
        if command[0] == "squeue":
            return done(json.dumps({"jobs": self.jobs}))
        if command[:3] == ["scontrol", "show", "job"]:
            return done(f"JobId={command[3]} JobState={self.show_state} UserId=u({self.show_uid}) GroupId=g(1) "
                        f"Partition={self.asked} ExcNodeList={self.exc}")
        if command[:3] == ["scontrol", "show", "partition"]:
            return done(f"PartitionName={command[3]} AllowGroups=ALL Nodes={self.partitions[command[3]]} State=UP")
        if command[:3] == ["scontrol", "show", "hostnames"]:
            return done(HOSTLISTS[command[3]])
        if command[:2] == ["scontrol", "update"] and command[-1].startswith("Comment="):
            self.comment = command[-1].removeprefix("Comment=")
            return done()
        self.calls.append(command)
        if command[:2] == ["scontrol", "requeue"] and self.requeue_fails:
            return done(code=1)
        if command[0] == "docker" and getattr(self, "fail_docker", False):
            return done(code=1)
        if command[0] == "scancel":
            self.jobs = [entry for entry in self.jobs if entry["job_id"] != command[1]]
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
        "log_path": str(tmp_path / "check.log"), "state_dir": str(tmp_path / "state"),
    }
    path = tmp_path / "node_check.json"
    path.write_text(json.dumps(settings))
    events: list[tuple] = []

    class Box:
        settings_path, log, state_dir = path, Path(settings["log_path"]), tmp_path / "state"
        pushes = events
        containers: list[gpu_jobs.JobContainer] | None = None  # what docker lists; None: no docker, or it did not answer
        container_map = tmp_path / "run" / "containers.json"

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
                path, slurm, lambda *args, **fields: (events.append((*args, fields)), Path("/events") / args[0])[1],
                hostname=NODE + ".lan", gpu_busy=lambda: 0.4, sleep=lambda _s: None,
                containers=lambda: Box.containers, container_map=Box.container_map,
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
    assert slurm.comment.startswith("available_mb=1000;total_mb=") and ";cores_busy=" in slurm.comment and slurm.comment.endswith(";gpu_busy=0.40")  # what the page shows
    assert box.pushes == [("2", gpu_jobs.EventAction.PUSH_BACK, {"node": NODE, "free_mb": 1000})]
    assert [d.split(" (")[0] for d in decisions] == ["push back 2", "would close"]  # and the foreign drain left alone
    slurm.jobs[1].update(job_state=["RUNNING"], nodes=NODE)  # it started here again
    box.run(slurm)
    assert slurm.calls[-1] == ["scontrol", "update", "JobId=2", f"ExcNodeList={NODE}"]
    assert slurm.updates() == [slurm.calls[-1]]  # barred only at the second push
    # ...and only where the job has another node to go to: barred from its only node it would never run again
    alone = FakeSlurm([job("1", 10)], drained="gpujobs-admin: held by the owner")
    alone.partitions["cpu"] = NODE  # its only node
    barred = lambda s: [c for c in s.calls if c[:2] == ["scontrol", "update"] and "ExcNodeList" in c[-1]]  # noqa: E731
    for _ in range(2):
        box.run(alone)
        alone.jobs[0].update(job_state=["RUNNING"], nodes=NODE)
    assert barred(alone) == []
    alone.asked = "cpu,part_time"  # --anywhere: the main node is another it may go to, even from the part-time one
    alone.partitions["cpu"], alone.partitions["part_time"] = "n9", NODE
    box.run(alone)
    assert len(barred(alone)) == 1
    # a job that already carries an exclusion list keeps it, joined raw (never split on commas); one already barred is left
    for before, updates in [("n[1,3],x", [f"ExcNodeList=n[1,3],x,{NODE}"]), (f"n[1,3],{NODE}", [])]:
        again = FakeSlurm([job("6", 99, excluded=before)], drained="gpujobs-admin: held")
        box.run(again)
        slurm.jobs, again.jobs = [], [job("6", 99, excluded=before)]
        box.run(again)  # the second push of job 6 from this node
        assert [c[3] for c in again.updates()] == updates


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


def test_a_part_time_machine_runs_only_batch_requeueable_resume_jobs(box, monkeypatch):
    chowned: list[tuple] = []
    monkeypatch.setattr(node_check.os, "chown", lambda *args: chowned.append(args))
    box.free(20000)  # plenty of memory: the rule applies every run, not only when memory is low
    pt = gpu_jobs.Partition.PART_TIME
    slurm = FakeSlurm([
        job("1", 1, partition=pt), job("2", 2, partition=pt, batch=False), job("3", 3, partition=pt, requeue=False),
        job("4", 4, partition=pt, name="p:restart:r"), job("5", 5, partition=pt, name="not-ours"),
        job("6", 6, partition="cpu", batch=False),  # not on a part-time machine: not this rule's business
    ])
    decisions = box.run(slurm)
    assert slurm.calls == [["scancel", j] for j in ("2", "3", "4", "5")]
    assert [d.split(" (")[0] for d in decisions] == [f"cancel {j}" for j in ("2", "3", "4", "5")]
    assert [e[0] for e in box.pushes] == ["2", "3", "4", "5"]
    assert chowned == [(Path("/events") / j, JOB_UID, -1) for j in ("2", "3", "4", "5")]  # each event goes to the job's user
    assert all(e[1] is gpu_jobs.EventAction.KILL and e[2] == {"node": NODE, "reason": node_check.KILL_REASON} for e in box.pushes)
    # a push-back that slurm refuses to requeue falls back to cancelling, and the check still completes
    box.pushes.clear()
    box.free(1000)
    stuck = FakeSlurm([job("7", 7, partition=pt)], requeue_fails=True)
    decisions = box.run(stuck)
    assert stuck.calls[:2] == [["scontrol", "requeue", "7"], ["scancel", "7"]] and "cancelled" in decisions[0]
    assert box.pushes == [("7", gpu_jobs.EventAction.KILL, {"node": NODE, "reason": "requeue failed", "free_mb": 1000})]


def test_the_owner_manifest_lists_each_regular_state_file_with_its_owner_and_only_in_our_own_directory(box):
    state = box.state_dir
    for sub in node_check.MANIFEST_DIRS:
        (state / sub).mkdir(parents=True)
    (state / "jobs" / "5.json").write_text("{}")
    (state / "events" / "5-1-kill.json").write_text("{}")
    (state / "logs" / "n-5.log").symlink_to("/etc/hostname")  # a link is no file of anyone's
    (state / "other").mkdir()
    (state / "other" / "x.json").write_text("{}")  # not a directory the controller copies
    box.run(FakeSlurm())
    assert not (state / "owners").exists()  # no `owners/`: not a part-time machine, nothing is written
    (state / "owners").mkdir(mode=0o755)
    box.run(FakeSlurm())
    uid = os.getuid()
    assert json.loads((state / "owners" / "manifest.json").read_text()) == {"files": {"jobs/5.json": uid, "events/5-1-kill.json": uid}}
    assert (state / "owners" / "manifest.json").stat().st_mode & 0o777 == 0o644
    (state / "owners").chmod(0o777)  # a directory others can write is not ours to put the manifest in
    (state / "owners" / "manifest.json").unlink()
    box.run(FakeSlurm())
    assert not (state / "owners" / "manifest.json").exists() and "manifest not written" in box.log.read_text()


def test_a_launch_refusal_bars_the_node_from_the_pending_job_of_the_events_owner_only(box):
    events = box.state_dir / "events"
    events.mkdir(parents=True)

    def marker(job: str) -> Path:  # what the job's wrapper writes (as the job's user) before it requeues
        path = events / f"{job}-123-launch_refused.json"
        path.write_text("{}")
        return path

    slurm = FakeSlurm()
    slurm.show_uid = os.getuid()  # the marker's owner is the job's owner
    first = marker("5")
    assert "5 barred from laptop (launch check)" in box.run(slurm)
    assert slurm.updates() == [["scontrol", "update", "JobId=5", f"ExcNodeList={NODE}"]] and not first.exists()
    slurm.exc = "n[1,3],x"  # a job's own exclusion list is kept
    marker("6")
    box.run(slurm)
    assert slurm.updates()[-1] == ["scontrol", "update", "JobId=6", f"ExcNodeList=n[1,3],x,{NODE}"]
    before = len(slurm.updates())
    slurm.show_uid = JOB_UID  # a marker some other user planted for this job
    forged = marker("7")
    assert any("not written by the job's owner" in d for d in box.run(slurm)) and not forged.exists()
    slurm.show_uid, slurm.show_state = os.getuid(), "COMPLETING"  # the requeue has not landed yet: wait, then give up
    waiting = marker("8")
    box.run(slurm)
    assert waiting.exists()
    os.utime(waiting, (1, 1))
    box.run(slurm)
    assert not waiting.exists() and len(slurm.updates()) == before  # no exclusion for a job that is not waiting
    # slurm failing to expand a host list is an error (the run fails closed), never "not covered"
    with pytest.raises(RuntimeError, match="show hostnames 'n\\[1,3\\],x' failed"):
        node_check.exclusion(lambda command: subprocess.CompletedProcess(command, 1, "", "boom"), "n[1,3],x", NODE)


def test_free_mb_is_memavailable_in_decimal_megabytes_and_refuses_a_file_without_it(tmp_path: Path) -> None:
    meminfo = tmp_path / "meminfo"
    two_gb_kb = 2 * 10**9 // 1024  # 1953125 kB
    meminfo.write_text(f"MemTotal: 9 kB\nMemAvailable: {two_gb_kb} kB\n")
    assert node_check.free_mb(meminfo) == 2 * node_check.GB_MB  # decimal MB
    meminfo.write_text("MemTotal: 9 kB\n")
    with pytest.raises(ValueError):
        node_check.free_mb(meminfo)


def test_containers_of_ended_jobs_are_removed_and_the_rest_mapped_and_a_failure_never_closes(box, monkeypatch):
    now = 10_000.0
    monkeypatch.setattr(node_check.time, "time", lambda: now)
    ctr = {name: gpu_jobs.JobContainer(name * 64, job_id, now - age, "/system.slice/x.scope")
           for name, job_id, age in (("a", "1", 999), ("b", "9", 999), ("c", "9", 30))}  # job 1 runs; 9 has ended
    box.containers = list(ctr.values())
    slurm = FakeSlurm([job("1", 10)])
    decisions = box.run(slurm)
    assert slurm.calls == [["docker", "rm", "-f", "b" * 64]]  # old and of no unfinished job; c is too new to judge
    assert decisions == [f"removed container(s) of ended jobs: {'b' * 12} (job 9)",
                         f"kept new container(s) of no unfinished job, for now: {'c' * 12} (job 9)"]
    mapped = gpu_jobs.node_containers(box.container_map, owner=gpu_jobs.current_user(), now=lambda: now)
    assert mapped == [ctr["a"], ctr["c"]]  # what readers without docker see
    assert "removed container(s)" in box.log.read_text()  # one log line per run carries it
    rm_fails = FakeSlurm([job("1", 10)])
    rm_fails.fail_docker = True
    assert box.run(rm_fails)[0].startswith(f"could not remove container(s) of ended jobs {'b' * 12} (job 9): ")
    stuck = gpu_jobs.node_containers(box.container_map, owner=gpu_jobs.current_user(), now=lambda: now)
    assert ctr["b"] in stuck  # still in the map, so status.py keeps listing what could not be removed
    box.containers = None  # docker gone or hung: an old map must not be trusted
    assert box.run(FakeSlurm([job("1", 10)])) == [] and not box.container_map.exists()
    box.containers = [ctr["a"]]
    (blocker := box.container_map.parent / "file").write_text("")
    box.container_map = blocker / "containers.json"  # its folder cannot be made: the map cannot be written
    slurm = FakeSlurm([job("1", 10)])
    assert box.run(slurm)[0].startswith("containers not checked: ") and slurm.drained is None
