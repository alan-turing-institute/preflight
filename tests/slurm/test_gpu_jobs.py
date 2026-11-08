"""gpu_jobs: events keyed to file ownership, squeue rows, container labels, nodes, and where state lives."""

from __future__ import annotations

import importlib
import itertools
import json
import os
import pwd
import signal
import subprocess
import types
from collections.abc import Callable
from pathlib import Path

import gpu_jobs
import levels
import pytest
import site_config
import status
import urgent
import wait
from gpu_jobs import EventAction, Level, QueuedJob


def test_event_actor_is_the_file_owner_not_the_content(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpu_jobs, "STATE_DIR", tmp_path)
    forged = tmp_path / "events" / "7-100-kill.json"
    forged.parent.mkdir()
    forged.write_text(json.dumps({"job": "7", "action": "kill", "actor": "mallory", "at": 100, "why": "x"}))
    path = gpu_jobs.write_event("7", EventAction.PAUSE, why="urgent")
    assert path.stat().st_mode & 0o777 == 0o644
    gpu_jobs.update_sidecar("7", what="x")  # a record, like a request or the agent's state, is read by other users
    assert gpu_jobs.sidecar_path("7").stat().st_mode & 0o777 == 0o644
    found = gpu_jobs.events("7")
    assert [e.action for e in found] == ["kill", "pause"]  # sorted by the time in the name
    assert {e.actor for e in found} == {Path(os.path.expanduser("~")).owner()}
    assert found[1].fields == {"why": "urgent"}
    gpu_jobs.write_event("8", EventAction.RESUME)
    assert len(gpu_jobs.events()) == 3
    assert [e.job for e in gpu_jobs.events("8")] == ["8"]


def test_a_hostile_or_foreign_record_is_ignored_and_never_blocks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpu_jobs, "STATE_DIR", tmp_path)
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    target = tmp_path / "victim-file"
    target.write_text("theirs")
    (scripts / "x-1.sbatch").symlink_to(target)  # left under the predictable name: never written through
    assert gpu_jobs.write_new(scripts / "x-1.sbatch", "mine") == scripts / "x-1-2.sbatch"
    assert target.read_text() == "theirs" and (scripts / "x-1-2.sbatch").read_text() == "mine"
    monkeypatch.setattr(gpu_jobs, "NEW_NAME_TRIES", 1)
    with pytest.raises(FileExistsError, match="already exist"):
        gpu_jobs.write_new(scripts / "x-1.sbatch", "again")
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    os.mkfifo(jobs / "1.json")  # any user may leave one; opening it for reading would block for ever
    (jobs / "2.json").symlink_to(jobs / "6.json")
    (jobs / "3.json").mkdir()
    (jobs / "4.json").write_text('{"x": 1}' + " " * gpu_jobs.MAX_RECORD_BYTES)  # valid JSON, too big to be a record
    (jobs / "5.json").write_text("[" * 200_000)  # deep enough to overflow the parser's stack
    (jobs / "6.json").write_text(json.dumps({"gpu": True, "name": "n", "what": "forged"}))
    signal.signal(signal.SIGALRM, lambda *_: pytest.fail("read_json blocked"))
    signal.alarm(5)
    try:
        assert [gpu_jobs.read_sidecar(str(i)) for i in range(1, 6)] == [None] * 5
    finally:
        signal.alarm(0)
    me = gpu_jobs.current_user()
    assert gpu_jobs.read_sidecar("6", me) == {"gpu": True, "name": "n", "what": "forged"}
    assert gpu_jobs.read_sidecar("6", "somebody-else") is None  # the file's owner is its writer's identity
    job = QueuedJob("6", "normal", "PENDING", "None", "n", user="somebody-else")
    assert not gpu_jobs.needs_the_card(job)  # the record says gpu, slurm granted none
    assert gpu_jobs.needs_the_card(QueuedJob("7", "normal", "PENDING", "None", "n", gpus=1))  # no record needed
    assert gpu_jobs.needs_the_card(QueuedJob("7", "urgent", "PENDING", "None", "n"))
    assert not gpu_jobs.needs_the_card(QueuedJob("7", "service", "RUNNING", "None", "n"))  # a stand-in: no demand (D20)
    events = tmp_path / "events"
    events.mkdir()
    (events / "7-100-kill.json").write_text("[" * 200_000)
    os.mkfifo(events / "9-100-kill.json")
    signal.alarm(5)
    try:
        (events / "junk.json").write_text("[" * 200_000)
        opened: list[Path] = []
        real = gpu_jobs.read_json
        monkeypatch.setattr(gpu_jobs, "read_json", lambda path, user=None: opened.append(path.name) or real(path, user))
        assert gpu_jobs.events() == [] and gpu_jobs.events("7") == []
        assert "junk.json" not in opened  # judged by its name: a file that is not an event is never opened
        assert gpu_jobs.events("8") == []
    finally:
        signal.alarm(0)
    assert wait.all_sidecars({"6": "somebody-else"})["6"] == {}
    monkeypatch.setattr(status.gpu_jobs, "node_state", lambda: "idle")
    monkeypatch.setattr(status.audit, "unregistered", list)
    monkeypatch.setattr(status.audit, "report", lambda _u: [])
    assert not any("forged" in line for line in status.lines([job]))
    named = QueuedJob("8", "normal", "PENDING", "None", "p:resume:\x1b]0;forged\x07")
    assert all(line.isprintable() for line in status.lines([named]))  # a terminal shows a name, never acts on it
    assert "[status] 8 PENDING    ?          normal 0 GPU" in "\n".join(status.lines([named]))  # the level after the user
    running = QueuedJob("9", "part_time", "RUNNING", "None", "p:resume:n", node="laptop")
    assert " p:resume:n on laptop |" in "\n".join(status.lines([running]))  # a running job says where it is
    keeper = any(line.startswith("[status] keeper:") for line in status.lines([named]))
    monkeypatch.setattr(status.gpu_jobs, "LEASE_DIR", None)  # a site without a lease (Turing box) has no keeper line
    assert keeper and not any(line.startswith("[status] keeper:") for line in status.lines([named]))
    # an update never merges into a record that is not this user's, and a write that cannot replace says so
    monkeypatch.setattr(gpu_jobs, "current_user", lambda: "somebody-else")
    gpu_jobs.update_sidecar("6", extra=1)
    assert gpu_jobs.read_sidecar("6") == {"job_id": "6", "extra": 1}
    with pytest.raises(OSError, match="belongs to"):
        gpu_jobs.update_sidecar("3", what="x")
    assert not list(jobs.glob(".*.tmp"))


def test_the_peak_for_mem_auto_comes_from_clean_finished_legs_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpu_jobs, "STATE_DIR", tmp_path)
    (tmp_path / "jobs").mkdir()
    done = {"peak_gpu_mib": 1024, "peak_mem_bytes": 0, "ended_at": 1, "exit": 0, "signals": []}
    crashed = {**done, "peak_gpu_mib": 40, "ended_at": 3, "exit": 1}
    preempted = {**done, "peak_gpu_mib": 40, "ended_at": 2, "signals": ["SIGTERM"]}
    legs = [done, preempted, crashed]
    (tmp_path / "jobs" / "1.json").write_text(json.dumps({"name": "n", "legs": legs, "last_leg": crashed}))
    assert gpu_jobs.measured_peak_bytes("n") == 1024 * 1024**2
    # with no clean run: the LARGEST peak of an unclean leg that ran long enough; a leg that died loading is ignored
    long_crash = {**crashed, "peak_gpu_mib": 2048, "started_at": 0, "ended_at": gpu_jobs.UNCLEAN_MIN_S}
    early_crash = {**crashed, "peak_gpu_mib": 4096, "started_at": 0, "ended_at": 5}
    (tmp_path / "jobs" / "9.json").write_text(json.dumps({"name": "m", "legs": [long_crash, early_crash, preempted]}))
    assert gpu_jobs.measured_peak_bytes("m") is None
    assert gpu_jobs.measured_peak_bytes("m", clean=False) == 2048 * 1024**2
    (tmp_path / "jobs" / "2.json").write_text(json.dumps({"name": "n", "legs": 5, "last_leg": {"peak_gpu_mib": "x", "exit": 0}}))
    assert gpu_jobs.measured_peak_bytes("n") == 1024 * 1024**2  # wrong types inside a valid record are skipped
    monkeypatch.setattr(gpu_jobs, "current_user", lambda: "somebody-else")  # the file is not theirs
    assert gpu_jobs.measured_peak_bytes("n") is None


def squeue_json(*jobs: dict[str, object]) -> str:
    """`squeue --json` output in the shape slurm 23.11.4 writes, for the fields read."""
    base = {"partition": "normal", "job_state": ["PENDING"], "state_reason": "Priority", "dependency": "",
            "user_name": "ann", "priority": {"set": True, "infinite": False, "number": 1}, "tres_per_node": "", "name": "n"}
    return json.dumps({"jobs": [{**base, **job} for job in jobs]})


def test_completion_records_take_each_jobs_latest_record_and_a_name_forges_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    forged = "a b\nJobId=999 UserId=bob(2002) JobState=FAILED Name=forged"  # forged a record in the jobcomp log
    legs = {"7": {"job_id": 7, "state": {"current": ["FAILED"]}, "name": forged, "user": "alice"},
            "8": {"job_id": 8, "state": {"current": ["CANCELLED"]}, "name": "n", "user": "bob"}}
    asked: list[tuple[str, ...]] = []

    def sacct(*selection: str) -> list[dict[str, object]]:
        asked.append(selection)
        if selection[0] == "-S":  # the window can return an earlier leg of a requeued job (measured)
            return [legs["7"], {**legs["8"], "state": {"current": ["PENDING", "REQUEUED"]}}]
        return [legs[i] for i in selection[1].split(",")]

    monkeypatch.setattr(gpu_jobs, "_sacct_jobs", sacct)
    records = gpu_jobs.completion_records()
    assert set(records) == {"7", "8"} and records["8"]["JobState"] == "CANCELLED"
    assert records["7"] == {"JobId": "7", "JobState": "FAILED", "Name": forged, "UserId": "alice"}
    assert gpu_jobs.completion_records(["8"])["8"]["UserId"] == "bob" and asked[-1] == ("-j", "8")


def test_cores_from_history_are_the_highest_average_of_the_three_newest_completed_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    def job(end: int, core_s: int, elapsed: int, name: str = "p:restart:n", micro: int = 0) -> dict[str, object]:
        return {"name": name, "time": {"end": end, "elapsed": elapsed, "total": {"seconds": core_s, "microseconds": micro}}}

    jobs = [job(1, 900, 100), job(2, 300, 100), job(3, 100, 100), job(4, 249, 100, micro=500_000),
            job(5, 0, 0), job(6, 999, 1, name="p:restart:n2"), {"name": "p:restart:n", "time": "x"}]
    asked: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
    monkeypatch.setattr(gpu_jobs, "_sacct_jobs", lambda *sel, flags, timeout: asked.append((sel, flags, timeout)) or jobs)
    # 9.0 from run 1 is older than the newest three; no wall time, another name, a bad record: skipped
    assert gpu_jobs.measured_cores("p:restart:n") == (3.0, 3)
    selection, flags, timeout = asked[-1]
    assert timeout == gpu_jobs.SUBMIT_SACCT_S and flags == () and selection[:4] == ("-u", gpu_jobs.current_user(), "--name", "p:restart:n")  # with steps: -X has no core time
    assert ("-S", gpu_jobs._SACCT_EPOCH) == selection[6:8] and selection[-2:] == ("-E", "now")  # with -s both ends, or nothing
    monkeypatch.setattr(gpu_jobs, "_sacct_jobs", lambda *_sel, flags, timeout: None)
    assert gpu_jobs.measured_cores("p:restart:n") is None  # accounting does not answer: nothing proposed


def test_parse_squeue_reads_json_so_a_name_cannot_forge_a_row() -> None:
    # Slurm keeps a newline in a job name and lets the owner rename the job (measured, 23.11.4): in a text row this
    # name was a second, forged job 999 with priority 9999.
    forged = "x\n999|normal|PENDING|None||bob|9999|N/A|y"
    text = squeue_json(
        {"job_id": 12, "job_state": ["RUNNING"], "priority": {"set": True, "number": 4001}, "tres_per_node": "gres/gpu:2",
         "name": forged, "qos": "high", "nodes": "gpu-node-1", "memory_per_node": {"set": True, "number": 61440},
         "submit_time": {"set": True, "infinite": False, "number": 1792838841}},
        {"job_id": 13, "job_state": ["COMPLETED"]},  # squeue --json ignored -t: finished jobs are dropped here
        {"job_id": 14, "priority": {"set": False, "number": 7}, "state_reason": "JobHeldUser", "qos": "staff",
         "cpus": {"number": 4}, "memory_per_node": {"set": False, "number": 0}, "memory_per_cpu": {"set": True, "number": 2048}},
        # Still saving after a pause (requeued) or a kill: COMPLETING is a flag after the base state in JSON.
        {"job_id": 15, "job_state": ["PENDING", "COMPLETING"]},
        {"job_id": 16, "job_state": ["CANCELLED", "COMPLETING"]},
    )
    running, held, paused, killed = gpu_jobs.parse_squeue(text)
    assert (running.job_id, running.user, running.priority, running.gpus, running.name) == ("12", "ann", 4001, 2, forged)
    assert (held.job_id, held.priority, held.held) == ("14", 0, True)
    assert (running.level, held.level) == (Level.HIGH, Level.NORMAL)  # an admin's own QOS counts as normal
    # slurm's memory request as it stands (a later `scontrol update` shows), per cpu times cores, None when it gave none
    assert (running.mem_mb, held.mem_mb, paused.mem_mb) == (61440, 8192, None)
    assert (running.submitted, held.submitted) == (1792838841, 0)
    assert (running.node, held.node) == ("gpu-node-1", "")  # only a running job has a node
    assert [(j.job_id, j.state, j.active) for j in (paused, killed)] == [("15", "COMPLETING", True), ("16", "COMPLETING", True)]
    # slurm 26.05.5's own output (`squeue --json` of a running job, saved from the container trial)
    (sample,) = gpu_jobs.parse_squeue((Path(__file__).parent / "fixtures" / "squeue_26.05.json").read_text())
    assert sample.sluid == "s4EXM5Q0DHSC00" and running.sluid == ""
    assert sample.cores == {"testnode:0:0"}  # core 0 ALLOCATED, 1-3 UNALLOCATED
    old = {"job_id": 1, "job_state": ["RUNNING"], "job_resources": {"nodes": "n1", "allocated_nodes": [
        {"nodename": "n1", "sockets": {"0": {"cores": {"2": "allocated", "3": "allocated"}}}}]}}
    assert gpu_jobs.parse_squeue(squeue_json(old))[0].cores == {"n1:0:2", "n1:0:3"}  # the 23.11 shape
    old = QueuedJob("1", "normal", "PENDING", "Priority", "n", "afterok:3(unfulfilled)")
    assert (old.user, old.priority, old.gpus, old.depends_on) == ("", 0, 0, ("3",))
    with pytest.raises(ValueError):
        gpu_jobs.parse_squeue("12|normal|RUNNING|None||ann|1|N/A|n")


def test_machine_use_reads_this_host_and_capacity_says_whether_this_host_is_the_node() -> None:
    proc = {"/proc/meminfo": "MemTotal: 1000 kB\nMemFree: 100 kB\nMemAvailable: 400 kB\n",
            "/proc/stat": iter(["cpu  10 0 10 70 10 0 0 0 0 0\n", "cpu  40 0 10 100 10 0 0 0 0 0\n"])}
    use = gpu_jobs.machine_use(lambda path: proc[path] if isinstance(proc[path], str) else next(proc[path]), lambda _s: None)
    cores = os.cpu_count()
    assert use == {"mem_total": 1000 * 1024, "mem_used": 600 * 1024, "cores": cores, "cores_busy": cores * 30 / 60}
    assert gpu_jobs.machine_use(lambda _path: "", lambda _s: None) == {}  # unreadable is unknown, never zero
    here = os.uname().nodename.split(".")[0]
    nodes = f"NodeName={here} CPUTot=20 CPUAlloc=12 RealMemory=100000 AllocMem=61440 CfgTRES=gres/gpu=1 AllocTRES=gres/gpu=1\n"
    one = gpu_jobs.capacity(lambda *_a, **_k: subprocess.CompletedProcess([], 0, nodes))
    two = gpu_jobs.capacity(lambda *_a, **_k: subprocess.CompletedProcess([], 0, nodes.replace(here, "elsewhere") + nodes))
    assert (one["nodes"], one["here"], one["mem_used_mb"], two["nodes"], two["here"]) == (1, 1, 61440, 2, 1)
    shared = nodes.replace("CfgTRES=gres/gpu=1 AllocTRES=gres/gpu=1", "CfgTRES=gres/gpu=1,gres/shard=12 AllocTRES=gres/shard=8")
    both_shared = gpu_jobs.capacity(lambda *_a, **_k: subprocess.CompletedProcess([], 0, shared))
    assert (both_shared["shards"], both_shared["shards_used"], both_shared["gpus_used"]) == (12, 8, 0)  # shard use is not GPU use
    lent = "NodeName=lap CPUTot=16 CPUEfctv=8 RealMemory=31494 MemSpecLimit=19494 AllocMem=0 CPUAlloc=0\n"
    both = gpu_jobs.capacity(lambda *_a, **_k: subprocess.CompletedProcess([], 0, nodes + lent))
    assert (both["cpus"], both["mem_mb"]) == (20 + 8, 100000 + 12000)  # a part-time machine counts only what it lends
    # for one partition only the nodes in it that can start a job count: the lent machine is down and in part_time (2026-11-03:
    # its 8 cores made urgent.py think 10 were free when the box had 2, so a cpu_fill job it would pre-empt went unlisted)
    up = nodes.rstrip("\n") + " State=MIXED+PLANNED Partitions=normal,cpu_fill,urgent\n"
    down = lent.rstrip("\n") + " State=DOWN+NOT_RESPONDING Partitions=part_time\n"
    draining = up.replace(here, "other").replace("MIXED+PLANNED", "IDLE+DRAIN")
    run = lambda text: (lambda *_a, **_k: subprocess.CompletedProcess([], 0, text))  # noqa: E731
    mine, lent_cpus = gpu_jobs.capacity(run(up))["cpus"], gpu_jobs.capacity(run(down))["cpus"]
    assert gpu_jobs.capacity(run(up + down))["cpus"] == mine + lent_cpus  # no partition: every node, as before
    assert gpu_jobs.capacity(run(up + down), partition="urgent")["cpus"] == mine  # down, and in another partition
    awake = down.replace("DOWN+NOT_RESPONDING", "IDLE").replace("part_time", "urgent")
    assert gpu_jobs.capacity(run(up + awake), partition="urgent")["cpus"] == mine + lent_cpus  # up and in it: counted
    assert gpu_jobs.capacity(run(up + draining), partition="urgent")["nodes"] == 1  # a draining node starts nothing new
    assert gpu_jobs.capacity(run(down), partition="urgent")["nodes"] == 0
    rows = gpu_jobs.node_capacity(run(up + down + draining))
    assert gpu_jobs.usable_capacity(rows)["nodes"] == 1  # any partition: the down and the draining node are no room at all
    assert gpu_jobs.usable_capacity(rows, "cpu_fill,part_time")["cpus"] == mine  # a job's partition list: part_time's node is down
    assert gpu_jobs.usable_capacity(rows, "part_time")["nodes"] == 0


def test_a_level_always_wins_then_the_earlier_job_and_every_placed_priority_is_above_slurms_own() -> None:
    order = [(Level.HIGH, "100"), (Level.NORMAL, "2"), (Level.NORMAL, "3"), (Level.LOW, "2")]
    priorities = [levels.priority_for(level, job_id, 2) for level, job_id in order]
    assert priorities == sorted(priorities, reverse=True) and len(set(priorities)) == len(priorities)
    # Above TOP_PRIORITY, the most priority/basic gives by itself (a fresh or released job: behind every placed one),
    # and below NO_VAL, which scontrol refuses; a job past the span shares its level's lowest priority.
    lowest, highest = levels.priority_for(Level.LOW, str(2 + 10 * levels.SPAN), 2), levels.priority_for(Level.HIGH, "2", 2)
    assert 0xFFFF0000 < lowest < highest < 0xFFFFFFFE
    far = str(2 + 10 * levels.SPAN)
    assert lowest == levels.priority_for(Level.LOW, str(1 + levels.SPAN), 2)  # past the span: the level's lowest
    assert levels.priority_for(Level.NORMAL, far, 2) > levels.priority_for(Level.LOW, "2", 2)  # still above the next level
    # the base moves only when it must: a lower live id, or one a whole span past it
    assert levels.rebase(None, ["7", "9"]) == 7 and levels.rebase(5, ["7", "9"]) == 5 and levels.rebase(5, []) == 5
    assert levels.rebase(5, ["4"]) == 4 and levels.rebase(5, ["9", str(5 + levels.SPAN)]) == 9


@pytest.mark.parametrize(
    ("gres", "count"),
    [("gres/gpu:2", 2), ("gpu:gb10:2", 2), ("gres/gpu:gb10:2", 2), ("gres/gpu", 1), ("N/A", 0), ("", 0), ("gres/gpu:1", 1),
     ("gres/shard:4", 0), ("shard:gb10:4", 0), ("gres/gpu:1,gres/shard:2", 1)],
)
def test_parse_gpus(gres: str, count: int) -> None:
    assert gpu_jobs.parse_gpus(gres) == count


@pytest.mark.parametrize(
    ("gres", "count"),
    [("gres/shard:4", 4), ("shard:gb10:4", 4), ("gres/shard", 1), ("gres/gpu:1,gres/shard:2", 2), ("gres/gpu:1", 0), ("N/A", 0)],
)
def test_parse_shards_and_a_shard_job_holds_the_card(gres: str, count: int) -> None:
    assert gpu_jobs.parse_shards(gres) == count
    job = QueuedJob("7", "normal", "PENDING", "None", "n", gpus=gpu_jobs.parse_gpus(gres), shards=count)
    assert job.uses_card == bool(gpu_jobs.parse_gpus(gres) or count)
    assert gpu_jobs.needs_the_card(job) == job.uses_card  # the lease keeper yields for a shard job too
    assert not gpu_jobs.needs_the_card(QueuedJob("8", "cpu", "RUNNING", "None", "n"))  # a CPU job: no demand
    # slurm 26.05.5's own text for three running `--gres=shard:2` jobs (recorded by the container's `sharing` stage)
    real = gpu_jobs.parse_squeue((Path(__file__).parent / "fixtures" / "squeue_shard.json").read_text())
    assert [(j.shards, j.gpus, gpu_jobs.needs_the_card(j)) for j in real] == [(2, 0, True)] * 3


def test_measured_sharing_is_this_users_newest_and_a_probe_never_sizes_a_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpu_jobs, "STATE_DIR", tmp_path)
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    leg = {"peak_gpu_mib": 1024, "peak_mem_bytes": 0, "ended_at": 1, "exit": 0, "signals": []}
    (jobs / "1.json").write_text(json.dumps({"name": "n", "legs": [leg], "sharing": {"level": 2, "measured_at": 10}}))
    (jobs / "2.json").write_text(json.dumps({"name": "n", "sharing": {"level": 4, "measured_at": 20}}))
    (jobs / "3.json").write_text(json.dumps({"name": "n", "sharing": {"level": 3, "measured_at": 15}}))
    assert gpu_jobs.measured_sharing("n") == {"level": 4, "measured_at": 20}
    assert gpu_jobs.measured_sharing("other") is None
    # a probe's peak is the total over its copies: it must never size a real job, however new
    (jobs / "4.json").write_text(json.dumps({"name": "n", "kind": gpu_jobs.SHARING_PROBE, "legs": [{**leg, "peak_gpu_mib": 9999, "ended_at": 99}]}))
    assert gpu_jobs.measured_peak_bytes("n") == 1024 * 1024**2
    monkeypatch.setattr(gpu_jobs, "current_user", lambda: "somebody-else")  # none of these files is theirs
    assert gpu_jobs.measured_sharing("n") is None


def test_plan_packing_takes_the_smallest_of_compute_memory_and_cores() -> None:
    gpu = site_config.Gpu(0, "GPU-a", "/dev/nvidia0", "gb10", shards=12)
    node = site_config.Node("n", 20, 100_000, (gpu,), cores_per_gpu=12)
    record = {"level": 4, "gpu_type": "gb10", "node_memory_mb": 100_000, "shards": 12}
    gb = 1000**3

    def peak_for(copies: int) -> int:  # a peak at which the node's memory fits exactly this many, margin included
        return int(100_000 * 1048576 / (copies * 1.1) * 0.999)

    def plan(**given):
        arguments = {"gpu": gpu, "node": node, "record": record, "peak_bytes": 2 * gb, "fraction": None, "cpus_asked": None,
                     "explicit_level": None, "margin": 0.1}
        return gpu_jobs.plan_packing(**{**arguments, **given})

    assert (plan().level, plan().shards, plan().cores) == (4, 3, 3)
    assert plan(peak_bytes=peak_for(5)).level == 4 and plan(peak_bytes=peak_for(3)).level == 3  # memory binds
    assert plan(fraction=0.5, peak_bytes=peak_for(5)).level == 2  # the site's fraction tightens memory
    assert plan(cpus_asked=5).level == 2  # cores: 12 // 5
    assert plan(cpus_asked=5).shards == 6  # ceil(12 / 2)
    assert plan(record={**record, "level": 5}).shards == 3 and plan(record={**record, "level": 5}).level == 5  # ceil(12/5) = 3
    assert plan(record={**record, "level": 5}).cores == 3
    assert "level 4 (measured 4; memory allows" in plan().line and "--gres=shard:3 of 12, 3 cores" in plan().line
    # no record, another card type, or an unknown peak: the whole card, today's request
    for whole in (plan(record=None), plan(record={**record, "gpu_type": "h100"}), plan(record={**record, "shards": 6}),
                  plan(peak_bytes=None), plan(gpu=site_config.Gpu(0, "GPU-a", "/dev/nvidia0", "gb10"))):
        assert (whole.level, whole.shards, whole.line) == (1, 0, "")
    # --max-jobs replaces the measured level only; memory still bounds it
    assert plan(explicit_level=6).level == 6 and plan(explicit_level=2).level == 2 and plan(record=None, explicit_level=3).level == 3
    assert plan(explicit_level=50, peak_bytes=peak_for(5)).level == 5
    # asked for sharing and not given it: the whole card, and the line says why
    refused = plan(explicit_level=3, peak_bytes=peak_for(1))
    assert (refused.level, refused.shards) == (1, 0) and refused.line.startswith("asked 3 copies, but memory allows 1")


def test_container_label_uses_the_scope_id_and_degrades_to_empty() -> None:
    cid = "ab" * 32
    calls: list[tuple[str, str]] = []

    def runner(container: str, label: str) -> str:
        calls.append((container, label))
        return "proj"

    assert gpu_jobs.container_label(f"/system.slice/docker-{cid}.scope", "gpu-lease-project", runner) == "proj"
    assert calls == [(cid, "gpu-lease-project")]
    assert gpu_jobs.container_label("/user.slice/session-3.scope", "l", runner) == ""

    def broken(container: str, label: str) -> str:
        raise FileNotFoundError

    assert gpu_jobs.container_label(f"/docker-{cid}.scope", "l", broken) == ""


def test_nodes_from_scontrol_one_line_records() -> None:
    def fake(out: str, code: int = 0):
        return lambda *a, **k: subprocess.CompletedProcess(a, code, out, "")

    text = "NodeName=spark-1 Arch=aarch64 CoresPerSocket=20\nNodeName=spark-2 Arch=x\n"
    assert gpu_jobs.nodes(fake(text)) == ["spark-1", "spark-2"]
    assert gpu_jobs.nodes(fake(text, 1)) == []

    def missing(*a, **k):
        raise FileNotFoundError

    assert gpu_jobs.nodes(missing) == []


def test_state_dir_resolution_env_over_site_file_and_no_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    site = tmp_path / "site.toml"
    site.write_text(
        f'name = "t"\nstate_dir = "{tmp_path / "site-state"}"\n'
        '[[nodes]]\nname = "n1"\ncpus = 4\nreal_memory_mb = 1024\n'
    )
    for var in ("SLURM_GPU_STATE_DIR", "GPU_LEASE_DIR"):
        monkeypatch.delenv(var, raising=False)
    try:
        monkeypatch.setattr(site_config, "SITE_FILE", tmp_path / "absent.toml")
        with pytest.raises(SystemExit, match=r"^\[gpujobs\] REFUSED: no site file at"):  # a message, not a traceback
            importlib.reload(gpu_jobs)
        monkeypatch.setenv("SLURM_GPU_STATE_DIR", str(tmp_path / "s"))
        monkeypatch.setenv("GPU_LEASE_DIR", str(tmp_path / "l"))
        with pytest.raises(SystemExit):  # both directories given still needs the site file
            importlib.reload(gpu_jobs)
        for var in ("SLURM_GPU_STATE_DIR", "GPU_LEASE_DIR"):
            monkeypatch.delenv(var)
        monkeypatch.setattr(site_config, "SITE_FILE", site)
        reloaded = importlib.reload(gpu_jobs)
        assert (reloaded.STATE_DIR, reloaded.LEASE_DIR) == (tmp_path / "site-state", None)  # no [lease]: no lease dir
        assert not reloaded.slurm_lease_admitted() and urgent.other_leases() == []  # and nothing reads one
        monkeypatch.setenv("GPU_LEASE_DIR", str(tmp_path / "env-lease"))
        assert importlib.reload(gpu_jobs).LEASE_DIR == tmp_path / "env-lease"
        monkeypatch.setenv("SLURM_GPU_STATE_DIR", str(tmp_path / "env-state"))
        assert importlib.reload(gpu_jobs).STATE_DIR == tmp_path / "env-state"
    finally:
        monkeypatch.undo()
        importlib.reload(gpu_jobs)


@pytest.mark.parametrize(("out", "busy"), [("40\n60\n", 0.5), ("40\n[N/A]\n", None), ("", None)])
def test_gpu_busy_is_unknown_unless_every_gpu_answers(out: str, busy: float | None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=out))
    assert gpu_jobs.gpu_busy() == busy  # an average over only the GPUs that answered would misstate the box


def part_time_copy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ran: dict[str, tuple[str, str]]) -> Path:
    """A state dir with pt1's copied-back files under `remote/pt1`, and a sacct that says where each job ran:
    `ran` maps a job id to (node, user). SlurmUser is the test user, whose files the copy "made"; ann and bob have uids
    of their own (`list_in_manifest` writes pt1's owner manifest)."""
    monkeypatch.setattr(gpu_jobs, "STATE_DIR", tmp_path)
    monkeypatch.setattr(gpu_jobs, "SLURM_USER", gpu_jobs.current_user())
    real = pwd.getpwnam
    monkeypatch.setattr(pwd, "getpwnam", lambda name: types.SimpleNamespace(pw_uid=UIDS[name]) if name in UIDS else real(name))
    jobs = [{"job_id": int(j), "nodes": node, "user": user} for j, (node, user) in ran.items()]
    monkeypatch.setattr(gpu_jobs, "_sacct_jobs", lambda *sel: jobs if "-D" in sel else None)
    remote = tmp_path / "remote" / "pt1"
    for sub in ("jobs", "events", "logs", "owners"):
        (remote / sub).mkdir(parents=True)
        (tmp_path / sub).mkdir(exist_ok=True)
    return remote


UIDS = {"ann": 1001, "bob": 1002}


def list_in_manifest(remote: Path, owners: dict[str, str]) -> None:
    """pt1's manifest as node_check.py writes it: `{"jobs/5.json": "ann"}` lists that file with ann's uid."""
    (remote / "owners" / "manifest.json").write_text(json.dumps({"files": {p: UIDS[u] for p, u in owners.items()}}))


def test_job_state_copied_back_from_a_part_time_machine_counts_only_for_jobs_that_ran_there(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    remote = part_time_copy(tmp_path, monkeypatch, {"5": ("pt1", "ann"), "6": ("dgx", "bob")})
    me = gpu_jobs.current_user()
    done = {"peak_gpu_mib": 0, "peak_mem_bytes": 7, "ended_at": 9, "exit": 0, "signals": []}
    (tmp_path / "jobs" / "5.json").write_text(json.dumps({"name": "n", "legs": [{"leg": 0, "ended_at": 1, "exit": 1}]}))
    (remote / "jobs" / "5.json").write_text(json.dumps({"legs": [{"leg": 1, "ended_at": 9, **done}], "last_leg": done}))
    (remote / "jobs" / "6.json").write_text(json.dumps({"legs": [done], "last_leg": done}))  # sacct: ran on dgx, not pt1
    (remote / "jobs" / "7.json").write_text(json.dumps({"legs": [done]}))  # sacct knows no job 7
    (remote / "events" / "5-100-push_back.json").write_text(json.dumps({"job": "5", "action": "push_back", "free_mb": 1}))
    (remote / "events" / "6-100-push_back.json").write_text("{}")
    (remote / "logs" / "n-1.log").write_text("from pt1")
    list_in_manifest(remote, {"jobs/5.json": "ann", "jobs/6.json": "bob", "jobs/7.json": "ann", "events/5-100-push_back.json": "ann",
                              "events/6-100-push_back.json": "bob", "logs/n-1.log": "ann"})
    # the sidecar: the node's leg joins the local one; for a job accounting did not put there, nothing joins
    assert [leg["leg"] for leg in gpu_jobs.read_sidecar("5")["legs"]] == [0, 1]
    assert [leg["leg"] for leg in gpu_jobs.read_sidecar("5", me)["legs"]] == [0]  # accounting says it is ann's job
    assert gpu_jobs.read_sidecar("6") is None and gpu_jobs.read_sidecar("7") is None
    # events are attributed to accounting's user (copied files lose their owner), and only for jobs that ran there
    found = gpu_jobs.events()
    assert [(e.job, e.actor, e.fields) for e in found] == [("5", "ann", {"free_mb": 1})] and gpu_jobs.events("6") == []
    # the log only for the job that ran there, and a symlink is never followed
    assert gpu_jobs.remote_log("5", "/anywhere/n-1.log") == remote / "logs" / "n-1.log"
    assert gpu_jobs.remote_log("6", "/anywhere/n-1.log") is None and gpu_jobs.remote_log("5", "/x/n-1.log", "bob") is None
    # the manifest is the proof of ownership: a file it omits, or lists under another uid (a user on the node made
    # it), and a file that is not SlurmUser's copy, never count, though accounting puts the job there
    list_in_manifest(remote, {"jobs/5.json": "bob", "logs/n-1.log": "bob"})
    assert [leg["leg"] for leg in gpu_jobs.read_sidecar("5")["legs"]] == [0] and gpu_jobs.events() == []
    assert gpu_jobs.remote_log("5", "/anywhere/n-1.log") is None
    list_in_manifest(remote, {"jobs/5.json": "ann", "logs/n-1.log": "ann", "events/5-100-push_back.json": "ann"})
    assert gpu_jobs.remote_log("5", "/anywhere/n-1.log") == remote / "logs" / "n-1.log"
    with monkeypatch.context() as other:
        other.setattr(gpu_jobs, "_copier_uid", lambda: os.getuid() + 1)
        assert gpu_jobs.read_sidecar("5") == {"name": "n", "legs": [{"leg": 0, "ended_at": 1, "exit": 1}]}  # the local record only
        assert gpu_jobs.events() == [] and gpu_jobs.remote_log("5", "/anywhere/n-1.log") is None
    (remote / "logs" / "n-1.log").unlink()
    (remote / "logs" / "n-1.log").symlink_to("/etc/hostname")
    assert gpu_jobs.remote_log("5", "/anywhere/n-1.log") is None
    # the footprint for --mem auto: a clean leg a part-time machine ran counts, for the user accounting names
    assert gpu_jobs.measured_peak_bytes("n") is None  # accounting says ann, not me: pt1's leg does not count for me
    monkeypatch.setattr(gpu_jobs, "_ran_on", lambda ids: {"5": {"pt1": me}})
    assert gpu_jobs.measured_peak_bytes("n") is None  # ... and the manifest must list it with my uid too
    (remote / "owners" / "manifest.json").write_text(json.dumps({"files": {"jobs/5.json": os.getuid()}}))
    assert gpu_jobs.measured_peak_bytes("n") == 7


def test_with_nothing_copied_back_the_readers_never_ask_sacct(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpu_jobs, "STATE_DIR", tmp_path)
    monkeypatch.setattr(gpu_jobs, "_sacct_jobs", lambda *sel: pytest.fail("sacct asked"))
    (tmp_path / "jobs").mkdir()
    (tmp_path / "jobs" / "5.json").write_text(json.dumps({"name": "n"}))
    assert gpu_jobs.read_sidecar("5") == {"name": "n"} and gpu_jobs.events() == []
    assert gpu_jobs.state_dirs("jobs") == [tmp_path / "jobs"]


@pytest.mark.parametrize(("label", "gpus", "service", "no_anywhere"),
                         list(itertools.product(("resume", "restart"), (0, 1), (False, True), (False, True))))
def test_only_an_eligible_job_is_offered_the_part_time_machines(label, gpus, service, no_anywhere) -> None:
    job_class = gpu_jobs.JobClass(label)  # by label: an earlier test reloads gpu_jobs, which makes new enum members
    eligible = label == "resume" and not gpus and not service and not no_anywhere
    allowed = gpu_jobs.part_time_eligible(job_class, gpus, service=service, no_anywhere=no_anywhere)
    asked = gpu_jobs.partition_for(job_class, gpus, service=service, part_time=allowed)
    assert allowed is eligible and (asked == "cpu_fill,part_time") is eligible
    # whatever the caller claims, an ineligible job (restart, GPU, service) never gets them, and urgent is not a choice here
    forced = gpu_jobs.partition_for(job_class, gpus, service=service, part_time=True)
    assert ("part_time" in forced) is (label == "resume" and not gpus and not service)
    assert "urgent" not in forced


def test_a_job_listing_two_partitions_reads_right_and_one_on_a_part_time_machine_is_not_in_urgent_works_way() -> None:
    waiting = QueuedJob("1", "part_time,cpu_fill", "PENDING", "Resources", "p:resume:n")  # squeue's order differs from submission
    assert "part-time machine" in status._pauses(waiting) and status._pauses(waiting) != waiting.partition
    assert status._pauses(QueuedJob("2", "cpu_fill", "RUNNING", "None", "p:resume:n")).startswith("fills idle cores;")
    running = QueuedJob("3", "part_time", "RUNNING", "None", "p:resume:n", node="laptop")
    lines, victims, steady, wait_only = urgent.plan(
        [running], gpu=True, whole_box=True, leases=[], unregistered=[], services={}, memory_short=True, cores_short=True
    )
    assert (victims, steady, wait_only) == ([], [], [])  # it holds none of this machine's cores, memory or card


def test_meminfo_gives_bytes_and_refuses_a_file_without_memavailable(tmp_path: Path) -> None:
    path = tmp_path / "meminfo"
    path.write_text("MemTotal: 1000 kB\nMemAvailable: 400 kB\nHugePages_Total: 0\nShmem: 7 kB\n")
    assert gpu_jobs.meminfo(path) == {"MemTotal": 1000 * 1024, "MemAvailable": 400 * 1024, "Shmem": 7 * 1024}
    with pytest.raises(ValueError):
        gpu_jobs.meminfo_text("MemTotal: 1000 kB\n")


def test_held_bytes_drops_only_cache_the_host_can_take_back(make_cgroup: Callable[..., Path]) -> None:
    cgroup = make_cgroup()
    assert gpu_jobs.held_bytes(cgroup) == 3_800_000_000
    assert gpu_jobs.held_bytes(make_cgroup("x", current=1, cache=9, shmem=0)) == 0  # cache above current: floored
    (cgroup / "memory.stat").unlink()
    assert gpu_jobs.held_bytes(cgroup) is None  # a missing file is unknown, never zero


def test_job_memory_measure_takes_unlimited_as_the_jobs_share_and_trusts_no_label(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_cgroup: Callable[..., Path]) -> None:
    monkeypatch.setattr(gpu_jobs, "container_label", lambda *_a, **_k: "5")  # every foreign container claims job 5
    job = QueuedJob("5", "gpu", "RUNNING", "", "n", user="ann", start=77)
    usage = {100: 3.0, 999: 50.0}  # pid 999 is some other job's process; pid 101 is not on the card
    memory = gpu_jobs.JobMemory.measure(job, make_cgroup(), usage, 8_000_000_000)
    assert memory == gpu_jobs.JobMemory("5", "ann", "n", 77, 3_800_000_000, 3 * gpu_jobs.SLURM_MB, 8_000_000_000)
    assert memory.total == memory.held + memory.card and memory.excess == memory.total - 8_000_000_000
    limited = gpu_jobs.JobMemory.measure(job, make_cgroup("b", limit="123"), usage, 8_000_000_000)
    assert limited is not None and limited.ask == 123
    assert gpu_jobs.JobMemory.measure(job, tmp_path / "gone", usage, 1) is None


def test_queued_start_is_the_start_time_of_a_running_job_only() -> None:
    started = {"set": True, "infinite": False, "number": 1700}
    running, pending = gpu_jobs.parse_squeue(squeue_json(
        {"job_id": 1, "job_state": ["RUNNING"], "start_time": started}, {"job_id": 2, "start_time": started}))
    assert (running.start, pending.start) == (1700, 0)  # a waiting job's start_time is only an estimate


def test_agent_notices_count_only_files_the_agent_user_wrote(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpu_jobs, "STATE_DIR", tmp_path)
    gpu_jobs.write_event("7", EventAction.MEMORY_STOP, message="stopped")
    gpu_jobs.write_event("7", EventAction.PAUSE, message="not a notice")
    kinds = frozenset({EventAction.MEMORY_STOP})
    monkeypatch.setattr(gpu_jobs, "SLURM_USER", "somebody-else")  # the file's owner is not the agent's user
    assert gpu_jobs.agent_notices("7", kinds) == []
    monkeypatch.setattr(gpu_jobs, "SLURM_USER", gpu_jobs.current_user())
    assert [e.action for e in gpu_jobs.agent_notices("7", kinds)] == ["memory_stop"]
    assert gpu_jobs.idle_notices("7") == []


def test_memory_kill_reads_the_stop_gaps_event_and_ignores_a_forged_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpu_jobs, "STATE_DIR", tmp_path)
    monkeypatch.setattr(gpu_jobs, "_sacct_jobs", lambda *_a, **_k: [])
    record = {"name": "n", "job_id": "5", "mem": "8G", "legs": [{"exit": 143, "signals": ["SIGTERM"], "ended_at": 2_000}]}
    monkeypatch.setattr(gpu_jobs, "own_records", lambda _name: [record])
    gpu_jobs.write_event("5", EventAction.MEMORY_STOP, message="m", total=3 * gpu_jobs.SLURM_MB + 1)
    monkeypatch.setattr(gpu_jobs, "SLURM_USER", gpu_jobs.current_user())
    kill = gpu_jobs.memory_kill("n")
    assert kill is not None and (kill.used_mb, kill.by, kill.asked_mb) == (4, "the memory stop-gap", 8192)
    monkeypatch.setattr(gpu_jobs, "SLURM_USER", "somebody-else")  # the job's own user wrote it: a forgery
    assert gpu_jobs.memory_kill("n") is None


def test_container_label_never_waits_on_a_hung_docker(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def run(*_a: object, **kwargs: object) -> None:
        seen.update(kwargs)
        raise subprocess.TimeoutExpired("docker", 5)

    monkeypatch.setattr(subprocess, "run", run)
    assert gpu_jobs.container_label(f"/docker-{'ab' * 32}.scope", "l") == "" and seen["timeout"] == gpu_jobs.DOCKER_TIMEOUT_S
