"""The agent's decisions, against a fake queue, a fake scontrol, a fake clock and a real state directory."""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import agent as agent_module
import gpu_jobs
import levels
import pytest
import site_config


def job(
    job_id: str, state: str = "RUNNING", reason: str = "None", priority: int | None = None,
    level: gpu_jobs.Level = gpu_jobs.Level.NORMAL,
) -> gpu_jobs.QueuedJob:
    """A job; by default already at the place its level gives it (the World's agent counts from base 1)."""
    placed = levels.priority_for(level, job_id, 1) if job_id.isdigit() else 1
    return gpu_jobs.QueuedJob(
        job_id, "normal", state, reason, f"p:resume:{job_id}", priority=placed if priority is None else priority,
        level=level,
    )


def done(
    argv: list[str], code: int = 0, out: str = "", err: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(argv, code, out, err)


class World:
    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(gpu_jobs, "STATE_DIR", tmp_path)
        self.admins: set[str] = set()
        monkeypatch.setattr(
            site_config, "is_admin", lambda _site, user: user in self.admins
        )
        self.dir = tmp_path
        self.queue: list[gpu_jobs.QueuedJob] | None = []
        self.t, self.boot, self.ctld_s = 1000.0, "boot-1", 60.0
        self.has_gpus = True
        self.lap: dict[str, object] | None = None  # the part-time node `lap` as `scontrol show node --json` gives it; None: no such node
        self.shared_state = False
        self.copies: list[list[str]] = []  # every rsync the agent ran, and what it answers: (exit status, stderr)
        self.copy_answer: tuple[int, str] = (0, "")
        self.running = ""  # squeue's "%i|%j|%P" lines for the jobs running on it
        self.nodes = {"n1": "IDLE", "n2": "IDLE+DRAIN"}
        self.writes: list[list[str]] = []  # every scontrol update / release, failed ones included
        self.totals = ""  # CPUTot=.. fields of node n1 in `scontrol show nodes -o`, for the idle check
        self.fail: set[str] = set()  # scontrol subcommands that answer with an error
        self.records: dict[str, dict[str, str]] = {}
        self.notify_command: tuple[str, ...] = ()
        self.metrics_mode = "off"
        self.smi: dict[int, float] | None = {}  # nvidia-smi's GPU MiB per pid; None when it does not answer
        self.cgroups = tmp_path / "cgroup"
        self.live: dict[int, str] = {}  # pid -> job of the processes that still exist when the OOM guard names them
        (tmp_path / "agent").mkdir()
        (tmp_path / "agent" / "boot_id").write_text(
            "boot-1\n"
        )  # this boot is already reconciled
        gpu_jobs.write_json(tmp_path / "agent" / "state.json", {"base": 1})  # placed priorities count from job 1
        self.agent = self.new_agent()

    def scontrol(
        self, argv: list[str], **_kw: object
    ) -> subprocess.CompletedProcess[str]:
        match argv[1:3]:
            case ["show", "node"] if "--json" in argv:
                return done(argv, out=json.dumps({"nodes": [self.lap]}))
            case ["-h", "-t"]:  # squeue -h -t R -w NODE
                return done(argv, out=self.running)
            case ["show", "nodes"]:
                return done(
                    argv, out="".join(f"NodeName={n} Arch=x {self.totals if n == 'n1' else ''}\n" for n in self.nodes)
                )
            case ["show", "node"]:
                return done(
                    argv,
                    out=f"NodeName={argv[3]} State={self.nodes[argv[3]]} Reason=x\n",
                )
        self.writes.append(argv if argv[0] == "scancel" else argv[1:])
        return done(argv, int(argv[1] in self.fail))

    def new_agent(self) -> agent_module.Agent:
        gpus = [{"index": 0, "uuid": "GPU-x", "device": "/dev/nvidia0", "type": "t"}] if self.has_gpus else []
        nodes = [{"name": "n1", "cpus": 4, "real_memory_mb": 1000, "gpus": gpus}]
        site = site_config.parse(
            {"name": "t", "nodes": nodes + ([{"name": "lap", "cpus": 8, "real_memory_mb": 30000, "part_time": True,
                                               "address": "10.0.0.2"}] if self.lap else []),
             "notify_command": list(self.notify_command), "metrics": {"mode": self.metrics_mode},
             "shared_state": self.shared_state}
        )
        return agent_module.Agent(
            site,
            self.dir,
            read_queue=lambda: self.queue,
            scontrol=self.scontrol,
            ctld_active_s=lambda: self.ctld_s,
            clock=lambda: self.t,
            boot_id=lambda: self.boot,
            completion=lambda: self.records,
            gpu_usage=lambda: self.smi,
            cgroup_root=self.cgroups,
            pid_job=lambda pid: self.live.get(pid),
            part_time_conf=self.dir / "part_time_nodes.conf",
            copier=lambda argv, **_kw: self.copies.append(argv) or done(argv, self.copy_answer[0], err=self.copy_answer[1]),
        )

    def step(self) -> dict[str, object]:
        return self.agent.step()

    def part_time(self, name: str, **request: object) -> Path:
        path = self.dir / "requests" / f"part_time-lap-{name}.json"
        gpu_jobs.write_json(path, request)
        return path

    def request(self, urgent: str, victims: list[str], held: list[str]) -> Path:
        path = self.dir / "requests" / "restore_order-1.json"
        gpu_jobs.write_json(path, {"urgent": urgent, "victims": victims, "held": held})
        return path


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> World:
    return World(tmp_path, monkeypatch)


def test_a_fresh_install_notices_its_first_failure_and_an_unanswered_sacct_is_not_no_jobs(world: World) -> None:
    world.records = {}  # first look on a fresh install: nothing has ended yet
    world.step()
    world.records = None  # sacct down: not "no jobs", so nothing is settled
    world.t += agent_module.NOTICES_EVERY_S
    assert "sacct did not answer" in str(world.step())
    world.records = {"5": {"JobId": "5", "JobState": "FAILED", "Name": "n", "UserId": "ann"}}
    world.t += agent_module.NOTICES_EVERY_S
    world.step()
    assert [(e.job, e.action) for e in gpu_jobs.events()] == [("5", "failed")]


def test_the_agent_writes_the_metrics_of_running_jobs_with_the_queues_user_and_removes_the_rest(world: World) -> None:
    scope = world.cgroups / "system.slice" / "n1_slurmstepd.scope"
    (scope / "job_5" / "step_0").mkdir(parents=True)
    (scope / "job_5" / "step_0" / "cgroup.procs").write_text("10\n")
    (scope / "job_5" / "memory.current").write_text("4096\n")
    sluid = world.cgroups / "system.slice" / "slurmstepd.scope" / "s8VSM5Q2WYAV00"  # 26.05: named by the job's SLUID
    sluid.mkdir(parents=True)
    (sluid / "memory.current").write_text("2048\n")
    (scope / "job_7").mkdir()  # job 7 is being torn down: its cgroup lingers, its series should not
    world.metrics_mode, world.smi = "local", {10: 300.0, 20: 50.0}
    world.agent = world.new_agent()
    metrics = world.dir / "metrics"
    metrics.mkdir()
    for planted in ("job_5.prom", "job_9.prom", "job_8.prom.tmp"):  # another user's forgery, a dead job's file, a stale temp
        (metrics / planted).write_text("forged\n")
    running = dataclasses.replace(job("5"), user="ann", name="proj:train")
    world.queue = [running, job("7", "COMPLETING"), dataclasses.replace(job("6"), user="bob"),
                   dataclasses.replace(job("8"), sluid="s8VSM5Q2WYAV00")]
    world.step()
    assert sorted(path.name for path in metrics.iterdir()) == ["job_5.prom", "job_8.prom"]  # job 6 runs, but on another node (no cgroup here), 7 is ending, the rest is gone
    text = (metrics / "job_5.prom").read_text()
    assert 'jobid="5",user="ann",project="proj"' in text  # the queue's user, never the writer's
    assert f"}} {300 * agent_module.MIB}\n" in text and "} 4096\n" in text  # pid 20 is nobody's here
    assert "} 2048\n" in (metrics / "job_8.prom").read_text()
    world.queue, world.smi = [], None  # job 5 ended, but nvidia-smi is silent: unknown is not zero
    assert "nvidia-smi did not answer" in str(world.step())
    assert (metrics / "job_5.prom").exists()
    world.smi = {}
    world.step()
    assert not list(metrics.iterdir())
    world.queue, world.has_gpus, world.smi = [running], False, None  # a site without GPUs has no nvidia-smi: CPU memory only, no error
    world.agent = world.new_agent()
    world.step()  # without GPUs nothing waits for nvidia-smi, so the file is written
    assert "} 0\n" in (metrics / "job_5.prom").read_text() and "} 4096\n" in (metrics / "job_5.prom").read_text()


def kill(pid: int) -> str:
    return json.dumps({"MESSAGE": f'sending SIGTERM to process {pid} uid 1000 "python": badness 1508, VmRSS 695 MiB',
                       "__REALTIME_TIMESTAMP": "1792500000000000"})


def test_oom_guard_kills_are_recorded_for_the_job_that_lost_the_process_and_dropped_once_it_leaves(world: World) -> None:
    # earlyoom's own line, measured in the journal 2026-10-22 13:45:55 (job 428); anything else is no kill
    assert agent_module.kill_from_entry(kill(42)) == (42, "SIGTERM", 1792500000.0)
    assert agent_module.kill_from_entry('{"MESSAGE": "process exited after 2.6 seconds"}') is None
    assert agent_module.kill_from_entry(json.dumps({"MESSAGE": list(b'sending SIGKILL to process 7 uid 0 "\xff"')})) == (
        7, "SIGKILL", pytest.approx(agent_module.time.time(), abs=60))  # non-UTF-8 text comes as bytes
    scope = world.cgroups / "system.slice" / "n1_slurmstepd.scope" / "job_5"
    scope.mkdir(parents=True)
    (scope / "cgroup.procs").write_text("10\n11\n")
    world.queue = [job("5"), job("6", "PENDING")]
    world.step()  # learns job 5's processes
    world.live = {12: "6"}
    watch = agent_module.OomWatch(world.agent.record_kill)
    lines = [kill(12) + "\n", kill(11) + "\n", kill(99) + "\n",
             "No journal files were opened due to insufficient permissions.\n"]
    watch.follow(lambda *_a, **_k: subprocess.Popen(["printf", "%s", "".join(lines)], stdout=subprocess.PIPE, text=True))
    kills = world.dir / "agent" / "oom_kills"
    # pid 12 named while alive; pid 11 gone already, found in the last step's processes; pid 99 belongs to no job
    assert sorted(p.name for p in kills.iterdir()) == ["job_5.json", "job_6.json"]
    assert json.loads((kills / "job_5.json").read_text())["pid"] == 11
    world.agent.watch = watch
    assert "insufficient permissions" in str(world.step())  # a follower that cannot read the journal says so
    world.queue = [job("6", "PENDING")]  # job 5 has left the queue: its wrapper has read its record already
    world.step()
    assert sorted(p.name for p in kills.iterdir()) == ["job_6.json"]


def test_slurmctld_uptime_probe_reads_the_monotonic_timestamp() -> None:
    out = "ActiveState=active\nActiveEnterTimestampMonotonic=23287769\n"
    assert agent_module.slurmctld_active_s(
        lambda argv, **_kw: done(argv, out=out), lambda: 100.0
    ) == pytest.approx(76.712231)
    assert (
        agent_module.slurmctld_active_s(
            lambda argv, **_kw: done(argv, out="ActiveState=inactive\n")
        )
        is None
    )


def test_nodes_are_undrained_once_per_boot_after_slurmctld_settles_and_drained_ones_are_skipped(
    world: World,
) -> None:
    world.lap = {"name": "lap", "state": ["IDLE"], "reason": "", "extra": ""}
    world.agent = world.new_agent()
    world.nodes["lap"] = "DOWN*"  # a part-time node switched off: left to its health check, it does not stall the boot
    world.boot, world.ctld_s = "boot-2", 10.0
    world.step()
    assert (
        world.writes == []
        and (world.dir / "agent" / "boot_id").read_text() == "boot-1\n"
    )  # waits for slurmctld
    world.ctld_s, world.fail = 60.0, {"update"}
    world.step()  # the undrain fails: the boot is not marked reconciled, so the next look tries again
    assert (world.dir / "agent" / "boot_id").read_text() == "boot-1\n"
    world.fail = set()
    world.writes.clear()
    beat = world.step()
    assert world.writes == [
        ["update", "nodename=n1", "state=undrain"]
    ]  # n2 is drained on purpose
    assert beat["boot_reconciled_at"] == int(world.t)
    world.step()
    world.agent = world.new_agent()  # a restart in the same boot
    world.step()
    assert len(world.writes) == 1
    world.boot = "boot-3"
    world.step()
    assert (
        world.writes == [["update", "nodename=n1", "state=undrain"]] * 2
    )  # once more for the new boot


def test_waiting_jobs_get_the_place_their_level_and_id_give_them_and_nothing_else_is_touched(world: World) -> None:
    high = levels.priority_for(gpu_jobs.Level.HIGH, "5", 1)
    world.queue = [
        job("5", "PENDING", "Priority", 4294901630, gpu_jobs.Level.HIGH),  # just submitted: slurm's own number
        job("6", "PENDING", "Priority", 50),  # an admin's hand-set priority: overwritten (D18)
        job("7", "PENDING", "JobHeldUser", 0),  # held: slurm refuses a priority, and it waits for its owner anyway
        job("8", "RUNNING", priority=3),  # running: its priority no longer matters
        job("9", "PENDING", "Priority"),  # already in place
    ]
    world.fail = {"update"}
    assert "placing job 5 failed; will retry" in str(world.step()["last_error"])
    world.fail = set()
    world.writes.clear()
    world.agent = world.new_agent()  # nothing to remember: the place comes from the level and id alone
    world.step()
    assert world.writes == [["update", "jobid=5", f"priority={high}"],
                            ["update", "jobid=6", f"priority={levels.priority_for(gpu_jobs.Level.NORMAL, '6', 1)}"]]
    # a fresh agent, with no base saved, counts from the lowest live job: jobs far past the span keep their order
    (world.dir / "agent" / "state.json").unlink()
    world.agent = world.new_agent()
    world.queue = [job("40000", "PENDING", "Priority", 5), job("40001", "PENDING", "Priority", 5)]
    world.writes.clear()
    world.step()
    first, second = (int(w[2].split("=")[1]) for w in world.writes)
    assert first > second and agent_module.gpu_jobs.read_json(world.dir / "agent" / "state.json")["base"] == 40000


def test_a_silent_squeue_keeps_the_state_and_acts_on_nothing(world: World) -> None:
    world.queue = [job("5", "PENDING", "JobHeldUser", 0)]
    world.step()
    saved = (world.dir / "agent" / "state.json").read_text()
    world.queue = None
    world.request("9", ["1"], ["7"])
    world.admins = {gpu_jobs._owner(world.dir / "requests" / "restore_order-1.json")}
    beat = world.step()
    assert "squeue did not answer" in str(beat["last_error"])
    assert (world.dir / "agent" / "state.json").read_text() == saved
    assert [w for w in world.writes if w[0] == "release"] == []


def test_restore_order_ignores_a_non_admin_then_releases_for_an_admin_once_the_urgent_job_ended_and_victims_run(
    world: World,
) -> None:
    # Victim 1 is pre-empted; victim 2 is one slurm did not need and never stops; urgent job 9.
    path = world.request("9", ["1", "2"], ["7", "-x", "7 8"])  # only a plain job id reaches scontrol
    world.admins = set()
    world.queue = [job("1"), job("2")]
    world.step()  # not an admin's request: ignored whatever the queue says
    assert world.writes == [] and path.exists()
    world.admins = {gpu_jobs._owner(path)}
    world.queue = [job("9", "PENDING", "Resources"), job("1"), job("2")]
    world.step()  # every victim still runs: the pre-emption has not happened yet
    assert world.writes == []
    world.queue = [job("9"), job("1", "PENDING", "BeginTime"), job("2")]
    world.step()  # the urgent job runs, the victim is requeued
    world.queue = [job("1", "PENDING", "BeginTime"), job("2")]
    world.step()  # the urgent job ended, the victim has not started again yet
    assert world.writes == []
    world.queue = [job("1"), job("2"), job("7", "PENDING", "JobHeldAdmin", 0)]
    world.fail = {"release"}
    world.step()  # the release fails: the request stays, to be tried again
    assert path.exists()
    world.fail = set()
    world.writes.clear()
    world.step()  # victim 2 never stopped, and still does not keep job 7 held
    placed = ["update", "jobid=7", f"priority={levels.priority_for(gpu_jobs.Level.NORMAL, '7', 1)}"]
    assert world.writes == [["release", "7"], placed] and not path.exists()
    path = world.request("10", ["3"], ["8", "11"])
    world.queue = [job("3", "PENDING", "JobHeldAdmin", 0), job("8", "PENDING", "JobHeldUser", 0, gpu_jobs.Level.LOW),
                   job("11", "PENDING", "JobHeldUser", 0, gpu_jobs.Level.HIGH)]
    world.writes.clear()
    world.step()  # a held victim will not start by itself: it must not keep jobs 8 and 11 held forever
    # one at a time, best place first (with the card free, the first released takes it), each placed at once
    assert world.writes == [["release", "11"], ["update", "jobid=11", f"priority={levels.priority_for(gpu_jobs.Level.HIGH, '11', 1)}"],
                            ["release", "8"], ["update", "jobid=8", f"priority={levels.priority_for(gpu_jobs.Level.LOW, '8', 1)}"]]
    gpu_jobs.write_json(path, {"victims": ["3"], "held": ["8", "11"]})
    world.writes.clear()
    world.step()  # a request naming no urgent job is ignored
    assert world.writes == []


def test_restore_order_releases_once_when_the_victim_left_and_the_file_cannot_be_deleted(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = world.request("9", ["1"], ["7", "8"])
    world.admins = {gpu_jobs._owner(path)}
    monkeypatch.setattr(
        gpu_jobs, "remove_quietly", lambda _path: None
    )  # SlurmUser owns none of the request files
    world.queue = [job("7", "PENDING", "JobHeldAdmin", priority=0)]  # 8 has left the queue: slurm would refuse it
    world.step()
    world.step()
    assert [w for w in world.writes if w[0] == "release"] == [["release", "7"]]  # once, and done (no retry for ever)


def test_restore_order_waits_for_a_queue_read_after_the_request_was_written(world: World) -> None:
    # urgent.py submits, then writes its request; a queue read from before the submission lacks the urgent job, and
    # judged against it the request looks finished (measured in the e2e, 2026-10-20: the held job was released at once).
    paths: list[Path] = []

    def read_queue() -> list:
        paths.append(world.request("9", ["1"], ["7"]))  # written while this step is between its read and its look
        world.admins = {gpu_jobs._owner(paths[0])}
        return [job("1")]

    world.agent.read_queue = read_queue
    world.step()
    assert world.writes == [] and paths[0].exists()  # judged on the next step, against a queue that has job 9


def test_failed_jobs_raise_an_event_and_run_the_notify_command_once(
    world: World, tmp_path: Path
) -> None:
    sink = tmp_path / "notified"
    script = "import os,sys; open(sys.argv[1],'a').write(' '.join(os.environ[k] for k in sys.argv[2:])+'\\n')"
    world.notify_command = (
        sys.executable,
        "-c",
        script,
        str(sink),
        "GPUJOBS_JOB",
        "GPUJOBS_STATE",
        "GPUJOBS_USER",
        "GPUJOBS_NAME",
    )
    world.agent = world.new_agent()
    world.agent.notify = agent_module.notify_run
    world.records = {"3": {"JobId": "3", "JobState": "FAILED", "Name": "old"}}
    world.step()  # the log's history is not news
    world.records |= {
        "9": {
            "JobId": "9",
            "JobState": "OUT_OF_MEMORY",
            "Name": "p:resume:train\x1b[2J",
            "UserId": "user1(1000)",
        },
        "10": {
            "JobId": "10",
            "JobState": "COMPLETED",
            "Name": "ok",
            "UserId": "user1(1000)",
        },
        "11": {
            "JobId": "11",
            "JobState": "PENDING",
            "Name": "requeued",
            "UserId": "user1(1000)",
        },
    }
    world.step()
    assert gpu_jobs.events() == []  # accounting is asked at most every NOTICES_EVERY_S, not every loop
    world.t += agent_module.NOTICES_EVERY_S
    world.step()
    world.step()
    assert [(e.job, e.action, e.fields) for e in gpu_jobs.events()] == [
        ("9", "failed", {"state": "OUT_OF_MEMORY", "name": "p:resume:train\x1b[2J"})
    ]
    assert sink.read_text() == "9 OUT_OF_MEMORY user1 p:resume:train?[2J\n"  # the notify command gets it shown, not live
    assert json.loads((world.dir / "agent" / "state.json").read_text())[
        "ended_seen"
    ] == ["3", "9", "10"]


def q(job_id: str, partition: str, state: str = "RUNNING", cpus: int = 2, mem: int | None = 100, gpus: int = 0,
      reason: str = "None", cores: str = "") -> gpu_jobs.QueuedJob:
    """A job; `cores` like "0-1" names the cores slurm gave it (by default, two of its own past every other job's)."""
    if not cores and state in {"RUNNING", "SUSPENDED"}:
        cores = f"{10 * int(job_id)}-{10 * int(job_id) + cpus - 1}"
    first, _, last = cores.partition("-")
    given = frozenset(f"n:0:{c}" for c in range(int(first), int(last or first) + 1)) if cores else frozenset()
    return gpu_jobs.QueuedJob(job_id, partition, state, reason, f"p:resume:{job_id}", user=gpu_jobs.current_user(),
                              priority=1, gpus=gpus, cpus=cpus, mem_mb=mem, cores=given)


def plan(queue: list, suspended: set, cpus: int = 8, cap: int = 8, consent: dict | None = None,
         given_up: frozenset = frozenset()) -> tuple:
    room = gpu_jobs.plan_room(queue, suspended, cpus, 1000, cap, consent, given_up)
    return list(room.suspend), list(room.resume)


def test_room_for_a_service_job_comes_from_the_newest_cpu_jobs_and_is_given_back_only_on_free_cores() -> None:
    full = [q("1", "normal", cpus=4, gpus=1), q("2", "cpu"), q("3", "cpu")]  # 8 cores, all taken
    waiting = q("9", "service", "PENDING", reason="Resources")
    assert plan([*full, waiting], set()) == (["3"], [])  # the newest cpu job, enough for its 2 cores
    assert plan([*full, q("9", "service", "PENDING", cpus=6, reason="Resources")], set()) == ([], [])  # cannot cover 6
    assert plan([*full, q("9", "service", "PENDING", mem=950, reason="Resources")], set()) == ([], [])  # no memory
    for reason in ("Dependency", "ReqNodeNotAvail", "PartitionConfig", "QOSMaxCpuPerUserLimit"):  # not cores
        assert plan([*full, q("9", "service", "PENDING", reason=reason)], set()) == ([], []), reason
    assert plan([*full, waiting], set(), given_up=frozenset({"9"})) == ([], [])
    assert plan([*full, q("4", "cpu", mem=None), waiting], set(), cpus=10) == ([], [])  # free memory unknown
    # the owner's consent (D9): only the jobs named, or none at all; and only within the service partition's cores
    assert plan([*full, waiting], set(), consent={"9": frozenset({"2"})}) == (["2"], [])
    assert plan([*full, waiting], set(), consent={"9": frozenset()}) == ([], [])
    assert plan([*full, q("8", "service"), waiting], set(), cpus=10, cap=3) == ([], [])  # 8 holds 2 of its 3
    # resumed only on its own cores (slurm puts it back there unchecked): case A of the review, the service job on them
    frozen = [full[0], full[1], q("3", "cpu", "SUSPENDED", cores="30-31")]
    assert plan([full[1], frozen[2], q("9", "service", cores="30-31")], {"3"}) == ([], [])  # the GPU job ended: still
    assert plan([*frozen, q("9", "service", cores="40-41")], {"3"}) == ([], ["3"])  # its own cores are free
    assert plan(frozen, {"3"}) == ([], ["3"])  # the service job has gone
    assert plan([*frozen, waiting], {"3"}) == ([], [])  # a service job it could help waits: its cores are free for it
    assert plan([*frozen, q("8", "service", "PENDING", cpus=6, reason="Resources")], {"3"}) == ([], ["3"])  # cannot
    unknown = dataclasses.replace(full[1], cores=frozenset())  # a running job whose cores squeue did not give
    assert plan([full[0], unknown, frozen[2]], {"3"}) == ([], [])


def test_the_agent_holds_waiting_cpu_jobs_one_at_a_time_before_suspending_and_undoes_it_all_after(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    world.agent.site = dataclasses.replace(world.agent.site, service=site_config.Service(2, "svc"))
    room = lambda: [w for w in world.writes if w[0] in {"hold", "suspend", "resume", "release", "uhold"}]  # noqa: E731
    others = [q("6", "normal", "PENDING", gpus=1, reason="Resources"), q("7", "urgent", "PENDING", reason="Resources")]
    full = [q("1", "normal", gpus=1), q("2", "cpu"), q("4", "cpu", "PENDING", reason="Resources"),
            q("5", "cpu", "PENDING", reason="Resources"), *others]
    world.queue = [*full, q("9", "service", "PENDING", reason="Resources")]
    gpu_jobs.update_sidecar("9", may_suspend=[])  # submit.py --no-suspend: its owner agreed to suspend nothing (D9)
    world.step()
    assert room() == []
    gpu_jobs.update_sidecar("9", may_suspend=["2"])  # --suspend 2
    world.fail = {"hold"}
    world.step()
    assert room() == [["hold", "4"], ["hold", "5"]]  # failed: nothing suspended, as a cpu job not held could take them
    world.fail, world.writes = set(), []
    world.step()
    assert room() == [["hold", "4"], ["hold", "5"], ["suspend", "2"]]  # cpu jobs only: GPU and urgent ones never
    saved = json.loads((world.dir / "agent" / "state.json").read_text())
    assert saved["suspended"] == ["2"] and set(saved["held"]) == {"4", "5"}
    world.writes.clear()
    held = [q("4", "cpu", "PENDING", reason="JobHeldAdmin"), q("5", "cpu", "PENDING", reason="JobHeldAdmin")]
    world.queue = [full[0], q("2", "cpu", "SUSPENDED", cores="20-21"), *held, *others, q("9", "service", cores="20-21")]
    world.step()
    assert room() == []  # the service job runs on its cores
    gpu_jobs.write_event("5", gpu_jobs.EventAction.PAUSE)  # its owner paused it meanwhile
    monkeypatch.setattr(gpu_jobs, "events", lambda job_id=None: [
        dataclasses.replace(e, actor="mallory") if e.job == "4" else e for e in events(job_id)])
    gpu_jobs.write_event("4", gpu_jobs.EventAction.PAUSE)  # someone else's PAUSE event names job 4: not its owner's
    world.queue = [full[0], q("2", "cpu", "SUSPENDED", cores="20-21"), *held, *others]  # the service job has gone
    world.step()
    assert room() == [["resume", "2"], ["uhold", "5"], ["release", "4"]]  # one call: the owner's hold, never a gap
    assert json.loads((world.dir / "agent" / "state.json").read_text())["held"] == {}


events = gpu_jobs.events


def test_a_service_job_that_does_not_start_once_room_is_made_gets_it_taken_back(world: World) -> None:
    world.agent.site = dataclasses.replace(world.agent.site, service=site_config.Service(2, "svc"))
    stuck = q("9", "service", "PENDING", reason="Resources")
    world.queue = [q("1", "normal", gpus=1), q("2", "cpu"), stuck]
    world.step()
    world.queue = [q("1", "normal", gpus=1), q("2", "cpu", "SUSPENDED", cores="20-21"), stuck]  # cores free, it waits
    world.t += agent_module.ROOM_WAIT_S + 1
    world.writes.clear()
    world.step()
    assert ["resume", "2"] in world.writes
    assert set(json.loads((world.dir / "agent" / "state.json").read_text())["given_up"]) == {"9"}
    world.queue = [q("1", "normal", gpus=1), q("2", "cpu"), stuck]  # resumed; 9 still waits
    world.writes.clear()
    world.step()
    assert ["suspend", "2"] not in world.writes  # passed over for a while...
    world.t += agent_module.ROOM_RETRY_S
    world.step()
    assert ["suspend", "2"] in world.writes  # ...then helped again


def test_part_time_requests_are_an_admins_and_each_does_exactly_its_own_thing(world: World) -> None:
    world.lap = {"name": "lap", "state": ["IDLE"], "reason": "", "extra": ""}
    world.agent = world.new_agent()
    asked = world.part_time("1", action="close")
    world.step()
    assert world.writes == []  # the writer is not an admin: nothing is closed
    world.admins = {gpu_jobs._owner(asked)}
    world.step()
    assert world.writes == [["update", "NodeName=lap", "State=DRAIN", f"Reason={agent_module.CLOSED_FROM_PAGE}"]]
    assert not asked.exists()
    world.writes.clear()
    # accept resumes only the page's own drain: never the health check's, never an operator's
    world.lap.update(state=["IDLE", "DRAIN"], reason="gpujobs-check: 4 GB free")
    world.part_time("2", action="accept")
    world.step()
    assert world.writes == []
    world.lap["reason"] = agent_module.CLOSED_FROM_PAGE
    world.part_time("3", action="accept")
    world.step()
    assert world.writes == [["update", "NodeName=lap", "State=RESUME"]]
    # give back: drain, then requeue the `resume` jobs there and cancel every other job (a part-time machine runs only those)
    world.writes.clear()
    # a forged name (a newline and a fake row) is just a name in the JSON queue, never a job to requeue
    world.queue = [gpu_jobs.QueuedJob(i, part, "RUNNING", "None", name, node="lap") for i, name, part in (
        ("11", "p:resume:a", "part_time"), ("12", "p:steady:b", "part_time"), ("13", "p:resume:c", "urgent"),
        ("14", "p:steady:x\n99|p:resume:z|part_time", "part_time"))] + [
        gpu_jobs.QueuedJob("15", "cpu", "RUNNING", "None", "p:resume:d", node="n1")]
    world.part_time("4", action="give_back")
    world.step()
    assert world.writes == [
        ["update", "NodeName=lap", "State=DRAIN", f"Reason={agent_module.CLOSED_FROM_PAGE}"], ["requeue", "11"],
        ["scancel", "12"], ["scancel", "13"], ["scancel", "14"]]


def test_part_time_limits_are_set_again_when_slurm_forgets_them_and_a_bad_pair_is_refused(world: World) -> None:
    world.lap = {"name": "lap", "state": ["IDLE"], "reason": "", "extra": ""}
    world.agent = world.new_agent()
    world.admins = {gpu_jobs._owner(world.part_time("1", action="thresholds", close_below_gb=8, push_back_below_gb=3))}
    world.step()
    extra = 'Extra={"close_below_mb":8000,"push_back_below_mb":3000}'
    assert world.writes == [["update", "NodeName=lap", extra]]
    world.writes.clear()
    world.lap["extra"] = json.dumps({"close_below_mb": 8000, "push_back_below_mb": 3000})
    world.step()
    assert world.writes == []  # slurm still has them
    world.lap["extra"] = ""
    world.step()
    assert world.writes == [["update", "NodeName=lap", extra]]
    world.writes.clear()
    world.lap["extra"] = json.dumps({"close_below_mb": 8000, "push_back_below_mb": 3000})
    world.part_time("2", action="thresholds", push_back_below_gb=9)  # above the close limit of 8
    world.step()
    assert world.writes == []
    assert world.agent.state.part_time_extra["lap"]["push_back_below_mb"] == 3000


def test_part_time_resources_rewrite_the_included_file_then_reconfigure(world: World) -> None:
    world.lap = {"name": "lap", "state": ["IDLE"], "reason": "", "extra": ""}
    world.agent = world.new_agent()
    world.admins = {gpu_jobs._owner(world.part_time("1", action="resources", jobs_memory_gb=20, keep_cpus="0-3"))}
    world.step()
    conf = (world.dir / "part_time_nodes.conf").read_text()
    assert "NodeName=lap NodeAddr=10.0.0.2 CPUs=8" in conf and "CpuSpecList=0-3 MemSpecLimit=10927" in conf
    assert world.writes == [["reconfigure"]]
    world.part_time("2", action="resources", jobs_memory_gb=99)  # more than the machine has
    world.step()
    assert world.writes == [["reconfigure"]] and world.agent.state.part_time_resources["lap"]["jobs_memory_gb"] == 20
    for n, bad in enumerate((float("nan"), float("inf"), -float("inf"), 10**400), start=3):  # JSON can carry these; round() and isfinite() cannot take them
        asked = world.part_time(str(n), action="resources", jobs_memory_gb=bad)
        world.step()
        assert not asked.exists()  # done: the agent did not crash, and did not retry it
    assert world.writes == [["reconfigure"]] and world.agent.state.part_time_resources["lap"]["jobs_memory_gb"] == 20


def test_a_part_times_job_state_is_copied_back_each_minute_and_a_failure_is_logged_once_per_outage(
        world: World, capsys: pytest.CaptureFixture[str]) -> None:
    world.lap = {"name": "lap", "state": ["IDLE"], "reason": "", "extra": ""}
    world.agent = world.new_agent()
    beat = world.step()
    ssh = (f"ssh -i {world.dir}/agent/ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=10 "
           f"-o UserKnownHostsFile={world.dir}/agent/ssh/known_hosts -o StrictHostKeyChecking=yes")  # the host key the admin saw
    only = ["--no-perms", "--chmod=D755,F644", "--no-links", "--max-size=50M", "--include=/jobs/***", "--include=/events/***",
            "--include=/logs/***", "--include=/footprints/***", "--include=/owners/***", "--exclude=*"]  # plain state files only
    assert world.copies == [["rsync", "-a", "--delete", "--timeout=20", *only, "-e", ssh, "slurm@10.0.0.2:", f"{world.dir}/remote/lap/"]]
    assert beat["copied"] == {"lap": 1000} and agent_module.read_copied(world.dir) == {"lap": 1000}  # the card's "last copied"
    world.t += 59
    world.step()
    assert len(world.copies) == 1  # once a minute, not every loop
    world.copy_answer = (255, "ssh: connect to host 10.0.0.2 port 22: Connection timed out\n")
    for _ in range(3):
        world.t += 60
        world.step()
    err = capsys.readouterr().err
    assert len(world.copies) == 4 and err.count("copying job state from lap failed: ssh: connect") == 1  # never an error loop
    assert world.agent.last_error == "" and agent_module.read_copied(world.dir) == {"lap": 1000}  # the old time stays
    world.copy_answer = (0, "")
    world.t += 60
    world.step()
    assert "copied from lap again" in capsys.readouterr().err and world.agent.copied == {"lap": 1299}
    # a partial copy (23: some files could not be read, 24: some vanished) counts as copied, and is logged once
    for _ in range(2):
        world.copy_answer = (24, "file has vanished: logs/x.log\n")
        world.t += 60
        world.step()
    assert capsys.readouterr().err.count("copied in part (exit 24)") == 1 and world.agent.copied == {"lap": 1419}
    world.shared_state, world.copies = True, []  # a shared state directory has nothing to copy
    world.agent = world.new_agent()
    world.step()
    assert world.copies == []


def idle_world(world: World, *, gpus_used: int = 0, mem_used: int = 22888) -> World:
    """The 596/611 case: a whole-box GPU job waits, a 24 GB `restart` CPU job runs, the GPU is unreserved."""
    world.totals = (f"CPUTot=4 CPUAlloc=2 RealMemory=100000 AllocMem={mem_used} "
                    f"CfgTRES=cpu=4,gres/gpu=1 AllocTRES=gres/gpu={gpus_used}")
    waiting = gpu_jobs.QueuedJob("596", "normal", "PENDING", "Resources", "wait:resume:big", user="a", gpus=1, cpus=4, mem_mb=0)
    holder = gpu_jobs.QueuedJob("611", "cpu", "RUNNING", "None", "crypto-quant-research:restart:scan", user="b", cpus=2, mem_mb=22888)
    world.queue = [waiting, holder]
    return world


def test_a_whole_box_job_waiting_beside_a_restart_cpu_job_is_found_noticed_hourly_and_shown(world: World, tmp_path: Path) -> None:
    sink = tmp_path / "notified"
    script = "import os,sys; open(sys.argv[1],'a').write(os.environ['GPUJOBS_JOB']+' '+os.environ['GPUJOBS_NAME']+'\\n')"
    world.notify_command = (sys.executable, "-c", script, str(sink))
    idle_world(world)
    world.agent = world.new_agent()
    world.agent.notify = agent_module.notify_run
    beat = world.step()
    [finding] = beat["idle"]
    assert finding["rule"] == "memory" and finding["since"] == 1000
    assert finding["text"] == "Waiting: #596 needs all memory; #611 crypto-quant-research holds 24.0 GB (restart, cannot be moved)"
    assert [(h["job"], h["movable"]) for h in finding["holders"]] == [("611", False)]
    assert agent_module.read_idle(world.dir) == beat["idle"]
    assert len(sink.read_text().splitlines()) == 1 and "wait job #596" in sink.read_text()
    world.step()  # inside the 600 s timer: not looked at again
    world.t += agent_module.IDLE_EVERY_S
    world.step()
    assert len(sink.read_text().splitlines()) == 1  # told once an hour per (waiting job, holder)
    world.t += agent_module.IDLE_NOTICE_EVERY_S
    world.step()
    assert len(sink.read_text().splitlines()) == 2
    assert world.step()["idle"][0]["since"] == 1000  # the same finding keeps its start
    # no finding for a job that waits on a dependency, nor on a machine with nothing idle
    idle_world(world)
    world.queue[0] = dataclasses.replace(world.queue[0], reason="Dependency")
    world.t += agent_module.IDLE_EVERY_S
    assert world.step()["idle"] == []
    idle_world(world, gpus_used=1)
    world.totals = world.totals.replace("CPUAlloc=2", "CPUAlloc=4").replace("AllocMem=22888", "AllocMem=99000")
    world.t += agent_module.IDLE_EVERY_S
    assert world.step()["idle"] == []
    # a shard job waits for shards: the card holds no GPU reservation, yet it is not "a GPU unreserved"
    world.totals = ("CPUTot=4 CPUAlloc=2 RealMemory=100000 AllocMem=1000 "
                    "CfgTRES=cpu=4,gres/gpu=1,gres/shard=12 AllocTRES=gres/shard=12")
    world.queue = [
        gpu_jobs.QueuedJob("700", "normal", "PENDING", "Resources", "p:resume:wait", user="a", shards=4, cpus=1, mem_mb=1000),
        gpu_jobs.QueuedJob("701", "normal", "RUNNING", "None", "p:resume:hold", user="b", shards=12, cpus=2, mem_mb=1000),
    ]
    world.t += agent_module.IDLE_EVERY_S
    [shard_finding] = world.step()["idle"]
    assert shard_finding["rule"] == "shards" and "needs 4 shards" in shard_finding["text"]
    assert [h["job"] for h in shard_finding["holders"]] == ["701"] and "GPU unreserved" not in shard_finding["idle"]
