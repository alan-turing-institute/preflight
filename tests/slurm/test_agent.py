"""The agent's decisions, against a fake queue, a fake scontrol, a fake clock and a real state directory."""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import agent as agent_module
import gpu_jobs
import pytest
import site_config


def job(
    job_id: str, state: str = "RUNNING", reason: str = "None", priority: int = 100
) -> gpu_jobs.QueuedJob:
    return gpu_jobs.QueuedJob(
        job_id, "normal", state, reason, f"p:resume:{job_id}", priority=priority
    )


def done(
    argv: list[str], code: int = 0, out: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(argv, code, out, "")


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
        self.nodes = {"n1": "IDLE", "n2": "IDLE+DRAIN"}
        self.writes: list[list[str]] = []  # every scontrol update / release, failed ones included
        self.fail: set[str] = set()  # scontrol subcommands that answer with an error
        self.records: dict[str, dict[str, str]] = {}
        self.notify_command: tuple[str, ...] = ()
        self.metrics_mode = "off"
        self.smi: dict[int, float] | None = {}  # nvidia-smi's GPU MiB per pid; None when it does not answer
        self.cgroups = tmp_path / "cgroup"
        (tmp_path / "agent").mkdir()
        (tmp_path / "agent" / "boot_id").write_text(
            "boot-1\n"
        )  # this boot is already reconciled
        self.agent = self.new_agent()

    def scontrol(
        self, argv: list[str], **_kw: object
    ) -> subprocess.CompletedProcess[str]:
        match argv[1:3]:
            case ["show", "nodes"]:
                return done(
                    argv, out="".join(f"NodeName={n} Arch=x\n" for n in self.nodes)
                )
            case ["show", "node"]:
                return done(
                    argv,
                    out=f"NodeName={argv[3]} State={self.nodes[argv[3]]} Reason=x\n",
                )
        self.writes.append(argv[1:])
        return done(argv, int(argv[1] in self.fail))

    def new_agent(self) -> agent_module.Agent:
        gpus = [{"index": 0, "uuid": "GPU-x", "device": "/dev/nvidia0", "type": "t"}] if self.has_gpus else []
        nodes = [{"name": "n1", "cpus": 4, "real_memory_mb": 1000, "gpus": gpus}]
        site = site_config.parse(
            {"name": "t", "nodes": nodes, "notify_command": list(self.notify_command), "metrics": {"mode": self.metrics_mode}}
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
        )

    def step(self) -> dict[str, object]:
        return self.agent.step()

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
    (scope / "job_7").mkdir()  # job 7 is being torn down: its cgroup lingers, its series should not
    world.metrics_mode, world.smi = "local", {10: 300.0, 20: 50.0}
    world.agent = world.new_agent()
    metrics = world.dir / "metrics"
    metrics.mkdir()
    for planted in ("job_5.prom", "job_9.prom", "job_8.prom.tmp"):  # another user's forgery, a dead job's file, a stale temp
        (metrics / planted).write_text("forged\n")
    running = dataclasses.replace(job("5"), user="ann", name="proj:train")
    world.queue = [running, job("7", "COMPLETING"), dataclasses.replace(job("6"), user="bob")]
    world.step()
    assert sorted(path.name for path in metrics.iterdir()) == ["job_5.prom"]  # job 6 runs, but on another node (no cgroup here), 7 is ending, the rest is gone
    text = (metrics / "job_5.prom").read_text()
    assert 'jobid="5",user="ann",project="proj"' in text  # the queue's user, never the writer's
    assert f"}} {300 * agent_module.MIB}\n" in text and "} 4096\n" in text  # pid 20 is nobody's here
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


def test_priority_returns_only_to_jobs_the_agent_saw_held_and_survives_a_restart(
    world: World,
) -> None:
    world.queue = [job("5", "PENDING", "Resources"), job("6", "PENDING", "Resources")]
    world.step()
    world.queue = [
        job("5", "PENDING", "JobHeldUser", 0),
        job("6", "PENDING", "Resources", 50),
    ]  # 6: an admin's change
    world.step()
    world.agent = world.new_agent()
    world.queue = [
        job("5", "PENDING", "Resources", 1),
        job("6", "PENDING", "Resources", 50),
    ]  # 5 released, re-ranked
    world.fail = {"update"}
    assert world.step()["restored"] == []  # the restore fails: 5 stays on the list of jobs seen held
    world.fail = set()
    world.writes.clear()
    beat = world.step()
    assert world.writes == [["update", "jobid=5", "priority=100"]]
    assert beat["restored"] == [{"job": "5", "from": 1, "to": 100, "at": int(world.t)}]
    world.step()
    assert len(world.writes) == 1  # once


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
    world.queue = [job("1"), job("2")]
    world.fail = {"release"}
    world.step()  # the release fails: the request stays, to be tried again
    assert path.exists()
    world.fail = set()
    world.writes.clear()
    world.step()  # victim 2 never stopped, and still does not keep job 7 held
    assert world.writes == [["release", "7"]] and not path.exists()
    path = world.request("10", ["3"], ["8"])
    world.queue = [job("3", "PENDING", "JobHeldAdmin", 0)]
    world.step()  # a held victim will not start by itself: it must not keep job 8 held forever
    assert world.writes[-1] == ["release", "8"]
    gpu_jobs.write_json(path, {"victims": ["3"], "held": ["8"]})
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
    world.queue = []
    world.step()
    world.step()
    assert world.writes == [["release", "7,8"]]


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
            "Name": "p:resume:train",
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
        ("9", "failed", {"state": "OUT_OF_MEMORY", "name": "p:resume:train"})
    ]
    assert sink.read_text() == "9 OUT_OF_MEMORY user1 p:resume:train\n"
    assert json.loads((world.dir / "agent" / "state.json").read_text())[
        "ended_seen"
    ] == ["3", "9", "10"]
