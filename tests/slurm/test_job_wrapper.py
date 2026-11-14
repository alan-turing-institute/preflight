"""The in-job wrapper: waits with a limit, re-checks the code, runs the command as a child through SIGTERM."""

from __future__ import annotations

import dataclasses
import json
import os
import pwd
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import gpu_jobs
import job_wrapper
import memory_guard
import pytest

WRAPPER = Path(job_wrapper.__file__)


@pytest.fixture(autouse=True)
def no_docker(monkeypatch):
    """No test asks this machine's docker daemon: a run's containers are none unless a test says otherwise."""
    monkeypatch.setattr(job_wrapper, "job_containers", lambda *_a, **_k: [])


def fake_clock():
    now = [0.0]
    return (lambda: now[0]), (lambda s: now.__setitem__(0, now[0] + s))


def test_gives_up_at_the_limit_saying_it_waited_for_the_lease() -> None:
    clock, sleep = fake_clock()
    ok, reason = job_wrapper.wait_for_card(
        needs_lease=True, limit_s=90, poll_s=30, lease_admitted=lambda: False,
        card_free=lambda: (True, ""), clock=clock, sleep=sleep,
    )
    assert not ok
    assert "lease" in reason


def test_waits_for_a_busy_card_and_names_what_holds_it() -> None:
    clock, sleep = fake_clock()
    ok, reason = job_wrapper.wait_for_card(
        needs_lease=True, limit_s=60, poll_s=30, lease_admitted=lambda: True,
        card_free=lambda: (False, "pid 7 on the card"), clock=clock, sleep=sleep,
    )
    assert not ok
    assert "pid 7" in reason


def test_proceeds_once_the_lease_is_held_and_the_card_is_free() -> None:
    clock, sleep = fake_clock()
    answers = iter([False, False, True])
    ok, _ = job_wrapper.wait_for_card(
        needs_lease=True, limit_s=600, poll_s=30, lease_admitted=lambda: next(answers),
        card_free=lambda: (True, ""), clock=clock, sleep=sleep,
    )
    assert ok


def test_job_cgroup_is_the_job_component_of_our_cgroup(tmp_path: Path) -> None:
    proc = tmp_path / "cgroup"
    proc.write_text("0::/system.slice/gpu-node-1_slurmstepd.scope/job_42/step_batch/user/task_0\n")
    path = job_wrapper.job_cgroup(proc)
    assert path == job_wrapper.CGROUP_ROOT / "system.slice/gpu-node-1_slurmstepd.scope/job_42"
    proc.write_text("0::/system.slice/slurmstepd.scope/s8VSM5Q2WYAV00/step_batch/user/task_0\n")  # 26.05
    assert job_wrapper.job_cgroup(proc) == job_wrapper.CGROUP_ROOT / "system.slice/slurmstepd.scope/s8VSM5Q2WYAV00"
    proc.write_text("0::/user.slice/session-1.scope\n")
    assert job_wrapper.job_cgroup(proc) is None


SMI = "100, GPU-aaa\n200, GPU-bbb\n300, GPU-bbb\n"


def test_occupancy_blocks_on_any_process_outside_the_job_and_names_it(monkeypatch) -> None:
    assert job_wrapper.foreign_gpu_processes({100, 200, 300}, SMI) == []
    assert job_wrapper.foreign_gpu_processes({100}, SMI) == [(200, "GPU-bbb"), (300, "GPU-bbb")]
    # nvidia-smi missing, hung or failing: occupancy unknown is not free (as dgx/assert_gpu_free.py refused)
    for fail in (FileNotFoundError("nvidia-smi"), subprocess.TimeoutExpired("nvidia-smi", 30)):
        monkeypatch.setattr(job_wrapper.subprocess, "run", lambda *_a, fail=fail, **_k: (_ for _ in ()).throw(fail))
        assert job_wrapper.card_is_free(set()) == (False, "nvidia-smi did not answer, so the card's occupancy is unknown")
    monkeypatch.setattr(job_wrapper.subprocess, "run", lambda argv, **_k: subprocess.CompletedProcess(argv, 9, "", "NVML"))
    assert job_wrapper.card_is_free(set())[0] is False


def test_declared_sharing_skips_the_occupancy_check(monkeypatch) -> None:
    monkeypatch.setattr(job_wrapper, "foreign_gpu_processes", lambda own: [(9, "GPU-x")])
    assert job_wrapper.card_is_free({1}) == (False, "compute process(es) outside this job: pid 9 on GPU-x")
    assert job_wrapper.card_is_free({1}, max_jobs=2) == (True, "")
    # a shard job shares the card with other slurm jobs: only a process outside slurm and the leases refuses it
    import audit

    monkeypatch.setattr(job_wrapper.gpu_jobs, "gpu_usage", lambda: {10: 300.0, 20: 50.0})
    monkeypatch.setattr(audit, "cgroup_of", lambda pid, *_a: "/x/slurmstepd.scope/job_5" if pid == 10 else "/user.slice")
    monkeypatch.setattr(audit, "lease_pids", lambda *_a: {})
    monkeypatch.setattr(audit, "descends_from", lambda *_a: False)
    monkeypatch.setattr(audit, "command_of", lambda *_a: "python")
    monkeypatch.setattr(job_wrapper.gpu_jobs, "container_label", lambda *_a: "")
    monkeypatch.setattr(job_wrapper.gpu_jobs, "known_containers", list)
    free, why = job_wrapper.card_is_free({1}, shard_job=True)
    assert not free and "pid 20" in why and "pid 10" not in why  # pid 10 is a neighbouring slurm job's
    monkeypatch.setattr(job_wrapper.gpu_jobs, "gpu_usage", lambda: {10: 300.0})
    assert job_wrapper.card_is_free({1}, shard_job=True) == (True, "")
    monkeypatch.setattr(job_wrapper.gpu_jobs, "gpu_usage", lambda: None)
    assert job_wrapper.card_is_free({1}, shard_job=True)[0] is False  # occupancy unknown is not free


def test_footprint_counts_the_runs_containers_gpu_and_host_memory(tmp_path: Path, monkeypatch) -> None:
    cg = tmp_path / "job_5"
    cg.mkdir()
    (cg / "cgroup.procs").write_text("10\n")
    (cg / "memory.current").write_text("4096\n")
    mine, theirs = ("/system.slice/docker-" + c * 64 + ".scope" for c in "ab")
    scope = tmp_path / "cgroups" / mine.lstrip("/")
    scope.mkdir(parents=True)
    (scope / "memory.current").write_text("1000\n")
    (scope / "memory.peak").write_text("3000\n")
    monkeypatch.setattr(job_wrapper.gpu_jobs, "CGROUP_ROOT", tmp_path / "cgroups")
    monkeypatch.setattr(job_wrapper, "gpu_mib_by_pid", lambda: {10: 300.0, 20: 50.0, 30: 70.0})
    monkeypatch.setattr(job_wrapper.gpu_jobs, "cgroup_of", lambda pid, *_a: {20: mine + "/sub", 30: theirs}.get(pid, "/user.slice"))
    asked = []
    run_containers = [job_wrapper.gpu_jobs.JobContainer("a" * 64, "5", 101.0, mine)]
    monkeypatch.setattr(job_wrapper, "job_containers", lambda job_id, since: asked.append((job_id, since)) or run_containers)
    row = job_wrapper.sample(cg, set(), "5", 100.0)
    assert asked == [("5", 100.0)]  # only containers made during this run
    assert row["gpu_mib"] == 350.0  # pid 10 (cgroup) + pid 20 (this run's container); pid 30's container is not this run's
    assert (row["mem_current"], row["ctr_current"], row["ctr_peak"], row["ctr_ids"]) == (4096, 1000, 3000, ["a" * 12])
    monkeypatch.setattr(job_wrapper, "job_containers", lambda job_id, since: None)  # docker absent or refused
    row = job_wrapper.sample(cg, set(), "5", 100.0)
    assert (row["gpu_mib"], row["ctr_peak"], row["ctr_ids"]) == (300.0, 0, None)  # unknown, not "none"
    # SIGTERM reaches the run's running containers, which slurm's signal never does
    calls = []
    monkeypatch.setattr(job_wrapper, "job_containers", lambda job_id, since, run: [*run_containers, job_wrapper.gpu_jobs.JobContainer("c" * 64, "5", 102.0, "")])
    assert job_wrapper.pass_term("5", 100.0, run=lambda argv, **_k: calls.append(argv)) == ["a" * 64]
    assert calls == [["docker", "kill", "--signal", "TERM", "a" * 64]]  # not the one that is not running
    assert job_wrapper.pass_term("5", 100.0, run=lambda argv, **_k: calls.append(argv), told=frozenset({"a" * 64})) == []
    # after slurm's SIGTERM the wrapper waits while they stop; slurm's SIGKILL at KillWait's end bounds it
    views = [run_containers, run_containers, []]
    monkeypatch.setattr(job_wrapper, "job_containers", lambda job_id, since: views.pop(0))
    naps: list[float] = []
    job_wrapper.wait_for_containers("5", 100.0, poll_s=0.5, sleep=naps.append)
    assert naps == [0.5, 0.5] and not views


SITE = 'name = "t"\n[[nodes]]\nname = "n"\ncpus = 1\nreal_memory_mb = 1\n[metrics]\nmode = "%s"\n'
SEEN = "import time, sys, pathlib; time.sleep(0.7); pathlib.Path(sys.argv[2]).write_text(str(pathlib.Path(sys.argv[1]).exists()))"


def run_wrapper(tmp_path: Path, job: str, mode: str) -> tuple[str, bool]:
    """(whether the job saw a metrics textfile while running, whether one is there afterwards)."""
    env = dict(os.environ, SLURM_GPU_STATE_DIR=str(tmp_path / "state"), SLURM_JOB_ID=job,
               GPUJOBS_SITE=str(tmp_path / "site.toml"))
    (tmp_path / "site.toml").write_text(SITE % mode)
    prom = tmp_path / "state" / "metrics" / f"job_{job}.prom"
    seen = tmp_path / f"seen_{job}"
    subprocess.run([sys.executable, str(WRAPPER), "--sample", "0.2", "--", sys.executable, "-c", SEEN, str(prom), str(seen)],
                   env=env, check=True)
    return seen.read_text(), prom.exists()


def test_the_wrapper_writes_no_metrics_textfile_whatever_the_site_says(tmp_path: Path) -> None:
    # the agent writes `metrics/job_<id>.prom` (tests/slurm/test_agent.py); the wrapper's user must not
    assert run_wrapper(tmp_path, "82", "local") == ("False", False)


CHILD = """
import os, signal, sys, time
def term(*_):
    time.sleep(1.0)  # "writing a checkpoint"
    open(sys.argv[1], "w").write("saved")
    sys.exit(0)
signal.signal(signal.SIGTERM, term)
open(sys.argv[2] + ".tmp", "w").write(open("/proc/self/oom_score_adj").read())
os.rename(sys.argv[2] + ".tmp", sys.argv[2])
time.sleep(30)
"""


def test_sigterm_is_passed_to_the_child_which_saves_then_the_leg_is_recorded(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    saved, up = tmp_path / "saved", tmp_path / "up"
    env = dict(os.environ, SLURM_GPU_STATE_DIR=str(tmp_path / "state"), SLURM_JOB_ID="77")
    wrapper = subprocess.Popen(
        [sys.executable, str(WRAPPER), "--sample", "0.2", "--", sys.executable, "-c", CHILD, str(saved), str(up)],
        env=env,
    )
    deadline = time.time() + 10
    while not up.exists() and time.time() < deadline:
        time.sleep(0.05)
    assert int(up.read_text()) >= 800  # the job, not a session, is the preferred memory victim
    os.kill(wrapper.pid, signal.SIGTERM)  # slurm signals only the job script's own process (stage C test 3)
    assert wrapper.wait(timeout=10) == 0
    assert saved.read_text() == "saved"
    record = json.loads((tmp_path / "state" / "jobs" / "77.json").read_text())
    assert record["last_leg"]["signals"] == ["SIGTERM"]
    me = f"SIGTERM from pid {os.getpid()} (user {pwd.getpwuid(os.getuid()).pw_name}: "  # the sender is named
    assert [s[:len(me)] for s in record["last_leg"]["senders"]] == [me]
    assert record["last_leg"]["exit"] == 0
    assert (tmp_path / "state" / "footprints" / "77.jsonl").exists()

    # A SIGTERM that lands before the child exists is passed on as soon as it does, not lost.
    real_popen, sent = subprocess.Popen, []

    def popen_after_a_sigterm(*args, **kwargs):  # the first start only: sampling runs subprocesses too
        if not sent:
            sent.append(True)
            os.kill(os.getpid(), signal.SIGTERM)
        return real_popen(*args, **kwargs)

    monkeypatch.setattr(job_wrapper.gpu_jobs, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(subprocess, "Popen", popen_after_a_sigterm)
    before = signal.signal(signal.SIGTERM, lambda *_: None)  # stands in for "no handler yet": the signal is lost
    try:
        leg = job_wrapper.run_and_measure([sys.executable, "-c", "import time; time.sleep(5)"], tmp_path / "f.jsonl", None, 0.2)
    finally:
        signal.signal(signal.SIGTERM, before)
    assert (leg["signals"], leg["exit"]) == (["SIGTERM"], -signal.SIGTERM)

    # Naming the sender failing (a user lookup, a full log) never stops the SIGTERM reaching the command.
    def full_log(_info: object) -> str:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(job_wrapper, "sender_text", full_log)
    threading.Timer(0.5, os.kill, (os.getpid(), signal.SIGTERM)).start()
    before = signal.getsignal(signal.SIGTERM)
    try:
        leg = job_wrapper.run_and_measure([sys.executable, "-c", "import time; time.sleep(5)"], tmp_path / "s.jsonl", None, 0.2)
    finally:
        signal.signal(signal.SIGTERM, before)
    assert (leg["signals"], leg["exit"], "senders" in leg) == (["SIGTERM"], -signal.SIGTERM, False)
    # the leg's memory peak is the job's own plus its containers', which slurm's cgroup does not see
    row = {"gpu_mib": 0.0, "mem_peak": 100, "mem_current": 90, "ctr_peak": 50, "ctr_ids": ["c1"]}
    sampled: list[tuple[object, ...]] = []
    monkeypatch.setattr(job_wrapper, "sample", lambda *a: sampled.append(a) or row)
    leg = job_wrapper.run_and_measure([sys.executable, "-c", "import time; time.sleep(0.5)"], tmp_path / "g.jsonl", None, 0.05, "5",
                                      leg_started=42.0)
    assert (leg["peak_mem_bytes"], leg["peak_container_bytes"]) == (150, 50)  # no held figure read: the kernel's peak
    # sized by what the guard counts: page cache the host could drop (a model file read at load) is left out
    row = {**row, "mem_held": 30}
    leg = job_wrapper.run_and_measure([sys.executable, "-c", "import time; time.sleep(0.5)"], tmp_path / "h.jsonl", None, 0.05, "5",
                                      leg_started=42.0)
    assert (leg["peak_mem_bytes"], leg["peak_mem_total_bytes"]) == (80, 150)
    assert sampled[0][2:] == ("5", 42.0)  # containers are counted only from this run's start
    # docker refusing this user (or hung) at the leg's one probe: said once, and no other docker call this leg
    monkeypatch.setattr(job_wrapper, "job_containers", lambda *_a, **_k: None)
    monkeypatch.setattr(job_wrapper.shutil, "which", lambda name: f"/usr/bin/{name}")
    sampled.clear()
    capsys.readouterr()
    leg = job_wrapper.run_and_measure([sys.executable, "-c", "import time; time.sleep(0.3)"], tmp_path / "g2.jsonl", None, 0.05,
                                      "5", leg_started=42.0)
    assert leg["docker"] is False and sampled and all(a[2] == "" for a in sampled)
    assert capsys.readouterr().err.count("docker did not answer") == 1
    sockets = job_wrapper.DOCKER_SOCKETS
    monkeypatch.setattr(job_wrapper, "DOCKER_SOCKETS", (tmp_path / "no-socket",))  # a user without access to docker
    job_wrapper.run_and_measure([sys.executable, "-c", "pass"], tmp_path / "g3.jsonl", None, 0.05, "5", leg_started=42.0)
    assert "docker did not answer" not in capsys.readouterr().err  # not using docker is no fault
    monkeypatch.setattr(job_wrapper, "DOCKER_SOCKETS", sockets)
    real_sample = job_wrapper.sample
    # docker going silent mid-run: one slow sample, then no docker call for the rest of the leg (the stop poll, D44)
    answers = iter([[], None])
    monkeypatch.setattr(job_wrapper, "job_containers", lambda *_a, **_k: next(answers, "asked again"))
    monkeypatch.setattr(job_wrapper, "sample", lambda c, p, j, s: {"gpu_mib": 0.0, "mem_peak": 1, "mem_current": 1, "ctr_peak": 0,
                                                                     "ctr_ids": job_wrapper.job_containers() if j else []})
    leg = job_wrapper.run_and_measure([sys.executable, "-c", "import time; time.sleep(0.4)"], tmp_path / "g4.jsonl", None, 0.05,
                                      "5", leg_started=42.0)
    assert leg["docker"] is False and next(answers, "unused") == "unused"
    assert "docker stopped answering" in capsys.readouterr().err
    monkeypatch.setattr(job_wrapper, "job_containers", lambda *_a, **_k: [])
    monkeypatch.setattr(job_wrapper, "sample", real_sample)
    # a command that ends at once on SIGTERM leaves the loop before its first poll: its containers are still told, once
    told_calls: list[tuple[str, frozenset[str]]] = []
    monkeypatch.setattr(job_wrapper, "pass_term", lambda job_id, since, told=frozenset(): told_calls.append((job_id, told)) or [])
    sent.clear()
    before = signal.signal(signal.SIGTERM, lambda *_: None)
    try:
        leg = job_wrapper.run_and_measure([sys.executable, "-c", "import time; time.sleep(5)"], tmp_path / "h.jsonl", None, 5.0, "5",
                                          leg_started=42.0)
    finally:
        signal.signal(signal.SIGTERM, before)
    assert leg["signals"] == ["SIGTERM"] and told_calls == [("5", frozenset())]
    monkeypatch.undo()  # back to the real sample for main() below, with no docker asked
    monkeypatch.setattr(job_wrapper.gpu_jobs, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(job_wrapper, "job_containers", lambda *_a, **_k: [])

    # legs copied back from a part-time machine are never written into the user's own sidecar (the merged view is not read)
    monkeypatch.setattr(job_wrapper.gpu_jobs, "read_sidecar", lambda *a, **k: {"legs": [{"leg": "copied"}]})
    monkeypatch.setenv("SLURM_JOB_ID", "81")
    monkeypatch.setattr(sys, "argv", ["job_wrapper.py", "--sample", "0.2", "--", "true"])
    capsys.readouterr()
    assert job_wrapper.main() == 0
    assert [leg["leg"] for leg in json.loads((tmp_path / "state" / "jobs" / "81.json").read_text())["legs"]] == [0]
    printed = capsys.readouterr().out.splitlines()  # the usage report follows the exit line, and is kept in the leg (D37)
    assert printed[-2].startswith("[job] exit 0;") and printed[-1].startswith("[job] used:")
    assert json.loads((tmp_path / "state" / "jobs" / "81.json").read_text())["last_leg"]["usage"]["lines"] == printed[-1:]

    # A leg records the cores it was allowed, by top speed, on a host whose cores differ and nothing on one whose cores are alike
    # (2026-11-03: one thread ran 4 to 6 times slower on the slower cores, and no run said which it had been given).
    monkeypatch.setattr(job_wrapper.gpu_jobs, "read_sidecar", lambda *a, **k: {})
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {0, 1, 5})
    slow, fast = 2_808_000, 3_900_000
    for job_id, khz, expected in (
        ("82", {0: slow, 1: slow, 5: fast}, {"cpus": "0-1,5", "cores_by_max_mhz": {"2808": 2, "3900": 1}}),
        ("83", {0: fast, 1: fast, 5: fast}, None),
    ):
        monkeypatch.setenv("SLURM_JOB_ID", job_id)
        monkeypatch.setattr(job_wrapper.cores.CoreSpeeds, "read", classmethod(lambda cls, root=None, khz=khz: cls(khz)))
        assert job_wrapper.main() == 0
        assert json.loads((tmp_path / "state" / "jobs" / f"{job_id}.json").read_text())["last_leg"].get("cores") == expected


def test_an_unverifiable_manifest_refuses_the_launch(tmp_path: Path) -> None:
    env = dict(os.environ, SLURM_GPU_STATE_DIR=str(tmp_path / "state"), SLURM_JOB_ID="78")
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")  # empty: the parity checker refuses it, as it refuses a changed file
    done = subprocess.run(
        [sys.executable, str(WRAPPER), "--manifest", str(manifest), "--repo", str(tmp_path), "--", "true"],
        env=env, check=False, capture_output=True, text=True,
    )
    assert done.returncode == job_wrapper.Exit.CODE_CHANGED
    assert "records no files" in done.stderr  # the checker ran and refused; a checker that could not run says otherwise


def test_bookkeeping_that_cannot_be_written_never_fails_the_job(tmp_path: Path) -> None:
    state = tmp_path / "state"
    for blocked in ("footprints/79.jsonl", "jobs/79.json"):
        (state / blocked).mkdir(parents=True)  # stand in for another user's files in the sticky directories
    (tmp_path / "site.toml").write_text(SITE % "local")
    env = dict(os.environ, SLURM_GPU_STATE_DIR=str(state), SLURM_JOB_ID="79", GPUJOBS_SITE=str(tmp_path / "site.toml"))
    done = subprocess.run(
        [sys.executable, str(WRAPPER), "--sample", "0.2", "--", sys.executable, "-c", "import time; time.sleep(0.7); raise SystemExit(3)"],
        env=env, check=False, capture_output=True, text=True,
    )
    assert done.returncode == 3  # the command's own status
    assert all(f"warning: {what} not recorded" in done.stderr for what in ("footprint", "sidecar"))


@pytest.mark.parametrize(("lease", "code", "refused"), [(False, 0, False), (True, job_wrapper.Exit.CARD_NOT_READY, True)])
def test_a_lease_less_site_never_waits_for_a_keeper_that_does_not_exist(
    tmp_path: Path, lease: bool, code: int, refused: bool
) -> None:
    leases = tmp_path / "leases"
    leases.mkdir()
    (tmp_path / "site.toml").write_text(
        SITE % "off" + (f'[lease]\nlease_dir = "{leases}"\nkeeper_user = "k"\npython = "{sys.executable}"\n' if lease else "")
    )
    env = dict(os.environ, SLURM_GPU_STATE_DIR=str(tmp_path / "state"), SLURM_JOB_ID="80",
               GPUJOBS_SITE=str(tmp_path / "site.toml"), GPU_LEASE_DIR=str(leases))
    done = subprocess.run(
        [sys.executable, str(WRAPPER), "--needs-lease", "--wait-limit", "0", "--poll", "0.1", "--", "true"],
        env=env, check=False, capture_output=True, text=True,
    )
    assert done.returncode == code, done.stderr
    record = json.loads((tmp_path / "state" / "jobs" / "80.json").read_text())
    assert ("refused" in record["last_leg"]) is refused


KILLED = """
import json, os, pathlib, sys, time
record = pathlib.Path(sys.argv[1])  # the agent's record, as it writes it when earlyoom names this process
record.parent.mkdir(parents=True)
record.write_text(json.dumps({"job": "78", "pid": os.getpid(), "signal": "SIGTERM", "at": time.time(), "guard": "earlyoom.service"}))
sys.exit(0)  # Lightning ends a SIGTERMed fit with exit 0 (job 428, 2026-10-22)
"""


def test_a_run_the_oom_guard_killed_fails_or_is_requeued_never_passes(tmp_path: Path) -> None:
    state = tmp_path / "state"
    env = dict(os.environ, SLURM_GPU_STATE_DIR=str(state), SLURM_JOB_ID="78", SLURM_JOB_NAME="p:restart:x")
    record = state / "agent" / "oom_kills" / "job_78.json"
    done = subprocess.run([sys.executable, str(WRAPPER), "--sample", "0.2", "--", sys.executable, "-c", KILLED, str(record)],
                          env=env, capture_output=True, text=True, check=False)
    assert done.returncode == job_wrapper.Exit.KILLED_FOR_MEMORY, done.stderr
    assert "KILLED" in done.stderr and "not a resume job" in done.stderr
    assert json.loads((state / "jobs" / "78.json").read_text())["last_leg"]["killed_by"]["pid"] > 0

    # which records count: this leg's, of the command itself or of a worker killed shortly before the command ended
    kill = {"job": "78", "pid": 5, "at": 1000.0}
    clock = iter(range(100)).__next__

    def found(record: dict, ended: float = 1010.0) -> bool:
        return job_wrapper.guard_kill("78", 900.0, ended, 5, read=lambda _p: record, clock=clock, sleep=lambda _s: None) is not None

    assert found(kill) and found({**kill, "pid": 6}) and not found({**kill, "pid": 6}, 1000.0 + job_wrapper.KILL_WINDOW_S + 1)
    assert not found({**kill, "at": 800.0}) and not found({**kill, "job": "79"})  # an earlier leg's, another job's

    # a resume job is requeued, until it has been killed KILLED_LEGS_LIMIT legs in a row
    ran: list[list[str]] = []
    legs = [{"exit": 0}, *[{"killed_by": kill}] * (job_wrapper.KILLED_LEGS_LIMIT - 1)]
    assert job_wrapper.after_kill("78", kill, legs, "p:resume:x", lambda argv: ran.append(argv) or 0) == 5
    assert ran == [["scontrol", "requeue", "78"]]
    job_wrapper.after_kill("78", kill, [*legs, {"killed_by": kill}], "p:resume:x", lambda argv: ran.append(argv) or 0)
    job_wrapper.after_kill("78", kill, [{"killed_by": kill}], "p:restart:x", lambda argv: ran.append(argv) or 0)
    assert len(ran) == 1


def test_a_memory_stop_of_this_run_is_carried_out_by_its_wrapper_and_only_that_one(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    gb, state = 10**9, tmp_path / "state"
    monkeypatch.setattr(job_wrapper.gpu_jobs, "STATE_DIR", state)
    monkeypatch.setattr(gpu_jobs, "SLURM_USER", gpu_jobs.current_user())  # the agent's user wrote the requests below

    def stop(job_id: str, at: float) -> None:
        memory_guard.StopRequest(job_id, 0, at, 9 * gb, 2 * gb, 8 * gb, 3 * gb).write()

    def run(job_id: str, child: str, then=None) -> tuple[int, float, dict]:
        monkeypatch.setenv("SLURM_JOB_ID", job_id)
        monkeypatch.setenv("SLURM_JOB_NAME", "p:restart:x")
        monkeypatch.setattr(sys, "argv", ["job_wrapper.py", "--sample", "0.2", "--", sys.executable, "-c", child])
        if then:
            threading.Timer(0.5, then).start()
        began = time.monotonic()
        status = job_wrapper.main()
        return status, time.monotonic() - began, json.loads((state / "jobs" / f"{job_id}.json").read_text())

    before = signal.getsignal(signal.SIGTERM)
    try:
        status, took, record = run("91", "import time; time.sleep(30)", lambda: stop("91", time.time()))
        assert status == job_wrapper.Exit.KILLED_FOR_MEMORY and took < 0.5 + memory_guard.ACK_S  # inside the agent's wait
        assert record["stop_ack"]["at"] >= record["stop_ack"]["request_at"]  # acknowledged before the kill
        assert record["last_leg"]["killed_by"]["guard"] == "memory stop-gap" and "stopped" not in record["last_leg"]
        assert "stopped by the memory guard" in capsys.readouterr().err
        assert len((state / "footprints" / "91.jsonl").read_text().splitlines()) > 1  # --sample 0.2 still samples
        stop("92", time.time() - 100)  # an earlier run's request
        assert run("92", "import time; time.sleep(1)")[0] == 0
        monkeypatch.setattr(gpu_jobs, "SLURM_USER", "somebody-else")  # a request the agent did not write
        stop("93", time.time() + 100)
        assert run("93", "import time; time.sleep(1)")[0] == 0 and "ignoring" in capsys.readouterr().err
        monkeypatch.setattr(gpu_jobs, "SLURM_USER", gpu_jobs.current_user())
        # slurm's SIGTERM in the same moment (a child that ignores it): the stop still counts as a memory kill
        stubborn = "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)"
        status, _, record = run("94", stubborn, lambda: (os.kill(os.getpid(), signal.SIGTERM), stop("94", time.time())))
        assert status == job_wrapper.Exit.KILLED_FOR_MEMORY and record["last_leg"]["signals"] == ["SIGTERM"]
    finally:
        signal.signal(signal.SIGTERM, before)


def test_a_part_time_launch_that_no_longer_holds_is_requeued_with_a_marker_and_the_third_in_a_row_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(gpu_jobs, "STATE_DIR", tmp_path)
    ran: list[list[str]] = []
    again = lambda argv: ran.append(argv) or 0  # noqa: E731
    for refusals in (1, 2):
        assert job_wrapper.refuse_launch("90", ["/x does not exist here"], again) == job_wrapper.Exit.LAUNCH_CHECK_FAILED
        assert json.loads(gpu_jobs.sidecar_path("90").read_text())["launch_refusals"] == refusals
        assert len(ran) == refusals  # requeued, not failed
    markers = gpu_jobs.events("90")
    assert [e.action for e in markers] == ["launch_refused"] * 2  # what node_check.py turns into the node's exclusion
    assert markers[0].fields["problems"] == ["/x does not exist here"] and markers[0].actor == gpu_jobs.current_user()
    job_wrapper.refuse_launch("90", ["/x does not exist here"], again)
    assert len(ran) == 2 and len(gpu_jobs.events("90")) == 2  # the third: failed with the message, nothing requeued
    (tmp_path / "jobs" / "91.json").mkdir(parents=True)  # a count that cannot be kept must fail the job, not loop it
    job_wrapper.refuse_launch("91", ["p"], again)
    assert len(ran) == 2


def test_the_launch_check_runs_only_in_the_part_time_partition(tmp_path: Path) -> None:
    ran = tmp_path / "ran"
    require = json.dumps([["program", str(tmp_path / "missing"), ""]])
    results = []
    for partition in ("part_time", "cpu_fill"):
        env = dict(os.environ, SLURM_GPU_STATE_DIR=str(tmp_path / partition), SLURM_JOB_ID="92", SLURM_JOB_PARTITION=partition)
        done = subprocess.run([sys.executable, str(WRAPPER), "--sample", "0.2", "--require", require, "--", "touch", str(ran)],
                              env=env, check=False, capture_output=True, text=True)
        results.append((done.returncode, ran.exists(), "REFUSED at launch" in done.stderr))
        ran.unlink(missing_ok=True)
    assert results == [(job_wrapper.Exit.LAUNCH_CHECK_FAILED, False, True), (0, True, False)]


def test_the_end_of_run_report_measures_from_the_cgroup_and_advises_the_next_request(tmp_path: Path) -> None:
    gib, most_mb = 2**30, 100 * 1024
    cg = tmp_path / "job"
    cg.mkdir()
    (cg / "memory.max").write_text(f"{40 * gib}\n")
    (cg / "cpu.stat").write_text("usage_usec 3000000\nuser_usec 1\n")
    (cg / "memory.events").write_text("low 0\nmax 7\noom 2\noom_kill 1\n")
    (cg / "memory.pressure").write_text("some avg10=0.00 avg60=0.00 avg300=0.00 total=600000\n"
                                        "full avg10=0.00 avg60=0.00 avg300=0.00 total=200000\n")
    Counters = job_wrapper.Counters
    assert job_wrapper.cgroup_counters(cg) == Counters(3_000_000, 1, 7, 600_000, 200_000)
    assert job_wrapper.cgroup_counters(None) == Counters()
    outcome = {"peak_mem_bytes": 4 * 10**9, "peak_gpu_mib": 1024}

    def measured(**kw: object) -> job_wrapper.Usage:
        args = {"cgroup": cg, "before": Counters(1_000_000, 1, 2, 100_000, 0), "after": job_wrapper.cgroup_counters(cg),
                "wall_s": 2.0, "unified": True, **kw}
        return job_wrapper.measure_usage(args.pop("cgroup"), outcome, args.pop("before"), args.pop("after"), args.pop("wall_s"),
                                         guard_killed=False, unified=args.pop("unified"))

    u = measured()
    assert (u.mem_asked_bytes, u.mem_peak_bytes, u.cores_avg, u.cap_killed) == (40 * gib, 4 * 10**9 + gib, 1.0, False)  # 2 core-seconds in 2 s
    assert (u.mem_wait_some, u.mem_wait_full, u.cap_hits) == (0.25, 0.1, 5)  # D51: 0.5 s and 0.2 s of waiting in 2 s; 5 cap hits
    (cg / "memory.pressure").write_text("some avg10=0.00\n")  # unreadable (as on a kernel without PSI): unknown, not zero
    assert job_wrapper.cgroup_counters(cg).wait_some_usec is None
    fell = measured(before=Counters(1_000_000, 1, 9, 900_000, 900_000), after=Counters(3_000_000, 1, 3, 100, 100))
    assert (fell.mem_wait_some, fell.cap_hits) == (None, None)  # a counter that fell is another cgroup's: unknown, not negative
    assert measured(unified=False).mem_peak_bytes == 4 * 10**9  # the card's memory counts only where it shares the pool
    assert measured(after=Counters(3_000_000, 2)).cap_killed  # the cap killed a process during this run
    (cg / "memory.max").write_text("max\n")
    assert measured().mem_asked_bytes is None  # no cap: unknown
    assert measured(cgroup=None, before=Counters(), after=Counters()) == job_wrapper.Usage(None, None, u.cores_held, None)

    def lines(peak: float | None = 9.4e9, *, asked: int | None = 40 * gib, avg: float | None = 1.7, gpu: bool = False,
              cap: bool = False, guard: bool = False, cores_cap: int | None = 8) -> list[str]:
        return job_wrapper.usage_lines(job_wrapper.Usage(asked, None if peak is None else int(peak), 8, avg, cap, guard),
                                       gpu_job=gpu, cores_cap=cores_cap, most_mb=most_mb)

    assert lines() == ["[job] used: memory peak 9.4 GB of 42.9 GB asked (22%); cores 1.7 on average of 8 held (21%)",
                       "[job] for the same work next time: --mem 10G --cpus 3"]  # peak + 10%; ceil(1.7 x 1.25)
    assert lines(gpu=True)[1].endswith("--mem 10G")  # a GPU job's cores are held for the card: no --cpus advice
    gpu_node = job_wrapper.site_config.Node("n", 20, 1000, (job_wrapper.site_config.Gpu(0, "u", "/dev/nvidia0", "t"),),
                                            cores_per_gpu=12, fast_cpus="5-9,15-19")
    site = SimpleNamespace(nodes=(gpu_node,), service=None, cpu_job_cap=8)
    assert job_wrapper.cores_advice(site, card=True, shard=False, service=False) == (False, 20)  # D46: bound, so advised
    assert job_wrapper.cores_advice(site, card=False, shard=True, service=False) == (True, 8)  # its cores are its shards
    site.nodes = (dataclasses.replace(gpu_node, fast_cpus=None),)
    assert job_wrapper.cores_advice(site, card=True, shard=False, service=False) == (True, 8)  # held for the card (D33)
    assert job_wrapper.cores_advice(site, card=False, shard=False, service=False) == (False, 8)
    assert lines(avg=20.0)[1].endswith("--cpus 8")  # within the cores the job may have
    assert lines(peak=5e6)[0].startswith("[job] used: memory peak 5 MB") and "--mem 1G" in lines(peak=5e6)[1]
    assert "--mem 1G" in lines(peak=0)[1]  # never 0G: slurm reads --mem 0 as the whole box
    assert "--mem 0" in lines(peak=95 * gib)[1]  # more than any node gives jobs: the whole box, not a figure slurm refuses
    killed = lines(asked=32 * gib, cap=True)
    assert killed[0].endswith("; the memory cap killed a process") and "--mem 39G or more" in killed[1]  # 1.2 x 32 GiB
    assert "--mem 0 " in lines(asked=most_mb * 2**20, guard=True)[1] + " "  # killed with the whole box: no more to give
    assert lines(guard=True, asked=None)[0].endswith("killed to free the host's memory")
    contained = job_wrapper.Usage(40 * gib, int(9.4e9), 8, 1.7, containers_bytes=int(3.7e9))
    assert job_wrapper.usage_lines(contained, gpu_job=True, cores_cap=8, most_mb=most_mb)[0].startswith(
        "[job] used: memory peak 9.4 GB (containers 3.7 GB) of 42.9 GB asked")  # the share outside the job's cgroup
    assert "--mem 7G or more" in lines(peak=6.7e9, asked=2 * gib, guard=True)[1]  # stopped over its ask: its peak + 10% (D44)
    assert lines(peak=None, asked=None, avg=None) == ["[job] used: memory peak unknown; cores in use unknown of 8 held"]
    waited = dataclasses.replace(contained, containers_bytes=0, mem_wait_some=0.25, mem_wait_full=0.1, cap_hits=5)
    assert job_wrapper.usage_lines(waited, gpu_job=True, cores_cap=8, most_mb=most_mb)[0] == (
        "[job] used: memory peak 9.4 GB of 42.9 GB asked (22%), waited on memory 25.0% of the run (all work 10.0%), "
        "reaching its cap; cores 1.7 on average of 8 held (21%)")
    assert ", reaching" not in job_wrapper.usage_lines(dataclasses.replace(waited, cap_hits=0), gpu_job=True, cores_cap=8, most_mb=most_mb)[0]
    record = job_wrapper.usage_record(u, ["[job] x"])
    assert record == {"mem_asked_bytes": 40 * gib, "mem_peak_bytes": 4 * 10**9 + gib, "cores_held": u.cores_held,
                      "cores_avg": 1.0, "mem_wait_some": 0.25, "mem_wait_full": 0.1, "mem_cap_hits": 5,
                      "memory_cap_killed": False, "lines": ["[job] x"]}
