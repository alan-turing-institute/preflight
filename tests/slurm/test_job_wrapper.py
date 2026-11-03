"""The in-job wrapper: waits with a limit, re-checks the code, runs the command as a child through SIGTERM."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import gpu_jobs
import job_wrapper
import pytest

WRAPPER = Path(job_wrapper.__file__)


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


def test_footprint_sums_all_gpus_and_counts_labelled_containers(tmp_path: Path, monkeypatch) -> None:
    import audit

    cg = tmp_path / "job_5"
    cg.mkdir()
    (cg / "cgroup.procs").write_text("10\n")
    (cg / "memory.current").write_text("4096\n")
    box = "/system.slice/docker-" + "a" * 64 + ".scope"
    monkeypatch.setattr(job_wrapper, "gpu_mib_by_pid", lambda: {10: 300.0, 20: 50.0, 30: 70.0})
    monkeypatch.setattr(audit, "cgroup_of", lambda pid: box if pid == 20 else "/user.slice")
    labels = {"a" * 64: "5"}
    real = job_wrapper.gpu_jobs.container_label
    monkeypatch.setattr(
        job_wrapper.gpu_jobs, "container_label",
        lambda cgroup, label: real(cgroup, label, lambda cid, _l: labels.get(cid, "")),
    )
    row = job_wrapper.sample(cg, set(), "5")
    assert row["gpu_mib"] == 350.0  # pid 10 (cgroup) + pid 20 (container of job 5); pid 30 is someone else's
    assert row["mem_current"] == 4096


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
    tmp_path: Path, monkeypatch
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

    # legs copied back from a part-time machine are never written into the user's own sidecar (the merged view is not read)
    monkeypatch.setattr(job_wrapper.gpu_jobs, "read_sidecar", lambda *a, **k: {"legs": [{"leg": "copied"}]})
    monkeypatch.setenv("SLURM_JOB_ID", "81")
    monkeypatch.setattr(sys, "argv", ["job_wrapper.py", "--sample", "0.2", "--", "true"])
    assert job_wrapper.main() == 0
    assert [leg["leg"] for leg in json.loads((tmp_path / "state" / "jobs" / "81.json").read_text())["legs"]] == [0]

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
