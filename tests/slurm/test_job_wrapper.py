"""The in-job wrapper: waits with a limit, re-checks the code, runs the command as a child through SIGTERM."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import job_wrapper

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
    path = job_wrapper.job_cgroup("42", proc)
    assert path == job_wrapper.CGROUP_ROOT / "system.slice/gpu-node-1_slurmstepd.scope/job_42"
    assert job_wrapper.job_cgroup("43", proc) is None


CHILD = """
import signal, sys, time
def term(*_):
    time.sleep(1.0)  # "writing a checkpoint"
    open(sys.argv[1], "w").write("saved")
    sys.exit(0)
signal.signal(signal.SIGTERM, term)
open(sys.argv[2], "w").write("up")
time.sleep(30)
"""


def test_sigterm_waits_for_the_child_to_save_then_records_the_leg(tmp_path: Path) -> None:
    saved, up = tmp_path / "saved", tmp_path / "up"
    env = dict(os.environ, SLURM_GPU_STATE_DIR=str(tmp_path / "state"), SLURM_JOB_ID="77")
    wrapper = subprocess.Popen(
        [sys.executable, str(WRAPPER), "--sample", "0.2", "--", sys.executable, "-c", CHILD, str(saved), str(up)],
        env=env,
    )
    deadline = time.time() + 10
    while not up.exists() and time.time() < deadline:
        time.sleep(0.05)
    child = [int(p) for p in subprocess.check_output(["pgrep", "-P", str(wrapper.pid)]).split()]
    for pid in [wrapper.pid, *child]:  # slurm signals every process of the job, wrapper and child alike
        os.kill(pid, signal.SIGTERM)
    assert wrapper.wait(timeout=10) == 0
    assert saved.read_text() == "saved"
    record = json.loads((tmp_path / "state" / "jobs" / "77.json").read_text())
    assert record["last_leg"]["signals"] == ["SIGTERM"]
    assert record["last_leg"]["exit"] == 0
    assert (tmp_path / "state" / "footprints" / "77.jsonl").exists()


def test_an_unverifiable_manifest_refuses_the_launch(tmp_path: Path) -> None:
    env = dict(os.environ, SLURM_GPU_STATE_DIR=str(tmp_path / "state"), SLURM_JOB_ID="78")
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")  # empty: the parity checker refuses it, as it refuses a changed file
    done = subprocess.run(
        [sys.executable, str(WRAPPER), "--manifest", str(manifest), "--repo", str(tmp_path), "--", "true"],
        env=env, check=False,
    )
    assert done.returncode == job_wrapper.Exit.CODE_CHANGED
