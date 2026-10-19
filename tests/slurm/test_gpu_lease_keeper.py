"""The lease keeper's decisions, against a fake queue, a fake clock and a real lease directory."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import gpu_jobs
import pytest
import submit
from gpu_lease_keeper import Keeper, Phase, beat, remove_leases_from_before_boot
from localmodels.lease import GpuLease


def job(job_id: str, state: str = "RUNNING", reason: str = "None", dependency: str = "") -> gpu_jobs.QueuedJob:
    return gpu_jobs.QueuedJob(job_id, "normal", state, reason, f"p:resume:{job_id}", dependency)


class World:
    def __init__(self, lease_dir: Path) -> None:
        self.queue: list[gpu_jobs.QueuedJob] | None = []
        self.t = 1000.0
        self.lease_dir = lease_dir
        self.keeper = Keeper(
            GpuLease(lease_dir, "slurm"),
            linger_s=120,
            read_queue=lambda: self.queue,
            needs_the_card=lambda _job: True,
            clock=lambda: self.t,
        )

    def step(self) -> dict[str, object]:
        return self.keeper.step()

    def lease(self) -> dict[str, object] | None:
        path = self.lease_dir / "slurm.json"
        return json.loads(path.read_text()) if path.exists() else None


@pytest.fixture
def other():
    """A live process to stand for another project's lease holder."""
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    yield proc.pid
    proc.terminate()
    proc.wait()


def write_other(lease_dir: Path, project: str, pid: int, **fields: int) -> None:
    (lease_dir / f"{project}.json").write_text(json.dumps({"pid": pid, "project": project, **fields}))


def test_no_jobs_no_lease(tmp_path: Path) -> None:
    world = World(tmp_path)
    assert world.step()["phase"] == Phase.NONE
    assert world.lease() is None


def test_active_gpu_job_with_a_free_card_is_admitted_at_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpu_jobs, "STATE_DIR", tmp_path / "state")
    world = World(tmp_path)
    world.queue = [job("1")]
    state = beat(world.keeper, gpu_jobs.keeper_state_path())
    assert submit.keeper_running(now=world.t + 5)  # what the keeper writes is what every submitter reads as alive
    assert not submit.keeper_running(now=world.t + 10_000)
    assert state["phase"] == Phase.ADMITTED
    lease = world.lease()
    assert lease["pid"] == os.getpid()
    assert "requested_at" not in lease
    assert "started_at" in lease


def test_a_holder_makes_it_wait_with_a_request_until_it_goes(tmp_path: Path, other: int) -> None:
    write_other(tmp_path, "service-a", other)
    world = World(tmp_path)
    world.queue = [job("1")]
    state = world.step()
    assert state["phase"] == Phase.WAITING
    assert state["blockers"] == ["service-a"]
    assert "requested_at" in world.lease()  # what makes the service yield
    (tmp_path / "service-a.json").unlink()  # the service yielded
    world.t += 5
    assert world.step()["phase"] == Phase.ADMITTED
    assert "requested_at" not in world.lease()


def test_an_earlier_waiter_goes_first_and_ours_is_withdrawn_meanwhile(tmp_path: Path, other: int) -> None:
    write_other(tmp_path, "service-c", other, requested_at=500)
    world = World(tmp_path)
    world.queue = [job("1")]
    state = world.step()
    assert state["phase"] == Phase.WAITING
    requested_at = state["requested_at"]
    assert world.lease() is None  # withdrawn: service-c treats every other lease as busy
    write_other(tmp_path, "service-c", other)  # service-c admitted: now a holder
    world.t += 5
    world.step()
    assert world.lease()["requested_at"] == requested_at  # re-published with the original time
    (tmp_path / "service-c.json").unlink()
    world.t += 5
    assert world.step()["phase"] == Phase.ADMITTED


def test_a_later_waiter_does_not_block_us(tmp_path: Path, other: int) -> None:
    write_other(tmp_path, "service-c", other, requested_at=int(time.time()) + 3600)  # the client stamps real time
    world = World(tmp_path)
    world.queue = [job("1")]
    assert world.step()["phase"] == Phase.ADMITTED


def test_the_lease_is_held_across_a_chain_hand_off_then_released_after_the_linger(tmp_path: Path) -> None:
    world = World(tmp_path)
    world.queue = [job("1"), job("2", "PENDING", "Dependency")]
    world.step()
    for t, queue in [(1030, [job("2", "PENDING", "Dependency")]), (1100, [job("2", "PENDING", "BeginTime")])]:
        world.t, world.queue = t, queue
        assert world.step()["phase"] == Phase.ADMITTED
        assert world.lease() is not None  # no gap for a service to reload into
    world.t, world.queue = 1160, [job("2", "PENDING", "Dependency")]  # 130 s with nothing active
    assert world.step()["phase"] == Phase.NONE
    assert world.lease() is None
    world.t, world.queue = 2000, [job("2")]
    assert world.step()["phase"] == Phase.ADMITTED


def test_a_held_job_is_not_demand_nor_is_a_chain_waiting_only_on_it(tmp_path: Path) -> None:
    world = World(tmp_path)
    world.queue = [job("1", "PENDING", "JobHeldUser")]
    assert world.step()["phase"] == Phase.NONE
    # a job paused by hand with its next step waiting on it (and one after that): nothing can start, so no lease
    world.queue = [
        job("1", "PENDING", "job requeued in held state"),
        job("2", "PENDING", "Dependency", "afterok:1(unfulfilled)"),
        job("3", "PENDING", "Dependency", "afterok:2(unfulfilled)"),
    ]
    assert world.step()["phase"] == Phase.NONE
    world.queue[0] = job("1", "PENDING", "Resources")  # released: the chain can move again
    assert world.step()["phase"] == Phase.ADMITTED


def test_a_silent_controller_keeps_the_last_state(tmp_path: Path) -> None:
    world = World(tmp_path)
    world.queue = [job("1")]
    world.step()
    world.queue = None
    state = world.step()
    assert state["phase"] == Phase.ADMITTED
    assert "squeue did not answer" in state["note"]
    assert world.lease() is not None


def test_a_deleted_lease_is_rewritten_while_admitted(tmp_path: Path) -> None:
    world = World(tmp_path)
    world.queue = [job("1")]
    admitted_at = world.step()["admitted_at"]
    (tmp_path / "slurm.json").unlink()
    state = world.step()
    assert "requested_at" not in world.lease()
    assert state["admitted_at"] == admitted_at


def test_only_leases_from_before_the_boot_are_removed(tmp_path: Path) -> None:
    old, new = tmp_path / "old.json", tmp_path / "new.json"
    old.write_text("{}")
    new.write_text("{}")
    booted = time.time() - 100
    os.utime(old, (booted - 50, booted - 50))
    assert remove_leases_from_before_boot(tmp_path, booted) == ["old.json"]
    assert new.exists()
