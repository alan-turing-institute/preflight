#!/usr/bin/env python3
"""Hold the cross-project GPU lease for slurm as a whole, so the background services step aside.

Design 3.2 (`design notes`). Revision 1 had each job take
the lease itself; the review found three defects with one cause, a job's life being shorter than slurm's
need for the card: a deadlock with service-c's queue (which treats every other lease as busy), the lease
released while a pre-empted job was still writing its checkpoint, and a gap at every chain step and
pre-emption in which service-a reloads its ~21 GB model. So this one long-lived process holds the
`slurm` lease, and jobs only read it (`job_wrapper.py`).

Every `--interval` seconds:

* DEMAND: a GPU job or an urgent job is active, or one is pending (not held) within `--linger` seconds of
  the last active one ending, or of first being seen with nothing active. A job held by a person, one
  pending only on held jobs (its chain cannot move until someone releases them: `gpu_jobs.held_back`), or one
  pending past the linger while nothing of slurm's runs (waiting on a CPU step, say), is not demand.
* demand and no lease: write the lease with `requested_at` (services start yielding); wait by the fair
  rule (`GpuLease.blockers`: holders block, an earlier waiter goes first, our request withdrawn while one
  exists); when nothing blocks, mark it admitted (a holder). Admitted, it does not yield.
* no demand: remove the lease.
* `squeue` failing (controller down): keep the last state; never release the card under a job it cannot see.

On start it deletes lease files written before this boot (their pids belong to the previous boot and may
be reused) and reports the node's state, since `ReturnToService=1` leaves it DOWN after a reboot.
State for the other tools, rewritten every loop: `gpu_jobs.keeper_state_path()` (`STATE_DIR/keeper/keeper.json`).

Run by the `slurm-gpu-lease-keeper.service` user unit (`slurm/install/templates/lease/`) under the site's lease python.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gpu_jobs  # noqa: E402 - path set just above

from localmodels.lease import FreezeError, GpuLease, GpuLeaseClient, LeaseRecord  # noqa: E402


class Phase(StrEnum):
    NONE = "none"
    WAITING = "waiting"
    ADMITTED = "admitted"


class Keeper:
    """One step at a time, so the decisions can be tested with a fake queue and a fake clock."""

    def __init__(
        self,
        lease: GpuLease,
        *,
        linger_s: float,
        read_queue: Callable[[], list[gpu_jobs.QueuedJob] | None] = gpu_jobs.read_queue,
        needs_the_card: Callable[[gpu_jobs.QueuedJob], bool] = gpu_jobs.needs_the_card,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.lease = lease
        self.client = GpuLeaseClient(lease)
        self.linger_s = linger_s
        self.read_queue = read_queue
        self.needs_the_card = needs_the_card
        self.clock = clock
        self.phase = Phase.NONE
        self.requested_at: int | None = None
        self.admitted_at: int | None = None
        self.anchor: float | None = None  # last moment slurm had GPU work active, or a waiter first seen
        self.blockers: list[str] = []
        self.jobs: list[str] = []
        self.note = ""
        self.last_queue: list[gpu_jobs.QueuedJob] | None = None

    def demand(self, queue: list[gpu_jobs.QueuedJob], now: float) -> bool:
        relevant = [job for job in queue if self.needs_the_card(job)]
        self.jobs = [f"{job.job_id}:{job.state}" for job in relevant]
        if any(job.active for job in relevant):
            self.anchor = now
            return True
        jobs = {job.job_id: job for job in queue}
        if not any(job.waiting and not gpu_jobs.held_back(job, jobs) for job in relevant):
            self.anchor = None
            return False
        if self.anchor is None:
            self.anchor = now
        return now - self.anchor <= self.linger_s

    def _job_id(self) -> str:
        return ",".join(self.jobs)

    def step(self) -> dict[str, object]:
        now = self.clock()
        queue = self.last_queue = self.read_queue()
        if queue is None:
            self.note = "squeue did not answer; keeping the last state"
            return self.state(now)
        self.note = ""
        if not self.demand(queue, now):
            if self.phase != Phase.NONE:
                self.client.release()
            self.phase, self.requested_at, self.admitted_at, self.blockers = Phase.NONE, None, None, []
            return self.state(now)
        if self.phase == Phase.NONE:
            self.client.begin_wait(job_id=self._job_id())
            self.phase, self.requested_at = Phase.WAITING, self._own().requested_at
        if self.phase == Phase.WAITING:
            blocking = self.client.step()
            self.blockers = sorted(record.project for record in blocking)
            if not blocking:
                self.client.admit(job_id=self._job_id())
                self.phase, self.admitted_at = Phase.ADMITTED, self._own().started_at
        elif not self.lease.lease_path().exists():
            self.client.admit(job_id=self._job_id())  # someone deleted it; a holder keeps holding
        return self.state(now)

    def _own(self) -> LeaseRecord:
        """What the client just wrote, so keeper.json carries the lease's own times."""
        record = self.lease.read_own()
        assert record is not None, "the lease just written is unreadable"
        return record

    def state(self, now: float) -> dict[str, object]:
        return {
            "phase": str(self.phase),
            "requested_at": self.requested_at,
            "admitted_at": self.admitted_at,
            "blockers": self.blockers,
            "jobs": self.jobs,
            "note": self.note,
            "keeper_pid": os.getpid(),
            "updated_at": int(now),
        }


def remove_leases_from_before_boot(lease_dir: Path, booted_at: float) -> list[str]:
    """Delete lease files last written before this boot; their pids may now name unrelated processes."""
    removed = []
    for path in lease_dir.glob("*.json"):
        try:
            stale = path.stat().st_mtime < booted_at
        except OSError:
            continue
        if stale:
            gpu_jobs.remove_quietly(path)
            removed.append(path.name)
    return removed


def beat(keeper: Keeper, state_path: Path) -> dict[str, object]:
    """One look at the queue, and the state file `submit.keeper_running` reads as the heartbeat.

    First localmodels' safety net wakes any service frozen by a requester that died or held it too long;
    it runs before the step, which returns early when squeue does not answer.
    """
    try:
        woken = [record.unit for record in keeper.lease.wake_abandoned_freezes()]
    except FreezeError as exc:  # every record was tried; uncaught it would end the keeper's loop
        woken, failed = [], str(exc)
    else:
        failed = ""
    state = keeper.step()
    state["woken"], state["wake_failed"] = woken, failed
    gpu_jobs.write_json(state_path, state)
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--interval", type=float, default=5.0, help="seconds between looks at the queue")
    parser.add_argument(
        "--linger",
        type=float,
        default=120.0,
        help="seconds a pending GPU job keeps the lease with nothing active (the 60-s requeue wait plus a minute)",
    )
    arguments = parser.parse_args()

    if gpu_jobs.LEASE_DIR is None:
        print("[keeper] REFUSED: the site file has no [lease] table, so there is no lease to keep", file=sys.stderr)
        return 1
    gpu_jobs.LEASE_DIR.mkdir(parents=True, exist_ok=True)
    removed = remove_leases_from_before_boot(gpu_jobs.LEASE_DIR, gpu_jobs.boot_time())
    print(f"[keeper] start; removed leases from before this boot: {removed or 'none'}", flush=True)
    print(f"[keeper] node state: {gpu_jobs.node_state()}", flush=True)
    keeper = Keeper(GpuLease(gpu_jobs.LEASE_DIR, gpu_jobs.LEASE_PROJECT), linger_s=arguments.linger)
    state_path = gpu_jobs.keeper_state_path()
    last = None
    # No cleanup on exit: a lease left by a dead keeper names a dead pid, which every reader ignores,
    # while removing it at a restart would open exactly the gap this keeper exists to close.
    while True:
        state = beat(keeper, state_path)
        if state["woken"] or state["wake_failed"]:
            print(f"[keeper] abandoned freezes: woke {state['woken']} {state['wake_failed']}", flush=True)
        summary = (state["phase"], tuple(state["blockers"]), state["note"])
        if summary != last:
            print(f"[keeper] {state['phase']}; jobs {state['jobs']}; blockers {state['blockers']} {state['note']}",
                  flush=True)
            last = summary
        time.sleep(arguments.interval)


if __name__ == "__main__":
    sys.exit(main())
