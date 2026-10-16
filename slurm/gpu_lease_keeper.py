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
  the last active one ending, or of first being seen with nothing active. A job held by a person, or one
  pending past the linger while nothing of slurm's runs (waiting on a CPU step, say), is not demand.
* demand and no lease: write the lease with `requested_at` (services start yielding); wait by the fair
  rule (`GpuLease.blockers`: holders block, an earlier waiter goes first, our request withdrawn while one
  exists); when nothing blocks, mark it admitted (a holder). Admitted, it does not yield.
* no demand: remove the lease.
* `squeue` failing (controller down): keep the last state; never release the card under a job it cannot see.

On start it deletes lease files written before this boot (their pids belong to the previous boot and may
be reused) and reports the node's state, since `ReturnToService=1` leaves it DOWN after a reboot.
State for the other tools: `~/.local/state/slurm-gpu/keeper.json`.

Run by `machine/slurm/slurm-gpu-lease-keeper.service` under `~/dev/localmodels/.venv/bin/python`.
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

from localmodels.lease import GpuLease, LeaseRecord  # noqa: E402


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

    def demand(self, queue: list[gpu_jobs.QueuedJob], now: float) -> bool:
        relevant = [job for job in queue if self.needs_the_card(job)]
        self.jobs = [f"{job.job_id}:{job.state}" for job in relevant]
        if any(job.active for job in relevant):
            self.anchor = now
            return True
        if not any(job.waiting for job in relevant):
            self.anchor = None
            return False
        if self.anchor is None:
            self.anchor = now
        return now - self.anchor <= self.linger_s

    def _write(self, **fields: int | str | None) -> None:
        self.lease.write(LeaseRecord(pid=os.getpid(), project=self.lease.project, job_id=",".join(self.jobs), **fields))

    def step(self) -> dict[str, object]:
        now = self.clock()
        queue = self.read_queue()
        if queue is None:
            self.note = "squeue did not answer; keeping the last state"
            return self.state(now)
        self.note = ""
        if not self.demand(queue, now):
            if self.phase != Phase.NONE:
                self.lease.remove()
            self.phase, self.requested_at, self.admitted_at, self.blockers = Phase.NONE, None, None, []
            return self.state(now)
        if self.phase == Phase.NONE:
            self.phase, self.requested_at = Phase.WAITING, int(now)
            self._write(requested_at=self.requested_at)
        if self.phase == Phase.WAITING:
            blocking, earlier_waiter = self.lease.blockers(self.requested_at)
            self.blockers = sorted(record.project for record in blocking)
            if not blocking:
                self.phase, self.admitted_at = Phase.ADMITTED, int(now)
                self._write(started_at=self.admitted_at)
            elif earlier_waiter:
                self.lease.remove()  # so the earlier waiter is not blocked by us
            elif not self.lease.lease_path().exists():
                self._write(requested_at=self.requested_at)  # re-published with the original time
        elif not self.lease.lease_path().exists():
            self._write(started_at=self.admitted_at)  # someone deleted it; a holder keeps holding
        return self.state(now)

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

    gpu_jobs.LEASE_DIR.mkdir(parents=True, exist_ok=True)
    removed = remove_leases_from_before_boot(gpu_jobs.LEASE_DIR, gpu_jobs.boot_time())
    print(f"[keeper] start; removed leases from before this boot: {removed or 'none'}", flush=True)
    print(f"[keeper] node state: {gpu_jobs.node_state()}", flush=True)
    keeper = Keeper(GpuLease(gpu_jobs.LEASE_DIR, gpu_jobs.LEASE_PROJECT), linger_s=arguments.linger)
    state_path = gpu_jobs.STATE_DIR / "keeper.json"
    last = None
    # No cleanup on exit: a lease left by a dead keeper names a dead pid, which every reader ignores,
    # while removing it at a restart would open exactly the gap this keeper exists to close.
    while True:
        state = keeper.step()
        summary = (state["phase"], tuple(state["blockers"]), state["note"])
        if summary != last:
            print(f"[keeper] {state['phase']}; jobs {state['jobs']}; blockers {state['blockers']} {state['note']}",
                  flush=True)
            last = summary
        gpu_jobs.write_json(state_path, state)
        time.sleep(arguments.interval)


if __name__ == "__main__":
    sys.exit(main())
