#!/usr/bin/env python3
"""Pause a slurm job by hand: say what the pause loses, then requeue it held (design 3.7).

    slurm/pause.py JOBID [--lose-progress] [--dry-run] [--reason TEXT]

`scontrol requeuehold` requeues and holds in one step (a separate requeue then hold could start the job again
in between). A `normal` job resumes from its last checkpoint when released (`slurm/resume.py JOBID`); a
`steady` job cannot resume, so pausing one needs `--lose-progress`. A pending job (not running) loses nothing:
slurm refuses to requeue a pending job (`ESLURM_JOB_PENDING`), so it gets `scontrol uhold`, a hold its owner may
release even when an admin set it. Its dependents keep waiting. `--dry-run`
prints only what the pause would lose and acts on nothing. Who may pause is slurm's rule (owner or admin):
its refusal is printed as it is. A successful pause is recorded as a PAUSE event.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gpu_jobs  # noqa: E402 - path set just above
from urgent import elapsed  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("job_id")
    parser.add_argument("--lose-progress", action="store_true", help="allow pausing a job that cannot resume")
    parser.add_argument("--dry-run", action="store_true", help="print only what the pause would lose; do nothing")
    parser.add_argument("--reason", default="", help="why, recorded with the event")
    arguments = parser.parse_args()
    queue = gpu_jobs.read_queue()
    if queue is None:
        print("[pause] slurm does not answer (controller down); see slurm/status.py", file=sys.stderr)
        return 1
    job = next((j for j in queue if j.job_id == arguments.job_id), None)
    if job is None:
        print(f"[pause] job {arguments.job_id} is not in the queue", file=sys.stderr)
        return 1
    pending = job.state == gpu_jobs.State.PENDING
    steady = job.partition == gpu_jobs.Partition.STEADY and not pending
    if pending:
        loss = "[pause] it is not running, so a pause loses nothing; it waits held until released (slurm/resume.py)"
    elif steady:
        loss = "[pause] a steady job cannot resume: a pause loses everything it has done so far"
    else:
        loss = "[pause] it resumes from its last checkpoint when released (slurm/resume.py)"
    if arguments.dry_run:
        print(loss)
        return 0
    record = gpu_jobs.read_sidecar(job.job_id, job.user) or {}
    print(f"[pause] {job.job_id} {job.name} ({record.get('what', 'no sidecar')}), {job.state}, run {elapsed(job.job_id)}")
    print(loss)
    if steady and not arguments.lose_progress:
        print("[pause] STOPPED: pass --lose-progress to pause it anyway", file=sys.stderr)
        return 1
    status = gpu_jobs.run_slurm(["scontrol", "uhold" if pending else "requeuehold", job.job_id])
    if status == 0:
        gpu_jobs.write_event(job.job_id, gpu_jobs.EventAction.PAUSE, reason=arguments.reason)
    return status


if __name__ == "__main__":
    sys.exit(main())
