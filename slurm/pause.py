#!/usr/bin/env python3
"""Pause a slurm job by hand: say what the pause loses, then requeue it held (design 3.7).

    slurm/pause.py JOBID [--lose-progress]

`scontrol requeuehold` requeues and holds in one step (a separate requeue then hold could start the job again
in between). A `normal` job resumes from its last checkpoint when released (`slurm/resume.py JOBID`); a
`steady` job cannot resume, so pausing one needs `--lose-progress`. Its dependents keep waiting.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gpu_jobs  # noqa: E402 - path set just above
from urgent import elapsed  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("job_id")
    parser.add_argument("--lose-progress", action="store_true", help="allow pausing a job that cannot resume")
    arguments = parser.parse_args()
    queue = gpu_jobs.read_queue() or []
    job = next((j for j in queue if j.job_id == arguments.job_id), None)
    if job is None:
        print(f"[pause] job {arguments.job_id} is not in the queue", file=sys.stderr)
        return 1
    record = gpu_jobs.read_sidecar(job.job_id) or {}
    print(f"[pause] {job.job_id} {job.name} ({record.get('what', 'no sidecar')}), {job.state}, run {elapsed(job.job_id)}")
    if job.partition == gpu_jobs.Partition.STEADY:
        print("[pause] a steady job cannot resume: a pause loses everything it has done so far")
        if not arguments.lose_progress:
            print("[pause] STOPPED: pass --lose-progress to pause it anyway", file=sys.stderr)
            return 1
    else:
        print("[pause] it resumes from its last checkpoint when released (slurm/resume.py)")
    done = subprocess.run(["scontrol", "requeuehold", job.job_id], check=False)
    return done.returncode


if __name__ == "__main__":
    sys.exit(main())
