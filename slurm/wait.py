#!/usr/bin/env python3
"""Wait until a slurm job and every job chained after it have finished; print their final states.

Design 3.8-3.9: watchers wait on slurm, not on systemd units. A pre-empted job is pending again, so it
never reads as done; a job cancelled because its dependency failed (`kill_invalid_depend`) leaves the
queue and is reported CANCELLED. Exit 0 when every job COMPLETED, 1 otherwise.

    slurm/wait.py JOBID [--poll 60] [--once]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gpu_jobs  # noqa: E402 - path set just above

COMPLETED = "COMPLETED"


def chain(job_id: str, sidecars: dict[str, dict[str, object]]) -> list[str]:
    """`job_id` and every job whose sidecar names it (transitively) in `after`."""
    found, frontier = [job_id], [job_id]
    while frontier:
        current = frontier.pop()
        for other, record in sidecars.items():
            if other not in found and current in (record.get("after") or []):
                found.append(other)
                frontier.append(other)
    return found


def all_sidecars(users: dict[str, str] | None = None) -> dict[str, dict[str, object]]:
    """Every record. `users` maps job id to its slurm user for the jobs the caller has queued: a record of one of those
    that another user wrote is ignored. The rest (finished jobs) are read as they are."""
    users = users or {}
    return {
        p.stem: gpu_jobs.read_json(p, users.get(p.stem)) or {} for p in (gpu_jobs.STATE_DIR / "jobs").glob("*.json")
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("job_id")
    parser.add_argument("--poll", type=float, default=60.0)
    parser.add_argument("--once", action="store_true", help="report now and exit 2 if anything is unfinished")
    arguments = parser.parse_args()
    while True:
        jobs = chain(arguments.job_id, all_sidecars())
        queue = gpu_jobs.read_queue()
        if queue is not None:
            unfinished = {job.job_id for job in queue} & set(jobs)
            if not unfinished:
                break
            if arguments.once:
                print(f"[wait] unfinished: {sorted(unfinished)}")
                return 2
        time.sleep(arguments.poll)
    records = gpu_jobs.completion_records(jobs) or {}
    states = {job: records.get(job, {}).get("JobState", "UNKNOWN") for job in jobs}
    for job, state in states.items():
        print(f"[wait] {job}: {state}")
    return 0 if all(state == COMPLETED for state in states.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
