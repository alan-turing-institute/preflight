#!/usr/bin/env python3
"""Wait until a slurm job and every job chained after it have finished; print their final states, each with the
usage report its last run left (D37: what it used, what to ask for next time; see `job_wrapper.usage_lines`).

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
            after = record.get("after")
            if other not in found and isinstance(after, list) and current in after:
                found.append(other)
                frontier.append(other)
    return found


def all_sidecars(users: dict[str, str] | None = None) -> dict[str, dict[str, object]]:
    """Every record. `users` maps job id to its slurm user for the jobs the caller has queued: a record of one of those
    that another user wrote is ignored. The rest (finished jobs) are read as they are. A file not named by a job id
    is no record: anyone can create one, and its name would reach the terminal."""
    users = users or {}
    return {
        p.stem: gpu_jobs.read_json(p, users.get(p.stem)) or {}
        for p in (gpu_jobs.STATE_DIR / "jobs").glob("*.json") if gpu_jobs.JOB_ID.fullmatch(p.stem)
    }


#: Longest report line `usage_lines` prints: the wrapper's are about 120 characters.
LINE_MAX = 300
#: Report lines per run (`job_wrapper.usage_lines`: what it used, what to ask for next time).
REPORT_LINES = 2


def usage_lines(job_id: str, user: str) -> list[str]:
    """The end-of-run report the job wrapper kept for the job's last leg (D37), [] when there is none. Untrusted input:
    only a record its slurm `user` wrote counts (no user from accounting: nothing, since the owner check would be
    skipped), and only a list of at most `REPORT_LINES` strings that each start with "[job] ", each made printable."""
    if not user:
        return []
    legs = gpu_jobs.record_legs(gpu_jobs.read_sidecar(job_id, user) or {})
    usage = legs[-1].get("usage") if legs else None
    lines = usage.get("lines") if isinstance(usage, dict) else None
    if not isinstance(lines, list) or len(lines) > REPORT_LINES or not all(isinstance(x, str) and x.startswith("[job] ") for x in lines):
        return []
    return [gpu_jobs.printable(line)[:LINE_MAX] for line in lines]


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
        for line in usage_lines(job, records.get(job, {}).get("UserId", "")):
            print(line)
    return 0 if all(state == COMPLETED for state in states.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
