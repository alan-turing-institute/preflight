#!/usr/bin/env python3
"""What every GPU job is, how it pauses, and who holds the card (design 3.3, 3.8)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import agent  # noqa: E402 - path set just above
import audit  # noqa: E402
import gpu_jobs  # noqa: E402
import jobs_view  # noqa: E402
import remote  # noqa: E402
from wait import all_sidecars

#: How a job in each partition pauses, for the table.
PAUSES = {
    gpu_jobs.Partition.NORMAL: "pre-empted, resumes from its checkpoint",
    gpu_jobs.Partition.STEADY: "never pre-empted (a requeue redoes it)",
    gpu_jobs.Partition.CPU_FILL: "fills idle cores; requeued for GPU or urgent work, resumes from its checkpoint",
    gpu_jobs.Partition.URGENT: "urgent",
    gpu_jobs.Partition.SERVICE: "a service's CPU stand-in; requeued for urgent work",
    gpu_jobs.Partition.PART_TIME: "a part-time machine; requeued when its owner needs it, resumes from its checkpoint",
}
#: A job waiting in both (`gpu_jobs.ANYWHERE_PARTITIONS`; once running it is shown in the one it got).
ANYWHERE_PAUSE = "fills idle cores here or on a part-time machine; requeued for GPU or urgent work or when that machine's owner needs it, resumes from its checkpoint"


def _pauses(job: gpu_jobs.QueuedJob) -> str:
    """`PAUSES` by partition; a `cpu` job by its class (urgent.py requeues one that resumes, never one that restarts)."""
    if job.partition == gpu_jobs.Partition.CPU:
        return PAUSES[gpu_jobs.Partition.STEADY if job.restarts else gpu_jobs.Partition.NORMAL]
    if set(job.partition.split(",")) == {gpu_jobs.Partition.CPU_FILL, gpu_jobs.Partition.PART_TIME}:  # waiting: squeue lists both
        return ANYWHERE_PAUSE
    return PAUSES.get(job.partition, job.partition)


def lines(queue: list[gpu_jobs.QueuedJob] | None) -> list[str]:
    out = [f"[status] node: {gpu_jobs.node_state()}"]
    if gpu_jobs.LEASE_DIR is not None:  # a site without a [lease] table has no keeper to report
        keeper = gpu_jobs.read_json(gpu_jobs.keeper_state_path()) or {}
        out.append(
            f"[status] keeper: {keeper.get('phase', 'no state')}; blockers {keeper.get('blockers', [])};"
            f" updated {keeper.get('updated_at', '?')} {keeper.get('note', '')}".rstrip()
        )
    if queue is None:
        out.append("[status] squeue did not answer: the slurm controller is down")
        return out
    if not queue:
        out.append("[status] no jobs in slurm")
    records = all_sidecars({job.job_id: job.user for job in queue})
    by_id = {job.job_id: job for job in queue}
    events = gpu_jobs.events()
    for job in queue:
        record = records.get(job.job_id, {})
        why = jobs_view.waiting_reason(job, by_id, records, events)
        out.append(gpu_jobs.printable(  # names and descriptions are what their submitters typed
            f"[status] {job.job_id} {job.state:<10} {job.user or '?':<10} {job.level:<6} {f'{job.shards} shards' if job.shards else f'{job.gpus} GPU'} {job.name}"
            f"{f' on {job.node}' if job.node else ''}"
            f" | {record.get('mem') or 'default mem'}, {_pauses(job)}"
            f" | {record.get('what', '(no sidecar)')}{f' | {why}' if why else ''}"
        ))
    out += [f"[idle] {line}" for line in agent.idle_summary(agent.read_idle())]
    out += audit.report(audit.unregistered())
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--site", default=None, metavar="NAME", help="show another site's queue ([remote.NAME] in the site file)")
    arguments = parser.parse_args()
    if arguments.site:
        try:
            print(remote.status(gpu_jobs.load_site(), arguments.site))
        except remote.RemoteRefused as error:
            print(f"[status] REFUSED: {error}", file=sys.stderr)
            return 1
        return 0
    print("\n".join(lines(gpu_jobs.read_queue())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
