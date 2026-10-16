#!/usr/bin/env python3
"""What every GPU job on gpu-node-1 is, how it pauses, and who holds the card (design 3.3, 3.8)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import audit  # noqa: E402 - path set just above
import gpu_jobs  # noqa: E402

#: How a job in each partition pauses, for the table.
PAUSES = {
    gpu_jobs.Partition.NORMAL: "pre-empted, resumes from its checkpoint",
    gpu_jobs.Partition.STEADY: "never pre-empted (a requeue redoes it)",
    gpu_jobs.Partition.URGENT: "urgent",
}


def lines(queue: list[gpu_jobs.QueuedJob] | None) -> list[str]:
    out = [f"[status] node: {gpu_jobs.node_state()}"]
    keeper = gpu_jobs.read_json(gpu_jobs.STATE_DIR / "keeper.json") or {}
    out.append(
        f"[status] keeper: {keeper.get('phase', 'no state')}; blockers {keeper.get('blockers', [])};"
        f" updated {keeper.get('updated_at', '?')} {keeper.get('note', '')}".rstrip()
    )
    if queue is None:
        out.append("[status] squeue did not answer: the slurm controller is down")
        return out
    if not queue:
        out.append("[status] no jobs in slurm")
    for job in queue:
        record = gpu_jobs.read_sidecar(job.job_id) or {}
        held = " <- HELD after repeated requeues: look, then slurm/resume.py" if job.reason == "JobHeldAdmin" else ""
        out.append(
            f"[status] {job.job_id} {job.state:<10} {job.reason:<22} {job.name}"
            f" | {'GPU' if record.get('gpu') else 'CPU'}, {record.get('mem') or 'default mem'},"
            f" {PAUSES.get(job.partition, job.partition)} | {record.get('what', '(no sidecar)')}{held}"
        )
    out += audit.report(audit.unregistered())
    return out


def main() -> int:
    print("\n".join(lines(gpu_jobs.read_queue())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
