#!/usr/bin/env python3
"""Release a paused or held slurm job, or return the node to service after a reboot (design 3.7-3.8).

    slurm/resume.py JOBID...     # scontrol release (a job paused by pause.py, or held by the requeue limit); the
                                 # job waits at the back until the agent gives it its place again (D18)
    slurm/resume.py --node [NAME]  # after a reboot: ReturnToService=1 leaves a node DOWN until this (sudo);
                                   # without NAME, every node slurm knows
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gpu_jobs  # noqa: E402 - path set just above


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("job_ids", nargs="*")
    parser.add_argument("--node", nargs="?", const="", metavar="NAME", help="return a node (default: all) to service (sudo)")
    arguments = parser.parse_args()
    if arguments.node is not None:
        names = [arguments.node] if arguments.node else gpu_jobs.nodes()
        if not names:
            print("[resume] slurm names no nodes", file=sys.stderr)
            return 1
        print(f"[resume] node before: {gpu_jobs.node_state()}")
        status = 0
        for name in names:
            status = gpu_jobs.run_slurm(["sudo", "scontrol", "update", f"nodename={name}", "state=resume"]) or status
        print(f"[resume] node after: {gpu_jobs.node_state()}")
        return status
    if not arguments.job_ids:
        parser.error("name a job, or --node")
    status = gpu_jobs.run_slurm(["scontrol", "release", ",".join(arguments.job_ids)])
    if status == 0:
        for job_id in arguments.job_ids:
            gpu_jobs.write_event(job_id, gpu_jobs.EventAction.RESUME)
    return status


if __name__ == "__main__":
    sys.exit(main())
