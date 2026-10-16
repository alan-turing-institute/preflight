#!/usr/bin/env python3
"""Release a paused or held slurm job, or return the node to service after a reboot (design 3.7-3.8).

    slurm/resume.py JOBID...     # scontrol release (a job paused by pause.py, or held by the requeue limit)
    slurm/resume.py --node       # after a reboot: ReturnToService=1 leaves the node DOWN until this (sudo)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gpu_jobs  # noqa: E402 - path set just above


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("job_ids", nargs="*")
    parser.add_argument("--node", action="store_true", help="return gpu-node-1 to service (needs sudo)")
    arguments = parser.parse_args()
    if arguments.node:
        print(f"[resume] node before: {gpu_jobs.node_state()}")
        done = subprocess.run(["sudo", "scontrol", "update", "nodename=gpu-node-1", "state=resume"], check=False)
        print(f"[resume] node after: {gpu_jobs.node_state()}")
        return done.returncode
    if not arguments.job_ids:
        parser.error("name a job, or --node")
    return subprocess.run(["scontrol", "release", ",".join(arguments.job_ids)], check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
