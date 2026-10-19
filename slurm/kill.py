#!/usr/bin/env python3
"""Cancel a slurm job by hand and record it.

    slurm/kill.py JOBID [--reason TEXT]

`scancel` cancels it for good (a pause is `slurm/pause.py`). Who may cancel is slurm's rule (owner or admin):
its refusal is printed as it is. A successful cancel is recorded as a KILL event.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gpu_jobs  # noqa: E402 - path set just above
from pause import run_slurm  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("job_id")
    parser.add_argument("--reason", default="", help="why, recorded with the event")
    arguments = parser.parse_args()
    status = run_slurm(["scancel", arguments.job_id])
    if status == 0:
        gpu_jobs.write_event(arguments.job_id, gpu_jobs.EventAction.KILL, reason=arguments.reason)
        print(f"[kill] {arguments.job_id} cancelled")
    return status


if __name__ == "__main__":
    sys.exit(main())
