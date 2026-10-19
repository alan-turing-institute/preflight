#!/usr/bin/env python3
"""A job's last measured footprint, for `submit.py --mem auto` and for an urgent job's stated needs (design 3.5).

    slurm/footprint.py NAME
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gpu_jobs  # noqa: E402 - path set just above


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("name", help="the job name given to submit.py")
    name = parser.parse_args().name
    peak = gpu_jobs.measured_peak_bytes(name)
    if peak is None:
        print(f"[footprint] no measured run called {name!r}")
        return 1
    print(f"[footprint] {name}: {peak / 1024**3:.1f} GiB at its peak (GPU plus CPU side, newest finished run)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
