#!/usr/bin/env python3
"""A job's last measured footprint, for `submit.py --mem auto` and for an urgent job's stated needs (design 3.5).

    slurm/footprint.py NAME
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gpu_jobs  # noqa: E402 - path set just above


def main() -> int:
    if len(sys.argv) != 2:  # noqa: PLR2004 - the program name and NAME
        print(__doc__.strip().splitlines()[-1].strip(), file=sys.stderr)
        return 2
    peak = gpu_jobs.measured_peak_bytes(sys.argv[1])
    if peak is None:
        print(f"[footprint] no measured run called {sys.argv[1]!r}")
        return 1
    print(f"[footprint] {sys.argv[1]}: {peak / 1024**3:.1f} GiB at its peak (GPU plus CPU side, newest finished run)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
