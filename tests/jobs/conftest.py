"""Make `jobs/*.py` importable as plain modules from here.

These are standalone drivers, not an installed package, so tests need the path
entry the scripts give themselves when run directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

JOBS: Path = Path(__file__).resolve().parents[2] / "jobs"
if str(JOBS) not in sys.path:
    sys.path.insert(0, str(JOBS))
