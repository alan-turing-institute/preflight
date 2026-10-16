"""Make `slurm/*.py` importable as plain modules from here (standalone drivers, not a package)."""

from __future__ import annotations

import sys
from pathlib import Path

SLURM: Path = Path(__file__).resolve().parents[2] / "slurm"
if str(SLURM) not in sys.path:
    sys.path.insert(0, str(SLURM))
