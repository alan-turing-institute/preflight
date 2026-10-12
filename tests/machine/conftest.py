"""Make `machine/earlyoom/*.py` and `machine/flight_recorder/*.py` importable from here."""

from __future__ import annotations

import sys
from pathlib import Path

MACHINE: Path = Path(__file__).resolve().parents[2] / "machine"
for _sub in ("earlyoom", "flight_recorder"):
    if str(MACHINE / _sub) not in sys.path:
        sys.path.insert(0, str(MACHINE / _sub))
