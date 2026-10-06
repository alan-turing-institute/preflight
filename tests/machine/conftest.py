"""Make `machine/earlyoom/*.py` importable as plain modules from here."""

from __future__ import annotations

import sys
from pathlib import Path

EARLYOOM: Path = Path(__file__).resolve().parents[2] / "machine" / "earlyoom"
if str(EARLYOOM) not in sys.path:
    sys.path.insert(0, str(EARLYOOM))
