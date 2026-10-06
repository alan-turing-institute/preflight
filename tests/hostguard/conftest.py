"""Make `hostguard/hostguard.py` importable as a plain module from here."""

from __future__ import annotations

import sys
from pathlib import Path

HOSTGUARD: Path = Path(__file__).resolve().parents[2] / "hostguard"
if str(HOSTGUARD) not in sys.path:
    sys.path.insert(0, str(HOSTGUARD))
