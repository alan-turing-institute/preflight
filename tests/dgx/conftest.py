"""Make `dgx/*.py` importable as plain modules from here.

These are standalone drivers, not an installed package, so tests need the path
entry the scripts give themselves when run directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

DGX: Path = Path(__file__).resolve().parents[2] / "dgx"
if str(DGX) not in sys.path:
    sys.path.insert(0, str(DGX))
