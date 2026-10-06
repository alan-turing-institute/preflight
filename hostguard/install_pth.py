"""Load hostguard in EVERY Python a virtualenv runs, not only launched jobs.

A guard that relies on each launcher putting it on PYTHONPATH covers exactly the
launchers someone remembered. The 2026-10-06 13:51 hang came from a smoke test, and
smoke tests, scorers and ad-hoc scripts are the runs nobody wraps. A `.pth` line in
the venv's site-packages runs at every interpreter start, so the guard is on for
all of them. It survives `uv sync --frozen`, `--reinstall` and `uv remove`
(measured 2026-10-06); recreating the venv drops it, so re-run this after one.

    python3 install_pth.py /path/to/project/.venv [...]   # idempotent
    python3 install_pth.py --check /path/to/.venv [...]   # exit 1 if any lacks it
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PTH_NAME = "hostguard.pth"


def pth_line(guard_dir: Path = HERE) -> str:
    # `import` lines in a .pth are executed by site.py at interpreter start.
    return f"import sys; sys.path.append({str(guard_dir)!r}); import hostguard\n"


def site_packages(venv: Path) -> Path:
    found = sorted(venv.glob("lib/python*/site-packages"))
    if not found:
        msg = f"{venv}: no lib/python*/site-packages -- not a virtualenv?"
        raise SystemExit(msg)
    return found[-1]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("venvs", type=Path, nargs="+")
    ap.add_argument("--check", action="store_true", help="report only; exit 1 if missing")
    args = ap.parse_args(argv)
    missing = 0
    for venv in args.venvs:
        target = site_packages(venv) / PTH_NAME
        current = target.read_text() if target.exists() else None
        if current == pth_line():
            print(f"ok       {target}")
        elif args.check:
            print(f"MISSING  {target}")
            missing += 1
        else:
            target.write_text(pth_line())
            print(f"written  {target}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
