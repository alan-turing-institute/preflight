"""Gate leg 4: the box's venv loads the SAME memory guard at every interpreter start.

Piped to the box and run by the PROJECT's own venv python with PYTHONPATH cleared, so
what it sees in sys.modules is exactly what any python started from that venv gets --
a smoke test, a scorer, an ad-hoc script, not only the launched job. Two refusals:

* the guard was not loaded at startup: the venv has no hostguard.pth (a recreated venv
  drops it). Every unwrapped run on it is then uncapped -- the 2026-10-06 13:51 hang
  on gpu-node-1 came from exactly such a run.
* the guard it loaded differs from preflight's hostguard/hostguard.py: the box runs a
  stale deployed copy (Turing runs an rsynced one).

    <venv>/bin/python - --expect-sha <sha256 of preflight's hostguard.py> < assert_hostguard.py
"""

from __future__ import annotations

import argparse
import hashlib
import sys


def verdict(loaded_file: str | None, expected_sha: str) -> tuple[bool, str]:
    if loaded_file is None:
        return False, (
            "this venv did not load hostguard at startup (no hostguard.pth). Fix on the box:\n"
            "  python3 <guard dir>/install_pth.py <project>/.venv"
        )
    with open(loaded_file, "rb") as fh:
        sha = hashlib.sha256(fh.read()).hexdigest()
    if sha != expected_sha:
        return False, (
            f"the guard this venv loads ({loaded_file}) is a stale copy: it differs from\n"
            "preflight's hostguard/hostguard.py. Re-copy preflight/hostguard/ to the box."
        )
    return True, f"hostguard loaded at startup from {loaded_file}, identical to preflight's"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--expect-sha", required=True)
    args = ap.parse_args(argv)
    module = sys.modules.get("hostguard")
    ok, msg = verdict(getattr(module, "__file__", None), args.expect_sha)
    print(f"[hostguard] {'OK -- ' if ok else 'REFUSED: '}{msg}", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
