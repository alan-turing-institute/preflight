#!/usr/bin/env python3
"""Print ONE line when the project under the working directory has an upgrade waiting or just applied.

    session_notice.py [CWD]            (default: the current directory; for a SessionStart hook)

`held`: an upgrade whose check moved numbers or failed, waiting for the project's study boundary. `applied` in the last
7 days: what changed and the commit. Nothing is printed for anything else, or when the directory is in no registered
project. Reads ~/.local/state/upgrades/<name>.json, written by weekly.py.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import weekly

RECENT = dt.timedelta(days=7)


def project_for(cwd: Path, projects: list[weekly.Project]) -> weekly.Project | None:
    """The registered project whose directory contains `cwd` (the deepest, should one contain another)."""
    cwd = cwd.resolve()
    inside = [p for p in projects if cwd == p.path or p.path in cwd.parents]
    return max(inside, key=lambda p: len(p.path.parts), default=None)


def notice(record: dict, now: dt.datetime) -> str | None:
    status = record.get("status")
    changed = ", ".join(f"{c['name']} {c['old']} -> {c['new']}" for c in record.get("changed", [])[:6])
    more = len(record.get("changed", [])) - 6
    changed += f" and {more} more" if more > 0 else ""
    if status == weekly.Status.HELD:
        return (
            f"Dependency upgrade HELD for {record['name']} ({record['date'][:10]}): {record['reason']}. "
            f"Waiting: {changed}. The upgraded lock is {weekly.CACHE / record['name'] / 'uv.lock.new'}; apply it at the study boundary."
        )
    if status == weekly.Status.APPLIED:
        when = dt.datetime.fromisoformat(record["date"])
        if now - when <= RECENT:
            return (
                f"Dependencies of {record['name']} upgraded on {record['date'][:10]} "
                f"(commit {record['commit']}): {changed}."
            )
    return None


def main() -> None:
    cwd = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    try:
        project = project_for(cwd, weekly.load_registry())
        if project is None:
            return
        record = json.loads((weekly.STATE / f"{project.name}.json").read_text())
        line = notice(record, dt.datetime.now(dt.UTC))
    except (OSError, ValueError, KeyError):
        return  # a session start must never fail on this
    if line:
        print(line)


if __name__ == "__main__":
    main()
