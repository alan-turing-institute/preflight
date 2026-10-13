#!/usr/bin/env python3
"""Freeze a job queue (SIGSTOP) the moment its log says it is moving on, so the next job never starts.

The case (2026-10-13, project-a's shallow-water queue on gpu-node-1): the owner
wanted the current cell to finish and be scored, but the NEXT 24-h cell held until he
decides which box runs it. The queue logs a line when a cell ends, starts that cell's
scoring as a child process, then checks the card and launches the next cell, all within
about a second. So this follows the log from its end; at the first new line containing
``--line`` it waits (at most ``--settle`` s) until the queue has a child matching any
``--await-child`` (the scoring), freezes the queue, then SIGKILLs any child matching
``--kill-child`` (the card check in flight, so a resumed queue re-checks the card instead
of acting on a check made hours earlier). Children keep running: SIGSTOP stops one
process, not its children. Resume with ``kill -CONT <pid>``.

It prints the queue's children at the moment of the hold. If the next job had already
started, it says so (``--late-child``) and still holds; deciding what to do then is a
person's job. Stdlib only; polls the log every ``--poll`` s (default 0.02).

    jobs/hold_at_line.py --pid <queue pid> --log <queue log> --line '<substring>' \
        [--await-child '<substring>']... [--kill-child '<substring>']... \
        [--late-child '<substring>']... [--settle 5]
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from enum import IntEnum
from pathlib import Path


class Exit(IntEnum):
    """How the hold ended."""

    HELD = 0
    LATE = 2  # held, but the next job had already started


def log(message: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}", flush=True)


def children(pid: int, proc: Path = Path("/proc")) -> dict[int, str]:
    """{child pid: command line} of ``pid``'s children (a zombie's command line is empty)."""
    out = {}
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text()
            fields = stat[stat.rindex(")") + 2:].split()
            if int(fields[1]) == pid:
                out[int(entry.name)] = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        except (OSError, ValueError, IndexError):
            continue
    return out


def matching(kids: dict[int, str], needles: list[str]) -> list[int]:
    return [pid for pid, cmd in kids.items() if any(n in cmd for n in needles)]


def wait_for_line(path: Path, needle: str, poll: float) -> str:
    """Follow ``path`` from its current end; return the first new line containing ``needle``."""
    with open(path, "rb") as stream:
        stream.seek(0, os.SEEK_END)
        pending = b""
        while True:
            chunk = stream.read()
            if not chunk:
                time.sleep(poll)
                continue
            pending += chunk
            *lines, pending = pending.split(b"\n")
            for raw in lines:
                line = raw.decode(errors="replace")
                if needle in line:
                    return line


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--line", required=True)
    parser.add_argument("--await-child", action="append", default=[])
    parser.add_argument("--kill-child", action="append", default=[])
    parser.add_argument("--late-child", action="append", default=[])
    parser.add_argument("--settle", type=float, default=5.0)
    parser.add_argument("--poll", type=float, default=0.02)
    args = parser.parse_args(argv)
    log(f"armed: holding pid {args.pid} at the first new line of {args.log} containing {args.line!r}")
    line = wait_for_line(args.log, args.line, args.poll)
    log(f"line seen: {line.strip()}")
    deadline = time.monotonic() + args.settle
    while args.await_child and time.monotonic() < deadline:
        if matching(children(args.pid), args.await_child):
            break
        time.sleep(args.poll)
    os.kill(args.pid, signal.SIGSTOP)
    kids = children(args.pid)
    log(f"HELD pid {args.pid} (SIGSTOP); its children: {kids or 'none'}")
    for pid in matching(kids, args.kill_child):
        os.kill(pid, signal.SIGKILL)
        log(f"killed child {pid}: {kids[pid]}")
    late = matching(kids, args.late_child)
    if late:
        log(f"LATE: the next job had already started: {[kids[p] for p in late]}")
        return Exit.LATE
    log(f"resume with: kill -CONT {args.pid}")
    return Exit.HELD


if __name__ == "__main__":
    sys.exit(main())
