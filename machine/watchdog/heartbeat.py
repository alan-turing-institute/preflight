#!/usr/bin/python3
"""Reboot a GB10 box that has wedged, instead of leaving it hung until someone power-cycles it.

WHY
---
The Turing DGX Spark hung silently four times (2026-09-25, 10-03, 10-07, 10-10) and stayed
down up to 37 h. Its 60-s hardware watchdog was armed through the 10-10 hang and never
fired, and other GB10 owners report the same ("it has never fired",
NVIDIA/open-gpu-kernel-modules#1358). One of them recovered automatically in ~10 min with a
systemd SERVICE watchdog instead: a heartbeat that systemd (PID 1) must hear from, else it
reboots. That works whenever PID 1 still runs during the hang, which the 35 h of an
unfired hardware watchdog (petted by PID 1) suggests it did, though nothing proves it.

WHAT
----
Runs as ``preflight-heartbeat.service`` (Type=notify, WatchdogSec=360,
FailureAction=reboot-immediate, written by ``machine/install_hang_guards.sh``). Every
``--interval`` s it tells systemd it is alive, unless memory has been stalled: the share of
time in which ALL non-idle tasks waited on memory (/proc/pressure/memory "full") has been at
least ``--threshold-pct`` in every interval for ``--hold`` s. Then it stops reporting, logs
why, and writes the reason to /var/lib/preflight-heartbeat/withheld.json (fsynced, so it
survives the reboot). If the box is too far gone to run this process at all, the reports
stop by themselves. Either way systemd reboots WatchdogSec after the last report; if the
stall clears first, reporting resumes and nothing happens.

Defaults are the reporter's (>= 10 % for 300 s). Our box's 1-s flight recorder over 23 h of
GPU training (2026-10-12/13) never held >= 10 % for more than 18 s in a row, a 16x margin.
Idle cost: one 128-byte ``pread`` and one datagram per interval.

It renames itself ``pf-heartbeat`` so earlyoom's ``--prefer python`` rule, which overrides a
negative oom_score_adj (earlyoom(1)), can never choose it: killing it would reboot the box.
Any error inside the loop is logged and the loop carries on, for the same reason.

    heartbeat.py [--interval 10] [--threshold-pct 10] [--hold 300]
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path

PSI = "/proc/pressure/memory"
STATE = Path("/var/lib/preflight-heartbeat/withheld.json")
US_PER_S = 1_000_000


def full_total_us(fd: int) -> int:
    """Cumulative microseconds of "full" memory stall, from /proc/pressure/memory."""
    text = os.pread(fd, 128, 0)
    return int(text[text.rindex(b"total=") + 6:].split()[0])


def notifier() -> socket.socket | None:
    """A datagram socket connected to systemd's notify socket, if run under systemd."""
    address = os.environ.get("NOTIFY_SOCKET")
    if not address:
        return None
    if address[0] == "@":
        address = "\0" + address[1:]
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    sock.connect(address)
    return sock


class Stall:
    """Counts consecutive seconds in which the "full" stall share stayed over a threshold."""

    def __init__(self, threshold_pct: float, hold_s: float) -> None:
        self.threshold = threshold_pct / 100
        self.hold = hold_s
        self.seconds = 0.0

    def update(self, stalled_us: int, elapsed_s: float) -> bool:
        """Add one interval; True once the stall has lasted ``hold`` seconds."""
        share = stalled_us / (elapsed_s * US_PER_S) if elapsed_s > 0 else 0.0
        self.seconds = self.seconds + elapsed_s if share >= self.threshold else 0.0
        return self.seconds >= self.hold


def log(event: str, **fields) -> None:
    print(json.dumps({"event": event, **fields}), flush=True)


def record_withheld(fields: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE, "w") as stream:
        stream.write(json.dumps({"wall_time": time.time(), **fields}))
        stream.flush()
        os.fsync(stream.fileno())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--threshold-pct", type=float, default=10.0)
    parser.add_argument("--hold", type=float, default=300.0)
    args = parser.parse_args(argv)
    try:
        Path("/proc/self/comm").write_text("pf-heartbeat")
    except OSError as exc:
        log("rename_failed", error=str(exc))
    sock = notifier()
    stall = Stall(args.threshold_pct, args.hold)
    fd, total, then = -1, 0, time.monotonic()
    try:  # without PSI it still reports (loop_error each tick): never fail at start
        fd = os.open(PSI, os.O_RDONLY)
        total = full_total_us(fd)
    except OSError as exc:
        log("psi_unavailable", error=str(exc))
    if sock:
        sock.send(b"READY=1")
    log("started", interval_s=args.interval, threshold_pct=args.threshold_pct,
        hold_s=args.hold, systemd=sock is not None)
    withholding = False
    while True:
        time.sleep(args.interval)
        try:
            now_total, now = full_total_us(fd), time.monotonic()
            stalled = stall.update(now_total - total, now - then)
            total, then = now_total, now
            if stalled != withholding:
                withholding = stalled
                if stalled:
                    fields = {"stalled_s": round(stall.seconds, 1),
                              "threshold_pct": args.threshold_pct}
                    log("heartbeat_withheld", **fields)
                    try:  # a failed record must not turn into a report
                        record_withheld(fields)
                    except OSError as exc:
                        log("record_failed", error=str(exc))
                else:
                    log("heartbeat_resumed")
            if sock and not withholding:
                sock.send(b"WATCHDOG=1")
        except Exception as exc:  # noqa: BLE001 - dying here would reboot the box
            log("loop_error", error=repr(exc))
            if sock and not withholding:
                try:
                    sock.send(b"WATCHDOG=1")
                except OSError:
                    pass


if __name__ == "__main__":
    sys.exit(main())
