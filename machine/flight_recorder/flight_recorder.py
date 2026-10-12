#!/usr/bin/env python3
"""Record the box's memory state every second, so the next freeze explains itself.

WHY THIS EXISTS
---------------
Both DGX Sparks (NVIDIA GB10, the GPU pool IS system RAM) froze five times between
2026-10-03 and 2026-10-10, and the cause of four is unknown. The only record was
sysstat, which samples every 10 minutes: at 13:50:01 on 2026-10-06 it showed 89.9 GiB
available on our box a minute before a real memory exhaustion hung it, and the
Turing box's last sample on 2026-10-10 came 45 s before it froze. A collapse that
takes 30 s is invisible at that rate.

This writes one JSON line per second to a day file under ``--out``, fsynced as it
goes, so the last seconds before a freeze survive a hard reboot. It records what
separates the candidate causes: truly free memory versus "available" (on the GB10
the driver needs free pages, and file cache counts as available), the kernel's
memory and I/O stall gauges, reclaim and allocation-stall rates, the number of
processes stuck in uninterruptible sleep, and each GPU process's memory. The GPU
query runs in its own thread with a timeout and never piles up, because a stuck
driver hangs ``nvidia-smi`` too -- the age of the last good GPU reading is itself a
signal.

Stdlib only. Runs as a user service on each box (``--install-unit``), one per box,
independent of any job, so it covers every freeze whoever's job causes it.

    machine/flight_recorder/flight_recorder.py [--out DIR] [--interval 1] \
        [--gpu-every 10] [--gpu-timeout 5] [--keep-days 14] [--once]
    machine/flight_recorder/flight_recorder.py --install-unit
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

DEFAULT_OUT = Path.home() / ".local" / "state" / "flight_recorder"
UNIT_NAME = "flight-recorder.service"
KIB_PER_MIB = 1024

#: /proc/meminfo fields recorded, in MiB.
MEMINFO = {
    "MemFree": "free",
    "MemAvailable": "avail",
    "Cached": "cached",
    "Dirty": "dirty",
    "Writeback": "writeback",
    "AnonPages": "anon",
    "Unevictable": "unevictable",
    "SwapFree": "swapfree",
}
#: /proc/vmstat counters recorded as per-second rates (allocstall_* are summed).
VMSTAT = {
    "pgscan_kswapd": "scan_kswapd_s",
    "pgscan_direct": "scan_direct_s",
    "pgsteal_kswapd": "steal_kswapd_s",
    "pgsteal_direct": "steal_direct_s",
    "compact_stall": "compact_stall_s",
    "pgmajfault": "majflt_s",
    "oom_kill": "oom_kill_s",
}
ALLOCSTALL_PREFIX = "allocstall_"
GPU_QUERY = [
    "nvidia-smi",
    "--query-compute-apps=pid,used_memory",
    "--format=csv,noheader,nounits",
]


def read(path: str) -> str:
    """A /proc file's text, or "" if it cannot be read."""
    try:
        return Path(path).read_text()
    except OSError:
        return ""


def parse_meminfo(text: str) -> dict[str, float]:
    """The MEMINFO fields in MiB."""
    out = {}
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        if key in MEMINFO and rest.split():
            out[MEMINFO[key]] = round(int(rest.split()[0]) / KIB_PER_MIB, 1)
    return out


def parse_psi(text: str, name: str) -> dict[str, float]:
    """avg10 of each line of a /proc/pressure file, as ``<name>_some``/``<name>_full``."""
    out = {}
    for line in text.splitlines():
        kind, *fields = line.split()
        for field in fields:
            if field.startswith("avg10="):
                out[f"{name}_{kind}"] = float(field.split("=", 1)[1])
    return out


def parse_vmstat(text: str) -> dict[str, int]:
    """The VMSTAT counters, plus ``allocstall`` summed over every zone."""
    out = {"allocstall": 0}
    for line in text.splitlines():
        key, _, value = line.partition(" ")
        if key in VMSTAT:
            out[key] = int(value)
        elif key.startswith(ALLOCSTALL_PREFIX):
            out["allocstall"] += int(value)
    return out


def rates(before: dict[str, int], after: dict[str, int], seconds: float) -> dict:
    """Per-second change of each counter present in both snapshots."""
    names = {**VMSTAT, "allocstall": "allocstall_s"}
    return {
        names[k]: round((after[k] - before[k]) / seconds, 1)
        for k in after
        if k in before and seconds > 0
    }


def parse_blocked(stat_text: str) -> int | None:
    """Processes in uninterruptible sleep, from /proc/stat."""
    for line in stat_text.splitlines():
        if line.startswith("procs_blocked"):
            return int(line.split()[1])
    return None


def parse_gpu(text: str) -> dict[str, float | None]:
    """``nvidia-smi`` compute apps as {pid: MiB}; "[N/A]" becomes None."""
    out: dict[str, float | None] = {}
    for line in text.strip().splitlines():
        pid, _, mib = (part.strip() for part in line.partition(","))
        if pid:
            try:
                out[pid] = float(mib)
            except ValueError:
                out[pid] = None
    return out


class GpuPoller(threading.Thread):
    """Query the GPU every ``every`` s; never start a query while one is still stuck."""

    def __init__(self, every: float, timeout: float, command=None) -> None:
        super().__init__(daemon=True)
        self.every, self.timeout = every, timeout
        self.command = command or GPU_QUERY
        self.result: dict | str = "not yet"
        self.last_ok: float | None = None
        self.stuck: subprocess.Popen | None = None

    def poll(self) -> None:
        if self.stuck is not None and self.stuck.poll() is None:
            self.result = "previous query still stuck"
            return
        self.stuck = None
        try:
            proc = subprocess.Popen(
                self.command,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except OSError as exc:
            self.result = f"error: {exc}"
            return
        try:
            out, _ = proc.communicate(timeout=self.timeout)
        except subprocess.TimeoutExpired:
            # Kill but never wait: a query stuck in the driver cannot be reaped either.
            proc.kill()
            self.stuck = proc
            self.result = "timeout"
            return
        self.result = parse_gpu(out)
        self.last_ok = time.monotonic()

    def run(self) -> None:
        while True:
            self.poll()
            time.sleep(self.every)

    def snapshot(self) -> dict:
        age = None if self.last_ok is None else round(time.monotonic() - self.last_ok, 1)
        return {"gpu": self.result, "gpu_age_s": age}


class DayFile:
    """Append-and-fsync to ``flight-<date>.jsonl``; prune day files past ``keep_days``."""

    def __init__(self, out: Path, keep_days: int) -> None:
        self.out, self.keep_days = out, keep_days
        self.day: dt.date | None = None
        self.handle = None

    def write(self, line: str, today: dt.date) -> None:
        if today != self.day:
            if self.handle is not None:
                self.handle.close()
            self.out.mkdir(parents=True, exist_ok=True)
            self.handle = (self.out / f"flight-{today.isoformat()}.jsonl").open("a")
            self.day = today
            self.prune(today)
        self.handle.write(line + "\n")
        self.handle.flush()
        os.fsync(self.handle.fileno())

    def prune(self, today: dt.date) -> None:
        cutoff = today - dt.timedelta(days=self.keep_days)
        for path in self.out.glob("flight-*.jsonl"):
            try:
                day = dt.date.fromisoformat(path.stem.removeprefix("flight-"))
            except ValueError:
                continue
            if day < cutoff:
                path.unlink(missing_ok=True)


def sample(previous: dict[str, int] | None, seconds: float) -> tuple[dict, dict]:
    """One record, and this sample's raw counters for the next rate."""
    counters = parse_vmstat(read("/proc/vmstat"))
    record: dict = {"t": dt.datetime.now().isoformat(timespec="seconds")}
    record.update(parse_meminfo(read("/proc/meminfo")))
    for name in ("memory", "io", "cpu"):
        record.update(parse_psi(read(f"/proc/pressure/{name}"), f"psi_{name}"))
    if previous is not None:
        record.update(rates(previous, counters, seconds))
    record["blocked"] = parse_blocked(read("/proc/stat"))
    record["load1"] = float((read("/proc/loadavg").split() or ["nan"])[0])
    return record, counters


def unit_text(script: Path, out: Path, python: str = sys.executable) -> str:
    """The user unit that keeps this recorder running on this box, across reboots."""
    return (
        "[Unit]\n"
        "Description=Per-second memory flight recorder (preflight)\n\n"
        "[Service]\n"
        f"ExecStart={python} {script} --out {out}\n"
        "Restart=always\n"
        "RestartSec=5\n\n"
        "[Install]\n"
        "WantedBy=default.target\n"
    )


def install_unit(out: Path) -> int:
    """Write and start the user unit. Survives reboots only where linger is enabled."""
    unit_dir = Path.home() / ".config" / "systemd" / "user"
    unit_dir.mkdir(parents=True, exist_ok=True)
    (unit_dir / UNIT_NAME).write_text(unit_text(Path(__file__).resolve(), out))
    for args in (["daemon-reload"], ["enable", "--now", UNIT_NAME]):
        subprocess.run(["systemctl", "--user", *args], check=True)
    linger = subprocess.run(
        ["loginctl", "show-user", os.environ.get("USER", ""), "-p", "Linger"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    print(f"{UNIT_NAME} enabled; {linger or 'Linger unknown'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--gpu-every", type=float, default=10.0)
    ap.add_argument("--gpu-timeout", type=float, default=5.0)
    ap.add_argument("--keep-days", type=int, default=14)
    ap.add_argument("--once", action="store_true", help="write one sample and exit")
    ap.add_argument("--install-unit", action="store_true")
    a = ap.parse_args(argv)
    if a.install_unit:
        return install_unit(a.out)
    gpu = GpuPoller(a.gpu_every, a.gpu_timeout)
    gpu.start()
    sink = DayFile(a.out, a.keep_days)
    previous, last = None, time.monotonic()
    while True:
        try:
            now = time.monotonic()
            record, previous = sample(previous, now - last)
            last = now
            record.update(gpu.snapshot())
            sink.write(json.dumps(record, separators=(",", ":")), dt.date.today())
        except Exception as exc:  # noqa: BLE001 -- a recorder that dies records nothing
            print(f"flight_recorder: {type(exc).__name__}: {exc}", file=sys.stderr)
        if a.once:
            return 0
        time.sleep(max(0.0, a.interval - (time.monotonic() - last)))


if __name__ == "__main__":
    raise SystemExit(main())
