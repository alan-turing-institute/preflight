"""Record GPU occupancy/owners and host memory without importing a GPU framework."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import pwd
import socket
import subprocess
import time


def query(fields: str, kind: str) -> dict:
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-{kind}={fields}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"rows": [], "error": str(exc)[:500]}
    if result.returncode:
        return {"rows": [], "error": (result.stderr or result.stdout)[:500]}
    names = fields.split(",")
    rows = []
    for values in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
        if not values:
            continue
        if len(values) != len(names):
            return {"rows": rows, "error": "Malformed nvidia-smi output"}
        # Preserve N/A / Not Supported, rather than reporting missing values as zero.
        rows.append(dict(zip(names, (value.strip() for value in values))))
    return {"rows": rows, "error": None}


def owner(pid: int, proc: Path = Path("/proc")) -> dict:
    process = proc / str(pid)
    try:
        stat = (process / "stat").read_text()
        start = stat[stat.rindex(")") + 1:].split()[19]
        status = (process / "status").read_text().splitlines()
        uid = int(next(line for line in status if line.startswith("Uid:")).split()[1])
        after = (process / "stat").read_text()
        if start != after[after.rindex(")") + 1:].split()[19]:
            raise ValueError("PID changed during sample")
    except (OSError, ValueError, IndexError, StopIteration) as exc:
        return {"uid": None, "username": None, "owner_error": str(exc)[:200]}
    try:
        username = pwd.getpwuid(uid).pw_name
    except KeyError:
        username = None
    try:
        cgroup = (process / "cgroup").read_text().strip()
    except OSError:
        cgroup = None
    # No command lines or environment variables: they can contain credentials.
    return {"uid": uid, "username": username, "start_ticks": int(start),
            "cgroup": cgroup, "owner_error": None}


def host_memory(proc: Path = Path("/proc")) -> dict:
    keys = {"MemTotal", "MemFree", "MemAvailable", "Buffers", "Cached",
            "SReclaimable", "SwapTotal", "SwapFree"}
    result = {}
    for line in (proc / "meminfo").read_text().splitlines():
        name, _, value = line.partition(":")
        if name in keys:
            result[name] = int(value.split()[0])  # KiB
    try:
        pressure = (proc / "pressure" / "memory").read_text().strip()
    except OSError:
        pressure = None
    return {"kib": result, "pressure": pressure}


def sample() -> dict:
    processes = query("gpu_uuid,pid,process_name,used_gpu_memory", "compute-apps")
    for row in processes["rows"]:
        try:
            row.update(owner(int(row["pid"])))
        except ValueError:
            row.update(uid=None, username=None, owner_error="Invalid GPU PID")
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "gpu": query("index,uuid,utilization.gpu,temperature.gpu,power.draw", "gpu"),
        "processes": processes,
        "memory": host_memory(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--interval", type=float, default=5)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval < 1:
        parser.error("--interval must be at least 1 second")
    logger = logging.getLogger("gpu-usage")
    logger.setLevel(logging.INFO)
    handler = (RotatingFileHandler(args.log, maxBytes=10 * 1024**2, backupCount=4)
               if args.log else logging.StreamHandler())
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    while True:
        started = time.monotonic()
        logger.info(json.dumps(sample(), separators=(",", ":")))
        if args.log:
            # Keep the last completed sample across an unclean restart.
            os.fsync(handler.stream.fileno())
        if args.once:
            return
        time.sleep(max(0, args.interval - (time.monotonic() - started)))


if __name__ == "__main__":
    main()
