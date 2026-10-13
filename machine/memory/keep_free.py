#!/usr/bin/python3
"""Keep a buffer of truly free memory on a GB10 box, so GPU driver allocations do not stall.

WHY
---
On the DGX Spark the GPU allocates from system RAM, and the NVIDIA driver needs pages that
are FREE now, not "available": file cache counts as available but must first be reclaimed.
On our box (gpu-node-1) every NVIDIA driver allocation failure recorded by the 1-s flight
recorder (7 events on 2026-10-12, ``NVRM ... NV_ERR_NO_MEMORY ... _memdescAllocInternal``)
came when truly free memory had just fallen to 1.4-13 GiB while "available" memory, mostly
file cache, stood at 16-100 GiB, and the kernel was in direct reclaim and compaction in that
second or the next (compaction stalls occur in 3% of all seconds; in 7 of 7 error seconds).
One drop was 6 GiB in 3 s. That driver failure is the reported precursor of the GB10 silent
whole-box hang (NVIDIA/open-gpu-kernel-modules#1358), which froze the Turing box on
2026-10-10 and our box on 2026-10-06.

WHAT
----
Once a second: if free memory is under ``--min-free-gib`` and the file cache holds at least
``--min-cache-gib``, drop the clean page cache (``drop_caches=1``: no sync, so it never waits
on dirty data, and mapped pages are left alone). At most one action per ``--min-gap``
seconds. Every action logs one JSON line to stdout (the journal) with free memory, cache and
the count of free 4-MiB-or-larger blocks before and after, and how long the drop took, so
each action can be checked. The idle cost is one 4-KiB ``pread`` of /proc/meminfo per second
(~2 us measured).

VALIDATED on gpu-node-1, 2026-10-13, with a 62-GiB training job running: one forced action
took free memory 43.07 -> 45.27 GiB (all 2 GiB of droppable cache) in 0.85 s; training
speed unchanged (0.27 it/s), no memory stalls, no driver error. NOT done, on measurement:
explicit compaction (``compact_memory``). Three runs in a row (0.02-0.06 s each) raised free
4-MiB-or-larger blocks only 178 -> 180 with 45 GiB free: while a GPU job is resident the free
memory sits between pages the kernel cannot move, so compaction has nothing to work with.

Root only (drop_caches is root-writable). Stdlib only. Installed as
``preflight-keep-free.service`` by ``machine/install_hang_guards.sh``.

    keep_free.py [--min-free-gib 16] [--min-cache-gib 4] [--min-gap 10] [--interval 1]
    keep_free.py --once ...     # one decision, one action at most, then exit (validation)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

MEMINFO = "/proc/meminfo"
BUDDYINFO = "/proc/buddyinfo"
DROP_CACHES = "/proc/sys/vm/drop_caches"
KIB_PER_GIB = 1024 * 1024
#: Buddy order of a 4-MiB block with 4-KiB pages (getconf PAGESIZE = 4096 on both boxes).
LARGE_ORDER = 10
#: /proc/buddyinfo columns before the per-order counts: "Node", "0,", "zone", "<name>".
BUDDY_PREFIX = 4


def meminfo(fd: int) -> tuple[int, int]:
    """(MemFree, Cached) in KiB, from the first five lines of /proc/meminfo."""
    free = cached = 0
    for line in os.pread(fd, 4096, 0).split(b"\n", 5)[:5]:
        key, _, rest = line.partition(b":")
        if key == b"MemFree":
            free = int(rest.split()[0])
        elif key == b"Cached":
            cached = int(rest.split()[0])
    return free, cached


def large_blocks(text: str) -> int:
    """Free blocks of order >= LARGE_ORDER over every zone in /proc/buddyinfo."""
    total = 0
    for line in text.splitlines():
        counts = line.split()[BUDDY_PREFIX:]
        total += sum(int(c) for c in counts[LARGE_ORDER:])
    return total


def state(fd: int) -> dict:
    free, cached = meminfo(fd)
    with open(BUDDYINFO) as stream:
        blocks = large_blocks(stream.read())
    return {"free_gib": round(free / KIB_PER_GIB, 2),
            "cached_gib": round(cached / KIB_PER_GIB, 2), "large_blocks": blocks}


def act(fd: int) -> dict:
    """Drop the clean page cache; return the before/after record."""
    record = {"event": "cache_dropped", "before": state(fd)}
    start = time.monotonic()
    with open(DROP_CACHES, "w") as stream:
        stream.write("1")
    record["drop_s"] = round(time.monotonic() - start, 3)
    record["after"] = state(fd)
    return record


def needs_action(free_kib: int, cached_kib: int, min_free_kib: int,
                 min_cache_kib: int) -> bool:
    return free_kib < min_free_kib and cached_kib >= min_cache_kib


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--min-free-gib", type=float, default=16.0)
    parser.add_argument("--min-cache-gib", type=float, default=4.0)
    parser.add_argument("--min-gap", type=float, default=10.0)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    try:  # out of earlyoom's "--prefer python" rule; it is tiny and must keep running
        Path("/proc/self/comm").write_text("pf-keep-free")
    except OSError:
        pass
    min_free = int(args.min_free_gib * KIB_PER_GIB)
    min_cache = int(args.min_cache_gib * KIB_PER_GIB)
    fd = os.open(MEMINFO, os.O_RDONLY)
    print(json.dumps({"event": "started", "min_free_gib": args.min_free_gib,
                      "min_cache_gib": args.min_cache_gib, "min_gap_s": args.min_gap}),
          flush=True)
    last = -float("inf")
    while True:
        free, cached = meminfo(fd)
        now = time.monotonic()
        if now - last >= args.min_gap and needs_action(free, cached, min_free, min_cache):
            try:
                print(json.dumps(act(fd)), flush=True)
            except OSError as exc:  # never die: a failed action is logged and retried
                print(json.dumps({"event": "action_failed", "error": str(exc)}), flush=True)
            last = now
        elif args.once:
            print(json.dumps({"event": "no_action", **state(fd)}), flush=True)
        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
