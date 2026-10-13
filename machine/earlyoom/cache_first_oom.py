"""Reclaim cache before enabling earlyoom; retain its victim selection and limits."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time

GIB_KIB = 1024**2
GPU_LOG = Path("/var/log/preflight/gpu.jsonl")


def memory() -> dict:
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, _, value = line.partition(":")
        if key in {"MemAvailable", "MemFree", "Cached", "SReclaimable"}:
            values[key] = int(value.split()[0])
    return values


def log(event: str, **fields) -> None:
    print(json.dumps({"timestamp_utc": datetime.now(timezone.utc).isoformat(),
                      "event": event, **fields}), flush=True)


def gpu_active() -> bool:
    try:
        with GPU_LOG.open("rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - 65536))
            record = json.loads(stream.read().splitlines()[-1])
        age = datetime.now(timezone.utc) - datetime.fromisoformat(record["timestamp_utc"])
        return 0 <= age.total_seconds() <= 15 and bool(record["processes"]["rows"])
    except (OSError, ValueError, KeyError, IndexError):
        return False


def fork_reclaim() -> int:
    pid = os.fork()
    if pid == 0:
        try:
            os.sync()
            Path("/proc/sys/vm/drop_caches").write_text("3\n")
        except OSError as exc:
            log("cache_reclaim_error", error=str(exc))
            os._exit(1)
        os._exit(0)
    return pid


class Guard:
    def __init__(self, args: list[str]):
        self.args = args
        self.term, self.kill = map(int, args[args.index("-M") + 1].split(","))
        if not 0 < self.kill <= self.term or args[args.index("-s") + 1] != "100,100":
            raise ValueError("requires fixed -M TERM,KILL and -s 100,100")
        self.worker = None
        self.timed_out = False
        self.reclaim_ok = False
        self.killer = None
        self.last_reclaim = -float("inf")

    def begin_reclaim(self, before: dict, now: float) -> None:
        self.before = before
        self.last_reclaim = now
        # Do not hold off an emergency kill for a slow sync/cache syscall.
        self.deadline = now + (0.2 if before["MemAvailable"] <= self.kill else 2)
        self.timed_out = False
        self.reclaim_ok = False
        self.worker = fork_reclaim()
        log("cache_reclaim_started", worker_pid=self.worker, before_kib=before,
            timeout_seconds=self.deadline - now)

    def reclaim_pending(self, now: float) -> bool:
        if self.worker is None:
            return False
        pid, status = os.waitpid(self.worker, os.WNOHANG)
        if pid:
            self.reclaim_ok = status == 0 and not self.timed_out
            log("cache_reclaim_finished", success=status == 0, timed_out=self.timed_out,
                before_kib=self.before, after_kib=memory())
            self.worker = None
            self.timed_out = False
            return False
        if memory()["MemAvailable"] <= self.kill:
            self.deadline = min(self.deadline, now + 0.2)
        if not self.timed_out and now >= self.deadline:
            os.kill(self.worker, signal.SIGKILL)
            self.timed_out = True
            log("cache_reclaim_timeout", before_kib=self.before, after_kib=memory())
        # A stuck syscall may remain unkillable. Never block waiting for it.
        return not self.timed_out

    def step(self, now: float) -> None:
        before = memory()
        if self.killer is not None:
            if self.killer.poll() is not None:
                log("earlyoom_exited", returncode=self.killer.returncode)
                self.killer = None
            elif before["MemAvailable"] > self.term:
                self.killer.terminate()
                log("memory_recovered", memory_kib=before)
                return
            else:
                return
        had_worker = self.worker is not None
        if self.reclaim_pending(now):
            return
        after = memory()
        if after["MemAvailable"] <= self.term:
            if not had_worker:
                try:
                    self.begin_reclaim(after, now)
                    return
                except OSError as exc:
                    log("cache_reclaim_failed", error=str(exc), memory_kib=after)
            log("earlyoom_enabled", reason="memory still low after reclaim attempt",
                memory_kib=after, term_kib=self.term, kill_kib=self.kill)
            self.killer = subprocess.Popen(["/usr/bin/earlyoom", *self.args])
        elif self.worker is None and now - self.last_reclaim >= 60:
            # Reclaim early at 12 GiB available on the current 8/4 GiB policy.
            # Also cover GPU allocations with little raw free RAM but large cache.
            low = after["MemAvailable"] <= self.term + 4 * GIB_KIB
            gpu_cache = (after["MemFree"] <= self.kill and
                         after["Cached"] >= self.term and gpu_active())
            if low or gpu_cache:
                self.begin_reclaim(after, now)

    def close(self) -> None:
        if self.killer is not None and self.killer.poll() is None:
            self.killer.terminate()
        if self.worker is not None:
            try:
                os.kill(self.worker, signal.SIGKILL)
            except ProcessLookupError:
                pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--test-reclaim", action="store_true",
                        help="one bounded cache reclaim; never start earlyoom")
    options = parser.parse_args()
    guard = Guard(shlex.split(os.environ["EARLYOOM_ARGS"]))
    if options.check:
        log("configuration_valid", term_kib=guard.term, kill_kib=guard.kill)
        return
    def stop(signum, frame):
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, stop)
    try:
        if options.test_reclaim:
            guard.begin_reclaim(memory(), time.monotonic())
            while guard.reclaim_pending(time.monotonic()):
                time.sleep(0.05)
            if not guard.reclaim_ok:
                raise SystemExit(1)
            return
        log("cache_first_guard_started", term_kib=guard.term, kill_kib=guard.kill)
        while True:
            guard.step(time.monotonic())
            time.sleep(0.1)
    finally:
        guard.close()


if __name__ == "__main__":
    main()
