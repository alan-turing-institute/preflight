#!/usr/bin/env python3
"""Measure how many copies of a job share one GPU card to advantage, and show the result (design 3(ii)).

    slurm/sharing.py measure --project P --name NAME [--levels 1,2,3,4] [--what TEXT] -- <probe command>
    slurm/sharing.py show NAME

`measure` queues one whole-card job that runs `run`: the probe (a finite command doing the same fixed work every time,
exiting 0) is run alone twice, then as L pinned copies at once for each larger level L. The level is chosen from the
measured wall times; nothing is projected and no constant is chosen by hand.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import statistics
import subprocess
import sys
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gpu_jobs  # noqa: E402 - path set just above
import site_config  # noqa: E402

GB = 1e9  # decimal, as everywhere in the slurm tools
DEFAULT_LEVELS = (1, 2, 3, 4)
SAMPLE_S = 1.0  # how often the box's memory and the copies' GPU memory are read
SOLO_RUNS = 2  # level 1 is run this many times: the spread of the runs is the noise a gain must beat
SOLO = 1  # the level of one job alone


@dataclass(frozen=True)
class LevelResult:
    """One run of `level` copies at once: every copy's wall time and exit code, and what the box used meanwhile."""

    level: int
    walls_s: tuple[float, ...]
    exit_codes: tuple[int, ...]
    cores_per_copy: int
    peak_used_gb: float  # the box's memory in use (MemTotal - MemAvailable), highest sample
    total_gb: float  # the box's MemTotal
    gpu_gb: tuple[float, ...]  # each copy's highest GPU memory (0 when its processes cannot be seen, e.g. in a container)

    @property
    def clean(self) -> bool:
        return all(code == 0 for code in self.exit_codes)

    @property
    def slowest_s(self) -> float:
        return max(self.walls_s)

    def as_record(self) -> dict[str, object]:
        return {"level": self.level, "walls_s": list(self.walls_s), "exit_codes": list(self.exit_codes),
                "cores_per_copy": self.cores_per_copy, "peak_used_gb": self.peak_used_gb, "total_gb": self.total_gb,
                "gpu_gb": list(self.gpu_gb)}


def noise_of(solo_walls_s: Sequence[float]) -> float:
    """The relative run-to-run spread of the level-1 runs: max - min over their mean."""
    return (max(solo_walls_s) - min(solo_walls_s)) / statistics.fmean(solo_walls_s)


def speedup(level: int, solo_s: float, slowest_s: float) -> float:
    """S(L) = L x t1 / tL: work done per second compared with one job alone (1: no gain, below 1: a loss)."""
    return level * solo_s / slowest_s


def choose_level(results: Sequence[LevelResult], noise: float, fraction: float = 1.0) -> tuple[int, str]:
    """(level, one-line reason). Candidates: levels whose copies all exited 0 and whose peak memory in use stayed at
    most `fraction` of the box's total. The level is the smallest candidate whose S is within `noise` of the best S;
    if no candidate's S exceeds 1 + noise the level is 1 (no sharing)."""
    solo = [r.slowest_s for r in results if r.level == SOLO]
    if not solo:
        msg = "no level-1 result: nothing to compare against"
        raise ValueError(msg)
    t1 = statistics.fmean(solo)
    scores: dict[int, float] = {}
    for result in results:
        if result.level == SOLO or not result.clean or result.peak_used_gb > fraction * result.total_gb:
            continue
        scores[result.level] = speedup(result.level, t1, result.slowest_s)
    if not scores:
        return SOLO, "no level above 1 ran cleanly within the memory limit: no sharing"
    best = max(scores.values())
    if best <= 1 + noise:
        return SOLO, f"best S {best:.2f} does not beat 1 by more than the run-to-run noise {noise:.1%}: no sharing"
    level = min(level for level, score in scores.items() if score >= best * (1 - noise))
    return level, f"level {level}: S {scores[level]:.2f}, within the noise {noise:.1%} of the best S {best:.2f}"


def descendants(pid: int) -> set[int]:
    """`pid` and every process below it (from /proc/<pid>/task/*/children)."""
    found, todo = set(), [pid]
    while todo:
        current = todo.pop()
        if current in found:
            continue
        found.add(current)
        for children in Path(f"/proc/{current}/task").glob("*/children"):
            try:
                todo.extend(int(child) for child in children.read_text(encoding="ascii").split())
            except (OSError, ValueError):
                pass  # the process ended between the listing and the read
    return found


class SharingProbe:
    """Runs one level: starts L copies of the probe at once, each pinned to its own cores, and samples while they run."""

    def __init__(self, command: Sequence[str], cores: Sequence[int], sample_s: float = SAMPLE_S) -> None:
        self.command = list(command)
        self.cores = list(cores)
        self.sample_s = sample_s

    def run(self, level: int) -> LevelResult:
        per_copy = len(self.cores) // level
        if per_copy < 1:
            msg = f"level {level} needs at least {level} cores; {len(self.cores)} available"
            raise ValueError(msg)
        pins = [set(self.cores[i * per_copy:(i + 1) * per_copy]) for i in range(level)]
        started = time.monotonic()
        procs = [subprocess.Popen(self.command, preexec_fn=lambda pin=pin: os.sched_setaffinity(0, pin)) for pin in pins]  # noqa: PLW1509 - no thread runs yet
        walls = [0.0] * level
        finished = threading.Event()
        remaining, lock = [level], threading.Lock()

        def wait(i: int) -> None:
            procs[i].wait()
            walls[i] = time.monotonic() - started
            with lock:
                remaining[0] -= 1
                if remaining[0] == 0:
                    finished.set()

        waiters = [threading.Thread(target=wait, args=(i,)) for i in range(level)]
        for waiter in waiters:
            waiter.start()
        peak_used, total, gpu = 0.0, 0.0, [0.0] * level
        while True:
            use = gpu_jobs.machine_use(sleep=lambda _s: None)
            peak_used, total = max(peak_used, use.get("mem_used", 0.0)), use.get("mem_total", total)
            mib = gpu_jobs.gpu_mib_by_pid()
            for i, proc in enumerate(procs):
                gpu[i] = max(gpu[i], sum(mib.get(pid, 0.0) for pid in descendants(proc.pid)) * 1024**2)
            if finished.wait(self.sample_s):
                break
        for waiter in waiters:
            waiter.join()
        return LevelResult(level, tuple(walls), tuple(p.returncode for p in procs), per_copy,
                           peak_used / GB, total / GB, tuple(g / GB for g in gpu))


def table(results: Sequence[LevelResult], solo_s: float) -> list[str]:
    """One line per run: S, the slowest copy, every copy's exit code, memory in use and the GPU memory of the copies."""
    lines = []
    for r in results:
        score = f"S {speedup(r.level, solo_s, r.slowest_s):.2f}" if r.level != SOLO else "S 1.00"
        codes = "ok" if r.clean else f"exit codes {list(r.exit_codes)}"
        lines.append(f"[sharing] level {r.level}: {score}, slowest {r.slowest_s:.1f}s, {r.cores_per_copy} cores each, {codes}, "
                     f"box memory peak {r.peak_used_gb:.1f} of {r.total_gb:.1f} GB, copies' GPU {[round(g, 1) for g in r.gpu_gb]} GB")
    return lines


def digest(command: Sequence[str]) -> str:
    return hashlib.sha256("\0".join(command).encode()).hexdigest()


def parse_levels(text: str) -> tuple[int, ...]:
    levels = tuple(sorted({int(part) for part in text.split(",") if part}))
    if not levels or levels[0] < SOLO:
        msg = f"--levels {text!r}: whole numbers from 1 up, e.g. 1,2,3,4"
        raise ValueError(msg)
    return levels


def gpu_node(site: site_config.Site) -> site_config.Node:
    for node in site.nodes:
        if node.gpus and node.cores_per_gpu:
            return node
    sys.exit("[sharing] REFUSED: no node in the site file has a GPU with cores_per_gpu set")


def run(arguments: argparse.Namespace) -> int:
    command = arguments.command[1:] if arguments.command[:1] == ["--"] else arguments.command
    site = gpu_jobs.load_site()
    node = gpu_node(site)
    levels = parse_levels(arguments.levels)
    cores = sorted(os.sched_getaffinity(0))[:node.cores_per_gpu]
    probe = SharingProbe(command, cores)
    results: list[LevelResult] = []
    for level in [SOLO] * SOLO_RUNS + [lv for lv in levels if lv != SOLO]:
        if len(cores) // level < 1:
            print(f"[sharing] level {level}: skipped, only {len(cores)} cores")
            continue
        results.append(probe.run(level))
        if level == SOLO and not results[-1].clean:
            print(f"[sharing] REFUSED: the probe alone exited {list(results[-1].exit_codes)}; it must exit 0\n"
                  "           nothing recorded", file=sys.stderr)
            return 1
        print(table(results[-1:], statistics.fmean(r.slowest_s for r in results if r.level == SOLO))[0], flush=True)
    solo = [r.slowest_s for r in results if r.level == SOLO]
    noise = noise_of(solo)
    level, reason = choose_level(results, noise, site.sharing_memory_fraction or 1.0)
    print(f"[sharing] noise {noise:.1%} (level-1 runs {solo[0]:.1f}s, {solo[1]:.1f}s); {reason}")
    sharing = {
        "level": level,
        "levels": [r.as_record() for r in results],
        "solo_wall_s": solo,
        "cores_per_copy": len(cores) // level,
        "gpu_type": node.gpus[0].type,
        "node_memory_mb": node.real_memory_mb,
        "shards": node.gpus[0].shards,
        "measured_at": int(time.time()),
        "probe_command_digest": digest(command),
    }
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id:
        own = gpu_jobs.read_json(gpu_jobs.sidecar_path(job_id), gpu_jobs.current_user()) or {}
        name = str(own.get("name") or "")
        # the named job's footprint now: `submit.py` warns when a later clean run has outgrown it
        sharing["peak_bytes"] = gpu_jobs.measured_peak_bytes(name) if name else None
        gpu_jobs.update_sidecar(job_id, kind=gpu_jobs.SHARING_PROBE, sharing=sharing)
    else:
        print("[sharing] not inside a slurm job: result printed, not recorded")
    return 0


def measure(arguments: argparse.Namespace) -> int:
    import submit  # lazy: submit reads the site file and slurm state on use

    command = arguments.command[1:] if arguments.command[:1] == ["--"] else arguments.command
    if not command:
        sys.exit("[sharing] REFUSED: no probe command after --")
    parse_levels(arguments.levels)
    body = [sys.executable, str(Path(__file__).resolve()), "run", "--levels", arguments.levels, "--", *command]
    what = arguments.what or f"sharing level measurement of {arguments.name}"
    try:
        job_id = submit.submit(["--project", arguments.project, "--what", what, "--class", gpu_jobs.JobClass.RESTART.value,
                                "--gpu", "--name", arguments.name, "--", *body])
    except submit.Refused as error:
        sys.exit(f"[sharing] REFUSED: {error}")
    print(f"[sharing] job {job_id} holds the whole card while it measures; read the result with: sharing.py show {arguments.name}")
    return 0


def show(arguments: argparse.Namespace) -> int:
    record = gpu_jobs.measured_sharing(arguments.name)
    if record is None:
        print(f"[sharing] no measurement called {arguments.name!r}")
        return 1
    results = [LevelResult(r["level"], tuple(r["walls_s"]), tuple(r["exit_codes"]), r["cores_per_copy"], r["peak_used_gb"],
                           r["total_gb"], tuple(r["gpu_gb"])) for r in record["levels"]]
    solo = record["solo_wall_s"]
    print("\n".join(table(results, statistics.fmean(solo))))
    print(f"[sharing] {arguments.name}: level {record['level']} ({record['cores_per_copy']} cores per copy), noise "
          f"{noise_of(solo):.1%}, on {record['gpu_type']}, {record['node_memory_mb'] / 1000:.0f} GB node, shards {record['shards']}, "
          f"measured at {time.strftime('%Y-%m-%d %H:%M', time.localtime(record['measured_at']))}")
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="action", required=True)
    m = sub.add_parser("measure", help="queue the whole-card measuring job")
    m.add_argument("--project", required=True)
    m.add_argument("--name", required=True, help="the job name submit.py is later given: the result is found by it")
    m.add_argument("--levels", default=",".join(map(str, DEFAULT_LEVELS)))
    m.add_argument("--what", default=None)
    m.add_argument("command", nargs=argparse.REMAINDER, help="-- the probe: a finite command that exits 0")
    m.set_defaults(handler=measure)
    r = sub.add_parser("run", help="the job body: measure inside the whole-card job")
    r.add_argument("--levels", default=",".join(map(str, DEFAULT_LEVELS)))
    r.add_argument("command", nargs=argparse.REMAINDER)
    r.set_defaults(handler=run)
    s = sub.add_parser("show", help="print the newest measurement of NAME")
    s.add_argument("name")
    s.set_defaults(handler=show)
    return p


def main() -> int:
    arguments = parser().parse_args()
    try:
        return arguments.handler(arguments)
    except ValueError as error:
        sys.exit(f"[sharing] REFUSED: {error}")


if __name__ == "__main__":
    sys.exit(main())
