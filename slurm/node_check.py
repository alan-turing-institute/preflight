#!/usr/bin/env python3
"""Node health check: slurmd runs it as root (`HealthCheckProgram`); it keeps a part-time machine usable by its owner.

    node_check.py [--settings /etc/gpujobs/node_check.json]

Rules, in order, each run (plan B2, `plan notes`):
1. An urgent job is pending: never close the node; reopen it if we closed it.
2. Free memory below `push_back_below_mb`: requeue the newest-started `resume` job here (one per run; at its second
   push from this node the job is also barred from the node).
3. Free memory below `close_below_mb`, or `mains_only` and on battery: drain, reason `gpujobs-check: ...`.
4. We closed it and the last 3 runs all passed rule 3: resume.
A drain whose reason does not start `gpujobs-check:` is never changed. Any exception drains the node (fail closed).
Every run logs one line, a decision to allow included.
"""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gpu_jobs  # noqa: E402 - path set just above

SETTINGS_PATH = Path("/etc/gpujobs/node_check.json")
LOG_PATH = Path("/var/log/gpujobs/node_check.log")  # the settings name it too; this is for a run that cannot read them
REASON_PREFIX = "gpujobs-check: "
#: slurm kills a health check after 60 s; every slurm call gets far less.
CALL_TIMEOUT_S = 10
#: Consecutive runs that must pass rule 3 before we reopen a node we closed.
REOPEN_AFTER = 3
#: A job is pushed back this many times from a node before it is barred from it.
PUSHES_BEFORE_EXCLUDE = 2
KB_BYTES = 1024
MB_BYTES = 1_000_000  # decimal, as everything shown to people
GB_MB = 1000
DRAIN_FLAG = "DRAIN"
#: The keys the node's `Extra` may override, and the type each must have.
EXTRA_KEYS: dict[str, type] = {"close_below_mb": int, "push_back_below_mb": int, "mains_only": bool}

Runner = Callable[[list[str]], subprocess.CompletedProcess]
Recorder = Callable[..., object]


def run_command(command: list[str]) -> subprocess.CompletedProcess:
    """Every subprocess call of this tool; tests pass a fake in its place. A timeout raises (and so fails closed)."""
    return subprocess.run(command, capture_output=True, text=True, timeout=CALL_TIMEOUT_S, check=False)


@dataclass(frozen=True)
class Settings:
    node: str
    close_below_mb: int
    push_back_below_mb: int
    mains_only: bool | None  # None: on when the machine has a battery
    meminfo_path: Path
    power_supply_dir: Path
    state_path: Path
    log_path: Path


def load_settings(path: Path, names: tuple[str, ...]) -> Settings:
    """The settings file: one object, or an object keyed by node name (the first of `names` present is used)."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if "node" not in raw:
        raw = next(raw[name] for name in names if name in raw)
    return Settings(
        node=str(raw["node"]), close_below_mb=int(raw["close_below_mb"]),
        push_back_below_mb=int(raw["push_back_below_mb"]), mains_only=raw.get("mains_only"),
        meminfo_path=Path(raw["meminfo_path"]), power_supply_dir=Path(raw["power_supply_dir"]),
        state_path=Path(raw["state_path"]), log_path=Path(raw["log_path"]),
    )


def apply_extra(settings: Settings, extra: str) -> tuple[Settings, str]:
    """The settings with the node's `Extra` applied; (unchanged, note) when it is set but not valid."""
    if not extra.strip():
        return settings, ""
    try:
        found = json.loads(extra)
        valid = (
            isinstance(found, dict) and found
            and all(k in EXTRA_KEYS and type(v) is EXTRA_KEYS[k] for k, v in found.items())
        )
    except ValueError:
        valid = False
    if not valid:
        return settings, "Extra ignored (invalid)"
    return Settings(**{**settings.__dict__, **found}), ""


def free_mb(meminfo: Path) -> int:
    """MemAvailable in decimal megabytes."""
    for line in meminfo.read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * KB_BYTES // MB_BYTES
    raise ValueError("no MemAvailable line")


def power(directory: Path) -> tuple[bool, bool]:
    """(on mains, has a battery). No Mains and no Battery counts as mains."""
    kinds = {}
    for supply in directory.glob("*"):
        try:
            kind = (supply / "type").read_text(encoding="utf-8").strip()
            online = (supply / "online").read_text(encoding="utf-8").strip() if kind == "Mains" else ""
        except OSError:
            continue
        kinds.setdefault(kind, []).append(online == "1")
    battery = "Battery" in kinds
    return any(kinds.get("Mains", [])) or ("Mains" not in kinds and not battery), battery


@dataclass(frozen=True)
class NodeInfo:
    name: str
    drained: bool
    reason: str
    extra: str

    @property
    def ours(self) -> bool:
        return self.drained and self.reason.startswith(REASON_PREFIX)

    @property
    def foreign(self) -> bool:
        return self.drained and not self.reason.startswith(REASON_PREFIX)


def read_node(run: Runner, hostname: str) -> NodeInfo:
    done = run(["scontrol", "show", "node", hostname, "--json"])
    if done.returncode != 0:
        raise RuntimeError(f"scontrol show node failed: {done.stderr.strip()}")
    node = json.loads(done.stdout)["nodes"][0]
    states = node.get("state") or []
    return NodeInfo(str(node["name"]), DRAIN_FLAG in states, str(node.get("reason") or ""), str(node.get("extra") or ""))


@dataclass(frozen=True)
class Running:
    job: gpu_jobs.QueuedJob
    started: int
    excluded: str


def read_queue(run: Runner, node: str) -> tuple[bool, list[Running], set[str]]:
    """(an urgent job is pending, the jobs running on `node` with their start times, every unfinished job id)."""
    done = run(["squeue", "--json"])
    if done.returncode != 0:
        raise RuntimeError(f"squeue failed: {done.stderr.strip()}")
    jobs = gpu_jobs.parse_squeue(done.stdout)
    raw = {str(entry["job_id"]): entry for entry in json.loads(done.stdout)["jobs"]}
    urgent = any(j.state == gpu_jobs.State.PENDING and j.partition == gpu_jobs.Partition.URGENT for j in jobs)
    here = []
    for job in jobs:
        entry = raw[job.job_id]
        if job.state == gpu_jobs.State.RUNNING and node in str(entry.get("nodes") or "").split(","):
            started = entry.get("start_time") or {}
            here.append(Running(job, int(started.get("number") or 0), str(entry.get("excluded_nodes") or "")))
    return urgent, here, {job.job_id for job in jobs}


def push_victim(running: list[Running]) -> Running | None:
    """The `resume` job started last, outside the urgent and service partitions."""
    candidates = [
        r for r in running
        if r.job.job_class is gpu_jobs.JobClass.RESUME
        and r.job.partition not in {gpu_jobs.Partition.URGENT, gpu_jobs.Partition.SERVICE}
    ]
    return max(candidates, key=lambda r: r.started, default=None)


def drain(run: Runner, node: str, why: str) -> None:
    done = run(["scontrol", "update", f"NodeName={node}", "State=DRAIN", f"Reason={REASON_PREFIX}{why}"])
    if done.returncode != 0:
        raise RuntimeError(f"drain failed: {done.stderr.strip()}")


def reopen(run: Runner, node: str) -> None:
    done = run(["scontrol", "update", f"NodeName={node}", "State=RESUME"])
    if done.returncode != 0:
        raise RuntimeError(f"resume failed: {done.stderr.strip()}")


def read_state(path: Path) -> dict:
    found = gpu_jobs.read_json(path) if path.exists() else None
    return {"pushes": dict((found or {}).get("pushes") or {}), "passes": int((found or {}).get("passes") or 0)}


def log_line(path: Path, line: str) -> None:
    """One line per run; a log that cannot be written never stops the check."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {line}\n")
    except OSError as error:
        print(f"node_check: cannot log: {error}", file=sys.stderr)


def check(
    settings_path: Path, run: Runner = run_command, record: Recorder = gpu_jobs.write_event,
    hostname: str | None = None,
) -> list[str]:
    """One run of the rules; the decisions taken (empty = allow). Raises on any failure: the caller fails closed."""
    host = (hostname or socket.gethostname()).split(".")[0]
    info = read_node(run, host)
    settings, note = apply_extra(load_settings(settings_path, (info.name, host)), info.extra)
    free, (on_mains, battery) = free_mb(settings.meminfo_path), power(settings.power_supply_dir)
    urgent, running, unfinished = read_queue(run, info.name)
    state = read_state(settings.state_path)
    state["pushes"] = {k: n for k, n in state["pushes"].items() if k.split("@")[0] in unfinished}  # a pushed job waits in the queue: keep it
    decisions = [note] if note else []

    if free < settings.push_back_below_mb and (victim := push_victim(running)):
        job = victim.job.job_id
        key = f"{job}@{info.name}"
        state["pushes"][key] = state["pushes"].get(key, 0) + 1
        done = run(["scontrol", "requeue", job])
        if done.returncode != 0:
            raise RuntimeError(f"requeue {job} failed: {done.stderr.strip()}")
        record(job, gpu_jobs.EventAction.PUSH_BACK, node=info.name, free_mb=free)
        decisions.append(f"push back {job} ({free} MB free)")
        if state["pushes"][key] >= PUSHES_BEFORE_EXCLUDE:
            excluded = sorted({*filter(None, victim.excluded.split(",")), info.name})
            done = run(["scontrol", "update", f"JobId={job}", f"ExcNodeList={','.join(excluded)}"])
            if done.returncode != 0:
                raise RuntimeError(f"exclude on {job} failed: {done.stderr.strip()}")
            decisions.append(f"{job} barred from {info.name}")

    mains_only = battery if settings.mains_only is None else settings.mains_only
    why = ""
    if free < settings.close_below_mb:
        why = f"{free / GB_MB:.1f} GB free"
    elif mains_only and not on_mains:
        why = f"on battery, {free / GB_MB:.1f} GB free"
    state["passes"] = 0 if why else state["passes"] + 1

    if urgent:
        if info.ours:
            reopen(run, info.name)
            decisions.append("reopen (urgent job waiting)")
        elif why:
            decisions.append(f"stay open (urgent job waiting; {why})")
    elif info.foreign:
        if why:
            decisions.append(f"would close ({why}) but drained by another reason: left alone")
    elif why and not info.ours:
        drain(run, info.name, why)
        decisions.append(f"close ({why})")
    elif info.ours and state["passes"] >= REOPEN_AFTER:
        reopen(run, info.name)
        decisions.append(f"reopen ({REOPEN_AFTER} passing runs)")

    gpu_jobs.write_json(settings.state_path, state)
    log_line(settings.log_path, f"{info.name} free={free} MB {'; '.join(decisions) or 'allow'}")
    return decisions


def fail_closed(error: Exception, run: Runner, hostname: str, log_path: Path) -> None:
    """Drain the node because the check itself failed, unless it is drained for a reason that is not ours."""
    try:
        try:
            info = read_node(run, hostname)
        except Exception:  # noqa: BLE001 - cannot tell: closing is the safe answer
            info = None
        if info is not None and info.foreign:
            decision = "left alone (drained by another reason)"
        else:
            drain(run, info.name if info else hostname, f"check failed: {type(error).__name__}")
            decision = "drained"
    except Exception as second:  # noqa: BLE001 - nothing left to try; the log is the record
        decision = f"could not drain: {type(second).__name__}"
    log_line(log_path, f"{hostname} check failed: {type(error).__name__}: {error}; {decision}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--settings", type=Path, default=SETTINGS_PATH)
    arguments = parser.parse_args()
    try:
        check(arguments.settings)
    except Exception as error:  # noqa: BLE001 - fail closed
        fail_closed(error, run_command, socket.gethostname().split(".")[0], LOG_PATH)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
