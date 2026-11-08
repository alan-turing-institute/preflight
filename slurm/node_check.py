#!/usr/bin/env python3
"""Node health check: slurmd runs it as root (`HealthCheckProgram`); it keeps a part-time machine usable by its owner.

    node_check.py [--settings /etc/gpujobs/node_check.json]

Rules, in order, each run (plan B2, `plan notes`):
0. A running job on a part-time machine that is not batch, requeueable and `resume`-class is cancelled (B7.1).
1. An urgent job is pending: never close the node; reopen it if we closed it.
2. Free memory below `push_back_below_mb`: requeue the newest-started `resume` job here (one per run; at its second
   push from this node the job is also barred from the node).
2b. A `launch_refused` event in this machine's `events/` (written by a job's wrapper that found the machine unfit for
   it and requeued the job): once the job waits again, and only if the event's owner is the job's owner, bar the job from
   this node (`ExcNodeList`; a running job cannot be changed, measured 2026-10-31) and delete the event.
3. Free memory below `close_below_mb`, or `mains_only` and on battery: drain, reason `gpujobs-check: ...`.
4. We closed it and the last 3 runs all passed rule 3: resume.
5. Docker containers labelled `slurm-job=J` (README "A job's docker containers"): written to the root map
   `gpu_jobs.CONTAINER_MAP` for readers that cannot ask docker (the agent, a job's user outside the docker group); one
   whose job J is in no unfinished job and that is older than `STALE_CONTAINER_S` is removed: what slurm's Epilog
   missed (docker hung, a restart policy after a reboot, a container made after the Epilog listed). Never closes the node.
A drain whose reason does not start `gpujobs-check:` is never changed. Any exception drains the node (fail closed).
Every run logs one line, a decision to allow included, and (when `owners/` exists, made by `apply --part-time`) writes
`owners/manifest.json`: who owns each state file here, which the controller trusts for the copy of them (B7.1).
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import os
import re
import stat
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
#: The node's `Comment` carries what the check just read, for the page: `available_mb=N;total_mb=N;cores=N;cores_busy=X`
#: (`gpu_busy=X`, 0-1, too on a machine with a GPU; no spaces: `scontrol -o` splits on them). A comment that does not start with the prefix is someone else's: left alone.
COMMENT_PREFIX = "available_mb="
#: slurm kills a health check after 60 s; every slurm call gets far less.
CALL_TIMEOUT_S = 10
#: Consecutive runs that must pass rule 3 before we reopen a node we closed.
REOPEN_AFTER = 3
#: A job is pushed back this many times from a node before it is barred from it (when its partition has another node).
PUSHES_BEFORE_EXCLUDE = 2
MB_BYTES = 1_000_000  # decimal, as everything shown to people
GB_MB = 1000
DRAIN_FLAG = "DRAIN"
#: The state subdirectories whose files the manifest lists (what the controller copies back), and where it is written.
MANIFEST_DIRS = ("jobs", "events", "logs", "footprints")
OWNERS_DIR = "owners"
MANIFEST_NAME = "manifest.json"
KILL_REASON = "not allowed on a part-time machine"
LAUNCH_REFUSED_FILE = re.compile(rf"(\d+)-\d+-{gpu_jobs.EventAction.LAUNCH_REFUSED}\.json", re.ASCII)
PENDING = gpu_jobs.State.PENDING.value
#: A `launch_refused` event whose job is not yet waiting is kept this long: `scontrol requeue` leaves the job COMPLETING
#: for a few seconds (measured 2026-10-31, ~6 s) before it waits.
LAUNCH_EVENT_WAIT_S = 60
#: The keys the node's `Extra` may override, and the type each must have.
EXTRA_KEYS: dict[str, type] = {"close_below_mb": int, "push_back_below_mb": int, "mains_only": bool}

#: A labelled container whose job is in no unfinished job is removed only once it is this old: a job that started after
#: this run read the queue can already have made one, and its Epilog had time to act on it.
STALE_CONTAINER_S = 120

Runner = Callable[[list[str]], subprocess.CompletedProcess]
Lister = Callable[[], "list[gpu_jobs.JobContainer] | None"]
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
    state_dir: Path = gpu_jobs.STATE_DIR


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
        **({"state_dir": Path(raw["state_dir"])} if raw.get("state_dir") else {}),
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
    return gpu_jobs.meminfo(meminfo)["MemAvailable"] // MB_BYTES


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
    comment: str = ""

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
    return NodeInfo(str(node["name"]), DRAIN_FLAG in states, str(node.get("reason") or ""), str(node.get("extra") or ""),
                    str(node.get("comment") or ""))


@dataclass(frozen=True)
class Running:
    job: gpu_jobs.QueuedJob
    started: int
    excluded: str
    batch: bool  # a batch script: the only kind that runs the wrapper, which is what makes a job safe to requeue
    requeueable: bool
    uid: int | None  # the submitter's uid, which owns the events written for the job

    @property
    def sanctioned(self) -> bool:
        """A job a part-time machine may run: batch, requeueable and a `resume` job."""
        return self.batch and self.requeueable and self.job.job_class is gpu_jobs.JobClass.RESUME


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
            uid = entry.get("user_id")
            here.append(Running(
                job, int(started.get("number") or 0), str(entry.get("excluded_nodes") or ""),
                entry.get("batch_flag") is True, entry.get("requeue") is True,
                uid if isinstance(uid, int) and not isinstance(uid, bool) else None,
            ))
    return urgent, here, {job.job_id for job in jobs}


def push_victim(running: list[Running]) -> Running | None:
    """The `resume` job started last, outside the urgent and service partitions."""
    candidates = [
        r for r in running
        if r.job.job_class is gpu_jobs.JobClass.RESUME
        and r.job.partition not in {gpu_jobs.Partition.URGENT, gpu_jobs.Partition.SERVICE}
    ]
    return max(candidates, key=lambda r: r.started, default=None)


def exclusion(run: Runner, existing: str, node: str) -> str | None:
    """`ExcNodeList` value that adds `node` to a job's `existing` one (a slurm host list, joined raw, never split on
    commas: `n[1,3]` is one entry); None when it already covers `node`."""
    if not existing:
        return node
    done = run(["scontrol", "show", "hostnames", existing])
    if done.returncode != 0:
        raise RuntimeError(f"show hostnames {existing!r} failed: {done.stderr.strip()}")
    return None if node in done.stdout.split() else f"{existing},{node}"


def has_other_node(run: Runner, job_id: str, node: str) -> bool:
    """Whether the job may run on a node besides `node`, by the partitions it asked for (a queued job lists them all):
    a job barred from the only node it can run on never runs."""
    done = run(["scontrol", "show", "job", job_id, "-o"])
    asked = re.search(r"\bPartition=(\S+)", done.stdout) if done.returncode == 0 else None
    if not asked:
        raise RuntimeError(f"show job {job_id} failed: {done.stderr.strip() or 'no Partition'}")
    for partition in asked[1].split(","):
        listed = re.search(r"\bNodes=(\S+)", (shown := run(["scontrol", "show", "partition", partition, "-o"])).stdout)
        if shown.returncode != 0 or not listed:
            raise RuntimeError(f"show partition {partition!r} failed: {shown.stderr.strip() or 'no Nodes'}")
        names = run(["scontrol", "show", "hostnames", listed[1]])
        if names.returncode != 0:
            raise RuntimeError(f"show hostnames {listed[1]!r} failed: {names.stderr.strip()}")
        if any(name != node for name in names.stdout.split()):
            return True
    return False


def bar_refusing_jobs(run: Runner, events_dir: Path, node: str) -> list[str]:
    """Apply the `launch_refused` events in `events_dir`: bar each pending job from `node`, owner-checked. An event of a
    job that is gone, not the event owner's, or not waiting a minute after the event, is deleted without effect; a
    younger one whose job is not waiting yet (the requeue takes seconds to land) waits for the next run."""
    decisions = []
    for path in sorted(events_dir.glob(f"*-{gpu_jobs.EventAction.LAUNCH_REFUSED}.json")):
        try:
            status = path.lstat()
        except OSError:
            continue
        match = LAUNCH_REFUSED_FILE.fullmatch(path.name)
        if not match or not stat.S_ISREG(status.st_mode):
            continue
        job = match[1]
        done = run(["scontrol", "show", "job", job, "-o"])
        shown = done.stdout if done.returncode == 0 else ""
        state = re.search(r"\bJobState=(\S+)", shown)
        owner = re.search(r"\bUserId=\S+\((\d+)\)", shown)
        if state and state[1] != PENDING and time.time() - status.st_mtime < LAUNCH_EVENT_WAIT_S:
            continue  # the requeue has not taken effect yet
        if not state or state[1] != PENDING:
            decisions.append(f"launch event of {job}: job {'is not waiting' if state else 'gone'}, removed")
        elif not owner or int(owner[1]) != status.st_uid:
            decisions.append(f"launch event of {job}: not written by the job's owner, removed")
        else:
            existing = re.search(r"\bExcNodeList=(\S+)", shown)
            value = exclusion(run, "" if not existing or existing[1] == "(null)" else existing[1], node)
            if value:
                done = run(["scontrol", "update", f"JobId={job}", f"ExcNodeList={value}"])
                if done.returncode != 0:  # a person's job must never close the node: say so, try again next run
                    decisions.append(f"bar {job} from {node} failed ({done.stderr.strip()}): event kept")
                    continue
            decisions.append(f"{job} barred from {node} (launch check)")
        gpu_jobs.remove_quietly(path)
    return decisions


def end_job(run: Runner, record: Recorder, victim: Running, node: str, reason: str, **fields: object) -> None:
    """Cancel a job here and record the kill, the event owned by the job's submitter (the manifest says so)."""
    job = victim.job.job_id
    done = run(["scancel", job])
    if done.returncode != 0:
        raise RuntimeError(f"scancel {job} failed: {done.stderr.strip()}")
    event(record, victim, gpu_jobs.EventAction.KILL, node=node, reason=reason, **fields)


def event(record: Recorder, victim: Running, action: gpu_jobs.EventAction, **fields: object) -> None:
    """Record an event for a job. The file is root's, and the controller trusts a copied file only when the manifest
    lists it with the uid of the job's user, so it is handed to that user (best effort: unlisted, it is not shown)."""
    path = record(victim.job.job_id, action, **fields)
    if isinstance(path, Path) and victim.uid is not None:
        with contextlib.suppress(OSError):
            os.chown(path, victim.uid, -1)


def write_manifest(state_dir: Path) -> str:
    """Write `owners/manifest.json`: every regular file under the state subdirectories (lstat; a link is skipped) and
    the uid that owns it, for the controller to trust the copy by (B7.1). Only when `owners/` exists and is ours alone
    (made by `apply --part-time`); returns why not, "" when written."""
    owners = state_dir / OWNERS_DIR
    try:
        status = owners.lstat()
    except FileNotFoundError:
        return ""  # not a part-time machine
    if not stat.S_ISDIR(status.st_mode) or status.st_uid != os.geteuid() or status.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        return f"{owners} is not a directory only we can write: manifest not written"
    files: dict[str, int] = {}
    for sub in MANIFEST_DIRS:
        top = state_dir / sub
        if top.is_symlink():
            continue
        for directory, _, names in os.walk(top):
            for name in names:
                path = Path(directory) / name
                with contextlib.suppress(OSError):
                    status = path.lstat()
                    if stat.S_ISREG(status.st_mode):
                        files[path.relative_to(state_dir).as_posix()] = status.st_uid
    gpu_jobs.write_json(owners / MANIFEST_NAME, {"files": files})
    return ""


def sweep_containers(containers: Lister, unfinished: set[str], run: Runner, map_path: Path, now: float) -> list[str]:
    """Rule 5: write the map of this node's labelled containers and remove those whose job has ended. `unfinished` and
    `now` come from the same moment (the queue read), so a job that started after it is protected by the age. The
    decisions taken; never raises (a failure is a decision line), because any exception here would drain the node.
    A container whose removal failed stays in the map, so `status.py` keeps listing it."""
    try:
        found = containers()
        if found is None:  # no docker here, or it did not answer: readers must not trust an old map
            map_path.unlink(missing_ok=True)
            return []
        stale = [c for c in found if c.job_id not in unfinished and now - c.created > STALE_CONTAINER_S]
        decisions, removed = [], []
        if stale:
            done = run(["docker", "rm", "-f", *(c.id for c in stale)])
            if done.returncode == 0:
                removed = stale
                decisions.append(f"removed container(s) of ended jobs: {_names(stale)}")
            else:
                decisions.append(f"could not remove container(s) of ended jobs {_names(stale)}: {done.stderr.strip()[:200]}")
        kept = [c for c in found if c not in removed]
        if young := [c for c in kept if c.job_id not in unfinished and c not in stale]:
            decisions.append(f"kept new container(s) of no unfinished job, for now: {_names(young)}")
        map_path.parent.mkdir(mode=0o755, exist_ok=True)
        gpu_jobs.write_json(map_path, {"at": int(now), "containers": [dataclasses.asdict(c) for c in kept]})
        return decisions
    except Exception as error:  # noqa: BLE001 - rule 5 never closes the node; the log line is the record
        return [f"containers not checked: {type(error).__name__}: {error}"]


def _names(containers: list[gpu_jobs.JobContainer]) -> str:
    return ", ".join(f"{c.short} (job {c.job_id})" for c in containers)


def drain(run: Runner, node: str, why: str) -> None:
    done = run(["scontrol", "update", f"NodeName={node}", "State=DRAIN", f"Reason={REASON_PREFIX}{why}"])
    if done.returncode != 0:
        raise RuntimeError(f"drain failed: {done.stderr.strip()}")


def read_comment(text: object) -> dict[str, float]:
    """The reading in a node's comment (`report_free`), {} when it is not one."""
    found = dict(part.split("=", 1) for part in str(text or "").split(";") if "=" in part)
    if not str(text or "").startswith(COMMENT_PREFIX) or not all(re.fullmatch(r"\d+(\.\d+)?", v) for v in found.values()):
        return {}
    return {k: float(v) for k, v in found.items()}


def report_free(run: Runner, info: NodeInfo, free: int, use: dict[str, float], gpu_busy: float | None = None) -> None:
    """Put what the check judged by, and what the machine is doing, on the node, where the page reads it (slurm's own
    `FreeMem` leaves out the cache, so it is not the figure the limits are about). Display only: a failure never closes
    the node."""
    value = f"{COMMENT_PREFIX}{free}"
    if use:
        value += f";total_mb={use['mem_total'] // MB_BYTES};cores={int(use['cores'])};cores_busy={use['cores_busy']:.1f}"
        if gpu_busy is not None:
            value += f";gpu_busy={gpu_busy:.2f}"
    if value != info.comment and (not info.comment or info.comment.startswith(COMMENT_PREFIX)):
        run(["scontrol", "update", f"NodeName={info.name}", f"Comment={value}"])


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
    hostname: str | None = None, gpu_busy: Callable[[], float | None] = gpu_jobs.gpu_busy,
    sleep: Callable[[float], None] = time.sleep, containers: Lister = gpu_jobs.job_containers,
    container_map: Path = gpu_jobs.CONTAINER_MAP,
) -> list[str]:
    """One run of the rules; the decisions taken (empty = allow). Raises on any failure: the caller fails closed."""
    host = (hostname or gpu_jobs.this_host()).split(".")[0]
    info = read_node(run, host)
    settings, note = apply_extra(load_settings(settings_path, (info.name, host)), info.extra)
    free, (on_mains, battery) = free_mb(settings.meminfo_path), power(settings.power_supply_dir)
    urgent, running, unfinished = read_queue(run, info.name)
    queue_read = time.time()  # rule 5 judges a container's age from here: a job started after this has no entry
    state = read_state(settings.state_path)
    state["pushes"] = {k: n for k, n in state["pushes"].items() if k.split("@")[0] in unfinished}  # a pushed job waits in the queue: keep it
    decisions = [note] if note else []

    for stray in [r for r in running if r.job.partition == gpu_jobs.Partition.PART_TIME and not r.sanctioned]:
        end_job(run, record, stray, info.name, KILL_REASON)
        decisions.append(f"cancel {stray.job.job_id} ({KILL_REASON})")
    running = [r for r in running if r.job.partition != gpu_jobs.Partition.PART_TIME or r.sanctioned]

    if free < settings.push_back_below_mb and (victim := push_victim(running)):
        job = victim.job.job_id
        key = f"{job}@{info.name}"
        done = run(["scontrol", "requeue", job])
        if done.returncode != 0:  # slurm will not requeue it: a job that cannot be pushed back must not stay and starve the owner
            end_job(run, record, victim, info.name, "requeue failed", free_mb=free)
            decisions.append(f"requeue {job} failed ({done.stderr.strip()}): cancelled ({free} MB free)")
        else:
            state["pushes"][key] = state["pushes"].get(key, 0) + 1
            event(record, victim, gpu_jobs.EventAction.PUSH_BACK, node=info.name, free_mb=free)
            decisions.append(f"push back {job} ({free} MB free)")
            if (state["pushes"][key] >= PUSHES_BEFORE_EXCLUDE and has_other_node(run, job, info.name)
                    and (value := exclusion(run, victim.excluded, info.name))):
                done = run(["scontrol", "update", f"JobId={job}", f"ExcNodeList={value}"])
                if done.returncode != 0:
                    raise RuntimeError(f"exclude on {job} failed: {done.stderr.strip()}")
                decisions.append(f"{job} barred from {info.name}")

    decisions += bar_refusing_jobs(run, settings.state_dir / "events", info.name)

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
    report_free(run, info, free, gpu_jobs.machine_use(lambda path: (settings.meminfo_path if path == "/proc/meminfo" else Path(path)).read_text(encoding="ascii"), sleep), gpu_busy())
    try:
        if skipped := write_manifest(settings.state_dir):
            log_line(settings.log_path, f"{info.name} {skipped}")
    except OSError as error:  # a manifest that is stale only hides copied files; it never closes the node
        log_line(settings.log_path, f"{info.name} manifest not written: {error}")
    # last before the log line: a slow docker must not keep the rules' state, the report or the manifest unwritten
    decisions += sweep_containers(containers, unfinished, run, container_map, queue_read)
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
        fail_closed(error, run_command, gpu_jobs.this_host(), LOG_PATH)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
