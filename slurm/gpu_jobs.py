"""Shared pieces of the GPU-job tools: where their state lives, and what slurm says.

Design: `design notes` (revision 2). Every GPU job on this
box goes through slurm; each submission leaves a SIDECAR record (project, class, GPU or not, command),
because slurm's own completion records carry no GPU field, no project and no comment (review finding 15).
The lease keeper, the in-job wrapper, `status.py` and `wait.py` read the same files through this module.

Stdlib only: every tool here runs under the system `python3`, except the keeper, which runs under
`~/dev/localmodels/.venv` so it can use the lease protocol's one implementation.
"""

from __future__ import annotations

import contextlib
import json
import os
import pwd
import re
import subprocess
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import site_config

_LEGACY_STATE_DIR = Path("~/.local/state/slurm-gpu")
_LEGACY_LEASE_DIR = Path("~/.local/state/gpu-lease")


def _site_dirs() -> tuple[Path, Path | None]:
    """(state dir, lease dir) from env, then the site file, then the pre-site-file locations.

    Never fails for a missing site file: until the installer writes one, the box keeps its legacy paths.
    """
    state, lease = os.environ.get("SLURM_GPU_STATE_DIR"), os.environ.get("GPU_LEASE_DIR")
    site = site_config.load() if site_config.SITE_FILE.exists() and not (state and lease) else None
    state_dir = Path(state) if state else site.state_dir if site else _LEGACY_STATE_DIR
    lease_dir = Path(lease) if lease else site.lease.lease_dir if site and site.lease else _LEGACY_LEASE_DIR
    return state_dir.expanduser(), lease_dir.expanduser()


_STATE, _LEASE = _site_dirs()
#: Where the tools keep their records (sidecars, events, footprints, the keeper's state). Overridable for tests.
STATE_DIR = _STATE
#: The cross-project GPU lease directory (service-c's protocol document); same override as localmodels.
LEASE_DIR = _LEASE
#: The project name slurm's keeper holds the lease under.
LEASE_PROJECT = "slurm"
#: slurm's completion log on this box (`JobCompLoc` in machine/slurm/slurm.conf).
JOBCOMP_LOG = Path(os.environ.get("SLURM_GPU_JOBCOMP", "/var/log/slurm/jobcomp.log"))


class Partition(StrEnum):
    """The three partitions of machine/slurm/slurm.conf."""

    NORMAL = "normal"  # resumes from its own checkpoint; pre-emptable
    STEADY = "steady"  # cannot resume; never pre-empted
    URGENT = "urgent"  # urgent.py only; pre-empts `normal`


class JobClass(StrEnum):
    """How a job pauses: the closed set `submit.py` accepts."""

    RESUME = "resume"
    RESTART = "restart"


#: The partition each class runs in.
PARTITION_OF = {JobClass.RESUME: Partition.NORMAL, JobClass.RESTART: Partition.STEADY}


class State(StrEnum):
    """The squeue job states these tools act on (`man squeue`, JOB STATE CODES)."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    CONFIGURING = "CONFIGURING"
    COMPLETING = "COMPLETING"
    SUSPENDED = "SUSPENDED"


#: States in which a job holds (or is about to hold) its resources.
ACTIVE_STATES = frozenset({State.RUNNING, State.CONFIGURING, State.COMPLETING, State.SUSPENDED})
#: Pending reasons meaning a person or the requeue limit holds the job: it will not start by itself.
#: "job requeued in held state" is what squeue's %r shows after `scontrol requeuehold` (stage C test 9).
HELD_REASONS = frozenset(
    {"JobHeldUser", "JobHeldAdmin", "DependencyNeverSatisfied", "job requeued in held state"}
)

#: The pending reason of a job waiting for the jobs it depends on (`man squeue`, JOB REASON CODES).
DEPENDENCY_REASON = "Dependency"

#: Name last, so a `|` inside it is kept. User, priority and GRES (`%b`) follow the original six fields.
_SQUEUE_FORMAT = "%i|%P|%T|%r|%E|%u|%Q|%b|%j"
#: Explicit, because with MinJobAge a day slurm keeps finished jobs listable; these tools want unfinished ones only.
_UNFINISHED = ",".join(state.lower() for state in State)


@dataclass(frozen=True)
class QueuedJob:
    """One row of `squeue`."""

    job_id: str
    partition: str
    state: str
    reason: str
    name: str
    dependency: str = ""  # squeue's %E, e.g. "afterok:65(unfulfilled)"; "(null)" or "" for none
    user: str = ""
    priority: int = 0
    gpus: int = 0

    @property
    def depends_on(self) -> tuple[str, ...]:
        """The ids of the jobs this one still waits for."""
        return tuple(re.findall(r":(\d+)", self.dependency))

    @property
    def held(self) -> bool:
        """Pending and held: it will not start until a person (or the requeue limit) releases it."""
        return self.state == State.PENDING and self.reason in HELD_REASONS

    @property
    def active(self) -> bool:
        return self.state in ACTIVE_STATES

    @property
    def waiting(self) -> bool:
        """Pending and able to start by itself (not held)."""
        return self.state == State.PENDING and self.reason not in HELD_REASONS


def parse_gpus(gres: str) -> int:
    """GPUs in squeue's `%b`: `gres/gpu:2`, `gpu:gb10:2`, `gres/gpu:gb10:2`; `gres/gpu` alone is 1; `N/A` is 0."""
    total = 0
    for item in gres.split(","):
        match = re.fullmatch(r"(?:gres/)?gpu(?::[A-Za-z][\w.-]*)?(?::(\d+))?(?:\(.*\))?", item.strip())
        if match:
            total += int(match[1] or 1)
    return total


def parse_squeue(text: str) -> list[QueuedJob]:
    """Rows of `squeue -h -o _SQUEUE_FORMAT`; the name is last, so a `|` inside it is kept."""
    jobs = []
    for line in text.splitlines():
        parts = line.strip().split("|", 8)
        if len(parts) == 9:  # noqa: PLR2004 - the nine fields of _SQUEUE_FORMAT
            job_id, partition, state, reason, dependency, user, priority, gres, name = parts
            jobs.append(
                QueuedJob(
                    job_id, partition, state, reason, name, dependency, user,
                    int(priority) if priority.isdigit() else 0, parse_gpus(gres),
                )
            )
    return jobs


def held_back(job: QueuedJob, jobs: dict[str, QueuedJob], _seen: frozenset[str] = frozenset()) -> bool:
    """Pending only on jobs that are held, directly or down a chain: it cannot start until someone releases them.

    A dependency no longer in the queue has just finished, so the job may be about to start: not held back.
    """
    if job.reason != DEPENDENCY_REASON or not job.depends_on or job.job_id in _seen:
        return False
    before = [jobs.get(job_id) for job_id in job.depends_on]
    return all(
        other is not None and (other.held or held_back(other, jobs, _seen | {job.job_id})) for other in before
    )


def read_queue() -> list[QueuedJob] | None:
    """Every job slurm knows that has not finished, or None when slurm does not answer."""
    try:
        done = subprocess.run(
            ["squeue", "-h", "-t", _UNFINISHED, "-o", _SQUEUE_FORMAT],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return parse_squeue(done.stdout) if done.returncode == 0 else None


def write_json(path: Path, payload: dict[str, object]) -> None:
    """Write atomically (temp file, then rename), so a reader never sees half a record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1, sort_keys=True)
    Path(temp).replace(path)


def read_json(path: Path) -> dict[str, object] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return data if isinstance(data, dict) else None


def sidecar_path(job_id: str) -> Path:
    return STATE_DIR / "jobs" / f"{job_id}.json"


def read_sidecar(job_id: str) -> dict[str, object] | None:
    return read_json(sidecar_path(job_id))


def update_sidecar(job_id: str, **fields: object) -> None:
    """Merge fields into a job's sidecar (read, update, atomic write)."""
    record = read_sidecar(job_id) or {"job_id": job_id}
    record.update(fields)
    write_json(sidecar_path(job_id), record)


def needs_the_card(job: QueuedJob) -> bool:
    """A job the background services must step aside for: a GPU job, or any urgent job (design 3.2)."""
    if job.partition == Partition.URGENT:
        return True
    record = read_sidecar(job.job_id)
    return bool(record and record.get("gpu"))


def pid_alive(pid: object) -> bool:
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, OverflowError):
        return False
    except PermissionError:  # alive, another user's
        return True
    return True


def slurm_lease_admitted(lease_dir: Path | None = None) -> bool:
    """Whether the keeper holds the GPU lease for slurm: a live pid and no outstanding request."""
    record = read_json((lease_dir or LEASE_DIR) / f"{LEASE_PROJECT}.json")
    return bool(record) and record.get("requested_at") is None and pid_alive(record.get("pid"))


def boot_time() -> float:
    """When this boot began (`btime` in /proc/stat)."""
    for line in Path("/proc/stat").read_text(encoding="utf-8").splitlines():
        if line.startswith("btime "):
            return float(line.split()[1])
    return 0.0


def remove_quietly(path: Path) -> None:
    with contextlib.suppress(OSError):
        path.unlink(missing_ok=True)


def completion_records(log: Path | None = None) -> dict[str, dict[str, str]]:
    """The LAST completion record of each job id in slurm's `jobcomp/filetxt` log (one leg per line)."""
    records: dict[str, dict[str, str]] = {}
    try:
        lines = (log or JOBCOMP_LOG).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return records
    for line in lines:
        fields = dict(part.split("=", 1) for part in line.split() if "=" in part)
        if "JobId" in fields:
            records[fields["JobId"]] = fields
    return records


def measured_peak_bytes(name: str) -> int | None:
    """The footprint of the newest finished run called `name`: GPU plus CPU side, from its sidecar."""
    best: tuple[int, int] | None = None
    for path in (STATE_DIR / "jobs").glob("*.json"):
        record = read_json(path) or {}
        leg = record.get("last_leg")
        if record.get("name") != name or not isinstance(leg, dict) or "peak_gpu_mib" not in leg:
            continue
        total = int(float(leg["peak_gpu_mib"]) * 1024 * 1024) + int(leg.get("peak_mem_bytes") or 0)
        ended = int(leg.get("ended_at") or 0)
        if best is None or ended > best[0]:
            best = (ended, total)
    return None if best is None else best[1]


def node_state() -> str:
    """The node's state and reason (`ReturnToService=1` leaves it DOWN after a reboot)."""
    done = subprocess.run(["sinfo", "-h", "-N", "-o", "%T|%E"], capture_output=True, text=True, check=False)
    return done.stdout.splitlines()[0] if done.returncode == 0 and done.stdout else "unknown (sinfo failed)"


class EventAction(StrEnum):
    """What happened to a job, as recorded in `STATE_DIR/events/`."""

    PAUSE = "pause"
    RESUME = "resume"
    KILL = "kill"
    URGENT_HOLD = "urgent_hold"
    RELEASE = "release"
    FAILED = "failed"


@dataclass(frozen=True)
class Event:
    """One event file. `actor` is the file's owner, read from `stat`, never from the content."""

    job: str
    action: str
    actor: str
    at: int  # time_ns, from the file name
    fields: dict[str, object] = field(default_factory=dict)


def write_event(job_id: str, action: EventAction, **fields: object) -> Path:
    """Record an event as `events/<job>-<time_ns>-<action>.json` (0644; the directory is sticky and shared)."""
    at = time.time_ns()
    path = STATE_DIR / "events" / f"{job_id}-{at}-{action}.json"
    write_json(path, {"job": job_id, "action": str(action), "at": at, **fields})
    path.chmod(0o644)
    return path


def _owner(path: Path) -> str:
    uid = path.stat().st_uid
    try:
        return pwd.getpwuid(uid).pw_name
    except KeyError:
        return str(uid)


def events(job_id: str | None = None) -> list[Event]:
    """Every event (of one job, if given), oldest first. Who did it comes from file ownership alone."""
    found = []
    for path in (STATE_DIR / "events").glob("*.json"):
        match = re.fullmatch(r"(.+)-(\d+)-([a-z_]+)\.json", path.name)
        record = read_json(path)
        if not match or record is None or (job_id is not None and match[1] != job_id):
            continue
        extra = {k: v for k, v in record.items() if k not in {"job", "action", "at"}}
        try:
            found.append(Event(match[1], match[3], _owner(path), int(match[2]), extra))
        except OSError:
            continue
    return sorted(found, key=lambda event: event.at)


def _docker_label(container_id: str, label: str) -> str:
    done = subprocess.run(
        ["docker", "inspect", "-f", f'{{{{index .Config.Labels "{label}"}}}}', container_id],
        capture_output=True,
        text=True,
        check=False,
    )
    return done.stdout.strip() if done.returncode == 0 else ""


_DOCKER_SCOPE = re.compile(r"/docker-([0-9a-f]{64})\.scope$")


def container_label(cgroup: str, label: str, runner: Callable[[str, str], str] = _docker_label) -> str:
    """The `label` of the docker container whose scope ends `cgroup`; "" if it is not one or docker fails."""
    match = _DOCKER_SCOPE.search(cgroup)
    if not match:
        return ""
    try:
        return runner(match[1], label)
    except (OSError, subprocess.SubprocessError):
        return ""


def nodes(runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run) -> list[str]:
    """Node names slurm knows (`scontrol show nodes -o`); [] when slurm does not answer."""
    try:
        done = runner(["scontrol", "show", "nodes", "-o"], capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return []
    if done.returncode != 0:
        return []
    return [m[1] for line in done.stdout.splitlines() if (m := re.search(r"(?:^|\s)NodeName=(\S+)", line))]
