"""Shared pieces of the GPU-job tools on gpu-node-1: where their state lives, and what slurm says.

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
import subprocess
import tempfile
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

#: Where the tools keep their records (sidecars, footprints, the keeper's state). Overridable for tests.
STATE_DIR = Path(os.environ.get("SLURM_GPU_STATE_DIR", "~/.local/state/slurm-gpu")).expanduser()
#: The cross-project GPU lease directory (service-c's protocol document); same override as localmodels.
LEASE_DIR = Path(os.environ.get("GPU_LEASE_DIR", "~/.local/state/gpu-lease")).expanduser()
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
HELD_REASONS = frozenset({"JobHeldUser", "JobHeldAdmin", "DependencyNeverSatisfied"})

_SQUEUE_FORMAT = "%i|%P|%T|%r|%j"


@dataclass(frozen=True)
class QueuedJob:
    """One row of `squeue`."""

    job_id: str
    partition: str
    state: str
    reason: str
    name: str

    @property
    def active(self) -> bool:
        return self.state in ACTIVE_STATES

    @property
    def waiting(self) -> bool:
        """Pending and able to start by itself (not held)."""
        return self.state == State.PENDING and self.reason not in HELD_REASONS


def parse_squeue(text: str) -> list[QueuedJob]:
    """Rows of `squeue -h -o '%i|%P|%T|%r|%j'`; the name is last, so a `|` inside it is kept."""
    jobs = []
    for line in text.splitlines():
        parts = line.strip().split("|", 4)
        if len(parts) == 5:  # noqa: PLR2004 - the five fields of _SQUEUE_FORMAT
            jobs.append(QueuedJob(*parts))
    return jobs


def read_queue() -> list[QueuedJob] | None:
    """Every job slurm knows that has not finished, or None when slurm does not answer."""
    try:
        done = subprocess.run(
            ["squeue", "-h", "-o", _SQUEUE_FORMAT], capture_output=True, text=True, timeout=30, check=False
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
