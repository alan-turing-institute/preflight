"""Shared pieces of the GPU-job tools: where their state lives, and what slurm says.

Design: `design notes` (revision 2). Every GPU job on the
machine goes through slurm; each submission leaves a SIDECAR record (project, class, GPU or not, command),
because slurm's own completion records carry no GPU field, no project and no comment (review finding 15).
The lease keeper, the in-job wrapper, `status.py` and `wait.py` read the same files through this module.

Stdlib only: every tool here runs under the system `python3`, except the keeper, which runs under the site's
`[lease] python` so it can use the lease protocol's one implementation (localmodels).
"""

from __future__ import annotations

import contextlib
import errno
import json
import os
import pwd
import re
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import site_config
from levels import Level, level_of  # noqa: F401 - re-exported for the tools


def load_site() -> site_config.Site:
    """The site file. Without a usable one the tool stops here with the reason, not a traceback: there is no
    fallback, every machine gets its site file from the installer."""
    try:
        return site_config.load()
    except site_config.SiteError as error:
        sys.exit(f"[gpujobs] REFUSED: {error}")


def _site_dirs() -> tuple[Path, Path | None]:
    """(state dir, lease dir), each from its environment variable, else from the site file, which is read on import
    either way: no tool runs without one."""
    site = load_site()
    state, lease = os.environ.get("SLURM_GPU_STATE_DIR"), os.environ.get("GPU_LEASE_DIR")
    lease_dir = Path(lease) if lease else site.lease.lease_dir if site.lease else None
    return Path(state or site.state_dir).expanduser(), lease_dir.expanduser() if lease_dir else None


_STATE, _LEASE = _site_dirs()
#: Where the tools keep their records (sidecars, events, footprints, the keeper's state). Overridable for tests.
STATE_DIR = _STATE
#: The cross-project GPU lease directory (service-c's protocol document); same override as localmodels. None for a
#: site without a `[lease]` table.
LEASE_DIR = _LEASE
#: The project name slurm's keeper holds the lease under.
LEASE_PROJECT = "slurm"
#: A slurm job id (an array task as `<job>_<task>`). Files in the shared sticky directories are named by whoever made
#: them, so a tool acts on or prints only names that match this.
JOB_ID = re.compile(r"\d+(?:_\d+)?", re.ASCII)


class Partition(StrEnum):
    """The partitions of the rendered `slurm.conf` (`slurm/install/templates/slurm/slurm.conf`)."""

    NORMAL = "normal"  # resumes from its own checkpoint; pre-emptable
    STEADY = "steady"  # cannot resume; never pre-empted
    CPU = "cpu"  # a job without GPUs, either class: kept off the cores held for GPU jobs; slurm never pre-empts it
    URGENT = "urgent"  # urgent.py only; pre-empts `normal`
    SERVICE = "service"  # submit.py --service only (D20): a service's CPU stand-in; suspends `cpu` jobs to start


class JobClass(StrEnum):
    """How a job pauses: the closed set `submit.py` accepts."""

    RESUME = "resume"
    RESTART = "restart"


def class_of(name: str) -> JobClass | None:
    """The class `submit.py` put in a job name (`project:class:name`); None for a job it did not submit."""
    try:
        _, label, _ = name.split(":", 2)
    except ValueError:
        return None
    return JobClass(label) if label in set(JobClass) else None


def partition_for(job_class: JobClass, gpus: int, *, service: bool = False) -> Partition:
    """The partition a job runs in: a service's stand-in in `service`; one without GPUs in `cpu` (one partition, so its
    MaxCPUsPerNode caps all CPU-only jobs together: slurm counts that limit per partition), else by its class."""
    if service:
        return Partition.SERVICE
    if not gpus:
        return Partition.CPU
    return Partition.NORMAL if job_class is JobClass.RESUME else Partition.STEADY


class State(StrEnum):
    """The squeue job states these tools act on (`man squeue`, JOB STATE CODES)."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    CONFIGURING = "CONFIGURING"
    COMPLETING = "COMPLETING"
    SUSPENDED = "SUSPENDED"


class Reason(StrEnum):
    """The squeue pending reasons these tools and the jobs pages act on (`man squeue`, JOB REASON CODES)."""

    NONE = "None"
    DEPENDENCY = "Dependency"
    DEPENDENCY_NEVER = "DependencyNeverSatisfied"
    HELD_USER = "JobHeldUser"
    HELD_REQUEUE = "job requeued in held state"  # squeue's %r after `scontrol requeuehold` (stage C test 9)
    HELD_ADMIN = "JobHeldAdmin"
    RESOURCES = "Resources"
    PRIORITY = "Priority"
    BEGIN_TIME = "BeginTime"
    PARTITION_CONFIG = "PartitionConfig"
    ACCOUNT_NOT_ALLOWED = "AccountNotAllowed"


#: States in which a job holds (or is about to hold) its resources.
ACTIVE_STATES = frozenset({State.RUNNING, State.CONFIGURING, State.COMPLETING, State.SUSPENDED})
#: Pending reasons meaning a person or the requeue limit holds the job: it will not start by itself.
HELD_REASONS = frozenset({Reason.HELD_USER, Reason.HELD_ADMIN, Reason.DEPENDENCY_NEVER, Reason.HELD_REQUEUE})


def printable(text: str) -> str:
    """`text` with every character a terminal acts on (escape sequences, newlines, direction overrides) as `?`.

    Job names and descriptions are whatever their submitter typed, and they reach admins' terminals and the site's
    notify command.
    """
    return "".join(c if c.isprintable() else "?" for c in text)

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
    cpus: int = 0  # cores slurm gave it (asked, while it waits)
    level: Level = Level.NORMAL
    mem_mb: int | None = None  # slurm's memory request now (MB per node; 0 is the whole node), None when it gave none
    cores: frozenset[str] = frozenset()  # "node:socket:core" slurm gave it; a suspended job keeps them for its resume
    submitted: int = 0  # epoch seconds it last entered the queue (a requeue resets it, measured); 0 when unknown

    @property
    def job_class(self) -> JobClass | None:
        return class_of(self.name)

    @property
    def restarts(self) -> bool:
        """A requeue loses its run: a `steady` job, or a `cpu` job not known to resume from a checkpoint."""
        return self.partition == Partition.STEADY or (
            self.partition == Partition.CPU and self.job_class is not JobClass.RESUME
        )

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


def mem_mb(text: str) -> int:
    """slurm memory notation (`24G`, `512M`, plain megabytes) in megabytes."""
    scale = {"K": 1 / 1024, "M": 1, "G": 1024, "T": 1024 * 1024}
    unit = text[-1].upper()
    return int(float(text[:-1]) * scale[unit]) if unit in scale else int(text)


def _number(value: object) -> int | None:
    """A slurm JSON number (`{"set": true, "number": N}`); None when it is not set."""
    return int(value.get("number") or 0) if isinstance(value, dict) and value.get("set") else None


def _memory_mb(raw: dict[str, object], cpus: int) -> int | None:
    """The job's memory request as slurm holds it now, so a later `scontrol update MinMemoryNode=` shows."""
    per_node, per_cpu = _number(raw.get("memory_per_node")), _number(raw.get("memory_per_cpu"))
    if per_node is not None:
        return per_node
    return per_cpu * cpus if per_cpu is not None else None


def parse_squeue(text: str) -> list[QueuedJob]:
    """The unfinished jobs in `squeue --json` output; raises ValueError when it is not that.

    JSON, never a `-o` text format: slurm keeps a newline (or a form feed, NEL, ...) in a job name, and an owner may
    rename their job at any time, so a name could forge whole rows, and the agent acts on rows as an operator
    (measured on 23.11.4, review R). `-t` was not applied with --json there either (finished jobs came back), so
    the states are filtered here.
    """
    jobs = []
    for raw in json.loads(text)["jobs"]:
        states = raw.get("job_state") or [""]
        states = states if isinstance(states, list) else [str(states)]
        # The base state comes first and flags follow (23.11.4 `parsers.c`, JOB_STATE). COMPLETING and CONFIGURING
        # are flags there, and win over the base state as in slurm's text output (`job_state_string`): a cancelled
        # or requeued job still holds its GPU while it saves and stops, so it must stay active.
        state = next((flag for flag in (State.COMPLETING, State.CONFIGURING) if flag in states), states[0])
        if state not in set(State):
            continue
        priority = raw.get("priority") or {}
        submitted = raw.get("submit_time") or {}
        cpus = int((raw.get("cpus") or {}).get("number") or 0)
        jobs.append(
            QueuedJob(
                str(raw["job_id"]), raw.get("partition", ""), state, raw.get("state_reason", ""), raw.get("name", ""),
                raw.get("dependency", ""), raw.get("user_name", ""),
                int(priority.get("number", 0)) if priority.get("set") else 0, parse_gpus(raw.get("tres_per_node") or ""),
                cpus, level_of(str(raw.get("qos") or "")), _memory_mb(raw, cpus), _cores(raw),
                int(submitted.get("number", 0)) if submitted.get("set") else 0,
            )
        )
    return jobs


def _cores(raw: dict) -> frozenset[str]:
    """The cores slurm gave a job (`job_resources.allocated_nodes[].sockets.<s>.cores.<c>`, 23.11 --json)."""
    found = set()
    for node in (raw.get("job_resources") or {}).get("allocated_nodes") or []:
        for socket, entry in (node.get("sockets") or {}).items():
            found.update(f"{node.get('nodename')}:{socket}:{core}" for core in (entry.get("cores") or {}))
    return frozenset(found)


def held_back(job: QueuedJob, jobs: dict[str, QueuedJob], _seen: frozenset[str] = frozenset()) -> bool:
    """Pending only on jobs that are held, directly or down a chain: it cannot start until someone releases them.

    A dependency no longer in the queue has just finished, so the job may be about to start: not held back.
    """
    if job.reason != Reason.DEPENDENCY or not job.depends_on or job.job_id in _seen:
        return False
    before = [jobs.get(job_id) for job_id in job.depends_on]
    return all(
        other is not None and (other.held or held_back(other, jobs, _seen | {job.job_id})) for other in before
    )


def read_queue() -> list[QueuedJob] | None:
    """Every job slurm knows that has not finished, or None when slurm does not answer."""
    try:
        done = subprocess.run(
            ["squeue", "--json", "-t", _UNFINISHED],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if done.returncode != 0:
        return None
    try:
        return parse_squeue(done.stdout)
    except (ValueError, KeyError, TypeError, AttributeError):
        return None


def write_json(path: Path, payload: dict[str, object]) -> None:
    """Write atomically (temp file, then rename), so a reader never sees half a record, readable by everyone.

    The state directories are shared and each file belongs to its writer (plan 3.3, mode 0644): another user's
    status, the web app and the agent (a different user) must be able to read a record or a request they did not write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        os.fchmod(fd, 0o644)  # mkstemp creates 0600
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=1, sort_keys=True)
        Path(temp).replace(path)
    except OSError as error:
        remove_quietly(Path(temp))
        if not os.path.lexists(path):
            raise
        # In a sticky directory a rename over another user's file is refused: say whose file is in the way.
        msg = f"{path} exists and belongs to {_owner(path)}; it cannot be replaced ({error.strerror})"
        raise OSError(error.errno, msg) from error


#: How many numbered names `write_new` tries after the first (`NAME-2.sbatch`, ...).
NEW_NAME_TRIES = 99


def write_new(path: Path, text: str) -> Path:
    """Create a new file holding `text` at `path`, or at the first free `<stem>-2<suffix>`, `-3`, ...; the path used.

    The scripts and logs directories are shared by every user and the names are predictable on purpose, so a file
    (or link) already under a name is never written into or through: a batch script written into another user's file
    could be changed by them before sbatch reads it. A name taken by the same user's earlier submission in the same
    second just moves on to the next number. FileExistsError when every name is taken.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    for n in range(1, NEW_NAME_TRIES + 2):
        candidate = path if n == 1 else path.with_name(f"{path.stem}-{n}{path.suffix}")
        try:
            fd = os.open(candidate, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
        except FileExistsError:
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        return candidate
    msg = f"{path} and its {NEW_NAME_TRIES} numbered names already exist; nothing was written"
    raise FileExistsError(errno.EEXIST, msg)


#: A record is a few kilobytes; a bigger file is not one.
MAX_RECORD_BYTES = 1 << 20


def read_json(path: Path, user: str | None = None) -> dict[str, object] | None:
    """The JSON object in `path`, or None for anything else, whatever the file is.

    The state directories are sticky and writable by everyone, so any user can leave a FIFO, a symlink, a huge or
    hostile file under a name another user's job will use. It is opened without blocking or following a link, and
    only a regular file of at most `MAX_RECORD_BYTES` is read. With `user`, a file that user does not own is
    ignored too: the owner is the identity, never the content (plan 3.3).
    """
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    except OSError:
        return None
    try:
        status = os.fstat(fd)
        if not stat.S_ISREG(status.st_mode):
            return None
        if user and _user_name(status.st_uid) != user:
            return None
        with os.fdopen(fd, "rb", closefd=False) as handle:
            raw = handle.read(MAX_RECORD_BYTES + 1)  # never more, whatever the file's size or growth
        data = json.loads(raw.decode("utf-8")) if len(raw) <= MAX_RECORD_BYTES else None
    except (OSError, ValueError, RecursionError):  # ValueError: bad JSON or bad UTF-8
        return None
    finally:
        os.close(fd)
    return data if isinstance(data, dict) else None


def keeper_state_path() -> Path:
    """The lease keeper's state and heartbeat, in a directory of its own: the keeper runs as the site's keeper user, and
    the state directory itself is root's (the lease component makes `keeper/` that user's)."""
    return STATE_DIR / "keeper" / "keeper.json"


def oom_kill_path(job_id: str, state_dir: Path | None = None) -> Path:
    """The agent's record of the host's OOM guard killing a process of this job (agent/ is the agent's alone, D11's
    reason: a record any user could write could end any job)."""
    return (state_dir or STATE_DIR) / "agent" / "oom_kills" / f"job_{job_id}.json"


def job_of_cgroup(cgroup: str) -> str | None:
    """The slurm job a cgroup path belongs to (its `job_<id>` component), None for one outside every job."""
    match = re.search(r"/job_(\d+)(?:/|$)", cgroup)
    return match[1] if match else None


def sidecar_path(job_id: str) -> Path:
    return STATE_DIR / "jobs" / f"{job_id}.json"


#: Job states that hold their cores (a SUSPENDED job keeps only its memory and GPUs, and its cores for its resume).
ON_CORES = frozenset({State.RUNNING, State.CONFIGURING, State.COMPLETING})


def free_room(queue: list[QueuedJob], cpus: int, memory_mb: int) -> tuple[int, int | None]:
    """(free cores, free memory MB) on a one-node site, from what its jobs hold. Memory is None (unknown) when a job's
    is unknown or the whole node; a suspended job keeps its memory but not its cores."""
    active = [job for job in queue if job.active]
    sizes = [job.mem_mb for job in active]
    free_mb = None if None in sizes or 0 in sizes else memory_mb - sum(s for s in sizes if s)
    return cpus - sum(job.cpus for job in active if job.state in ON_CORES), free_mb


@dataclass(frozen=True)
class Room:
    """What `plan_room` decides: jobs to suspend for `helping`, suspended jobs to resume."""

    suspend: tuple[str, ...] = ()
    resume: tuple[str, ...] = ()
    helping: str | None = None


def plan_room(
    queue: list[QueuedJob], suspended: set[str], cpus: int, memory_mb: int, service_cores: int,
    may_suspend: dict[str, frozenset[str] | None] | None = None, given_up: frozenset[str] = frozenset(),
) -> Room:
    """Room for service jobs on a one-node site of `cpus` cores and `memory_mb` (D20; the agent's step 3b, and
    `submit.py --service --dry-run`).

    Who gets it: the first service job (by priority, then id) waiting for `Resources` only (another reason, such as a
    drained node or a limit, is not cores), not in `given_up`, that fits in the free memory (suspending frees none)
    and in the partition's `service_cores`. Room comes from running cpu jobs, newest first, among those
    `may_suspend[job]` allows (absent or None: any; empty: none, the owner's consent, D9), and only when they cover the
    whole shortfall. Otherwise a suspended job is resumed, oldest first, once none of ITS cores is held by another job:
    slurm resumes a job on the cores it had and does not check them (23.11 `_job_resume_test`; `job_resources.c`:
    "resumed and it's allocated resources are already in use by some other job"); while a running job's cores are
    unknown nothing is resumed. With memory unknown nobody is given room.
    """
    allowed = may_suspend or {}
    free_cores, free_mb = free_room(queue, cpus, memory_mb)
    service = Partition.SERVICE
    service_free = service_cores - sum(j.cpus for j in queue if j.partition == service and j.state in ON_CORES)
    waiting = sorted(
        (j for j in queue if j.partition == service and j.waiting and j.reason == Reason.RESOURCES
         and j.job_id not in given_up and j.cpus <= service_free
         and free_mb is not None and j.mem_mb and j.mem_mb <= free_mb),
        key=lambda j: (-j.priority, int(j.job_id)),
    )
    if waiting:
        job = waiting[0]
        short = job.cpus - free_cores
        if short <= 0:
            return Room()  # it fits: slurm starts it
        consent = allowed.get(job.job_id)
        running = sorted((j for j in queue if j.partition == Partition.CPU and j.state == State.RUNNING
                          and (consent is None or j.job_id in consent)), key=lambda j: int(j.job_id), reverse=True)
        picked, freed = [], 0
        for victim in running:
            if freed >= short:
                break
            picked.append(victim.job_id)
            freed += victim.cpus
        if freed >= short:
            return Room(suspend=tuple(picked), helping=job.job_id)
        # cannot be covered: what is suspended goes back (it would only add to the shortfall if kept)
    on_cores = [j for j in queue if j.state in ON_CORES]
    if any(not j.cores for j in on_cores):
        return Room()  # whose cores a running job holds is unknown: resuming could land on them
    busy = set().union(*(j.cores for j in on_cores))
    resume = []
    for job in sorted((j for j in queue if j.job_id in suspended), key=lambda j: int(j.job_id)):
        if job.cores and not job.cores & busy:
            resume.append(job.job_id)
            busy |= job.cores
    return Room(resume=tuple(resume))


def read_sidecar(job_id: str, user: str | None = None) -> dict[str, object] | None:
    """The job's record. Pass the job's slurm `user` whenever it is known: a record some other user wrote (anyone can
    create `jobs/<id>.json` for an id not yet issued) is then ignored."""
    return read_json(sidecar_path(job_id), user)


def update_sidecar(job_id: str, **fields: object) -> None:
    """Merge fields into a job's sidecar (read, update, atomic write). Only this user's own record is merged into."""
    record = read_sidecar(job_id, current_user()) or {"job_id": job_id}
    record.update(fields)
    write_json(sidecar_path(job_id), record)


def record_queued_job(job_id: str, tool: str, **fields: object) -> None:
    """`update_sidecar` for a job `sbatch` has just queued: the job is real whatever happens here, so a record that
    cannot be written (another user's file in the way, a full disk) is a warning that names the job, not a crash."""
    try:
        update_sidecar(job_id, **fields)
    except OSError as error:
        print(f"[{tool}] WARNING: job {job_id} is queued but its record was not written ({error}); "
              "it runs, without a record", file=sys.stderr)


def needs_the_card(job: QueuedJob) -> bool:
    """A job the background services must step aside for: one that holds GPUs, or any urgent job (design 3.2).

    Decided from what slurm says (`gres`), never from the job's sidecar: any user can leave a record under an id.
    """
    return job.partition == Partition.URGENT or job.gpus > 0


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
    lease_dir = lease_dir or LEASE_DIR
    record = read_json(lease_dir / f"{LEASE_PROJECT}.json") if lease_dir else None
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


def _sacct_jobs(*selection: str) -> list[dict[str, object]] | None:
    try:
        done = subprocess.run(["sacct", "--json", "-X", "-a", *selection], capture_output=True, text=True, timeout=60,
                              check=False)
        return json.loads(done.stdout)["jobs"] if done.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired, ValueError, KeyError, TypeError):
        return None


def completion_records(job_ids: Iterable[str] | None = None) -> dict[str, dict[str, str]] | None:
    """The latest accounting record of each job, `{id: {"JobId", "JobState", "Name", "UserId"}}`; without ids, every
    job of the last day. None when sacct does not answer (never mistaken for "no jobs ended").

    From `sacct --json`. These tools read slurm's jobcomp/filetxt log before, which writes a job name unescaped, so a
    name with a newline forged whole records (another user's job "FAILED"; measured on 23.11.4, review R). A time
    window alone can return an earlier leg of a requeued job (measured), so it only gives the ids, and `-j` their
    latest records. The keys keep the log's field names.
    """
    ids = sorted(set(job_ids)) if job_ids is not None else None
    if ids is None:
        recent = _sacct_jobs("-S", "now-1days")
        if recent is None:
            return None
        ids = sorted({str(job["job_id"]) for job in recent})
    if not ids:
        return {}
    latest = _sacct_jobs("-j", ",".join(ids))
    if latest is None:
        return None
    records: dict[str, dict[str, str]] = {}
    for job in latest:
        states = job.get("state", {}).get("current") or [""]
        records[str(job["job_id"])] = {"JobId": str(job["job_id"]), "JobState": states[0],
                                       "Name": str(job.get("name", "")), "UserId": str(job.get("user", ""))}
    return records


def measured_peak_bytes(name: str) -> int | None:
    """The footprint of the newest run called `name` that finished cleanly (exit 0, no signal): GPU plus CPU side.

    Only this user's own sidecars count: another user's file with the name could understate the peak, and a crashed
    or pre-empted leg stopped early, so its peak is too low to size the next run on.
    """
    best: tuple[int, int] | None = None
    me = current_user()
    for path in (STATE_DIR / "jobs").glob("*.json"):
        record = read_json(path, me) or {}
        if record.get("name") != name:
            continue
        legs = record.get("legs")
        for leg in legs if isinstance(legs, list) else [record.get("last_leg")]:
            if not isinstance(leg, dict) or "peak_gpu_mib" not in leg or leg.get("exit") != 0 or leg.get("signals"):
                continue
            try:
                total = int(float(leg["peak_gpu_mib"]) * 1024 * 1024) + int(leg.get("peak_mem_bytes") or 0)
                ended = int(leg.get("ended_at") or 0)
            except (TypeError, ValueError):
                continue
            if best is None or ended > best[0]:
                best = (ended, total)
    return None if best is None else best[1]


def run_slurm(command: list[str]) -> int:
    """Run a slurm command; on failure print slurm's own message. Returns its exit status."""
    done = subprocess.run(command, capture_output=True, text=True, check=False)
    if done.returncode != 0:
        print(done.stderr.strip() or done.stdout.strip() or f"{command[0]} failed", file=sys.stderr)
    return done.returncode


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
    return path


def _user_name(uid: int) -> str:
    try:
        return pwd.getpwuid(uid).pw_name
    except KeyError:
        return str(uid)


def current_user() -> str:
    return _user_name(os.getuid())


def _owner(path: Path) -> str:
    return _user_name(path.lstat().st_uid)  # a symlink is its creator's, never its target's owner's


def events(job_id: str | None = None) -> list[Event]:
    """Every event (of one job, if given), oldest first. Who did it comes from file ownership alone."""
    found = []
    for path in (STATE_DIR / "events").glob("*.json" if job_id is None else f"{job_id}-*.json"):
        match = re.fullmatch(r"(.+)-(\d+)-([a-z_]+)\.json", path.name)
        if not match or (job_id is not None and match[1] != job_id):
            continue  # judged by the name before the file is opened
        record = read_json(path)
        if record is None:
            continue
        extra = {k: v for k, v in record.items() if k not in {"job", "action", "at"}}
        try:
            found.append(Event(match[1], match[3], _owner(path), int(match[2]), extra))
        except OSError:
            continue
    return sorted(found, key=lambda event: event.at)


#: A container's labels never change, so each is asked of docker once (the job wrapper asks every sample).
_LABELS: dict[tuple[str, str], str] = {}


def _docker_label(container_id: str, label: str) -> str:
    if (container_id, label) in _LABELS:
        return _LABELS[container_id, label]
    done = subprocess.run(
        ["docker", "inspect", "-f", f'{{{{index .Config.Labels "{label}"}}}}', container_id],
        capture_output=True,
        text=True,
        check=False,
    )
    if done.returncode != 0:  # docker down or the container gone: not remembered, asked again next time
        return ""
    _LABELS[container_id, label] = value = done.stdout.strip()
    return value


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


CGROUP_ROOT = Path("/sys/fs/cgroup")
#: The docker label `submit.py`-launched containers carry so their GPU memory counts toward the job.
JOB_LABEL = "slurm-job"


def cgroup_pids(cgroup: Path) -> set[int]:
    pids: set[int] = set()
    for procs in cgroup.rglob("cgroup.procs"):
        try:
            pids |= {int(pid) for pid in procs.read_text(encoding="utf-8").split()}
        except (OSError, ValueError):
            continue
    return pids


def read_int(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def gpu_usage() -> dict[int, float] | None:
    """GPU memory (MiB) per pid, summed over every GPU the pid uses; None when `nvidia-smi` does not answer."""
    try:
        done = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if done.returncode != 0:
        return None
    usage: dict[int, float] = {}
    for line in done.stdout.splitlines():
        pid, _, mib = line.partition(",")
        try:
            usage[int(pid)] = usage.get(int(pid), 0.0) + float(mib)
        except ValueError:
            continue
    return usage


def gpu_busy() -> float | None:
    """How busy the GPUs are, 0-1, averaged over them (`nvidia-smi` utilization.gpu); None when it does not answer, or
    when any GPU gives no number (`[N/A]`): an average over the GPUs that did answer would misstate the box."""
    try:
        done = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    lines = [line.strip() for line in done.stdout.splitlines() if line.strip()] if done.returncode == 0 else []
    if not lines or not all(line.isdigit() for line in lines):
        return None
    return sum(int(line) for line in lines) / 100 / len(lines)


def gpu_mib_by_pid() -> dict[int, float]:
    """`gpu_usage`, with {} for an unanswered `nvidia-smi`."""
    return gpu_usage() or {}


def job_pids(cgroup: Path | None, fallback_pids: set[int], job_id: str, gpu_pids: set[int]) -> set[int]:
    """The cgroup's pids plus those of `gpu_pids` running in a docker container labelled `slurm-job=<id>`."""
    import audit  # lazy: audit imports job_wrapper, which imports this module

    pids = cgroup_pids(cgroup) if cgroup else set(fallback_pids)
    for pid in gpu_pids - pids:
        if container_label(audit.cgroup_of(pid), JOB_LABEL) == job_id:
            pids.add(pid)
    return pids


def capacity(runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run) -> dict[str, int]:
    """Over every node: the cores, memory (MB) and GPUs slurm has to give, and how much of each it has given to jobs
    (`scontrol show nodes -o`), with the number of nodes and how many of them are this host. {} when slurm does not
    answer."""
    try:
        done = runner(["scontrol", "show", "nodes", "-o"], capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return {}
    if done.returncode != 0:
        return {}
    total = dict.fromkeys(("cpus", "cpus_used", "mem_mb", "mem_used_mb", "gpus", "gpus_used", "nodes", "here"), 0)
    host = os.uname().nodename.split(".")[0]
    for line in done.stdout.splitlines():
        fields = dict(part.split("=", 1) for part in line.split() if "=" in part)
        total["nodes"] += 1
        total["here"] += host in {fields.get("NodeName"), fields.get("NodeHostName")}
        gpus = [re.search(r"gres/gpu=(\d+)", fields.get(key, "")) for key in ("CfgTRES", "AllocTRES")]
        for key, value in (("cpus", fields.get("CPUTot")), ("cpus_used", fields.get("CPUAlloc")),
                           ("mem_mb", fields.get("RealMemory")), ("mem_used_mb", fields.get("AllocMem")),
                           ("gpus", gpus[0] and gpus[0][1]), ("gpus_used", gpus[1] and gpus[1][1])):
            total[key] += int(value) if value and str(value).isdigit() else 0
    return total


def _cpu_ticks(stat_text: str) -> tuple[int, int]:
    """(busy, total) jiffies over all cores, from /proc/stat's first line (idle and iowait are not busy)."""
    values = [int(v) for v in stat_text.splitlines()[0].split()[1:]]
    idle = values[3] + (values[4] if len(values) > 4 else 0)
    return sum(values) - idle, sum(values)


def machine_use(
    read: Callable[[str], str] = lambda path: Path(path).read_text(encoding="ascii"),
    sleep: Callable[[float], None] = time.sleep,
    interval_s: float = 0.2,
) -> dict[str, float]:
    """What this host is really using, slurm or not: memory in use (MemTotal - MemAvailable, bytes, as `free`'s
    "available") and how many cores were busy over `interval_s` (/proc/stat). {} when /proc cannot be read."""
    try:
        meminfo = {key: value.split() for key, _, value in (line.partition(":") for line in read("/proc/meminfo").splitlines())}
        total, available = int(meminfo["MemTotal"][0]) * 1024, int(meminfo["MemAvailable"][0]) * 1024  # kB in meminfo
        before = _cpu_ticks(read("/proc/stat"))
        sleep(interval_s)
        after = _cpu_ticks(read("/proc/stat"))
    except (OSError, ValueError, IndexError, KeyError):
        return {}
    cores, ticks = os.cpu_count() or 0, after[1] - before[1]
    busy = cores * (after[0] - before[0]) / ticks if ticks > 0 else 0.0
    return {"mem_total": total, "mem_used": total - available, "cores": cores, "cores_busy": busy}


def nodes(runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run) -> list[str]:
    """Node names slurm knows (`scontrol show nodes -o`); [] when slurm does not answer."""
    try:
        done = runner(["scontrol", "show", "nodes", "-o"], capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return []
    if done.returncode != 0:
        return []
    return [m[1] for line in done.stdout.splitlines() if (m := re.search(r"(?:^|\s)NodeName=(\S+)", line))]
