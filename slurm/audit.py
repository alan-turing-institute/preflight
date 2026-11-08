#!/usr/bin/env python3
"""List GPU work on this machine that belongs to no slurm job and no live GPU lease.

Design 3.10. A GPU compute process is ACCOUNTED FOR when its cgroup is a slurm job's (`job_<id>` in
its path), or it shares the systemd unit cgroup of a live lease's pid, or it descends from that pid
(a lease names the OWNER process, a service's or a queue's, not the GPU process itself), or it runs in
a docker container labelled `gpu-lease-project=<project>` while that project holds a live lease (a
container's processes descend from containerd, not from the lease owner), or in a container labelled
`slurm-job=<J>` while J runs on this node holding a GPU or shards and the container was made during J's current
run (`job_container_reason`). Anything else is printed as unregistered, a labelled container with the reason.
Prints only; the in-job occupancy check is what blocks a launch.
"""

from __future__ import annotations

import argparse
import functools
import sys
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gpu_jobs  # noqa: E402 - path set just above
from gpu_jobs import cgroup_of  # noqa: E402 - also used as audit.cgroup_of
from job_wrapper import gpu_mib_by_pid  # noqa: E402


def parent_of(pid: int, proc: Path = Path("/proc")) -> int:
    try:
        stat = (proc / str(pid) / "stat").read_text(encoding="utf-8")
    except OSError:
        return 0
    return int(stat.rsplit(")", 1)[1].split()[1])


def descends_from(pid: int, ancestor: int, proc: Path = Path("/proc")) -> bool:
    seen = set()
    while pid > 1 and pid not in seen:
        if pid == ancestor:
            return True
        seen.add(pid)
        pid = parent_of(pid, proc)
    return False


def lease_pids(lease_dir: Path) -> dict[int, str]:
    pids = {}
    for path in lease_dir.glob("*.json"):
        record = gpu_jobs.read_json(path) or {}
        pid = record.get("pid")
        if gpu_jobs.pid_alive(pid):
            pids[int(pid)] = str(record.get("project") or path.stem)
    return pids


def command_of(pid: int, proc: Path = Path("/proc")) -> str:
    try:
        text = (proc / str(pid) / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
    except OSError:
        return "?"
    return gpu_jobs.printable(text)  # any user's process: shown in admins' terminals


class Stray(NamedTuple):
    """GPU work no job and no lease accounts for; `why` says why a labelled container is not excused."""

    pid: int
    mib: float
    command: str
    why: str = ""


def job_container_reason(container: gpu_jobs.JobContainer, queue: list[gpu_jobs.QueuedJob] | None, here: str,
                         gpus_here: int | None) -> str:
    """Why GPU work in a container labelled `slurm-job=J` is not slurm's allocation of this card; "" when it is: J runs
    on this node, holds a GPU or shards, the node has one GPU (docker is not held to slurm's devices, so on several
    the GPU J holds would have to be matched), and the container was made during J's current run (not left by an
    earlier run of the same id). A multi-node job's node list never equals one host: never excused."""
    job_id = container.job_id
    if queue is None:
        return f"container of job {job_id}; the queue could not be read"
    job = next((j for j in queue if j.job_id == job_id), None)
    if job is None or job.state != gpu_jobs.State.RUNNING:
        return f"container of job {job_id}, which is not running"
    if job.node != here:
        return f"container of job {job_id}, which runs on {job.node or 'no node'}, not here"
    if not job.uses_card:
        return f"container of job {job_id}, which holds no GPU or shards"
    if gpus_here != 1:
        return f"container of job {job_id}, on a node with {gpus_here or 'an unknown number of'} GPUs: which one it holds is not checked"
    if not container.of_run(job):
        return f"container of job {job_id}, made before its current run started"
    return ""


def unregistered(
    usage: dict[int, float] | None = None,
    lease_dir: Path | None = None,
    proc: Path = Path("/proc"),
    label_of: Callable[[str, str], str] = gpu_jobs.container_label,
    *,
    containers: Callable[[], list[gpu_jobs.JobContainer] | None] | None = None,  # default gpu_jobs.known_containers
    queue: Callable[[], list[gpu_jobs.QueuedJob] | None] | None = None,  # default gpu_jobs.read_queue
    here: str | None = None,
    gpus_here: Callable[[str], int | None] | None = None,
) -> list[Stray]:
    """GPU work no job and no lease accounts for, a labelled container's with the reason. The job
    containers and the queue are asked for once, and only when some process is not otherwise accounted for; either not
    answering (None) excuses nothing."""
    usage = gpu_mib_by_pid() if usage is None else usage
    containers, queue = containers or gpu_jobs.known_containers, queue or gpu_jobs.read_queue  # looked up when called
    lease_dir = lease_dir or gpu_jobs.LEASE_DIR
    leases = lease_pids(lease_dir) if lease_dir else {}
    lease_cgroups = {cgroup_of(pid, proc) for pid in leases} - {""}
    found = []
    host = here or gpu_jobs.this_host()
    # each asked once, and only when a process is not otherwise accounted for
    listed, read_queue = functools.cache(containers), functools.cache(queue)
    gpus = functools.cache(lambda: (gpus_here or _gpus_on)(host))
    for pid, mib in sorted(usage.items()):
        cgroup = cgroup_of(pid, proc)
        if "/job_" in cgroup or "slurmstepd.scope/" in cgroup or cgroup in lease_cgroups or any(descends_from(pid, owner, proc) for owner in leases):
            continue
        if label_of(cgroup, "gpu-lease-project") in leases.values():
            continue
        container = next((c for c in listed() or [] if gpu_jobs.within(cgroup, c.cgroup)), None)
        why = ""
        if container is not None:
            why = job_container_reason(container, read_queue(), host, gpus())
            if not why:
                continue
        found.append(Stray(pid, mib, command_of(pid, proc), why))
    return found


def _gpus_on(host: str) -> int | None:
    """How many GPUs the site file gives this host's node; None when it is not a node of the site."""
    return next((len(n.gpus) for n in gpu_jobs.load_site().nodes if n.name == host), None)


def leftover_containers(containers: list[gpu_jobs.JobContainer] | None, queue: list[gpu_jobs.QueuedJob] | None,
                        here: str, now: float) -> list[str]:
    """Report lines for labelled containers no current run on this node owns (its job ended, runs elsewhere, or the
    container is an earlier run's). Slurm's Epilog removes them when the job ends and the health check sweeps what it
    missed, so one listed for minutes is one neither could remove. Nothing when the containers or the queue cannot be
    read (no health-check map and no docker access)."""
    if containers is None or queue is None:
        return []
    by_id = {job.job_id: job for job in queue if job.active and job.node == here}  # an ending job's are still its own
    stale = [c for c in containers if not (job := by_id.get(c.job_id)) or not c.of_run(job)]
    return [f"[audit] container {c.short} of job {c.job_id}, made {int(now - c.created) // 60} min ago, "
            f"{'running' if c.cgroup else 'stopped'}: its job's current run is not on this node" for c in stale]


def report(found: list[Stray]) -> list[str]:
    if not found:
        return ["[audit] no GPU work outside slurm and the lease protocol"]
    lines = ["[audit] GPU work outside slurm and outside the lease protocol (unregistered):"]
    lines += [f"[audit]   pid {s.pid}, {s.mib:.0f} MiB: {s.command[:160]}" + (f" ({s.why})" if s.why else "") for s in found]
    return lines


def main() -> int:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    print("\n".join(report(unregistered())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
