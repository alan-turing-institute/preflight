#!/usr/bin/env python3
"""List GPU work on this machine that belongs to no slurm job and no live GPU lease.

Design 3.10. A GPU compute process is ACCOUNTED FOR when its cgroup is a slurm job's (`job_<id>` in
its path), or it shares the systemd unit cgroup of a live lease's pid, or it descends from that pid
(a lease names the OWNER process, a service's or a queue's, not the GPU process itself), or it runs in
a docker container labelled `gpu-lease-project=<project>` while that project holds a live lease (a
container's processes descend from containerd, not from the lease owner). Anything else is printed as
unregistered. Prints only; the in-job occupancy check is what blocks a launch.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gpu_jobs  # noqa: E402 - path set just above
from job_wrapper import gpu_mib_by_pid  # noqa: E402


def cgroup_of(pid: int, proc: Path = Path("/proc")) -> str:
    try:
        return (proc / str(pid) / "cgroup").read_text(encoding="utf-8").strip().split("::", 1)[-1]
    except OSError:
        return ""


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


def unregistered(
    usage: dict[int, float] | None = None,
    lease_dir: Path | None = None,
    proc: Path = Path("/proc"),
    label_of: Callable[[str, str], str] = gpu_jobs.container_label,
) -> list[tuple[int, float, str]]:
    usage = gpu_mib_by_pid() if usage is None else usage
    lease_dir = lease_dir or gpu_jobs.LEASE_DIR
    leases = lease_pids(lease_dir) if lease_dir else {}
    lease_cgroups = {cgroup_of(pid, proc) for pid in leases} - {""}
    found = []
    for pid, mib in sorted(usage.items()):
        cgroup = cgroup_of(pid, proc)
        if "/job_" in cgroup or "slurmstepd.scope/" in cgroup or cgroup in lease_cgroups or any(descends_from(pid, owner, proc) for owner in leases):
            continue
        if label_of(cgroup, "gpu-lease-project") in leases.values():
            continue
        found.append((pid, mib, command_of(pid, proc)))
    return found


def report(found: list[tuple[int, float, str]]) -> list[str]:
    if not found:
        return ["[audit] no GPU work outside slurm and the lease protocol"]
    lines = ["[audit] GPU work outside slurm and outside the lease protocol (unregistered):"]
    lines += [f"[audit]   pid {pid}, {mib:.0f} MiB: {command[:160]}" for pid, mib, command in found]
    return lines


def main() -> int:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    print("\n".join(report(unregistered())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
