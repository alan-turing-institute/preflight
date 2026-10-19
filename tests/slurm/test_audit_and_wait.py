"""audit.py attributes GPU processes; wait.py follows a chain."""

from __future__ import annotations

import json
import os
from pathlib import Path

import audit
import wait


def fake_proc(root: Path, pid: int, cgroup: str, parent: int, command: str = "python x") -> None:
    d = root / str(pid)
    d.mkdir(parents=True)
    (d / "cgroup").write_text(f"0::{cgroup}\n")
    (d / "stat").write_text(f"{pid} (python) S {parent} 0 0\n")
    (d / "cmdline").write_bytes(command.replace(" ", "\0").encode())


def test_attribution_by_job_cgroup_service_cgroup_and_ancestry(tmp_path: Path) -> None:
    proc, leases = tmp_path / "proc", tmp_path / "leases"
    leases.mkdir()
    owner = os.getpid()  # a live pid standing for a service's lease owner
    (leases / "service-a.json").write_text(json.dumps({"pid": owner, "project": "service-a"}))
    fake_proc(proc, owner, "/user.slice/service-a-models.service", 1)
    fake_proc(proc, 101, "/system.slice/gpu-node-1_slurmstepd.scope/job_5/step_batch", 1)
    fake_proc(proc, 102, "/user.slice/service-a-models.service", 1)  # same unit as the lease owner
    fake_proc(proc, 103, "/user.slice/other.scope", owner)  # a child of the lease owner
    fake_proc(proc, 104, "/user.slice/autocast-swe32-patch4.service", 1, "python train patch4")
    # containers: one labelled for the leased project, one for a project with no lease, one unlabelled
    labelled, other, bare = ("a" * 64), ("b" * 64), ("c" * 64)
    fake_proc(proc, 105, f"/system.slice/docker-{labelled}.scope", 2)
    fake_proc(proc, 106, f"/system.slice/docker-{other}.scope", 2)
    fake_proc(proc, 107, f"/system.slice/docker-{bare}.scope", 2)
    labels = {labelled: "service-a", other: "service-b"}
    usage = {101: 100.0, 102: 200.0, 103: 300.0, 104: 400.0, 105: 1.0, 106: 2.0, 107: 3.0}
    found = audit.unregistered(usage, leases, proc, lambda cid: labels.get(cid, ""))
    assert [(pid, mib) for pid, mib, _ in found] == [(104, 400.0), (106, 2.0), (107, 3.0)]
    assert "patch4" in found[0][2]


def test_chain_follows_dependents_transitively() -> None:
    sidecars = {"1": {}, "2": {"after": ["1"]}, "3": {"after": ["2"]}, "4": {"after": ["9"]}}
    assert wait.chain("1", sidecars) == ["1", "2", "3"]
