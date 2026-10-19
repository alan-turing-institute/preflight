"""gpu_jobs: events keyed to file ownership, squeue rows, container labels, nodes, and where state lives."""

from __future__ import annotations

import importlib
import json
import os
import subprocess
from pathlib import Path

import gpu_jobs
import pytest
import site_config
from gpu_jobs import EventAction, QueuedJob


def test_event_actor_is_the_file_owner_not_the_content(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpu_jobs, "STATE_DIR", tmp_path)
    forged = tmp_path / "events" / "7-100-kill.json"
    forged.parent.mkdir()
    forged.write_text(json.dumps({"job": "7", "action": "kill", "actor": "mallory", "at": 100, "why": "x"}))
    path = gpu_jobs.write_event("7", EventAction.PAUSE, why="urgent")
    assert path.stat().st_mode & 0o777 == 0o644
    found = gpu_jobs.events("7")
    assert [e.action for e in found] == ["kill", "pause"]  # sorted by the time in the name
    assert {e.actor for e in found} == {Path(os.path.expanduser("~")).owner()}
    assert found[1].fields == {"why": "urgent"}
    gpu_jobs.write_event("8", EventAction.RESUME)
    assert len(gpu_jobs.events()) == 3
    assert [e.job for e in gpu_jobs.events("8")] == ["8"]


def test_parse_squeue_new_fields_and_six_field_construction() -> None:
    row = "12|normal|RUNNING|None|(null)|ann|4001|gres/gpu:2|a|b"
    [job] = gpu_jobs.parse_squeue(row)
    assert (job.user, job.priority, job.gpus, job.name) == ("ann", 4001, 2, "a|b")
    old = QueuedJob("1", "normal", "PENDING", "Priority", "n", "afterok:3(unfulfilled)")
    assert (old.user, old.priority, old.gpus, old.depends_on) == ("", 0, 0, ("3",))
    assert gpu_jobs.parse_squeue("1|normal|PENDING|x|y|short") == []


@pytest.mark.parametrize(
    ("gres", "count"),
    [("gres/gpu:2", 2), ("gpu:gb10:2", 2), ("gres/gpu:gb10:2", 2), ("gres/gpu", 1), ("N/A", 0), ("", 0), ("gres/gpu:1", 1)],
)
def test_parse_gpus(gres: str, count: int) -> None:
    assert gpu_jobs.parse_gpus(gres) == count


def test_container_label_uses_the_scope_id_and_degrades_to_empty() -> None:
    cid = "ab" * 32
    calls: list[tuple[str, str]] = []

    def runner(container: str, label: str) -> str:
        calls.append((container, label))
        return "proj"

    assert gpu_jobs.container_label(f"/system.slice/docker-{cid}.scope", "gpu-lease-project", runner) == "proj"
    assert calls == [(cid, "gpu-lease-project")]
    assert gpu_jobs.container_label("/user.slice/session-3.scope", "l", runner) == ""

    def broken(container: str, label: str) -> str:
        raise FileNotFoundError

    assert gpu_jobs.container_label(f"/docker-{cid}.scope", "l", broken) == ""


def test_nodes_from_scontrol_one_line_records() -> None:
    def fake(out: str, code: int = 0):
        return lambda *a, **k: subprocess.CompletedProcess(a, code, out, "")

    text = "NodeName=spark-1 Arch=aarch64 CoresPerSocket=20\nNodeName=spark-2 Arch=x\n"
    assert gpu_jobs.nodes(fake(text)) == ["spark-1", "spark-2"]
    assert gpu_jobs.nodes(fake(text, 1)) == []

    def missing(*a, **k):
        raise FileNotFoundError

    assert gpu_jobs.nodes(missing) == []


def test_state_dir_resolution_env_over_site_file_over_legacy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    site = tmp_path / "site.toml"
    site.write_text(
        f'name = "t"\nstate_dir = "{tmp_path / "site-state"}"\n'
        '[[nodes]]\nname = "n1"\ncpus = 4\nreal_memory_mb = 1024\n'
    )
    for var in ("SLURM_GPU_STATE_DIR", "GPU_LEASE_DIR"):
        monkeypatch.delenv(var, raising=False)
    try:
        monkeypatch.setattr(site_config, "SITE_FILE", tmp_path / "absent.toml")
        assert importlib.reload(gpu_jobs).STATE_DIR == Path("~/.local/state/slurm-gpu").expanduser()
        monkeypatch.setattr(site_config, "SITE_FILE", site)
        assert importlib.reload(gpu_jobs).STATE_DIR == tmp_path / "site-state"
        monkeypatch.setenv("SLURM_GPU_STATE_DIR", str(tmp_path / "env-state"))
        assert importlib.reload(gpu_jobs).STATE_DIR == tmp_path / "env-state"
    finally:
        monkeypatch.undo()
        importlib.reload(gpu_jobs)
