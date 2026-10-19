"""gpu_jobs: events keyed to file ownership, squeue rows, container labels, nodes, and where state lives."""

from __future__ import annotations

import importlib
import json
import os
import signal
import subprocess
from pathlib import Path

import gpu_jobs
import pytest
import site_config
import status
import wait
from gpu_jobs import EventAction, QueuedJob


def test_event_actor_is_the_file_owner_not_the_content(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpu_jobs, "STATE_DIR", tmp_path)
    forged = tmp_path / "events" / "7-100-kill.json"
    forged.parent.mkdir()
    forged.write_text(json.dumps({"job": "7", "action": "kill", "actor": "mallory", "at": 100, "why": "x"}))
    path = gpu_jobs.write_event("7", EventAction.PAUSE, why="urgent")
    assert path.stat().st_mode & 0o777 == 0o644
    gpu_jobs.update_sidecar("7", what="x")  # a record, like a request or the agent's state, is read by other users
    assert gpu_jobs.sidecar_path("7").stat().st_mode & 0o777 == 0o644
    found = gpu_jobs.events("7")
    assert [e.action for e in found] == ["kill", "pause"]  # sorted by the time in the name
    assert {e.actor for e in found} == {Path(os.path.expanduser("~")).owner()}
    assert found[1].fields == {"why": "urgent"}
    gpu_jobs.write_event("8", EventAction.RESUME)
    assert len(gpu_jobs.events()) == 3
    assert [e.job for e in gpu_jobs.events("8")] == ["8"]


def test_a_hostile_or_foreign_record_is_ignored_and_never_blocks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpu_jobs, "STATE_DIR", tmp_path)
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    os.mkfifo(jobs / "1.json")  # any user may leave one; opening it for reading would block for ever
    (jobs / "2.json").symlink_to(jobs / "6.json")
    (jobs / "3.json").mkdir()
    (jobs / "4.json").write_text('{"x": 1}' + " " * gpu_jobs.MAX_RECORD_BYTES)  # valid JSON, too big to be a record
    (jobs / "5.json").write_text("[" * 200_000)  # deep enough to overflow the parser's stack
    (jobs / "6.json").write_text(json.dumps({"gpu": True, "name": "n", "what": "forged"}))
    signal.signal(signal.SIGALRM, lambda *_: pytest.fail("read_json blocked"))
    signal.alarm(5)
    try:
        assert [gpu_jobs.read_sidecar(str(i)) for i in range(1, 6)] == [None] * 5
    finally:
        signal.alarm(0)
    me = gpu_jobs.current_user()
    assert gpu_jobs.read_sidecar("6", me) == {"gpu": True, "name": "n", "what": "forged"}
    assert gpu_jobs.read_sidecar("6", "somebody-else") is None  # the file's owner is its writer's identity
    job = QueuedJob("6", "normal", "PENDING", "None", "n", user="somebody-else")
    assert not gpu_jobs.needs_the_card(job)  # the record says gpu, slurm granted none
    assert gpu_jobs.needs_the_card(QueuedJob("7", "normal", "PENDING", "None", "n", gpus=1))  # no record needed
    assert gpu_jobs.needs_the_card(QueuedJob("7", "urgent", "PENDING", "None", "n"))
    events = tmp_path / "events"
    events.mkdir()
    (events / "7-100-kill.json").write_text("[" * 200_000)
    os.mkfifo(events / "9-100-kill.json")
    signal.alarm(5)
    try:
        (events / "junk.json").write_text("[" * 200_000)
        opened: list[Path] = []
        real = gpu_jobs.read_json
        monkeypatch.setattr(gpu_jobs, "read_json", lambda path, user=None: opened.append(path.name) or real(path, user))
        assert gpu_jobs.events() == [] and gpu_jobs.events("7") == []
        assert "junk.json" not in opened  # judged by its name: a file that is not an event is never opened
        assert gpu_jobs.events("8") == []
    finally:
        signal.alarm(0)
    assert wait.all_sidecars({"6": "somebody-else"})["6"] == {}
    monkeypatch.setattr(status.gpu_jobs, "node_state", lambda: "idle")
    monkeypatch.setattr(status.audit, "unregistered", list)
    monkeypatch.setattr(status.audit, "report", lambda _u: [])
    assert not any("forged" in line for line in status.lines([job]))
    # an update never merges into a record that is not this user's, and a write that cannot replace says so
    monkeypatch.setattr(gpu_jobs, "current_user", lambda: "somebody-else")
    gpu_jobs.update_sidecar("6", extra=1)
    assert gpu_jobs.read_sidecar("6") == {"job_id": "6", "extra": 1}
    with pytest.raises(OSError, match="belongs to"):
        gpu_jobs.update_sidecar("3", what="x")
    assert not list(jobs.glob(".*.tmp"))


def test_the_peak_for_mem_auto_comes_from_clean_finished_legs_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpu_jobs, "STATE_DIR", tmp_path)
    (tmp_path / "jobs").mkdir()
    done = {"peak_gpu_mib": 1024, "peak_mem_bytes": 0, "ended_at": 1, "exit": 0, "signals": []}
    crashed = {**done, "peak_gpu_mib": 40, "ended_at": 3, "exit": 1}
    preempted = {**done, "peak_gpu_mib": 40, "ended_at": 2, "signals": ["SIGTERM"]}
    legs = [done, preempted, crashed]
    (tmp_path / "jobs" / "1.json").write_text(json.dumps({"name": "n", "legs": legs, "last_leg": crashed}))
    assert gpu_jobs.measured_peak_bytes("n") == 1024 * 1024**2
    (tmp_path / "jobs" / "2.json").write_text(json.dumps({"name": "n", "legs": 5, "last_leg": {"peak_gpu_mib": "x", "exit": 0}}))
    assert gpu_jobs.measured_peak_bytes("n") == 1024 * 1024**2  # wrong types inside a valid record are skipped
    monkeypatch.setattr(gpu_jobs, "current_user", lambda: "somebody-else")  # the file is not theirs
    assert gpu_jobs.measured_peak_bytes("n") is None


def squeue_json(*jobs: dict[str, object]) -> str:
    """`squeue --json` output in the shape slurm 23.11.4 writes, for the fields read."""
    base = {"partition": "normal", "job_state": ["PENDING"], "state_reason": "Priority", "dependency": "",
            "user_name": "ann", "priority": {"set": True, "infinite": False, "number": 1}, "tres_per_node": "", "name": "n"}
    return json.dumps({"jobs": [{**base, **job} for job in jobs]})


def test_completion_records_take_each_jobs_latest_record_and_a_name_forges_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    forged = "a b\nJobId=999 UserId=bob(2002) JobState=FAILED Name=forged"  # forged a record in the jobcomp log
    legs = {"7": {"job_id": 7, "state": {"current": ["FAILED"]}, "name": forged, "user": "alice"},
            "8": {"job_id": 8, "state": {"current": ["CANCELLED"]}, "name": "n", "user": "bob"}}
    asked: list[tuple[str, ...]] = []

    def sacct(*selection: str) -> list[dict[str, object]]:
        asked.append(selection)
        if selection[0] == "-S":  # the window can return an earlier leg of a requeued job (measured)
            return [legs["7"], {**legs["8"], "state": {"current": ["PENDING", "REQUEUED"]}}]
        return [legs[i] for i in selection[1].split(",")]

    monkeypatch.setattr(gpu_jobs, "_sacct_jobs", sacct)
    records = gpu_jobs.completion_records()
    assert set(records) == {"7", "8"} and records["8"]["JobState"] == "CANCELLED"
    assert records["7"] == {"JobId": "7", "JobState": "FAILED", "Name": forged, "UserId": "alice"}
    assert gpu_jobs.completion_records(["8"])["8"]["UserId"] == "bob" and asked[-1] == ("-j", "8")


def test_parse_squeue_reads_json_so_a_name_cannot_forge_a_row() -> None:
    # Slurm keeps a newline in a job name and lets the owner rename the job (measured, 23.11.4): in a text row this
    # name was a second, forged job 999 with priority 9999.
    forged = "x\n999|normal|PENDING|None||bob|9999|N/A|y"
    text = squeue_json(
        {"job_id": 12, "job_state": ["RUNNING"], "priority": {"set": True, "number": 4001}, "tres_per_node": "gres/gpu:2",
         "name": forged},
        {"job_id": 13, "job_state": ["COMPLETED"]},  # squeue --json ignored -t: finished jobs are dropped here
        {"job_id": 14, "priority": {"set": False, "number": 7}, "state_reason": "JobHeldUser"},
        # Still saving after a pause (requeued) or a kill: COMPLETING is a flag after the base state in JSON.
        {"job_id": 15, "job_state": ["PENDING", "COMPLETING"]},
        {"job_id": 16, "job_state": ["CANCELLED", "COMPLETING"]},
    )
    running, held, paused, killed = gpu_jobs.parse_squeue(text)
    assert (running.job_id, running.user, running.priority, running.gpus, running.name) == ("12", "ann", 4001, 2, forged)
    assert (held.job_id, held.priority, held.held) == ("14", 0, True)
    assert [(j.job_id, j.state, j.active) for j in (paused, killed)] == [("15", "COMPLETING", True), ("16", "COMPLETING", True)]
    old = QueuedJob("1", "normal", "PENDING", "Priority", "n", "afterok:3(unfulfilled)")
    assert (old.user, old.priority, old.gpus, old.depends_on) == ("", 0, 0, ("3",))
    with pytest.raises(ValueError):
        gpu_jobs.parse_squeue("12|normal|RUNNING|None||ann|1|N/A|n")


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
