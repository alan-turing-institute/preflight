"""urgent.py's plan: what is in the way, what can be paused, and when it stops without submitting."""

from __future__ import annotations

import json
from pathlib import Path

import gpu_jobs
import kill
import pause
import pytest
import resume
import urgent


def job(job_id: str, partition: str, state: str = "RUNNING", gpus: int = 1) -> gpu_jobs.QueuedJob:
    return gpu_jobs.QueuedJob(job_id, partition, state, "None", f"p:x:{job_id}", gpus=gpus)


@pytest.fixture(autouse=True)
def sidecars(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpu_jobs, "STATE_DIR", tmp_path)
    monkeypatch.setattr(urgent, "elapsed", lambda _job: "1:00")
    for job_id, gpu in [("1", True), ("2", True), ("3", True), ("4", True)]:  # what slurm grants decides, not this flag
        gpu_jobs.update_sidecar(job_id, gpu=gpu, what=f"job {job_id}")


def test_classes_of_what_is_in_the_way() -> None:
    queue = [job("1", "normal"), job("2", "steady"), job("3", "normal", gpus=0), job("4", "urgent", "PENDING")]
    lines, victims, steady, wait_only = urgent.plan(
        queue, gpu=True, whole_box=False,
        leases=[("service-a", False), ("service-b", False), ("service-c", False), ("slurm", False)],
        unregistered=[(99, 500.0, "python patch4")],
    )
    assert victims == ["1"]  # job 3 holds no GPU, so a GPU-only urgent job leaves it alone
    assert steady == ["2"]
    assert wait_only == ["4", "lease:service-c", "pid:99"]
    text = "\n".join(lines)
    assert "PAUSABLE  service service-a" in text
    assert "PAUSABLE  service service-b" in text
    assert "loses all of that run" in text


def test_whole_box_puts_every_running_job_in_the_way() -> None:
    _, victims, _, _ = urgent.plan(
        [job("1", "normal"), job("3", "normal", gpus=0)], gpu=True, whole_box=True, leases=[], unregistered=[]
    )
    assert victims == ["1", "3"]


def test_stops_without_submitting_behind_a_wait_only_blocker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpu_jobs, "read_queue", lambda: [job("2", "steady")])
    monkeypatch.setattr(urgent, "other_leases", list)
    monkeypatch.setattr(urgent.audit, "unregistered", list)
    monkeypatch.setattr(urgent, "free_memory_mb", lambda: None)
    monkeypatch.setattr(urgent.site_config, "SITE_FILE", Path("/nonexistent/site.toml"))
    submitted = []
    monkeypatch.setattr(urgent.subprocess, "run", lambda *a, **k: submitted.append(a))
    assert urgent.run(["--project", "p", "--what", "w", "--time", "0:10:00", "--", "true"]) == 1
    assert submitted == []


def test_a_memory_only_urgent_job_that_does_not_fit_puts_running_jobs_in_the_way() -> None:
    _, victims, _, _ = urgent.plan(
        [job("3", "normal")], gpu=False, whole_box=False, leases=[], unregistered=[], memory_short=True
    )
    assert victims == ["3"]


def test_memory_notation() -> None:
    assert (urgent.mem_mb("24G"), urgent.mem_mb("512M"), urgent.mem_mb("2048")) == (24576, 512, 2048)


def site_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A minimal site whose lease lists one service."""
    path = tmp_path / "site.toml"
    path.write_text(
        'name = "t"\nadmin_group = "admins"\n[[nodes]]\nname = "n1"\ncpus = 4\nreal_memory_mb = 1000\n'
        f'[lease]\nlease_dir = "{tmp_path}"\nkeeper_user = "k"\npython = "/usr/bin/python3"\n'
        '[lease.services]\nsvc = "yields in a minute"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(urgent.site_config, "SITE_FILE", path)


def test_site_admin_check_multi_gpu_plan_and_request_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    queue = [
        gpu_jobs.QueuedJob("5", "normal", "RUNNING", "None", "p:x:5", gpus=2),
        gpu_jobs.QueuedJob("6", "normal", "PENDING", "Resources", "p:x:6", gpus=1),
    ]
    monkeypatch.setattr(gpu_jobs, "read_queue", lambda: queue)
    monkeypatch.setattr(urgent, "other_leases", lambda: [("svc", False)])
    monkeypatch.setattr(urgent.audit, "unregistered", list)
    monkeypatch.setattr(urgent, "free_memory_mb", lambda: None)
    monkeypatch.setattr(urgent.submit, "keeper_running", lambda: True)
    done = type("Done", (), {"returncode": 0, "stdout": "9", "stderr": ""})()
    monkeypatch.setattr(urgent.subprocess, "run", lambda *_a, **_k: done)
    argv = ["--project", "p", "--what", "w", "--time", "0:10:00", "--name", "u", "--mem", "1G", "--gpus", "2"]
    site_file(tmp_path, monkeypatch)

    monkeypatch.setattr(urgent.site_config, "is_admin", lambda _site, _user: False)
    assert urgent.run([*argv, "--", "true"]) == 1
    err = capsys.readouterr().err
    assert "only members of admins may submit urgent work" in err
    assert "sudo slurm/install/install.py admin add USER" in err

    monkeypatch.setattr(urgent.site_config, "is_admin", lambda _site, _user: True)
    assert urgent.run([*argv, "--dry-run", "--", "true"]) == 0
    out = capsys.readouterr().out
    assert "service svc: yields in a minute" in out
    assert "PAUSABLE  5 p:x:5" in out  # the running job holding 2 GPUs is a candidate
    assert "--gres=gpu:2" in out
    assert urgent.run([*argv, "--", "true"]) == 0
    request = next((tmp_path / "requests").glob("restore_order-*.json"))
    assert json.loads(request.read_text()) == {"urgent": "9", "victims": ["5"], "held": ["6"]}
    assert [e.action for e in gpu_jobs.events("6")] == ["urgent_hold"]

    # the keeper is required only where the site has a lease (end-to-end on a lease-less site: urgent always refused)
    monkeypatch.setattr(urgent.submit, "keeper_running", lambda: False)
    assert urgent.run([*argv, "--", "true"]) == 1
    assert "lease keeper" in capsys.readouterr().err
    (tmp_path / "site.toml").write_text(
        'name = "t"\nadmin_group = "admins"\n[[nodes]]\nname = "n1"\ncpus = 4\nreal_memory_mb = 1000\n', encoding="utf-8"
    )
    monkeypatch.setattr(urgent, "other_leases", list)
    assert urgent.run([*argv, "--", "true"]) == 0


class FakeSlurm:
    """Stands in for every scontrol/scancel call: records them, fails when told to."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, returncode: int = 0) -> None:
        self.calls: list[list[str]] = []
        self.returncode = returncode
        monkeypatch.setattr(gpu_jobs.subprocess, "run", self)

    def __call__(self, command: list[str], **_kwargs: object) -> object:
        self.calls.append(command)
        err = "scontrol: error: Access/permission denied" if self.returncode else ""
        return type("Done", (), {"returncode": self.returncode, "stdout": "", "stderr": err})()


def actions(job_id: str) -> list[str]:
    return [event.action for event in gpu_jobs.events(job_id)]


def test_pause_dry_run_acts_on_nothing_and_a_pause_is_recorded(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(gpu_jobs, "read_queue", lambda: [job("1", "normal"), job("2", "steady"), job("3", "steady", "PENDING")])
    slurm = FakeSlurm(monkeypatch)
    monkeypatch.setattr(pause, "elapsed", lambda _job: "1:00")
    monkeypatch.setattr("sys.argv", ["pause.py", "2", "--dry-run"])
    assert pause.main() == 0
    assert capsys.readouterr().out.strip() == "[pause] a steady job cannot resume: a pause loses everything it has done so far"
    assert slurm.calls == [] and actions("2") == []

    monkeypatch.setattr("sys.argv", ["pause.py", "1", "--reason", "need the card"])
    assert pause.main() == 0
    assert slurm.calls == [["scontrol", "requeuehold", "1"]]
    assert [(e.action, e.fields) for e in gpu_jobs.events("1")] == [("pause", {"reason": "need the card"})]

    # slurm refuses to requeue a pending job; one that has not started loses nothing, steady or not
    monkeypatch.setattr("sys.argv", ["pause.py", "3"])
    assert pause.main() == 0
    assert slurm.calls[-1] == ["scontrol", "uhold", "3"] and actions("3") == ["pause"]


def test_a_failing_scontrol_prints_slurms_text_and_records_nothing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(gpu_jobs, "read_queue", lambda: [job("1", "normal")])
    FakeSlurm(monkeypatch, returncode=1)
    monkeypatch.setattr("sys.argv", ["pause.py", "1"])
    assert pause.main() == 1
    assert "Access/permission denied" in capsys.readouterr().err
    monkeypatch.setattr("sys.argv", ["resume.py", "1"])
    assert resume.main() == 1
    monkeypatch.setattr("sys.argv", ["kill.py", "1"])
    assert kill.main() == 1
    assert actions("1") == []
    capsys.readouterr()
    monkeypatch.setattr(gpu_jobs, "read_queue", lambda: None)  # squeue down: said so, not "not in the queue"
    monkeypatch.setattr("sys.argv", ["pause.py", "1"])
    assert pause.main() == 1
    assert "slurm does not answer" in capsys.readouterr().err


def test_resume_and_kill_write_their_events_and_resume_node_covers_every_node(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    slurm = FakeSlurm(monkeypatch)
    monkeypatch.setattr(gpu_jobs, "node_state", lambda: "idle|")
    monkeypatch.setattr(gpu_jobs, "nodes", lambda: ["n1", "n2"])
    monkeypatch.setattr("sys.argv", ["resume.py", "1", "3"])
    assert resume.main() == 0
    assert (actions("1"), actions("3")) == (["resume"], ["resume"])
    monkeypatch.setattr("sys.argv", ["kill.py", "1", "--reason", "wrong config"])
    assert kill.main() == 0
    assert actions("1") == ["resume", "kill"]
    assert slurm.calls[-1] == ["scancel", "1"]

    slurm.calls.clear()
    monkeypatch.setattr("sys.argv", ["resume.py", "--node"])
    assert resume.main() == 0
    assert [c[-2] for c in slurm.calls] == ["nodename=n1", "nodename=n2"]
    slurm.calls.clear()
    monkeypatch.setattr("sys.argv", ["resume.py", "--node", "n2"])
    assert resume.main() == 0
    assert [c[-2] for c in slurm.calls] == ["nodename=n2"]
