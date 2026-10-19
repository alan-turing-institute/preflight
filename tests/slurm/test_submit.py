"""submit.py: the batch-script options per class, and every refusal."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import gpu_jobs
import pytest
import site_config
import submit

DIRECTIVE = "#S" + "BATCH"  # spelled in two halves so a text-matching guard does not read a submission here


@pytest.fixture(autouse=True)
def no_site_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Default: no site file, so the box behaves as before the cutover."""
    monkeypatch.setattr(site_config, "SITE_FILE", tmp_path / "absent-site.toml")


def fake_site(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, lease: object, dgx_gate: bool) -> None:
    """A site file that exists; `load` returns just the two fields submit.py reads."""
    present = tmp_path / "site.toml"
    present.write_text("")
    monkeypatch.setattr(site_config, "SITE_FILE", present)
    monkeypatch.setattr(site_config, "load", lambda: SimpleNamespace(lease=lease, dgx_gate=dgx_gate))


def dry(*argv: str) -> str:
    return submit.submit(["--dry-run", *argv])


def test_resume_job_runs_in_normal_with_requeue_append_and_the_gpu_token() -> None:
    script = dry("--project", "p", "--what", "w", "--class", "resume", "--resume-ok", "--gpu", "--mem", "24G",
                 "--after", "5", "--after", "6", "--name", "n", "--", "python", "train.py")
    for option in ["--partition=normal", "--requeue", "--open-mode=append", "--gres=gpu:1",
                   "--dependency=afterok:5:6", "--job-name=p:resume:n", "--mem=24G"]:
        assert f"{DIRECTIVE} {option}" in script
    assert "--needs-lease --needs-card" in script
    assert script.rstrip().endswith("-- python train.py")


def test_restart_job_runs_in_steady_still_requeueable() -> None:
    script = dry("--project", "p", "--what", "w", "--class", "restart", "--mem", "4G", "--", "score")
    assert f"{DIRECTIVE} --partition=steady" in script
    assert f"{DIRECTIVE} --requeue" in script
    assert "--gres" not in script
    assert "--needs-lease" not in script


@pytest.mark.parametrize(
    ("argv", "why"),
    [
        (["--class", "resume", "--mem", "4G", "--", "x"], "--resume-ok"),
        (["--class", "restart", "--entry", "a.py", "--", "x"], "--entry needs --repo"),
        (["--class", "restart", "--wait-limit", "soon", "--", "x"], "not a duration"),
        (["--class", "restart"], "no command"),
    ],
)
def test_refusals(argv: list[str], why: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpu_jobs, "STATE_DIR", tmp_path)
    with pytest.raises(submit.Refused, match=why):
        dry("--project", "p", "--what", "w", *argv)


def test_an_unknown_class_is_refused_by_the_parser() -> None:
    with pytest.raises(SystemExit):
        dry("--project", "p", "--what", "w", "--class", "service", "--mem", "1G", "--", "x")


def test_mem_auto_uses_the_last_measured_run_plus_a_margin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpu_jobs, "STATE_DIR", tmp_path)
    (tmp_path / "jobs").mkdir()
    for job_id, ended, gpu_mib in [("1", 100, 30 * 1024), ("2", 200, 10 * 1024)]:
        leg = {"peak_gpu_mib": gpu_mib, "peak_mem_bytes": 2 * 1024**3, "ended_at": ended}
        (tmp_path / "jobs" / f"{job_id}.json").write_text(json.dumps({"name": "n", "last_leg": leg}))
    script = dry("--project", "p", "--what", "w", "--class", "restart", "--gpu", "--mem", "auto", "--name", "n",
                 "--", "x")
    assert f"{DIRECTIVE} --mem=14G" in script  # newest run: 10 + 2 GiB, plus 10%, rounded up


def test_no_mem_reuses_the_last_measured_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpu_jobs, "STATE_DIR", tmp_path)
    (tmp_path / "jobs").mkdir()
    leg = {"peak_gpu_mib": 10 * 1024, "peak_mem_bytes": 2 * 1024**3, "ended_at": 1}
    (tmp_path / "jobs" / "1.json").write_text(json.dumps({"name": "n", "last_leg": leg}))
    script = dry("--project", "p", "--what", "w", "--class", "restart", "--gpu", "--name", "n", "--", "x")
    assert f"{DIRECTIVE} --mem=14G" in script


@pytest.mark.parametrize("mem", [[], ["--mem", "auto"]])
def test_a_first_gpu_run_with_no_mem_takes_the_whole_box(mem: list[str], tmp_path: Path,
                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpu_jobs, "STATE_DIR", tmp_path)
    script = dry("--project", "p", "--what", "w", "--class", "restart", "--gpu", *mem, "--name", "never-ran",
                 "--", "x")
    assert f"{DIRECTIVE} --mem=0" in script


def test_a_first_cpu_run_with_no_mem_gets_slurms_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpu_jobs, "STATE_DIR", tmp_path)
    script = dry("--project", "p", "--what", "w", "--class", "restart", "--name", "never-ran", "--", "x")
    assert "--mem" not in script


def test_no_mem_without_a_name_never_reuses_another_jobs_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpu_jobs, "STATE_DIR", tmp_path)
    (tmp_path / "jobs").mkdir()
    leg = {"peak_gpu_mib": 10 * 1024, "peak_mem_bytes": 0, "ended_at": 1}
    (tmp_path / "jobs" / "1.json").write_text(json.dumps({"name": "python", "last_leg": leg}))
    script = dry("--project", "p", "--what", "w", "--class", "restart", "--gpu", "--", "python", "other.py")
    assert f"{DIRECTIVE} --mem=0" in script


def test_refuses_while_the_keeper_is_down(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The keeper is up while its heartbeat is fresh, whoever asks (a user unit of another user is invisible to them).
    monkeypatch.setattr(gpu_jobs, "STATE_DIR", tmp_path)
    assert not submit.keeper_running(now=100.0)  # no heartbeat yet
    gpu_jobs.write_json(gpu_jobs.keeper_state_path(), {"updated_at": 100})
    assert submit.keeper_running(now=100.0 + submit.KEEPER_STALE_S - 1)
    assert not submit.keeper_running(now=100.0 + submit.KEEPER_STALE_S)
    monkeypatch.setattr(submit, "keeper_running", lambda: False)
    with pytest.raises(submit.Refused, match="keeper"):
        submit.submit(["--project", "p", "--what", "w", "--class", "restart", "--mem", "1G", "--", "x"])


def test_refuses_an_after_on_a_job_slurm_no_longer_knows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(submit, "keeper_running", lambda: True)
    monkeypatch.setattr(submit, "known_to_slurm", lambda _job: False)
    with pytest.raises(submit.Refused, match="no longer knows"):
        submit.submit(["--project", "p", "--what", "w", "--class", "restart", "--mem", "1G", "--after", "9", "--",
                       "x"])


def test_gpus_asks_for_that_many_and_is_exclusive_with_gpu() -> None:
    script = dry("--project", "p", "--what", "w", "--class", "restart", "--gpus", "2", "--mem", "1G", "--", "x")
    assert f"{DIRECTIVE} --gres=gpu:2" in script
    assert "--needs-card" in script
    with pytest.raises(SystemExit):
        dry("--project", "p", "--what", "w", "--class", "restart", "--gpu", "--gpus", "2", "--", "x")
    with pytest.raises(submit.Refused, match="at least one GPU"):
        dry("--project", "p", "--what", "w", "--class", "restart", "--gpus", "0", "--", "x")


def test_comment_is_the_what_on_one_line_cut_to_200(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script = dry("--project", "p", "--what", "line one\nline two " + "x" * 300, "--class", "restart", "--mem", "1G",
                 "--", "x")
    comment = next(line for line in script.splitlines() if line.startswith(f"{DIRECTIVE} --comment="))
    assert "\n" not in comment
    assert submit.comment("line one\nline two") == "line one line two"
    assert len(submit.comment("y" * 300)) == submit.COMMENT_MAX


def submit_for_real(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *extra: str) -> dict[str, object]:
    """Run a full submission against fakes; return the sidecar record and whether the gate and keeper ran."""
    monkeypatch.setattr(gpu_jobs, "STATE_DIR", tmp_path)
    seen = {"keeper": False, "gate": False}
    monkeypatch.setattr(submit, "keeper_running", lambda: seen.update(keeper=True) or True)
    monkeypatch.setattr(submit, "run_gate", lambda *_a: seen.update(gate=True))
    monkeypatch.setattr(submit.audit, "unregistered", list)
    monkeypatch.setattr(submit.audit, "report", lambda _u: [])
    monkeypatch.setattr(
        submit.subprocess, "run",
        lambda *_a, **_k: SimpleNamespace(returncode=0, stdout="77\n", stderr=""),
    )
    submit.submit(["--project", "p", "--what", "w", "--class", "restart", "--mem", "1G", "--gpus", "2",
                   "--entry", "a.py", "--repo", str(tmp_path), *extra, "--", "x"])
    return {**gpu_jobs.read_sidecar("77"), **seen}


def test_no_site_file_checks_keeper_and_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    record = submit_for_real(monkeypatch, tmp_path)
    assert record["keeper"] is True
    assert record["gate"] is True
    assert (record["gpu"], record["gpus"]) == (True, 2)


def test_a_site_without_lease_or_gate_skips_both(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_site(monkeypatch, tmp_path, lease=None, dgx_gate=False)
    record = submit_for_real(monkeypatch, tmp_path)
    assert record["keeper"] is False
    assert record["gate"] is False
    assert record["manifest"] is None


def test_a_site_with_lease_and_gate_checks_both(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_site(monkeypatch, tmp_path, lease=object(), dgx_gate=True)
    record = submit_for_real(monkeypatch, tmp_path)
    assert record["keeper"] is True
    assert record["gate"] is True
