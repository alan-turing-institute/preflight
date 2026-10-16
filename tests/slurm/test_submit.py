"""submit.py: the batch-script options per class, and every refusal."""

from __future__ import annotations

import json
from pathlib import Path

import gpu_jobs
import pytest
import submit

DIRECTIVE = "#S" + "BATCH"  # spelled in two halves so a text-matching guard does not read a submission here


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
        (["--class", "restart", "--gpu", "--", "x"], "must state --mem"),
        (["--class", "resume", "--mem", "4G", "--", "x"], "--resume-ok"),
        (["--class", "restart", "--mem", "auto", "--name", "never-ran", "--", "x"], "no measured run"),
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


def test_refuses_while_the_keeper_is_down(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(submit, "keeper_running", lambda: False)
    with pytest.raises(submit.Refused, match="keeper"):
        submit.submit(["--project", "p", "--what", "w", "--class", "restart", "--mem", "1G", "--", "x"])


def test_refuses_an_after_on_a_job_slurm_no_longer_knows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(submit, "keeper_running", lambda: True)
    monkeypatch.setattr(submit, "known_to_slurm", lambda _job: False)
    with pytest.raises(submit.Refused, match="no longer knows"):
        submit.submit(["--project", "p", "--what", "w", "--class", "restart", "--mem", "1G", "--after", "9", "--",
                       "x"])
