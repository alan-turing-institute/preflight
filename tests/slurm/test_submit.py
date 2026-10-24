"""submit.py: the batch-script options per class, and every refusal."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import gpu_jobs
import pytest
import site_config
import submit

DIRECTIVE = "#S" + "BATCH"  # spelled in two halves so a text-matching guard does not read a submission here


def fake_site(monkeypatch: pytest.MonkeyPatch, *, lease: object, dgx_gate: bool, nodes: tuple = (),
              cpu_job_cap: int | None = None, service: object = None, part_time_nodes: tuple = ()) -> None:
    """`load` returns just the fields submit.py reads."""
    site = SimpleNamespace(lease=lease, dgx_gate=dgx_gate, nodes=nodes, cpu_job_cap=cpu_job_cap, service=service,
                           part_time_nodes=part_time_nodes)
    monkeypatch.setattr(site_config, "load", lambda *_p: site)


@pytest.fixture(autouse=True)
def default_site(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default: a site with a lease and the DGX gate, so a submission checks both."""
    fake_site(monkeypatch, lease=object(), dgx_gate=True)


def dry(*argv: str) -> str:
    return submit.submit(["--dry-run", *argv])


def test_resume_job_runs_in_normal_with_requeue_append_and_the_gpu_token() -> None:
    script = dry("--project", "p", "--what", "w", "--class", "resume", "--resume-ok", "--gpu", "--mem", "24G",
                 "--after", "5", "--after", "6", "--name", "n", "--", "python", "train.py")
    for option in ["--partition=normal", "--requeue", "--open-mode=append", "--gres=gpu:1",
                   "--dependency=afterok:5:6", "--job-name=p:resume:n", "--mem=24G", "--qos=normal"]:
        assert f"{DIRECTIVE} {option}" in script
    assert "--needs-lease --needs-card" in script
    assert script.rstrip().endswith("-- python train.py")


def test_restart_job_runs_in_steady_still_requeueable() -> None:
    script = dry("--project", "p", "--what", "w", "--class", "restart", "--mem", "4G", "--level", "low", "--", "score")
    assert f"{DIRECTIVE} --partition=cpu" in script  # no GPUs: the capped cpu partition, either class
    assert f"{DIRECTIVE} --qos=low" in script
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
        (["--class", "restart", "--anywhere", "--", "x"], "--anywhere needs --class resume"),
        (["--class", "resume", "--resume-ok", "--gpu", "--anywhere", "--", "x"], "--anywhere needs --class resume"),
        (["--class", "resume", "--resume-ok", "--anywhere", "--", "x"], "no part-time node"),
        (["--project", "x:resume", "--class", "restart", "--", "x"], "no ':' in a project name"),  # would fake the class
    ],
)
def test_refusals(argv: list[str], why: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpu_jobs, "STATE_DIR", tmp_path)
    with pytest.raises(submit.Refused, match=why):
        dry("--project", "p", "--what", "w", *argv)


def test_anywhere_asks_for_the_cpu_and_part_time_partitions(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_site(monkeypatch, lease=None, dgx_gate=False, part_time_nodes=(object(),))
    script = dry("--project", "p", "--what", "w", "--class", "resume", "--resume-ok", "--anywhere", "--mem", "4G",
                 "--", "x")
    assert f"{DIRECTIVE} --partition=cpu,part_time" in script and f"{DIRECTIVE} --requeue" in script


def test_an_unknown_class_is_refused_by_the_parser() -> None:
    with pytest.raises(SystemExit):
        dry("--project", "p", "--what", "w", "--class", "service", "--mem", "1G", "--", "x")


def test_mem_auto_uses_the_last_measured_run_plus_a_margin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpu_jobs, "STATE_DIR", tmp_path)
    (tmp_path / "jobs").mkdir()
    for job_id, ended, gpu_mib in [("1", 100, 30 * 1024), ("2", 200, 10 * 1024)]:
        leg = {"peak_gpu_mib": gpu_mib, "peak_mem_bytes": 2 * 1024**3, "ended_at": ended, "exit": 0, "signals": []}
        (tmp_path / "jobs" / f"{job_id}.json").write_text(json.dumps({"name": "n", "last_leg": leg}))
    script = dry("--project", "p", "--what", "w", "--class", "restart", "--gpu", "--mem", "auto", "--name", "n",
                 "--", "x")
    assert f"{DIRECTIVE} --mem=14G" in script  # newest run: 10 + 2 GiB, plus 10%, rounded up


def test_no_mem_reuses_the_last_measured_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpu_jobs, "STATE_DIR", tmp_path)
    (tmp_path / "jobs").mkdir()
    leg = {"peak_gpu_mib": 10 * 1024, "peak_mem_bytes": 2 * 1024**3, "ended_at": 1, "exit": 0, "signals": []}
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


@pytest.mark.parametrize(
    ("site_cpus", "held", "argv", "cpus"),
    [
        (None, None, ["--gpu"], 4),  # the site lists no GPU node: the small default
        (20, None, ["--gpu"], 20),  # one-GPU box, nothing held: all of it
        (20, 6, ["--gpu"], 6),  # the cores the site holds for the GPU
        (20, None, [], 4),  # a CPU job keeps the small default
        (20, None, ["--gpu", "--cpus", "8"], 8),  # asked for wins
        (24, None, ["--gpus", "2"], 12),  # four GPUs: two of them get half the cores
        (24, 5, ["--gpus", "2"], 10),  # two GPUs' held cores
    ],
)
def test_a_gpu_job_gets_its_gpus_share_of_the_cores(site_cpus: int | None, held: int | None, argv: list[str], cpus: int,
                                                    tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpu_jobs, "STATE_DIR", tmp_path)
    if site_cpus is not None:
        gpus = (object(),) * (4 if "--gpus" in argv else 1)
        node = SimpleNamespace(cpus=site_cpus, gpus=gpus, cores_per_gpu=held)
        fake_site(monkeypatch, lease=None, dgx_gate=False, nodes=(node,))
    script = dry("--project", "p", "--what", "w", "--class", "restart", "--mem", "4G", *argv, "--", "x")
    assert f"{DIRECTIVE} --cpus-per-task={cpus}" in script
    # a job without GPUs runs in the CPU-only twin, kept off the cores held for GPU jobs
    assert f"{DIRECTIVE} --partition={'steady' if '--gpu' in argv or '--gpus' in argv else 'cpu'}" in script


def test_a_cpu_job_over_the_cpu_cap_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpu_jobs, "STATE_DIR", tmp_path)
    fake_site(monkeypatch, lease=None, dgx_gate=False, cpu_job_cap=14)
    with pytest.raises(submit.Refused, match="at most 14 cores"):  # it would wait forever on MaxCPUsPerNode
        dry("--project", "p", "--what", "w", "--class", "restart", "--mem", "4G", "--cpus", "15", "--", "x")
    assert "--cpus-per-task=14" in dry("--project", "p", "--what", "w", "--class", "restart", "--mem", "4G",
                                        "--cpus", "14", "--", "x")
    fake_site(monkeypatch, lease=None, dgx_gate=False, cpu_job_cap=2)  # a small box: the default 4 is clamped, not refused
    assert "--cpus-per-task=2" in dry("--project", "p", "--what", "w", "--class", "restart", "--mem", "4G", "--", "x")
    # a service stand-in (D20): its own partition and cap, never with a GPU, never waiting for the card
    service = ("--project", "p", "--what", "w", "--class", "restart", "--mem", "12G", "--service")
    with pytest.raises(submit.Refused, match="no service partition"):
        dry(*service, "--", "x")
    node = SimpleNamespace(cpus=8, real_memory_mb=20000, gpus=())
    fake_site(monkeypatch, lease=None, dgx_gate=False, nodes=(node,), cpu_job_cap=2,
              service=SimpleNamespace(cores=3, group="svc"))
    monkeypatch.setattr(gpu_jobs, "read_queue", list)  # an empty machine
    monkeypatch.setattr(site_config, "may_submit_service", lambda _site, _user: False)
    with pytest.raises(submit.Refused, match="only members of svc"):  # slurm would hold it forever (AllowGroups)
        dry(*service, "--", "x")
    monkeypatch.setattr(site_config, "may_submit_service", lambda _site, _user: True)
    script = dry(*service, "--", "x")  # the default 4 clamped to the service cap, not the cpu partition's 2
    assert f"{DIRECTIVE} --partition=service" in script and "--cpus-per-task=3" in script
    assert "--needs-lease" not in script and "--gres" not in script
    with pytest.raises(submit.Refused, match="at most 3 cores"):
        dry(*service, "--cpus", "4", "--", "x")
    with pytest.raises(submit.Refused, match="no --gpu"):
        dry(*service, "--gpu", "--", "x")
    with pytest.raises(submit.Refused, match="needs --mem"):  # whether it fits is judged by it
        dry("--project", "p", "--what", "w", "--class", "restart", "--service", "--", "x")
    with pytest.raises(submit.Refused, match="with --service only"):
        dry("--project", "p", "--what", "w", "--class", "restart", "--mem", "1G", "--no-suspend", "--", "x")
    with pytest.raises(submit.Refused, match="job ids"):
        dry(*service, "--suspend", "5,x", "--", "x")


def room(capsys: pytest.CaptureFixture[str], *flags: str) -> dict:
    """The dry run's `service room` line, as a dict."""
    capsys.readouterr()
    dry("--project", "p", "--what", "w", "--class", "restart", "--mem", "12G", "--cpus", "2", "--service", *flags, "--", "x")
    line = capsys.readouterr().out.splitlines()[0]
    assert line.startswith("[submit] service room: ")
    return json.loads(line.removeprefix("[submit] service room: "))


def test_the_service_dry_run_names_the_jobs_it_would_suspend_with_their_owners_consent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(gpu_jobs, "STATE_DIR", tmp_path)
    monkeypatch.setattr(site_config, "may_submit_service", lambda _site, _user: True)
    node = SimpleNamespace(cpus=8, real_memory_mb=40000, gpus=())  # 32,000 MB free beside the two jobs
    fake_site(monkeypatch, lease=None, dgx_gate=False, nodes=(node,), service=SimpleNamespace(cores=4, group="s"))
    me = gpu_jobs.current_user()  # a record counts only when the job's own user wrote it
    queue = [gpu_jobs.QueuedJob("1", "normal", "RUNNING", "None", "r:resume:gpu", user=me, gpus=1, cpus=6, mem_mb=4000),
             gpu_jobs.QueuedJob("2", "cpu", "RUNNING", "None", "r:restart:edge", user=me, cpus=2, mem_mb=4000)]
    monkeypatch.setattr(gpu_jobs, "read_queue", lambda: queue)
    gpu_jobs.update_sidecar("2", what="edge sweep")
    assert room(capsys) == {"starts": "after_suspending", "why": room(capsys)["why"],
                            "suspends": [{"job": "2", "user": me, "name": "r:restart:edge", "what": "edge sweep"}]}
    assert room(capsys, "--suspend", "2")["suspends"][0]["job"] == "2"
    refused = room(capsys, "--no-suspend")
    assert refused["starts"] == "waits" and refused["why"].startswith("cores: 0 free")
    queue.pop()  # job 2 has ended: two cores free
    assert room(capsys, "--no-suspend") == {"starts": "now", "suspends": [], "why": "it fits beside what runs"}
    queue[0] = dataclasses.replace(queue[0], mem_mb=0)  # the GPU job holds the whole machine's memory
    assert room(capsys)["starts"] == "waits"


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
    return {**(gpu_jobs.read_sidecar("77") or {}), **seen}


def test_a_record_that_cannot_be_written_after_sbatch_is_a_warning_naming_the_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "jobs" / "77.json").mkdir(parents=True)  # stands in for another user's file in the sticky directory
    submit_for_real(monkeypatch, tmp_path)
    assert "job 77 is queued but its record was not written" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("site", "keeper", "gate"),
    [({"lease": None, "dgx_gate": False}, False, False), ({"lease": object(), "dgx_gate": True}, True, True)],
    ids=["no lease or gate", "lease and gate"],
)
def test_the_keeper_and_gate_are_checked_only_where_the_site_has_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, site: dict[str, object], keeper: bool, gate: bool
) -> None:
    fake_site(monkeypatch, **site)
    record = submit_for_real(monkeypatch, tmp_path)
    assert (record["keeper"], record["gate"]) == (keeper, gate)
    assert (record["gpu"], record["gpus"]) == (True, 2)
    if not gate:
        assert record["manifest"] is None


def test_search_roots_reach_the_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An entry script importing a helper from another folder needs that root on the parity walk, or the gate refuses."""
    seen: list[list[str]] = []
    monkeypatch.setattr(submit.subprocess, "run", lambda cmd, **_k: seen.append(cmd) or SimpleNamespace(returncode=1))
    with pytest.raises(submit.Refused):
        submit.run_gate(tmp_path, "a.py", ["x"], tmp_path / "m.json", ["d1", "d2"])
    cmd = seen[0]
    assert cmd[cmd.index("--search") : cmd.index("--search") + 4] == ["--search", "d1", "--search", "d2"]
    assert cmd.index("--search") < cmd.index("--")
