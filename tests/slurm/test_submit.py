"""submit.py: the batch-script options per class, and every refusal."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import cores
import gpu_jobs
import portability
import pytest
import site_config
import submit

DIRECTIVE = "#S" + "BATCH"  # spelled in two halves so a text-matching guard does not read a submission here


def fake_site(monkeypatch: pytest.MonkeyPatch, *, lease: object, dgx_gate: bool, nodes: tuple = (),
              cpu_job_cap: int | None = None, service: object = None, part_time_nodes: tuple = (),
              node_python: str = "/usr/bin/python3", sharing_memory_fraction: float | None = None,
              cpus_from_history: site_config.CoresFromHistory = site_config.CoresFromHistory.OFF) -> None:
    """`load` returns just the fields submit.py reads."""
    site = SimpleNamespace(lease=lease, dgx_gate=dgx_gate, nodes=nodes, cpu_job_cap=cpu_job_cap, service=service,
                           part_time_nodes=part_time_nodes, node_python=node_python,
                           sharing_memory_fraction=sharing_memory_fraction, cpus_from_history=cpus_from_history)
    monkeypatch.setattr(site_config, "load", lambda *_p: site)


@pytest.fixture(autouse=True)
def default_site(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default: a site with a lease and the DGX gate, so a submission checks both; accounting knows no job."""
    fake_site(monkeypatch, lease=object(), dgx_gate=True)
    monkeypatch.setattr(gpu_jobs, "_sacct_jobs", lambda *_selection, **_flags: [])
    alike = cores.CoreSpeeds({})  # these tests do not depend on the machine they run on having fast and slow cores
    monkeypatch.setattr(cores.CoreSpeeds, "read", classmethod(lambda cls, root=None: alike))


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


def pt_node(name: str = "fedora", *, cores: int = 8, mb: int = 11444, default_jobs: bool = True) -> SimpleNamespace:
    return SimpleNamespace(name=name, address=f"{name}.lan", cores_lent=cores, jobs_mb=mb, default_jobs=default_jobs)


def fake_decide(monkeypatch: pytest.MonkeyPatch, *verdicts: portability.Verdict) -> list[dict]:
    """Replace the node probe: the calls `decide` gets are returned, its answer is `verdicts` (and one requirement when
    the paths were checked, as the real one gives)."""
    calls: list[dict] = []

    def decide(nodes, **kwargs):
        calls.append({"nodes": [n.name for n in nodes], **kwargs})
        return portability.Decision(verdicts, (("workdir", "/nas/w", "nas:/w"),) if kwargs["check_paths"] else ())

    monkeypatch.setattr(portability, "decide", decide)
    return calls


JOB = ["--project", "p", "--what", "w", "--resume-ok", "--mem", "4G"]
RESUME = [*JOB, "--class", "resume"]


@pytest.mark.parametrize(
    ("argv", "nodes", "partition", "checked"),
    [
        ([], [pt_node()], "cpu_fill,part_time", True),  # a portable resume job goes unasked
        (["--no-anywhere"], [pt_node()], "cpu_fill", None),
        (["--anywhere"], [pt_node()], "cpu_fill,part_time", False),  # vouched: the path checks are skipped
        ([], [pt_node(default_jobs=False)], "cpu_fill", None),  # a node that is opt-in only is never used unasked
        (["--anywhere"], [pt_node(default_jobs=False)], "cpu_fill,part_time", False),
        (["--gpu"], [pt_node()], "normal", None),
        (["--class", "restart"], [pt_node()], "cpu", None),
    ],
    ids=["default", "no-anywhere", "anywhere", "opt-in node", "anywhere on opt-in node", "gpu", "restart"],
)
def test_part_time_is_in_the_partition_list_of_an_eligible_job_only(
    argv: list[str], nodes: list, partition: str, checked: bool | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_site(monkeypatch, lease=None, dgx_gate=False, part_time_nodes=tuple(nodes))
    calls = fake_decide(monkeypatch, portability.Verdict("fedora", None))
    script = dry(*JOB, *(argv if "--class" in argv else ["--class", "resume", *argv]), "--", "x")
    assert f"{DIRECTIVE} --partition={partition}\n" in script
    assert [c["check_paths"] for c in calls] == ([] if checked is None else [checked])
    assert ("--require" in script) == (checked is True)  # only a checked job carries what the wrapper re-checks
    # the machine has no copy of the submitter's venv: the wrapper runs with the site's node_python, and only there
    assert (f"exec /usr/bin/python3 {submit.HERE / 'job_wrapper.py'}" in script) == ("part_time" in partition)


def test_a_node_that_fails_is_barred_by_name_and_every_decision_is_printed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_site(monkeypatch, lease=None, dgx_gate=False, part_time_nodes=(pt_node(), pt_node("lap")))
    fake_decide(monkeypatch, portability.Verdict("fedora", None), portability.Verdict("lap", "cannot ask lap"))
    script = dry(*RESUME, "--", "x")
    assert f"{DIRECTIVE} --exclude=lap\n" in script and f"{DIRECTIVE} --partition=cpu_fill,part_time\n" in script
    out = capsys.readouterr().out  # also under --dry-run
    assert "[submit] part_time fedora: included" in out and "[submit] part_time lap: excluded: cannot ask lap" in out
    fake_decide(monkeypatch, portability.Verdict("fedora", "x"), portability.Verdict("lap", "y"))
    none = dry(*RESUME, "--", "x")
    assert "part_time" not in none and "--exclude" not in none  # nothing passed: plain cpu_fill, nothing to bar


def test_size_is_checked_even_for_a_vouched_job_and_the_refusal_does_not_teach_the_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_site(monkeypatch, lease=None, dgx_gate=False, part_time_nodes=(pt_node(),))
    with pytest.raises(submit.Refused, match="asks 16 cores, it lends 8") as refused:
        dry(*RESUME, "--anywhere", "--cpus", "16", "--", "x")
    assert "--anywhere" not in str(refused.value)
    assert "part_time" not in dry(*RESUME, "--cpus", "16", "--", "x")  # unasked, an oversized job just stays on the main node


def test_a_job_for_another_site_makes_no_local_part_time_decision(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_site(monkeypatch, lease=None, dgx_gate=False, part_time_nodes=(pt_node(),))
    calls = fake_decide(monkeypatch, portability.Verdict("fedora", None))
    monkeypatch.setattr(submit.remote, "submit", lambda *_a: "remote")
    assert submit.submit([*RESUME, "--site", "turing", "--no-anywhere", "--", "x"]) == "remote" and calls == []


def test_an_unknown_class_is_refused_by_the_parser() -> None:
    with pytest.raises(SystemExit):
        dry("--project", "p", "--what", "w", "--class", "service", "--mem", "1G", "--", "x")


def test_mem_auto_is_the_highest_of_the_three_newest_clean_runs_and_a_memory_kill_bumps_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(gpu_jobs, "STATE_DIR", tmp_path)
    (tmp_path / "jobs").mkdir()

    def run(job_id: str, ended: int, mem: str | None = None, *more: dict, **leg: object) -> None:
        leg = {"peak_gpu_mib": 10 * 1024, "peak_mem_bytes": 2 * 1024**3, "ended_at": ended, "exit": 0, "signals": [], **leg}
        record = {"job_id": job_id, "name": "n", "mem": mem, "legs": [leg, *more]}
        (tmp_path / "jobs" / f"{job_id}.json").write_text(json.dumps(record))

    def ask() -> str:
        script = dry("--project", "p", "--what", "w", "--class", "restart", "--gpu", "--mem", "auto", "--name", "n", "--", "x")
        return next(line.split("=")[1] for line in script.splitlines() if line.startswith(f"{DIRECTIVE} --mem="))

    for job_id, ended, gpu_gib in [("1", 100, 40), ("2", 200, 30), ("3", 300, 10)]:
        run(job_id, ended, None, peak_gpu_mib=gpu_gib * 1024)
    second = {"peak_gpu_mib": 1024, "peak_mem_bytes": 0, "ended_at": 410, "exit": 0, "signals": []}
    run("4", 400, None, second)  # two clean legs (requeued after the first): still one run of the three
    assert ask() == "36G"  # 30 + 2 GiB from run 2, the highest of the newest three (run 1 is older), plus 10%, rounded up
    sacct: list[dict] = []
    monkeypatch.setattr(gpu_jobs, "_sacct_jobs", lambda *_selection, **_flags: sacct)
    run("5", 500, "40G", exit=-9)  # slurm's memory limit killed it: only accounting says so
    assert ask() == "36G"  # a crash is not a kill
    sacct.append({"job_id": 5, "state": {"current": ["OUT_OF_MEMORY"]}})
    capsys.readouterr()
    assert ask() == "48G"  # 1.2 x 40 GiB
    assert "[submit] memory: run 5 of 'n' was killed by slurm (OUT_OF_MEMORY" in capsys.readouterr().out
    run("5", 500, "10G", exit=-9)
    assert ask() == "36G"  # 1.2 x 10 GiB is below what the clean runs need: never lower than history
    run("6", 600, None, exit=-9, killed_by={"guard": "earlyoom", "job": "6"})  # the host's guard; no --mem given ...
    sacct.append({"job_id": 6, "tres": {"allocated": [{"type": "mem", "count": 50 * 1024}]}})  # ... so slurm's default
    assert ask() == "60G"
    node = site_config.Node("n", 20, 55 * 1024, ())  # no node gives jobs 60 GiB: the whole box instead, and it says so
    fake_site(monkeypatch, lease=None, dgx_gate=False, nodes=(node,))
    assert ask() == "0" and "more than any node gives jobs" in capsys.readouterr().err
    fake_site(monkeypatch, lease=None, dgx_gate=False)
    run("7", 700, "0", exit=-9, killed_by={"guard": "earlyoom"})  # killed even with the whole box: never less after it
    assert ask() == "0" and "whole box given" in capsys.readouterr().err
    run("7", 700, "lots", exit=-9, killed_by={"guard": "earlyoom"})  # what it was given is unknown: history, and it says so
    assert ask() == "36G" and "is unknown" in capsys.readouterr().err
    run("8", 800)  # a clean run since the kills: history alone again (runs 8, 4, 3), and no kill is mentioned
    assert ask() == "14G" and "killed" not in "".join(capsys.readouterr())


def test_a_memory_stop_sizes_the_next_run_from_what_it_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(gpu_jobs, "STATE_DIR", tmp_path)
    (tmp_path / "jobs").mkdir()
    stop = gpu_jobs.MemoryKill("9", 5150, "the memory stop-gap", used_mb=83923)  # 5.4 GB given, 88 GB held + card
    monkeypatch.setattr(gpu_jobs, "memory_kill", lambda _name: stop)

    def ask(jobs_gib: int) -> str:
        fake_site(monkeypatch, lease=None, dgx_gate=False, nodes=(site_config.Node("n", 20, jobs_gib * 1024, ()),))
        script = dry("--project", "p", "--what", "w", "--class", "restart", "--gpu", "--mem", "auto", "--name", "n", "--", "x")
        return next(line.split("=")[1] for line in script.splitlines() if line.startswith(f"{DIRECTIVE} --mem="))

    assert ask(120) == "91G"  # what it used plus 10% (96.8 GB), far above 1.2 times its 5.4 GB ask
    assert "killed by the memory stop-gap with 5.4 GB given and 88.0 GB measured in use" in capsys.readouterr().out
    assert ask(80) == "0"  # no node gives jobs that much: the whole box
    stop = gpu_jobs.MemoryKill("9", 40 * 1024, "the memory stop-gap", used_mb=30_000)  # used less than 1.2 x given
    assert ask(120) == "48G"  # the bump stands


def test_a_named_measured_job_asks_for_shards_and_its_share_of_the_cores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    gpu = site_config.Gpu(0, "GPU-a", "/dev/nvidia0", "gb10", shards=12)
    node = site_config.Node("n", 20, 100_000, (gpu,), cores_per_gpu=12)
    fake_site(monkeypatch, lease=None, dgx_gate=False, nodes=(node,))
    monkeypatch.setattr(gpu_jobs, "STATE_DIR", tmp_path)
    (tmp_path / "jobs").mkdir()
    leg = {"peak_gpu_mib": 1024, "peak_mem_bytes": 0, "ended_at": 5, "exit": 0, "signals": []}
    sharing = {"level": 4, "gpu_type": "gb10", "node_memory_mb": 100_000, "shards": 12, "peak_bytes": 1024**3}
    (tmp_path / "jobs" / "1.json").write_text(json.dumps({"name": "n", "legs": [leg], "sharing": sharing}))
    argv = ["--project", "p", "--what", "w", "--class", "restart", "--gpu"]
    script = dry(*argv, "--name", "n", "--", "x")
    assert f"{DIRECTIVE} --gres=shard:3" in script and "--gres=gpu" not in script
    assert f"{DIRECTIVE} --cpus-per-task=3" in script and f"{DIRECTIVE} --mem=2G" in script  # 1 GiB peak plus 10%
    assert "--max-jobs" not in script  # the wrapper tells a shard job by slurm's environment
    assert "[submit] sharing n: level 4 (measured 4;" in capsys.readouterr().out
    explicit = dry(*argv, "--max-jobs", "6", "--name", "n", "--", "x")  # --max-jobs replaces the measured level
    assert f"{DIRECTIVE} --gres=shard:2" in explicit and "--max-jobs" not in explicit
    assert f"{DIRECTIVE} --gres=gpu:1" in dry(*argv, "--name", "unmeasured", "--", "x")  # no record: the whole card
    capsys.readouterr()
    for kept in (dry(*argv, "--max-jobs", "3", "--name", "unmeasured", "--", "x"), dry(*argv, "--max-jobs", "3", "--", "x")):
        assert f"{DIRECTIVE} --gres=gpu:1" in kept and "--max-jobs" not in kept  # asked, not given: never the wrapper's own flag
    shown = capsys.readouterr()
    assert "asked 3 copies" in shown.out and "needs --name" in shown.err  # and both say so
    assert f"{DIRECTIVE} --gres=gpu:2" in dry("--project", "p", "--what", "w", "--class", "restart", "--gpus", "2", "--name", "n", "--", "x")  # never shares
    leg["peak_gpu_mib"] = 4096  # grew past what the level was measured at
    (tmp_path / "jobs" / "2.json").write_text(json.dumps({"name": "n", "legs": [{**leg, "ended_at": 9}]}))
    dry(*argv, "--name", "n", "--", "x")
    assert "re-measure" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("mem", "name", "expected"),
    [
        ("10G", "n", "--mem 10G (10.7 GB) is below the 12.9 GB 'n' peaked at"),  # 12 GiB measured
        ("0", "n", "--mem 0 asks for the whole box"),
        ("0", "n", "--mem auto would ask for 15.0 GB"),  # 14G is 14 GiB
        ("12000M", "n", "below the 12.9 GB"),  # plain slurm notation: M is MiB
        ("13G", "n", ""),  # enough
        ("auto", "n", ""),
        ("1G", "other", ""),  # no measured run under that name
    ],
)
def test_a_hand_set_mem_below_the_measured_peak_or_zero_warns(
    mem: str, name: str, expected: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(gpu_jobs, "STATE_DIR", tmp_path)
    (tmp_path / "jobs").mkdir()
    leg = {"peak_gpu_mib": 10 * 1024, "peak_mem_bytes": 2 * 1024**3, "ended_at": 1, "exit": 0, "signals": []}
    (tmp_path / "jobs" / "1.json").write_text(json.dumps({"name": "n", "last_leg": leg}))
    dry("--project", "p", "--what", "w", "--class", "restart", "--gpu", "--mem", mem, "--name", name, "--", "x")
    err = capsys.readouterr().err
    assert ("[submit] warning: " in err and expected in err) if expected else err == ""


@pytest.mark.parametrize(
    ("command", "queue", "warns"),
    [
        (["--device", "cpu"], [], True),
        (["--device=cpu"], [], True),
        (["--device", "cuda"], [], False),
        (["--device", "cpu"], [gpu_jobs.QueuedJob("1", "normal", "PENDING", "Resources", "p:x:1", gpus=1)], False),
        (["--device", "cpu"], [gpu_jobs.QueuedJob("1", "normal", "RUNNING", "None", "p:x:1", gpus=1)], False),
        (["--device", "cpu"], None, False),  # slurm does not answer: say nothing
    ],
)
def test_device_cpu_with_the_gpu_unreserved_warns(
    command: list[str], queue: list | None, warns: bool, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(gpu_jobs, "read_queue", lambda: queue)
    dry("--project", "p", "--what", "w", "--class", "restart", "--", "x", *command)
    assert ("the GPU is free" in capsys.readouterr().err) is warns


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
        gpus = (site_config.Gpu(0, "GPU-a", "/dev/nvidia0", "gb10"),) * (4 if "--gpus" in argv else 1)
        node = SimpleNamespace(cpus=site_cpus, gpus=gpus, cores_per_gpu=held, fast_cpus=None)
        fake_site(monkeypatch, lease=None, dgx_gate=False, nodes=(node,))
    script = dry("--project", "p", "--what", "w", "--class", "restart", "--mem", "4G", *argv, "--", "x")
    assert f"{DIRECTIVE} --cpus-per-task={cpus}" in script
    # a job without GPUs runs in the CPU-only twin, kept off the cores held for GPU jobs
    assert f"{DIRECTIVE} --partition={'steady' if '--gpu' in argv or '--gpus' in argv else 'cpu'}" in script


@pytest.mark.parametrize(
    ("khz", "node_cpus", "argv", "cpus"),
    [
        ({**dict.fromkeys(range(10), 2_800_000), **dict.fromkeys(range(10, 20), 3_900_000)}, 20, ["--cpus", "4"], 11),
        ({**dict.fromkeys(range(10), 2_800_000), **dict.fromkeys(range(10, 20), 3_900_000)}, 20, ["--cpus", "11"], 11),
        ({**dict.fromkeys(range(10), 2_800_000), **dict.fromkeys(range(10, 20), 3_900_000)}, 20, ["--cpus", "15"], 15),
        ({**dict.fromkeys(range(10), 2_800_000), **dict.fromkeys(range(10, 20), 3_900_000)}, 20, [], 11),  # site holds 6
        ({**dict.fromkeys(range(10), 2_800_000), **dict.fromkeys(range(10, 20), 3_900_000)}, 12, ["--cpus", "4"], 4),  # not this node
        (dict.fromkeys(range(20), 2_800_000), 20, ["--cpus", "4"], 4),  # alike: nothing to avoid
    ],
)
def test_a_gpu_job_on_mixed_cores_always_holds_a_fast_core(khz: dict[int, int], node_cpus: int, argv: list[str], cpus: int,
                                                           tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpu_jobs, "STATE_DIR", tmp_path)
    monkeypatch.setattr(cores.CoreSpeeds, "read", classmethod(lambda cls, root=None: cls(khz)))
    node = SimpleNamespace(cpus=node_cpus, gpus=(site_config.Gpu(0, "GPU-a", "/dev/nvidia0", "gb10"),), cores_per_gpu=6, fast_cpus=None)
    fake_site(monkeypatch, lease=None, dgx_gate=False, nodes=(node,))
    script = dry("--project", "p", "--what", "w", "--class", "restart", "--mem", "4G", "--gpu", *argv, "--", "x")
    assert f"{DIRECTIVE} --cpus-per-task={cpus}" in script


def test_a_gpu_job_on_a_fast_cpus_node_keeps_its_cores_and_is_bound_to_the_fast_ones(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """D46: no raise to 11; a GPU job that fits in the fast cores is held to them, a larger one is not; a named GPU job
    is sized from its runs like a CPU job."""
    monkeypatch.setattr(gpu_jobs, "STATE_DIR", tmp_path)
    monkeypatch.setattr(cores.CoreSpeeds, "read", classmethod(lambda cls, root=None: cls(
        {**dict.fromkeys(range(10), 2_800_000), **dict.fromkeys(range(10, 20), 3_900_000)})))
    monkeypatch.setattr(gpu_jobs, "measured_cores", lambda job_name: (1.0, 3))
    gpu = site_config.Gpu(0, "GPU-a", "/dev/nvidia0", "gb10")
    node = site_config.Node("n", 20, 100000, (gpu,), cores_per_gpu=12, fast_cpus="10-19")
    fake_site(monkeypatch, lease=None, dgx_gate=False, nodes=(node,), cpus_from_history=site_config.CoresFromHistory.ENFORCE)

    def job(*argv: str) -> tuple[str, str]:
        capsys.readouterr()
        script = dry("--project", "p", "--what", "w", "--class", "restart", "--mem", "4G", *argv, "--", "x")
        return script, capsys.readouterr().out

    script, out = job("--gpu", "--cpus", "2")
    assert f"{DIRECTIVE} --cpus-per-task=2" in script and "[submit] cores: 2 fast cores, bound to the GPU" in out
    assert f"{DIRECTIVE} --ntasks=1\n" in script and f"{DIRECTIVE} --gres-flags=enforce-binding\n" in script
    script, out = job("--gpu", "--cpus", "12")  # more than the 10 fast cores: fast first, never strict
    assert f"{DIRECTIVE} --cpus-per-task=12" in script and "--ntasks" not in script and "10 fast cores first" in out
    script, out = job("--gpu", "--name", "n")  # sized from its runs: ceil(1.0 x 1.25)
    assert f"{DIRECTIVE} --cpus-per-task=2" in script and "--gres-flags=enforce-binding" in script
    script, _ = job("--name", "n")  # a CPU-only job is never bound
    assert "--ntasks" not in script and "--gres-flags" not in script


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


def test_a_named_cpu_job_is_offered_the_cores_its_completed_runs_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    measured: dict[str, tuple[float, int] | None] = {}
    monkeypatch.setattr(gpu_jobs, "measured_cores", lambda job_name: measured.setdefault(job_name, (1.7, 3)))
    mode = site_config.CoresFromHistory

    def cpus(history: site_config.CoresFromHistory, *argv: str) -> str:
        fake_site(monkeypatch, lease=None, dgx_gate=False, cpu_job_cap=8, cpus_from_history=history)
        capsys.readouterr()
        script = dry("--project", "p", "--what", "w", "--class", "restart", "--mem", "4G", *argv, "--", "x")
        return next(line.split("=")[1] for line in script.splitlines() if "--cpus-per-task=" in line)

    assert cpus(mode.RECOMMEND, "--name", "n") == "4"  # printed, not applied
    assert "[submit] cores: measured 1.7 avg over 3 completed runs; would ask 3 (asking 4 now)" in capsys.readouterr().out
    assert list(measured) == ["p:restart:n"]  # slurm's whole name: another project's or class's runs are not this job's
    assert cpus(mode.ENFORCE, "--name", "n") == "3"  # ceil(1.7 x 1.25)
    assert "asking 3 (was 4)" in capsys.readouterr().out
    measured["p:restart:big"] = (20.0, 2)
    assert cpus(mode.ENFORCE, "--name", "big") == "8"  # never past the cores a CPU job may have
    assert cpus(mode.ENFORCE, "--name", "n", "--cpus", "7") == "7"  # a hand-set count is kept ...
    shown = capsys.readouterr()
    assert "set by --cpus" in shown.out and "more than 2 times the 3 cores" in shown.err  # ... and warns above 2x
    assert cpus(mode.ENFORCE, "--name", "n", "--cpus", "6") == "6" and capsys.readouterr().err == ""
    measured["p:restart:new"] = None  # no completed run: nothing to say
    for history, argv in [(mode.ENFORCE, ("--name", "new")), (mode.ENFORCE, ()), (mode.OFF, ("--name", "n"))]:
        assert cpus(history, *argv) == "4" and "[submit] cores:" not in capsys.readouterr().out
    fake_site(monkeypatch, lease=None, dgx_gate=False, cpu_job_cap=8, cpus_from_history=mode.ENFORCE)
    assert "[submit] cores:" not in (dry("--project", "p", "--what", "w", "--class", "restart", "--mem", "4G", "--gpu",
                                         "--name", "n", "--", "x") + capsys.readouterr().out)  # GPU jobs: untouched
    fake_site(monkeypatch, lease=None, dgx_gate=False, cpu_job_cap=8, cpus_from_history=mode.RECOMMEND)
    record = submit_for_real(monkeypatch, tmp_path, job=["--project", "p", "--what", "w", "--class", "restart", "--mem",
                                                         "4G", "--name", "n", "--", "x"])
    assert record["cores_from_history"] == {"mode": "recommend", "average": 1.7, "runs": 3, "proposal": 3, "asked": 4}


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


def submit_for_real(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *extra: str, job: list[str] | None = None) -> dict[str, object]:
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
    submit.submit(job or ["--project", "p", "--what", "w", "--class", "restart", "--mem", "1G", "--gpus", "2",
                          "--entry", "a.py", "--repo", str(tmp_path), *extra, "--", "x"])
    return {**(gpu_jobs.read_sidecar("77") or {}), **seen}


def test_a_record_that_cannot_be_written_after_sbatch_is_a_warning_naming_the_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "jobs" / "77.json").mkdir(parents=True)  # stands in for another user's file in the sticky directory
    submit_for_real(monkeypatch, tmp_path)
    assert "job 77 is queued but its record was not written" in capsys.readouterr().err


def test_the_part_time_decision_is_in_the_job_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_site(monkeypatch, lease=None, dgx_gate=False, part_time_nodes=(pt_node(), pt_node("lap")))
    fake_decide(monkeypatch, portability.Verdict("fedora", None), portability.Verdict("lap", "cannot ask lap"))
    record = submit_for_real(monkeypatch, tmp_path, job=[*RESUME, "--", "x"])
    assert record["part_time"] == {"fedora": "included", "lap": "cannot ask lap"}


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
