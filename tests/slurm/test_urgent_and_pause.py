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


def job(job_id: str, partition: str, state: str = "RUNNING", gpus: int = 1, cls: str = "x") -> gpu_jobs.QueuedJob:
    return gpu_jobs.QueuedJob(job_id, partition, state, "None", f"p:{cls}:{job_id}", gpus=gpus)


@pytest.fixture(autouse=True)
def sidecars(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpu_jobs, "STATE_DIR", tmp_path)
    monkeypatch.setattr(urgent, "elapsed", lambda _job: "1:00")
    for job_id, gpu in [("1", True), ("2", True), ("3", True), ("4", True)]:  # what slurm grants decides, not this flag
        gpu_jobs.update_sidecar(job_id, gpu=gpu, what=f"job {job_id}")


def test_classes_of_what_is_in_the_way() -> None:
    queue = [job("1", "normal"), job("2", "steady"), job("3", "cpu", gpus=0, cls="resume"), job("4", "urgent", "PENDING"),
             job("6", "cpu_fill", gpus=0, cls="resume")]
    lines, victims, steady, wait_only = urgent.plan(
        queue, gpu=True, whole_box=False,
        leases=[("service-a", False), ("service-b", False), ("service-c", False), ("slurm", False)],
        unregistered=[urgent.audit.Stray(99, 500.0, "python patch4")],
        services={"service-a": "yields within a minute", "service-b": "yields at its next job"},
    )
    assert victims == ["1"]  # job 3 holds no GPU, so a GPU-only urgent job leaves it alone
    assert steady == ["2"]
    assert wait_only == ["4", "lease:service-c", "pid:99"]
    text = "\n".join(lines)
    assert "PAUSABLE  service service-a" in text
    assert "PAUSABLE  service service-b" in text
    assert "loses all of that run" in text
    # job 6 is not in the way, but the normal job 1 it pre-empts restarts above it and may take its cores (2026-11-03: 1739 and 1687)
    assert "normal job(s) 1 " in text and "running now, not otherwise touched: 6" in text and "6" not in victims
    # the warning does not depend on a cpu_fill job running now: one that starts in the gap is hit too (seen the same day)
    alone, _, _, _ = urgent.plan([job("1", "normal")], gpu=True, whole_box=False, leases=[], unregistered=[], services={})
    assert any("KNOCK-ON  normal job(s) 1 " in line and "running now" not in line for line in alone)
    # no normal job pre-empted, no knock-on; a cpu_fill job already in the way (cores short) is not named twice
    quiet, _, _, _ = urgent.plan([job("6", "cpu_fill", gpus=0, cls="resume")], gpu=True, whole_box=False, leases=[], unregistered=[], services={})
    assert quiet == ["[urgent] nothing in the way"]
    short, victims_short, _, _ = urgent.plan([job("1", "normal"), job("6", "cpu_fill", gpus=0, cls="resume")], gpu=False, whole_box=False,
                                             leases=[], unregistered=[], services={}, cores_short=True)
    assert victims_short == ["1", "6"] and not any("running now" in line for line in short)
    # only a normal job's restart outranks cpu_fill: a cpu_fill job pre-empted alone brings no knock-on
    only_fill, victims_fill, _, _ = urgent.plan([job("6", "cpu_fill", gpus=0, cls="resume")], gpu=False, whole_box=False, leases=[],
                                                unregistered=[], services={}, cores_short=True)
    assert victims_fill == ["6"] and not any("KNOCK-ON" in line for line in only_fill)


def test_a_running_shard_job_is_in_the_way_of_an_urgent_gpu_job() -> None:
    shard = gpu_jobs.QueuedJob("8", "normal", "RUNNING", "None", "p:x:8", shards=4)  # holds shards, no GPU
    _, victims, _, _ = urgent.plan([shard, job("5", "cpu", gpus=0, cls="resume")], gpu=True, whole_box=False,
                                   leases=[], unregistered=[], services={})
    assert victims == ["8"]


def test_whole_box_puts_every_running_job_in_the_way() -> None:
    _, victims, steady, _ = urgent.plan(
        [job("1", "normal"), job("3", "cpu", gpus=0, cls="resume"), job("5", "cpu", gpus=0, cls="restart")],
        gpu=True, whole_box=True, leases=[], unregistered=[], services={},
    )
    assert victims == ["1", "3"]  # a cpu job that resumes is a victim (urgent.py requeues it; slurm never pre-empts cpu)
    assert steady == ["5"]  # one that restarts would lose its run, like a steady job


def test_stops_without_submitting_behind_a_wait_only_blocker_or_without_a_site_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(gpu_jobs, "read_queue", lambda: [job("2", "steady")])
    monkeypatch.setattr(urgent, "other_leases", list)
    monkeypatch.setattr(urgent.audit, "unregistered", list)
    monkeypatch.setattr(urgent, "free_on_nodes", lambda: None)
    monkeypatch.setattr(urgent.site_config, "is_admin", lambda _site, _user: True)
    submitted = []
    monkeypatch.setattr(urgent.subprocess, "run", lambda *a, **k: submitted.append(a))
    argv = ["--project", "p", "--what", "w", "--time", "0:10:00", "--", "true"]
    monkeypatch.setattr(urgent.site_config, "SITE_FILE", tmp_path / "absent-site.toml")
    with pytest.raises(SystemExit, match=r"^\[gpujobs\] REFUSED: no site file at"):  # so no admin check is skipped
        urgent.run(argv)
    site_file(tmp_path, monkeypatch)
    gpu_jobs.update_sidecar("2", what="\x1b]0;forged\x07")  # whatever its submitter typed
    assert urgent.run(argv) == 1
    printed = capsys.readouterr()
    assert "STOPPED, nothing submitted" in printed.err
    assert "forged" in printed.out and printed.out.rstrip("\n").replace("\n", "").isprintable()
    assert submitted == []


def test_a_job_without_gpus_whose_memory_or_cores_do_not_fit_puts_running_jobs_in_the_way() -> None:
    for short in ({"memory_short": True}, {"cores_short": True}):
        _, victims, _, _ = urgent.plan(
            [job("3", "normal")], gpu=False, whole_box=False, leases=[], unregistered=[], services={}, **short
        )
        assert victims == ["3"], short
    _, victims, _, _ = urgent.plan([job("3", "normal")], gpu=False, whole_box=False, leases=[], unregistered=[],
                                   services={})
    assert victims == []  # it fits: a GPU job is not in a CPU job's way
    # a service stand-in (D20) in the way is requeued by slurm (its partition's PreemptMode=REQUEUE), never wait-only
    lines, victims, _, wait_only = urgent.plan([job("3", "service", gpus=0)], gpu=False, whole_box=False, leases=[],
                                               unregistered=[], services={}, cores_short=True)
    assert (victims, wait_only) == (["3"], []) and "a service's CPU stand-in" in "\n".join(lines)


def test_free_room_is_counted_only_on_nodes_the_urgent_partition_can_start_a_job_on(monkeypatch: pytest.MonkeyPatch) -> None:
    asked: list[object] = []
    monkeypatch.setattr(gpu_jobs, "capacity", lambda *_a, **kw: asked.append(kw.get("partition")) or
                        {"nodes": 1, "mem_mb": 10, "mem_used_mb": 4, "cpus": 8, "cpus_used": 3})
    assert urgent.free_on_nodes() == (6, 5) and asked == [gpu_jobs.Partition.URGENT]
    monkeypatch.setattr(gpu_jobs, "capacity", lambda *_a, **_k: {"nodes": 0})  # every node down: unknown, never "plenty"
    assert urgent.free_on_nodes() is None
    # D46: a job bound to the fast cores (socket 1 here) counts only those; slow cores and waiting jobs free none of them
    node = urgent.site_config.Node("n", 20, 1000, (urgent.site_config.Gpu(0, "u", "/dev/nvidia0", "t"),), fast_cpus="5-9,15-19")
    queue = [gpu_jobs.QueuedJob("1", "cpu_fill", "RUNNING", "None", "p:x:1", cores=frozenset({"n:1:0", "n:1:1", "n:0:0"})),
             gpu_jobs.QueuedJob("2", "cpu", "PENDING", "Resources", "p:x:2", cores=frozenset({"n:1:2"})),
             gpu_jobs.QueuedJob("3", "cpu", "RUNNING", "None", "p:x:3", cores=frozenset({"other:1:3"})),
             gpu_jobs.QueuedJob("4", "cpu_fill", "COMPLETING", "None", "p:x:4", cores=frozenset({"n:1:4"}))]  # still on its cores
    assert urgent.jobs_view.free_fast_cores(queue, node) == 7


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
    write_new = gpu_jobs.write_new
    queue = [
        gpu_jobs.QueuedJob("5", "normal", "RUNNING", "None", "p:x:5", gpus=2),
        gpu_jobs.QueuedJob("6", "normal", "PENDING", "Resources", "p:x:6", gpus=1),
    ]
    monkeypatch.setattr(gpu_jobs, "read_queue", lambda: queue)
    monkeypatch.setattr(urgent, "other_leases", lambda: [("svc", False)])
    monkeypatch.setattr(urgent.audit, "unregistered", list)
    monkeypatch.setattr(urgent, "free_on_nodes", lambda: None)
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
    (tmp_path / "jobs" / "m.json").write_text(json.dumps({"name": "u", "last_leg": {
        "peak_gpu_mib": 2048, "peak_mem_bytes": 0, "ended_at": 1, "exit": 0, "signals": []}}))
    assert urgent.run([*argv, "--dry-run", "--", "true"]) == 0
    captured = capsys.readouterr()
    out = captured.out
    assert "[submit] warning: --mem 1G (1.1 GB) is below the 2.1 GB 'u' peaked at" in captured.err
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

    # a refused submission releases what it held; a steady job in the way stops it unless --requeue-steady
    calls: list[list[str]] = []

    def refusing_sbatch(command: list[str], **_kwargs: object) -> object:
        calls.append(command)
        return type("Done", (), {"returncode": int(command[0] == "sbatch"), "stdout": "", "stderr": "no"})()

    monkeypatch.setattr(urgent.subprocess, "run", refusing_sbatch)
    holds = len(gpu_jobs.events("6"))
    assert urgent.run([*argv, "--", "true"]) == 1
    assert ["scontrol", "hold", "6"] in calls and calls[-1] == ["scontrol", "release", "6"]
    assert len(gpu_jobs.events("6")) == holds  # a refused submission leaves no hold in the job's history

    # a hold slurm refuses stops it before anything is submitted, with slurm's words (no traceback)
    def refusing_hold(command: list[str], **_kwargs: object) -> object:
        calls.append(command)
        return type("Done", (), {"returncode": int(command[:2] == ["scontrol", "hold"]), "stdout": "", "stderr": "denied"})()

    calls.clear()
    capsys.readouterr()
    monkeypatch.setattr(urgent.subprocess, "run", refusing_hold)
    assert urgent.run([*argv, "--", "true"]) == 1
    err = capsys.readouterr().err
    assert calls == [["scontrol", "hold", "6"], ["scontrol", "release", "6"]]  # a partial hold is undone
    assert "denied" in err and "would not hold ['6']" in err

    queue[:] = [gpu_jobs.QueuedJob("2", "steady", "RUNNING", "None", "p:x:2", gpus=2)]
    monkeypatch.setattr(urgent.subprocess, "run", refusing_sbatch)
    calls.clear()
    assert urgent.run([*argv, "--", "true"]) == 1
    assert calls == []
    assert urgent.run([*argv, "--requeue-steady", "--", "true"]) == 1  # sbatch still refuses
    assert ["scontrol", "requeue", "2"] not in calls  # so the steady job keeps its run

    calls.clear()
    monkeypatch.setattr(urgent.subprocess, "run", lambda command, **_k: calls.append(command) or done)
    assert urgent.run([*argv, "--requeue-steady", "--", "true"]) == 0
    assert calls[-2][0] == "sbatch" and calls[-1] == ["scontrol", "requeue", "2"]  # requeued only once accepted
    refused = type("Done", (), {"returncode": 1, "stdout": "", "stderr": "denied"})()
    monkeypatch.setattr(urgent.subprocess, "run", lambda c, **_k: done if c[0] == "sbatch" else refused)
    capsys.readouterr()
    assert urgent.run([*argv, "--requeue-steady", "--", "true"]) == 0  # submitted; the steady job keeps the card
    assert "could not requeue steady job 2" in capsys.readouterr().err
    # no batch script (every name taken): refused before any requeue or hold, so nothing is left held
    calls.clear()
    capsys.readouterr()
    monkeypatch.setattr(urgent.gpu_jobs, "write_new", lambda *_a: (_ for _ in ()).throw(FileExistsError(17, "taken")))
    assert urgent.run([*argv, "--requeue-steady", "--", "true"]) == 1
    assert calls == [] and "REFUSED: the batch script taken" in capsys.readouterr().err

    # slurm never pre-empts the cpu partition: urgent.py requeues a resume-class cpu job in its way, once accepted
    queue[:] = [gpu_jobs.QueuedJob("3", "cpu", "RUNNING", "None", "p:resume:3", gpus=0)]
    monkeypatch.setattr(urgent.gpu_jobs, "write_new", write_new)
    monkeypatch.setattr(urgent.subprocess, "run", lambda command, **_k: calls.append(command) or done)
    calls.clear()
    assert urgent.run(["--project", "p", "--what", "w", "--time", "0:10:00", "--name", "u", "--whole-box", "--", "true"]) == 0
    assert ["scontrol", "requeue", "3"] in calls
    # ... and it stays out of the restore-order request: the held GPU jobs wait only for what slurm pre-empted
    queue[:] = [gpu_jobs.QueuedJob("5", "normal", "RUNNING", "None", "p:resume:5", gpus=2), queue[0],
                gpu_jobs.QueuedJob("6", "normal", "PENDING", "Resources", "p:resume:6", gpus=1)]
    calls.clear()
    assert urgent.run(["--project", "p", "--what", "w", "--time", "0:10:00", "--name", "u", "--whole-box", "--", "true"]) == 0
    request = max((tmp_path / "requests").glob("restore_order-*.json"), key=lambda p: p.stat().st_mtime_ns)
    assert json.loads(request.read_text())["victims"] == ["5"] and ["scontrol", "requeue", "3"] in calls
    # memory short (not the whole box): the cpu job is requeued only if pre-empting the normal one cannot free enough
    monkeypatch.setattr(urgent, "free_on_nodes", lambda: (1000, 99))
    for asked, requeued in (("4G", False), ("512M", True), ("lots", True)):  # needs 1048 MB more; its owner wrote "lots"
        gpu_jobs.update_sidecar("5", mem=asked)
        calls.clear()
        assert urgent.run([*argv[:-4], "--mem", "2G", "--", "true"]) == 0
        assert (["scontrol", "requeue", "3"] in calls) is requeued, asked
    # cores short (the memory fits): the same rule with the cores the normal job holds; needs 4, slurm has 1 left
    monkeypatch.setattr(urgent, "free_on_nodes", lambda: (10**6, 1))
    for held_cores, requeued in ((12, False), (2, True)):
        queue[0] = gpu_jobs.QueuedJob("5", "normal", "RUNNING", "None", "p:resume:5", gpus=2, cpus=held_cores)
        calls.clear()
        assert urgent.run([*argv[:-4], "--mem", "2G", "--cpus", "4", "--", "true"]) == 0
        assert (["scontrol", "requeue", "3"] in calls) is requeued, held_cores
    # D46, bound to the fast cores: only the fast cores a victim gives back count. The normal job's 12 are all slow, so
    # the cpu job on 8 of the 10 fast cores is requeued although 12 cores would be freed
    (tmp_path / "site.toml").write_text(
        'name = "t"\nadmin_group = "admins"\n[[nodes]]\nname = "n1"\ncpus = 20\nreal_memory_mb = 1000\nfast_cpus = "5-9,15-19"\n'
        '[[nodes.gpus]]\nindex = 0\nuuid = "GPU-x"\ndevice = "/dev/nvidia0"\ntype = "gb10"\n', encoding="utf-8")
    queue[0] = gpu_jobs.QueuedJob("5", "normal", "RUNNING", "None", "p:resume:5", cpus=12,
                                  cores=frozenset(f"n1:0:{c}" for c in range(10)) | {"n1:1:8", "n1:1:9"})
    queue[1] = gpu_jobs.QueuedJob("3", "cpu", "RUNNING", "None", "p:resume:3", cpus=8, cores=frozenset(f"n1:1:{c}" for c in range(8)))
    calls.clear()
    assert urgent.run([*argv[:-4], "--mem", "2G", "--gpu", "--cpus", "4", "--", "true"]) == 0
    assert ["scontrol", "requeue", "3"] in calls


def test_entry_runs_the_gate_before_anything_is_held_and_mem_auto_sizes_the_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "site.toml").write_text(
        'name = "t"\nadmin_group = "admins"\ndgx_gate = true\n[[nodes]]\nname = "n1"\ncpus = 20\nreal_memory_mb = 100000\n'
        'cores_per_gpu = 12\n[[nodes.gpus]]\nindex = 0\nuuid = "GPU-x"\ndevice = "/dev/nvidia0"\ntype = "gb10"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(urgent.site_config, "SITE_FILE", tmp_path / "site.toml")
    monkeypatch.setattr(urgent.site_config, "is_admin", lambda _site, _user: True)
    queue = [
        gpu_jobs.QueuedJob("5", "normal", "RUNNING", "None", "p:x:5", gpus=1),
        gpu_jobs.QueuedJob("6", "normal", "PENDING", "Resources", "p:x:6", gpus=1),
    ]
    monkeypatch.setattr(gpu_jobs, "read_queue", lambda: queue)
    monkeypatch.setattr(urgent, "other_leases", list)
    monkeypatch.setattr(urgent.audit, "unregistered", list)
    monkeypatch.setattr(urgent, "free_on_nodes", lambda: None)
    peak = {"bytes": 20 * 1024**3}
    monkeypatch.setattr(gpu_jobs, "measured_peak_bytes", lambda _name, *, clean=True: peak["bytes"])
    calls: list[list[str]] = []
    done = type("Done", (), {"returncode": 0, "stdout": "9", "stderr": ""})()
    monkeypatch.setattr(urgent.subprocess, "run", lambda command, **_k: calls.append(command) or done)
    gates: list[tuple] = []

    def gate(repo: Path, entry: str, command: list[str], manifest: Path, searches: list[str]) -> None:
        gates.append((repo, entry, command, manifest.parent, searches))
        if gates[-1][1] == "refused.py":
            raise urgent.submit.Refused("the DGX gate refused (exit 1); nothing was queued")
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(urgent.submit, "run_gate", gate)
    argv = ["--project", "p", "--what", "w", "--time", "0:10:00", "--name", "u", "--mem", "auto", "--gpu", "--repo", str(tmp_path),
            "--search", "third_party/x"]

    # a gate refusal holds, requeues and submits nothing
    assert urgent.run([*argv, "--entry", "refused.py", "--", "true"]) == 1
    assert calls == [] and "the DGX gate refused" in capsys.readouterr().err

    # the gate passes: its manifest copy goes to the job wrapper and the job's record; --mem auto is the measured peak plus 10%
    assert urgent.run([*argv, "--entry", "scripts/e.py", "--", "true"]) == 0
    assert gates[-1] == (tmp_path.resolve(), "scripts/e.py", ["true"], tmp_path / "manifests", ["third_party/x"])
    script = Path(calls[-1][-1]).read_text(encoding="utf-8")
    manifest = gpu_jobs.read_sidecar("9", gpu_jobs.current_user())["manifest"]
    assert f"--manifest {manifest} --repo {tmp_path}" in script and "--mem=22G" in script and "--exclusive" not in script
    assert "--cpus-per-task=12" in script  # as submit.py: the site's cores per GPU
    assert gpu_jobs.read_sidecar("9", gpu_jobs.current_user())["entry"] == "scripts/e.py"
    assert ["scontrol", "hold", "6"] in calls  # the other pending GPU job is held only after the gate

    # no measured run: --mem auto is the whole box, exactly as --whole-box
    peak["bytes"] = None
    assert urgent.run([*argv, "--", "true"]) == 0
    assert "--mem=0" in Path(calls[-1][-1]).read_text(encoding="utf-8") and "--exclusive" in Path(calls[-1][-1]).read_text(encoding="utf-8")

    # D46: on a node with fast_cpus a job that fits in the fast cores is bound to them; the whole box never is
    site = (tmp_path / "site.toml").read_text(encoding="utf-8").replace("cores_per_gpu = 12\n", 'cores_per_gpu = 12\nfast_cpus = "5-9,15-19"\n')
    (tmp_path / "site.toml").write_text(site, encoding="utf-8")
    peak["bytes"] = 20 * 1024**3
    assert urgent.run([*argv, "--cpus", "2", "--", "true"]) == 0
    bound = Path(calls[-1][-1]).read_text(encoding="utf-8")
    assert "--cpus-per-task=2" in bound and "--ntasks=1\n" in bound and "--gres-flags=enforce-binding\n" in bound
    peak["bytes"] = None
    assert urgent.run([*argv, "--cpus", "2", "--", "true"]) == 0
    assert "--gres-flags" not in Path(calls[-1][-1]).read_text(encoding="utf-8")


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

    monkeypatch.setattr("sys.argv", ["pause.py", "2"])  # a steady job loses its whole run: not without being told so
    assert pause.main() == 1
    assert slurm.calls == [] and actions("2") == []
    monkeypatch.setattr("sys.argv", ["pause.py", "2", "--lose-progress"])
    assert pause.main() == 0
    assert slurm.calls == [["scontrol", "requeuehold", "2"]]
    slurm.calls.clear()

    monkeypatch.setattr("sys.argv", ["pause.py", "1", "--reason", "need the card"])
    gpu_jobs.update_sidecar("1", what="\x1b[2Jforged")
    capsys.readouterr()
    assert pause.main() == 0
    out = capsys.readouterr().out
    assert "forged" in out and out.replace("\n", "").isprintable()  # a description is shown, never acted on
    assert slurm.calls == [["scontrol", "requeuehold", "1"]]
    assert [(e.action, e.fields) for e in gpu_jobs.events("1")] == [("pause", {"reason": "need the card"})]

    # slurm refuses to requeue a pending job; a pending job loses nothing, steady or not
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
