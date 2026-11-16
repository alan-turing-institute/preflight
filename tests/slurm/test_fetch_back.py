"""fetch_back.py: the rsync pull of one folder, its free-space check and how its exit is read (design 4, 5, 7, 8)."""

from __future__ import annotations

import argparse
import dataclasses
import fcntl
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "slurm"))

import fetch_back as fb
import fetch_record as fr
import gpu_jobs
import site_config
from fetch_record import Existing, FetchPath

HOST = "site-b"
FORBIDDEN = {"--delete", "--remove-source-files", "--trust-sender", "--old-args", "--secluded-args", "--partial", "-D"}
needs_rsync = pytest.mark.skipif(shutil.which("rsync") is None, reason="rsync is not installed")
BIG_BYTES = 20_000_000
SLOW_KBPS = 2000  # the copy of BIG_BYTES takes about 10 s at this rate, so it is still running when stopped
STOP_AFTER_S = 2
PLANNED_BYTES = 88_716_298  # a real dry run of a job folder (design 5)


def test_the_pull_argv_is_read_only_ordered_and_one_source() -> None:
    path = FetchPath("gpujobs-sync/proj/out", "/home/alice/proj/out/", (("+", "*resume_*"), ("-", "*")), "2G")
    ssh = fb.ssh_command(Path("/run/cm-%C"))
    argv = fb.rsync_argv(HOST, path, Existing.UPDATE, ssh)
    assert argv[-2:] == [f"{HOST}:gpujobs-sync/proj/out/", "/home/alice/proj/out/"]
    assert [a for a in argv if a.startswith(("--include", "--exclude"))] == ["--include=*resume_*", "--exclude=*"]
    assert {"--safe-links", "--partial-dir=.gpujobs-partial", "-u", "--max-size=2G", "--timeout=20"} <= set(argv)
    assert not FORBIDDEN & set(argv)
    assert argv[argv.index("-e") + 1].split()[:3] == ["ssh", "-o", "BatchMode=yes"]
    assert "ControlPath=/run/cm-%C" in argv[argv.index("-e") + 1]
    keep = fb.rsync_argv(HOST, path, Existing.KEEP, ssh)
    replace = fb.rsync_argv(HOST, path, Existing.REPLACE, ssh, dry_run=True)
    assert "--ignore-existing" in keep and "-u" not in keep
    assert not {"-u", "--ignore-existing"} & set(replace)
    assert {"-n", "--stats"} <= set(replace)


@pytest.mark.parametrize(
    ("code", "stderr", "expected"),
    [
        (0, "", fb.Outcome.DONE),
        (255, "ssh: connect to host site-b port 22: Connection refused", fb.Outcome.RETRY),
        (30, "timeout in data send/receive", fb.Outcome.RETRY),
        (23, 'rsync: [sender] change_dir "/x/out" failed: No such file or directory (2)', fb.Outcome.MISSING),
        (23, "rsync: read errors mapping", fb.Outcome.RETRY),
        (24, "file has vanished", fb.Outcome.VANISHED),
        (12, "protocol data stream", fb.Outcome.FAILED),
    ],
)
def test_rsync_exit_is_read_with_its_message(code: int, stderr: str, expected: fb.Outcome) -> None:
    assert fb.classify(code, stderr) == expected


def test_the_size_check_reads_a_dry_run_and_leaves_the_kept_space_free(tmp_path: Path) -> None:
    stats = "Number of files: 3\nTotal file size: 166,701,561 bytes\nTotal transferred file size: 88,716,298 bytes\n"
    assert fb.planned_bytes(stats) == PLANNED_BYTES
    assert fb.planned_bytes("nothing") is None
    target = tmp_path / "not" / "yet" / "there"
    free = shutil.disk_usage(tmp_path).free
    assert fb.short_of_space(target, 1, keep_free_gb=0) is None
    refusal = fb.short_of_space(target, 1, keep_free_gb=free / 1e9 + 1)
    assert refusal is not None and "must stay free" in refusal


def test_a_local_folder_swapped_for_a_link_out_of_the_repo_is_refused(tmp_path: Path) -> None:
    repo, elsewhere = tmp_path / "repo", tmp_path / "elsewhere"
    (repo / "out").mkdir(parents=True)
    elsewhere.mkdir()
    assert fb.inside(repo / "out", repo.resolve())
    assert fb.inside(repo / "new" / "deeper", repo.resolve())
    (repo / "out").rmdir()
    (repo / "out").symlink_to(elsewhere)
    assert not fb.inside(repo / "out", repo.resolve())


@pytest.fixture
def fake_ssh(tmp_path: Path) -> list[str]:
    """An `ssh` that drops the host and runs the remote rsync here: real rsync on both ends, no network."""
    script = tmp_path / "fake-ssh"
    script.write_text('#!/bin/sh\nwhile [ "${1#-}" != "$1" ]; do shift 2; done\nshift\nexec "$@"\n')
    script.chmod(0o755)
    return [str(script)]


def pull(source: Path, dest: Path, ssh: list[str], existing: Existing = Existing.UPDATE, *extra: str):
    argv = fb.rsync_argv(HOST, FetchPath(str(source), str(dest)), existing, ssh)
    return argv[:1] + list(extra) + argv[1:]


@needs_rsync
def test_an_interrupted_pull_leaves_no_truncated_file_and_the_rerun_completes_it(tmp_path: Path, fake_ssh) -> None:
    source, dest = tmp_path / "remote" / "out", tmp_path / "home" / "out"
    source.mkdir(parents=True)
    dest.mkdir(parents=True)
    (source / "big.bin").write_bytes(os.urandom(BIG_BYTES))
    running = subprocess.Popen(pull(source, dest, fake_ssh, Existing.UPDATE, f"--bwlimit={SLOW_KBPS}"),
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(STOP_AFTER_S)
    running.send_signal(signal.SIGTERM)
    running.wait()
    assert not (dest / "big.bin").exists()
    done = subprocess.run(pull(source, dest, fake_ssh), capture_output=True, text=True, check=False)
    assert fb.classify(done.returncode, done.stderr) == fb.Outcome.DONE
    assert (dest / "big.bin").read_bytes() == (source / "big.bin").read_bytes()


@needs_rsync
def test_a_real_pull_keeps_newer_local_files_skips_outward_links_and_names_a_missing_folder(tmp_path: Path, fake_ssh) -> None:
    source, dest = tmp_path / "remote" / "out", tmp_path / "home" / "out"
    source.mkdir(parents=True)
    dest.mkdir(parents=True)
    (source / "result.txt").write_text("from the job")
    (source / "edited.txt").write_text("from the job")
    (source / "escape").symlink_to("/etc/hostname")
    (dest / "edited.txt").write_text("edited here after the job")
    os.utime(dest / "edited.txt", (time.time() + 60, time.time() + 60))
    done = subprocess.run(pull(source, dest, fake_ssh), capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr
    assert (dest / "result.txt").read_text() == "from the job"
    assert (dest / "edited.txt").read_text() == "edited here after the job"
    assert not (dest / "escape").is_symlink()
    gone = subprocess.run(pull(tmp_path / "remote" / "never", dest, fake_ssh), capture_output=True, text=True, check=False)
    assert fb.classify(gone.returncode, gone.stderr) == fb.Outcome.MISSING


# ---------------------------------------------------------------------------------------------------------------------
# The run loop and the commands, over a fake ssh/sacct/rsync (design 2, 3, 8, 17).


ROOT_THERE = "/home/alice/gpujobs-sync/proj"
SETTINGS = fb.Settings("site-b", "gpujobs-sync", ("snapshot-*.ckpt",), 0.0, 7.0, (60, 300), 3600,
                       "Permission denied (publickey)", "run the renew command", 300)
RENEWED_AFTER_S = 120


class Remote:
    """A fake remote: answers `id`, `sacct` and rsync, and runs `--fetch-then` commands, recording every call."""

    def __init__(self) -> None:
        self.states = {"41": "RUNNING"}
        self.user, self.workdir, self.reachable, self.error = "alice", f"{ROOT_THERE}/run", True, ""
        self.calls: list[list[str]] = []
        self.then_exit = 0
        self.now = 1_000_000.0

    def __call__(self, argv, **kwargs):
        self.calls.append(list(argv))
        line = " ".join(argv)
        ok = lambda out="": subprocess.CompletedProcess(argv, 0, out, "")
        if argv[0] != "rsync" and argv[0] != "ssh":
            return subprocess.CompletedProcess(argv, self.then_exit, "", "")
        if "-O" in argv:
            return ok()
        if not self.reachable:
            return subprocess.CompletedProcess(argv, 255, "", self.error or "ssh: connect to host site-b: Connection refused")
        if "id -un" in line:
            return ok("alice\n/home/alice\n")
        if "sacct" in line:
            return ok("".join(f"{job}|{state}|{self.user}|{self.workdir}\n" for job, state in self.states.items()))
        if "-n" in argv:
            return ok("Total transferred file size: 5 bytes\n")
        dest = Path(argv[-1])
        (dest / "result.txt").write_text("from the job")
        return ok(">f+++++++++ result.txt\n")

    def clock(self) -> float:
        return self.now


@pytest.fixture
def there(tmp_path: Path) -> Remote:
    return Remote()


def context(remote_: Remote, tmp_path: Path, notices: list | None = None, command=("notify",)) -> fb.Context:
    return fb.Context(remote_, lambda argv, env: (notices if notices is not None else []).append(env), tuple(command),
                      "alice", {"XDG_STATE_HOME": str(tmp_path / "state")}, remote_.clock)


def added(there: Remote, tmp_path: Path, **options) -> fr.Record:
    repo = tmp_path / "proj"
    repo.mkdir(exist_ok=True)
    return fb.add_record("site-b", SETTINGS, ["41"], [(f"{ROOT_THERE}/out", str(repo / "out"))],
                         fb.Options(**options), context(there, tmp_path), repo=repo, workdir_root="gpujobs-sync/proj")


def test_a_record_waits_fetches_when_the_job_completes_then_rests(there: Remote, tmp_path: Path) -> None:
    record = added(there, tmp_path)
    assert (record.login, record.workdir_root) == ("alice", ROOT_THERE)
    assert record.paths[0].filters == (("-", "snapshot-*.ckpt"),)  # the site's exclusions come first
    ctx = context(there, tmp_path)
    there.calls.clear()
    fb.run_site("site-b", SETTINGS, ctx)
    folder = fr.store_dir("site-b", ctx.env)
    assert fr.find(folder, "41").state == fr.RecordState.WAITING
    there.states["41"] = "COMPLETED"
    lines = fb.run_site("site-b", SETTINGS, ctx)
    done = fr.find(folder, "41")
    assert (done.state, done.paths[0].bytes) == (fr.RecordState.DONE, 5)
    assert (tmp_path / "proj" / "out" / "result.txt").read_text() == "from the job"
    assert lines == ["[fetch] site-b job 41: done"]
    assert ">f+++++++++ result.txt" in (folder / "41.log").read_text()
    sacct = [c for c in there.calls if "sacct" in " ".join(c)]
    assert len(sacct) == len(("waiting", "done"))  # one sacct per run for every record
    there.calls.clear()
    assert fb.run_site("site-b", SETTINGS, ctx) == []
    assert there.calls == []  # a final record costs nothing


def test_a_second_run_leaves_a_locked_remote_alone(there: Remote, tmp_path: Path) -> None:
    added(there, tmp_path)
    ctx = context(there, tmp_path)
    there.calls.clear()
    with (fr.store_dir("site-b", ctx.env) / ".lock").open("w") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        assert "another run" in fb.run_site("site-b", SETTINGS, ctx)[0]
    assert there.calls == []


def test_an_expired_login_blocks_once_with_the_remedy_and_resumes_after_renewal(there: Remote, tmp_path: Path) -> None:
    added(there, tmp_path)
    notices: list = []
    ctx = context(there, tmp_path, notices)
    there.reachable, there.error = False, "alice@site-b: Permission denied (publickey)."
    fb.run_site("site-b", SETTINGS, ctx)
    there.now += RENEWED_AFTER_S
    fb.run_site("site-b", SETTINGS, ctx)
    record = fr.find(fr.store_dir("site-b", ctx.env), "41")
    assert record.state == fr.RecordState.BLOCKED and "run the renew command" in record.note
    assert [n["GPUJOBS_EVENT"] for n in notices] == ["fetch_blocked"]  # told once, not on every run
    assert notices[0]["GPUJOBS_JOB"] == "41"
    there.reachable, there.states["41"] = True, "COMPLETED"
    there.now += RENEWED_AFTER_S * 10
    fb.run_site("site-b", SETTINGS, ctx)
    assert fr.find(fr.store_dir("site-b", ctx.env), "41").state == fr.RecordState.DONE


@pytest.mark.parametrize(
    ("change", "fragment"),
    [
        (lambda r, a: setattr(r, "user", "bob"), "belongs to bob"),
        (lambda r, a: setattr(r, "workdir", "/home/alice/other"), "outside /home/alice/gpujobs-sync/proj"),
        (lambda r, a: a.update(sources=["../../etc"]), "no '..'"),
        (lambda r, a: a.update(targets=["relative/out"]), "absolute path"),
        (lambda r, a: a.update(jobs=["41;rm"]), "not a slurm job id"),
    ],
)
def test_add_refuses_another_persons_job_and_paths_that_leave_the_folders(there: Remote, tmp_path: Path, change, fragment) -> None:
    given = {"jobs": ["41"], "sources": [f"{ROOT_THERE}/out"], "targets": [str(tmp_path / "proj" / "out")]}
    change(there, given)
    with pytest.raises(fb.FetchRefused, match=re.escape(fragment)):
        fb.add_record("site-b", SETTINGS, given["jobs"], list(zip(given["sources"], given["targets"], strict=True)),
                      fb.Options(), context(there, tmp_path), repo=tmp_path / "proj", workdir_root="gpujobs-sync/proj")
    assert not fr.store_dir("site-b", {"XDG_STATE_HOME": str(tmp_path / "state")}).is_dir()


def test_the_then_command_runs_once_after_a_fetch_never_after_a_skip_and_again_only_when_asked(there: Remote, tmp_path: Path) -> None:
    added(there, tmp_path, then="python3 score.py", fetch_on=fr.FetchOn.COMPLETED)
    notices: list = []
    ctx = context(there, tmp_path, notices)
    there.states["41"] = "FAILED"
    fb.run_site("site-b", SETTINGS, ctx)
    folder = fr.store_dir("site-b", ctx.env)
    assert fr.find(folder, "41").state == fr.RecordState.SKIPPED
    assert not [c for c in there.calls if c[0] == "python3"]
    there.then_exit = 2
    fb.run_site("site-b", SETTINGS, ctx, force=frozenset({"41"}))
    fb.run_site("site-b", SETTINGS, ctx, force=frozenset({"41"}))
    assert len([c for c in there.calls if c[0] == "python3"]) == 1
    assert fr.find(folder, "41").then_result["exit"] == there.then_exit
    assert [n["GPUJOBS_EVENT"] for n in notices] == ["fetch_then_failed"]
    fb.run_site("site-b", SETTINGS, ctx, force=frozenset({"41"}), then_again=True)
    assert len([c for c in there.calls if c[0] == "python3"]) == len(("first", "asked again"))
    assert fb.summary_line("site-b", ctx.env) == "[fetch] fetch-back: 1 done"


def test_wait_ends_with_the_fetch_and_says_whether_the_folders_arrived(there: Remote, tmp_path: Path, monkeypatch) -> None:
    site = site_config.load(Path(__file__).resolve().parent / "fixtures" / "site.toml")
    site = dataclasses.replace(site, remotes={"site-b": site_config.Remote("site-b", "gpujobs-sync")})
    monkeypatch.setattr(gpu_jobs, "load_site", lambda: site)
    added(there, tmp_path)
    ctx = context(there, tmp_path)
    pauses: list = []

    def complete(seconds: float) -> None:
        pauses.append(seconds)
        there.states["41"] = "COMPLETED"
        there.now += seconds

    assert fb.main(["wait", "41", "--site", "site-b"], ctx, sleep=complete) == 0
    assert pauses == [site.tuning.fetch_every_s]
    assert fb.main(["wait", "99", "--site", "site-b"], ctx, sleep=complete) == 1


def test_a_submitted_job_gets_its_record_or_the_exact_command_that_writes_it(there: Remote, tmp_path: Path) -> None:
    repo = tmp_path / "proj"
    repo.mkdir()
    site = SimpleSite({"site-b": site_config.Remote("site-b", "gpujobs-sync", fetch_exclude=("snapshot-*.ckpt",))})
    given = argparse.Namespace(site="site-b", project="proj", repo=repo, fetch_to=[], fetch_filters=[("+", "*resume_*"), ("-", "*")],
                               fetch_on=fr.FetchOn.COMPLETED, fetch_existing=Existing.UPDATE, fetch_max_size=None,
                               fetch_keep_free=None, fetch_wait=None, fetch_then=None)
    pairs = [("gpujobs-sync/proj/out", str(repo / "out"))]
    there.states = {"41": "PENDING"}
    there.workdir = "/home/alice/gpujobs-sync/proj"
    ctx = context(there, tmp_path)
    said = fb.record_after_submit(given, site, "[submit] job 41 (normal)\n41\n", pairs, ctx)
    assert said.startswith("[fetch] job 41: 1 folder(s) come home")
    record = fr.find(fr.store_dir("site-b", ctx.env), "41")
    assert record.paths[0].filters == (("-", "snapshot-*.ckpt"), ("+", "*resume_*"), ("-", "*"))
    assert record.workdir_root == "/home/alice/gpujobs-sync/proj" and record.repo == str(repo.resolve())
    failed = fb.record_after_submit(given, site, "[submit] something else\n", pairs, ctx)
    assert failed.startswith("[fetch] NOT RECORDED") and "the job is queued" in failed
    assert "--from gpujobs-sync/proj/out --to" in failed and "--fetch-include '*resume_*' --fetch-exclude '*'" in failed


class SimpleSite:
    def __init__(self, remotes: dict) -> None:
        self.remotes, self.notify_command = remotes, ()
        self.tuning = site_config.Tuning()
