"""fetch_back.py: the rsync pull of one folder, its free-space check and how its exit is read (design 4, 5, 7, 8)."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "slurm"))

import fetch_back as fb
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
    assert fb.inside(repo / "out", repo)
    assert fb.inside(repo / "new" / "deeper", repo)
    (repo / "out").rmdir()
    (repo / "out").symlink_to(elsewhere)
    assert not fb.inside(repo / "out", repo)


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
