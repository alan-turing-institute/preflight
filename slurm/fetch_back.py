#!/usr/bin/env python3
"""Pull a remote job's declared folders home when it ends (D52; design machine/slurm/design_remote_fetch_back_2026-11-16.md).

Pull only and read only on the remote: `sacct` to learn the job's end, `rsync --server --sender` to read the folders.
"""

from __future__ import annotations

import re
import shlex
import shutil
import sys
from enum import StrEnum
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gpu_jobs
import remote
import transfer
from fetch_record import Existing, FetchPath

#: Where rsync keeps an interrupted file until the next run finishes it, inside the destination folder: never under the
#: real name, so a reader of the folder never sees a truncated file (measured with plain --partial: it does, design 4).
PARTIAL_DIR = ".gpujobs-partial"
#: ssh gives up on an unreachable host after this many seconds (ssh_config(5) ConnectTimeout).
CONNECT_TIMEOUT_S = 10
#: One ssh connection per run for every call to a host (design 17.3); it stays this long after the last call.
CONTROL_PERSIST_S = 60
#: rsync -n --stats: "Total transferred file size: 88,716,298 bytes" (rsync 3.2.7).
PLANNED = re.compile(r"^Total transferred file size: ([\d,]+) bytes", re.MULTILINE)
#: What each overwrite rule adds to rsync (design 7).
EXISTING_FLAGS = {Existing.UPDATE: ["-u"], Existing.KEEP: ["--ignore-existing"], Existing.REPLACE: []}
FILTER_FLAGS = {"+": "--include", "-": "--exclude"}


class Outcome(StrEnum):
    DONE = "done"
    RETRY = "retry"  # the network or a stalled transfer: try again later
    MISSING = "missing"  # the remote folder does not exist: the job wrote nothing there
    VANISHED = "vanished"  # files went away during the copy (nothing should change in a finished job's folder)
    FAILED = "failed"


def ssh_command(control: Path | None = None) -> list[str]:
    """The ssh rsync and sacct run through; with `control`, every call of one run shares one connection."""
    out = [*remote.SSH, "-o", f"ConnectTimeout={CONNECT_TIMEOUT_S}"]
    if control is not None:
        out += ["-o", "ControlMaster=auto", "-o", f"ControlPath={control}", "-o", f"ControlPersist={CONTROL_PERSIST_S}"]
    return out


def rsync_argv(host: str, path: FetchPath, existing: Existing, ssh: list[str], *, dry_run: bool = False) -> list[str]:
    """The pull of one folder. Never --delete or --remove-source-files (the remote is only read, nothing here is
    removed), never --trust-sender (rsync then checks the sender's file list stays inside what was asked), no -D,
    --safe-links (a link pointing out of the tree is not made), --partial-dir (never a truncated file under its name)."""
    filters = [f"{FILTER_FLAGS[sign]}={pattern}" for sign, pattern in path.filters]
    return [
        "rsync", "-rltE", "--itemize-changes", f"--timeout={transfer.COPY_IO_TIMEOUT_S}", "--safe-links",
        f"--partial-dir={PARTIAL_DIR}", *EXISTING_FLAGS[existing],
        *([f"--max-size={path.max_size}"] if path.max_size else []), *filters,
        *(["-n", "--stats"] if dry_run else []),
        "-e", shlex.join(ssh), f"{host}:{path.remote.rstrip('/')}/", f"{path.local.rstrip('/')}/",
    ]  # fmt: skip


def planned_bytes(stdout: str) -> int | None:
    """What a dry run says it would transfer; None when the line is not there."""
    found = PLANNED.search(stdout)
    return int(found.group(1).replace(",", "")) if found else None


def _existing_parent(path: Path) -> Path:
    while not path.exists() and path != path.parent:
        path = path.parent
    return path


def short_of_space(local: Path, need: int, keep_free_gb: float) -> str | None:
    """Why `need` bytes do not fit here while leaving `keep_free_gb` free; None when they fit (design 5)."""
    free = shutil.disk_usage(_existing_parent(local)).free
    keep = keep_free_gb * gpu_jobs.GB
    if free - keep >= need:
        return None
    size = gpu_jobs.size_text
    return f"needs {size(need)}; {size(free)} free at {local}, and {size(keep)} must stay free"


def classify(returncode: int, stderr: str) -> Outcome:
    """rsync's exit value, read with its message where the value alone is ambiguous (design 4)."""
    if returncode == 0:
        return Outcome.DONE
    if returncode in {transfer.RSYNC_SSH_FAILED, transfer.RSYNC_IO_TIMEOUT}:
        return Outcome.RETRY
    if returncode == transfer.RSYNC_PARTIAL_ERROR:
        return Outcome.MISSING if transfer.MISSING_SOURCE in stderr else Outcome.RETRY
    if returncode == transfer.RSYNC_VANISHED:
        return Outcome.VANISHED
    return Outcome.FAILED


def inside(local: Path, repo: Path) -> bool:
    """`local`, with every link on its way resolved now, is `repo` or below it (a folder swapped for a link out of it
    since the record was written is refused, design 8.5)."""
    return local.resolve().is_relative_to(repo.resolve())
