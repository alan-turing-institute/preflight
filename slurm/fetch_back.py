#!/usr/bin/env python3
"""Pull a remote job's declared folders home when it ends (D52): `add`, `run` (the timer), `status`, `wait`, `enable`.

Pull only and read only on the remote: `sacct` to learn the job's end, `rsync --server --sender` to read the folders.
"""

from __future__ import annotations

import argparse
import dataclasses
import fcntl
import getpass
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import agent
import fetch_record as fr
import gpu_jobs
import remote
import site_config
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
    """`local`, with every link on its way resolved now, is `repo` (as resolved when the record was written) or below
    it: a folder swapped for a link out of it since then is refused (design 8.5)."""
    return local.resolve().is_relative_to(repo)


# ---------------------------------------------------------------------------------------------------------------------
# One run: read the jobs' ends, pull what is due, record the result (design 2, 3, 4, 17).

Runner = Callable[..., subprocess.CompletedProcess[str]]
#: `sacct` columns read back, in this order (one row per job with -X).
SACCT_FIELDS = ("JobID", "State", "User", "WorkDir")
#: An ssh call that is not a copy (sacct, id) gives up after this long; a copy is bounded by rsync's own --timeout.
CALL_TIMEOUT_S = 60
#: How much of a failed command's error is kept in a record and a notice.
ERROR_TAIL_CHARS = 300
#: A file vanishing from a finished job's folder is retried once, then the folder is given up (design 4).
VANISHED_TRIES = 2
DAY_S = 86400
#: The notice sent when a record enters each state (owner decision 3: none for success).
NOTICES = {
    fr.RecordState.FAILED: agent.NotifyEvent.FETCH_FAILED,
    fr.RecordState.LOST: agent.NotifyEvent.FETCH_FAILED,
    fr.RecordState.BLOCKED: agent.NotifyEvent.FETCH_BLOCKED,
    fr.RecordState.EXPIRED: agent.NotifyEvent.FETCH_EXPIRED,
}
TIMER = "gpujobs-fetch.timer"


class FetchRefused(Exception):
    """A fetch record that cannot be written; the message names what to change."""


@dataclass(frozen=True)
class Settings:
    """A remote's fetch-back settings (`[remote.NAME]` in the site file) and the timer's pace (`[tuning]`)."""

    host: str
    root: str
    exclude: tuple[str, ...]
    keep_free_gb: float
    wait_days: float
    retry_s: tuple[int, ...]
    lost_after_s: int
    auth_expired_text: str
    renew_hint: str
    every_s: int

    @classmethod
    def from_site(cls, site: site_config.Site, name: str) -> Settings:
        r = remote.pick(site, name)
        return cls(r.host, r.root, r.fetch_exclude, r.fetch_keep_free_gb, r.fetch_wait_days, r.fetch_retry_s,
                   r.fetch_lost_after_s, r.auth_expired_text, r.renew_hint, site.tuning.fetch_every_s)

    @property
    def policy(self) -> fr.Policy:
        return fr.Policy(self.retry_s, self.lost_after_s)


@dataclass(frozen=True)
class Context:
    """What a run takes from outside, so a test swaps it and touches nothing real."""

    runner: Runner = subprocess.run
    notify: Callable[[list[str], dict[str, str]], None] = agent.notify_run
    notify_command: tuple[str, ...] = ()
    user: str = field(default_factory=getpass.getuser)
    env: Mapping[str, str] = field(default_factory=lambda: dict(os.environ))
    clock: Callable[[], float] = time.time


@dataclass(frozen=True)
class Options:
    """What a record asks for beyond its folders (the `--fetch-*` flags)."""

    filters: tuple[tuple[str, str], ...] = ()
    fetch_on: fr.FetchOn = fr.FetchOn.COMPLETED
    existing: Existing = Existing.UPDATE
    max_size: str | None = None
    keep_free_gb: float | None = None  # None: the site's
    wait_days: float | None = None  # None: the site's
    then: str | None = None
    project: str = ""


def _tail(text: str) -> str:
    text = " ".join(text.split())
    return text[-ERROR_TAIL_CHARS:]


def read_jobs(settings: Settings, ssh: list[str], jobs: list[str], ctx: Context) -> tuple[fr.Answer, str]:
    """One `sacct` call for every job (design 3): no answer when ssh fails, rows only for the ids it knows."""
    command = shlex.join(["sacct", "-j", ",".join(jobs), "-X", "-n", "-P", "-o", ",".join(SACCT_FIELDS)])
    done = agent.call(ctx.runner, [*ssh, settings.host, command], timeout=CALL_TIMEOUT_S)
    if done.returncode != 0:
        return fr.Answer(False), done.stderr
    rows = {}
    for line in done.stdout.splitlines():
        parts = line.split("|")
        if len(parts) == len(SACCT_FIELDS) and parts[0] in jobs:
            rows[parts[0]] = (parts[1], parts[2], parts[3])
    return fr.Answer(True, rows), ""


def owner_problem(record: fr.Record, user: str, workdir: str) -> str | None:
    """Why a job with this sacct User and WorkDir must not be fetched for this record (design 8), or None."""
    if record.login and user != record.login:
        return f"the job belongs to {user}, not to {record.login} who logs in there"
    root = record.workdir_root.rstrip("/")
    if root and workdir != root and not workdir.startswith(f"{root}/"):
        return f"the job ran in {workdir}, outside {root}"
    return None


def _pull_one(record: fr.Record, path: FetchPath, settings: Settings, ssh: list[str], ctx: Context, log: Path) -> FetchPath:
    local = Path(path.local)
    if not inside(local, Path(record.repo)):
        return dataclasses.replace(path, state=fr.PathState.FAILED, last_error=f"{local} now leads outside {record.repo}")
    local.mkdir(parents=True, exist_ok=True)
    tried = dataclasses.replace(path, attempts=path.attempts + 1)
    dry = agent.call(ctx.runner, rsync_argv(settings.host, path, record.existing, ssh, dry_run=True), timeout=None)
    outcome = classify(dry.returncode, dry.stderr)
    if outcome == Outcome.DONE:
        need = planned_bytes(dry.stdout) or 0
        short = short_of_space(local, need, record.keep_free_gb)
        if short:
            return dataclasses.replace(tried, state=fr.PathState.BLOCKED, last_error=short)
        done = agent.call(ctx.runner, rsync_argv(settings.host, path, record.existing, ssh), timeout=None)
        with log.open("a", encoding="utf-8") as out:
            out.write(done.stdout)
        outcome = classify(done.returncode, done.stderr)
        if outcome == Outcome.DONE:
            return dataclasses.replace(tried, state=fr.PathState.DONE, bytes=path.bytes + need, last_error="")
        dry = done
    error = _tail(dry.stderr) or f"rsync exit {dry.returncode}"
    if outcome == Outcome.MISSING:
        return dataclasses.replace(tried, state=fr.PathState.FAILED, last_error=f"the job wrote nothing at {path.remote}")
    if outcome == Outcome.RETRY or (outcome == Outcome.VANISHED and tried.attempts < VANISHED_TRIES):
        return dataclasses.replace(tried, state=fr.PathState.PENDING, last_error=error)
    return dataclasses.replace(tried, state=fr.PathState.FAILED, last_error=error)


def fetch_paths(record: fr.Record, settings: Settings, ssh: list[str], ctx: Context, folder: Path) -> fr.Record:
    """Pull every folder not yet pulled; the record is DONE when all are, BLOCKED when one waits for room here, FAILED
    when one cannot be pulled and nothing else is left, else still FETCHING with a pause (design 4, 5)."""
    log = folder / f"{record.jobs[0]}.log"
    paths = tuple(p if p.state == fr.PathState.DONE else _pull_one(record, p, settings, ssh, ctx, log) for p in record.paths)
    record = dataclasses.replace(record, paths=paths)
    now, states = ctx.clock(), {p.state for p in paths}
    if states == {fr.PathState.DONE}:
        return dataclasses.replace(record, state=fr.RecordState.DONE, finished_at=now, note="")
    blocked = [p for p in paths if p.state == fr.PathState.BLOCKED]
    if blocked:
        return dataclasses.replace(record, state=fr.RecordState.BLOCKED, note=f"{blocked[0].local}: {blocked[0].last_error}")
    failed = [p for p in paths if p.state == fr.PathState.FAILED]
    if failed and fr.PathState.PENDING not in states:
        return dataclasses.replace(record, state=fr.RecordState.FAILED, finished_at=now,
                                   note=f"{failed[0].remote}: {failed[0].last_error}")
    pending = next(p for p in paths if p.state == fr.PathState.PENDING)
    return fr.retry_after(dataclasses.replace(record, note=pending.last_error), now, settings.policy)


def run_then(record: fr.Record, folder: Path, ctx: Context, *, again: bool = False) -> fr.Record:
    """The record's `--fetch-then` command, once, from its repo, after a successful fetch only (design 17.2)."""
    if record.state != fr.RecordState.DONE or not record.then or (record.then_result is not None and not again):
        return record
    log = folder / f"{record.jobs[0]}.then.log"
    with log.open("a", encoding="utf-8") as out:
        try:
            code = ctx.runner(shlex.split(record.then), cwd=record.repo, stdout=out, stderr=subprocess.STDOUT,
                              check=False).returncode
        except OSError as exc:
            out.write(f"{exc}\n")
            code = 127  # the shell's "command not found"
    return dataclasses.replace(record, then_result={"exit": code, "at": ctx.clock(), "log": str(log)})


def step(record: fr.Record, answer: fr.Answer, error: str, settings: Settings, ssh: list[str], ctx: Context,
         folder: Path, *, then_again: bool = False) -> fr.Record:
    """One record through one run."""
    now = ctx.clock()
    if not answer.reached:
        expired = bool(settings.auth_expired_text) and settings.auth_expired_text in error
        if expired:
            note = f"the login to {settings.host} has expired; {settings.renew_hint or 'renew it'}, then it continues"
            return fr.retry_after(dataclasses.replace(record, state=fr.RecordState.BLOCKED, note=note), now, settings.policy)
        moved = fr.advance(record, answer, now, settings.policy)
        return dataclasses.replace(moved, note=f"ssh to {settings.host} failed: {_tail(error)}") if not fr.is_final(moved) else moved
    for job in record.jobs:
        row = answer.rows.get(job)
        problem = row and owner_problem(record, row[1], row[2])
        if problem:
            return dataclasses.replace(record, state=fr.RecordState.FAILED, finished_at=now, note=f"job {job}: {problem}")
    moved = fr.advance(record, answer, now, settings.policy)
    if moved.state in {fr.RecordState.WAITING, fr.RecordState.FETCHING}:
        moved = dataclasses.replace(moved, note="")
    if moved.state == fr.RecordState.FETCHING:
        moved = fetch_paths(moved, settings, ssh, ctx, folder)
    return run_then(moved, folder, ctx, again=then_again)


def _notice(old: fr.Record, new: fr.Record, ctx: Context) -> list[str]:
    """One line, and one notice, per change worth telling: a new state, or a failed `--fetch-then`."""
    job, lines = new.jobs[0], []
    if new.state != old.state or (new.note != old.note and new.note):
        lines.append(f"[fetch] {new.site} job {','.join(new.jobs)}: {new.state}{f' ({new.note})' if new.note else ''}")
    told = []
    if new.state != old.state and new.state in NOTICES:
        told.append((NOTICES[new.state], f"fetch-back of {new.site} job {job}: {new.state}; {new.note}".rstrip("; ")))
    if new.then_result and new.then_result != old.then_result and new.then_result["exit"] != 0:
        told.append((agent.NotifyEvent.FETCH_THEN_FAILED,
                     (f"after fetching {new.site} job {job}, `{new.then}` exited {new.then_result['exit']}; "
                      f"log {new.then_result['log']}")))
    for event, message in told if ctx.notify_command else []:
        env = agent.notice_env(event, job, ctx.user, f"fetch-back from {new.site}", new.project, message, site=new.site)
        try:
            ctx.notify(list(ctx.notify_command), env)
        except (OSError, subprocess.SubprocessError) as exc:
            lines.append(f"[fetch] the notify command failed: {exc}")
    return lines


def control_path(env: Mapping[str, str]) -> Path:
    """The shared connection's socket: short (a socket path is limited to about 100 bytes) and private to the user."""
    return Path(env.get("XDG_RUNTIME_DIR") or tempfile.gettempdir()) / "gpujobs-fetch-%C"


def run_site(name: str, settings: Settings, ctx: Context, *, force: frozenset[str] = frozenset(),
             then_again: bool = False) -> list[str]:
    """Every due record of one remote, over one ssh connection, at most one run at a time (design 2, 17.3)."""
    folder = fr.store_dir(name, ctx.env)
    if not folder.is_dir():
        return []
    with (folder / ".lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return [f"[fetch] {name}: another run is fetching from it; this one leaves it"]
        now = ctx.clock()
        records = [dataclasses.replace(r, force=True, next_try=0.0) if force & set(r.jobs) else r for r in fr.load_all(folder)]
        due = [r for r in records if (not fr.is_final(r) or r.force) and r.next_try <= now]
        if not due:
            return []
        ssh, lines = ssh_command(control_path(ctx.env)), []
        try:
            answer, error = read_jobs(settings, ssh, sorted({job for r in due for job in r.jobs}), ctx)
            for old in due:
                new = step(old, answer, error, settings, ssh, ctx, folder, then_again=then_again)
                fr.save(new, folder)
                lines += _notice(old, new, ctx)
        finally:
            agent.call(ctx.runner, [*ssh, "-O", "exit", settings.host])
        return lines


def add_record(name: str, settings: Settings, jobs: list[str], pairs: list[tuple[str, str]], options: Options,
               ctx: Context, *, repo: Path | None = None, workdir_root: str = "") -> fr.Record:
    """Write a record for jobs already queued there (`add`, and `submit.py --site --fetch-back`). Refuses what would
    leave the declared folders or fetch another person's job (design 8)."""
    for job in jobs:
        if not gpu_jobs.JOB_ID.fullmatch(job):
            raise FetchRefused(f"--job {job}: not a slurm job id")
    if not pairs:
        raise FetchRefused("nothing to fetch: give --from and --to (or --fetch-back)")
    for there, here in pairs:
        if ".." in Path(there).parts:
            raise FetchRefused(f"--from {there}: no '..' in it")
        if not Path(here).is_absolute():
            raise FetchRefused(f"--to {here}: give an absolute path")
    base = (repo or Path(os.path.commonpath([here for _, here in pairs]))).resolve()
    for _, here in pairs:
        if not inside(Path(here), base):
            raise FetchRefused(f"--to {here} is outside {base}")
    folder = fr.store_dir(name, ctx.env)
    taken = [job for job in jobs if fr.find(folder, job)]
    if taken:
        raise FetchRefused(f"job {', '.join(taken)} already has a fetch record (fetch_back.py status --site {name})")
    ssh = ssh_command()
    who = agent.call(ctx.runner, [*ssh, settings.host, "id -un && pwd"], timeout=CALL_TIMEOUT_S)
    if who.returncode != 0 or len(who.stdout.split()) < len(("user", "home")):
        raise FetchRefused(f"ssh to {settings.host} failed ({_tail(who.stderr)}); the record is checked there first")
    login, home = who.stdout.split()[:2]
    root = workdir_root if not workdir_root or workdir_root.startswith("/") else f"{home}/{workdir_root}"
    now = ctx.clock()
    filters = tuple(("-", pattern) for pattern in settings.exclude) + options.filters
    record = fr.Record(
        site=name, jobs=tuple(jobs), created=now,
        deadline=now + (options.wait_days if options.wait_days is not None else settings.wait_days) * DAY_S,
        fetch_on=options.fetch_on, existing=options.existing,
        keep_free_gb=options.keep_free_gb if options.keep_free_gb is not None else settings.keep_free_gb,
        repo=str(base), paths=tuple(FetchPath(there, str(Path(here)), filters, options.max_size) for there, here in pairs),
        then=options.then, project=options.project, login=login, workdir_root=root,
    )  # fmt: skip
    answer, error = read_jobs(settings, ssh, list(jobs), ctx)
    if not answer.reached:
        raise FetchRefused(f"sacct on {settings.host} did not answer ({_tail(error)})")
    for job, (_, user, workdir) in answer.rows.items():
        problem = owner_problem(record, user, workdir)
        if problem:
            raise FetchRefused(f"job {job}: {problem}")
    fr.save(record, folder)
    return record


# ---------------------------------------------------------------------------------------------------------------------
# What the user sees.


def status_lines(name: str, env: Mapping[str, str]) -> list[str]:
    out = []
    for r in fr.load_all(fr.store_dir(name, env)):
        moved = sum(p.bytes for p in r.paths)
        after = (f"; then exit {r.then_result['exit']} (log {r.then_result['log']})" if r.then_result else "")
        retry = (f"; next try {time.strftime('%H:%M', time.localtime(r.next_try))}"
                 if r.next_try and not fr.is_final(r) else "")
        out.append(gpu_jobs.printable(f"[fetch] {name} job {','.join(r.jobs)}: {r.state}, {gpu_jobs.size_text(moved)} "
                                      f"pulled{f'; {r.note}' if r.note else ''}{retry}{after}"))
    return out


def summary_line(name: str, env: Mapping[str, str] = os.environ) -> str | None:
    """`status.py --site NAME`'s last line: this user's records for that remote, by state."""
    records = fr.load_all(fr.store_dir(name, env))
    if not records:
        return None
    counts: dict[str, int] = {}
    for r in records:
        counts[r.state] = counts.get(r.state, 0) + 1
    return "[fetch] fetch-back: " + ", ".join(f"{n} {state}" for state, n in sorted(counts.items()))


def prune(name: str, days: float, env: Mapping[str, str], now: float) -> int:
    """Delete final records (and their logs) that ended more than `days` ago; returns how many."""
    folder, gone = fr.store_dir(name, env), 0
    for r in fr.load_all(folder):
        if fr.is_final(r) and (r.finished_at or r.created) < now - days * DAY_S:
            for path in folder.glob(f"{r.jobs[0]}.*"):
                path.unlink(missing_ok=True)
            gone += 1
    return gone


def add_fetch_arguments(parser: argparse.ArgumentParser) -> None:
    """The `--fetch-*` options, shared by `fetch_back.py add` and `submit.py --site`."""
    parser.add_argument("--fetch-include", dest="fetch_filters", action=_Filter, sign="+", default=[], metavar="PATTERN",
                        help="rsync pattern to fetch (repeatable; with --fetch-exclude, the first match wins, in the order given)")
    parser.add_argument("--fetch-exclude", dest="fetch_filters", action=_Filter, sign="-", default=[], metavar="PATTERN",
                        help="rsync pattern to leave there (repeatable), e.g. 'snapshot-*.ckpt'")
    parser.add_argument("--fetch-on", type=fr.FetchOn, choices=list(fr.FetchOn), default=fr.FetchOn.COMPLETED,
                        help="completed (default): only when every job ended COMPLETED; finished: whatever they ended as")
    parser.add_argument("--fetch-existing", type=Existing, choices=list(Existing), default=Existing.UPDATE,
                        help="a file already here: update (default) keeps it if newer, keep never touches it, replace "
                             "takes the remote's when they differ")
    parser.add_argument("--fetch-max-size", default=None, metavar="SIZE", help="leave files larger than this (rsync --max-size)")
    parser.add_argument("--fetch-keep-free", type=float, default=None, metavar="GB",
                        help="wait rather than leave less than this free here (default: the site's fetch_keep_free_gb)")
    parser.add_argument("--fetch-wait", type=float, default=None, metavar="DAYS",
                        help="give up if the jobs have not ended by then (default: the site's fetch_wait_days)")
    parser.add_argument("--fetch-then", default=None, metavar="CMD",
                        help="run once from the repo after a successful fetch (no shell); its exit code is kept")


class _Filter(argparse.Action):
    def __init__(self, *args, sign: str, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.sign = sign

    def __call__(self, parser, namespace, value, option_string=None) -> None:
        setattr(namespace, self.dest, [*getattr(namespace, self.dest, []), (self.sign, value)])


def options_from(arguments: argparse.Namespace, project: str = "") -> Options:
    return Options(tuple(arguments.fetch_filters), arguments.fetch_on, arguments.fetch_existing, arguments.fetch_max_size,
                   arguments.fetch_keep_free, arguments.fetch_wait, arguments.fetch_then, project)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="action", required=True)
    add = sub.add_parser("add", help="fetch the folders of jobs already queued there when they end")
    add.add_argument("--site", required=True, metavar="NAME", help="the remote ([remote.NAME] in the site file)")
    add.add_argument("--job", required=True, metavar="ID[,ID]", help="the jobs whose end is waited for (a chain: its last job)")
    add.add_argument("--from", dest="sources", action="append", default=[], metavar="DIR",
                     help="a folder there: absolute, or relative to the login home (repeatable, paired with --to)")
    add.add_argument("--to", dest="targets", action="append", default=[], metavar="DIR", help="where it lands here (absolute)")
    add.add_argument("--repo", type=Path, default=None, metavar="DIR",
                     help="the folder every --to must stay inside (default: their common folder)")
    add.add_argument("--project", default="", help="the project named in notices")
    add_fetch_arguments(add)
    run = sub.add_parser("run", help="process every due record once (what the timer runs)")
    run.add_argument("--site", default=None, metavar="NAME")
    run.add_argument("--force", action="append", default=[], metavar="ID", help="fetch a skipped job's folders anyway")
    run.add_argument("--then-again", action="store_true", help="with --force: run its --fetch-then command again")
    status = sub.add_parser("status", help="this user's records")
    status.add_argument("--site", default=None, metavar="NAME")
    status.add_argument("--prune", type=float, default=None, metavar="DAYS", help="delete final records older than this")
    wait = sub.add_parser("wait", help="run until this job's record is final; exit 0 only if its folders arrived")
    wait.add_argument("job", metavar="ID")
    wait.add_argument("--site", default=None, metavar="NAME")
    sub.add_parser("enable", help=f"start this user's timer ({TIMER})")
    return p


def _names(site: site_config.Site, given: str | None) -> list[str]:
    if given:
        remote.pick(site, given)
        return [given]
    return sorted(site.remotes)


def main(argv: list[str] | None = None, ctx: Context | None = None, sleep: Callable[[float], None] = time.sleep) -> int:
    arguments = parser().parse_args(argv)
    if arguments.action == "enable":
        command = ["systemctl", "--user", "enable", "--now", TIMER]
        done = subprocess.run(command, capture_output=True, text=True, check=False)
        print(f"[fetch] {TIMER} enabled" if done.returncode == 0 else
              f"[fetch] could not enable it ({_tail(done.stderr)}); run `{shlex.join(command)}`, or `fetch_back.py run` from cron")
        return done.returncode
    site = gpu_jobs.load_site()
    ctx = ctx or Context(notify_command=site.notify_command)
    try:
        names = _names(site, arguments.site)
        if arguments.action == "add":
            if len(arguments.sources) != len(arguments.targets):
                raise FetchRefused("give one --to for each --from")
            name = arguments.site
            record = add_record(name, Settings.from_site(site, name), arguments.job.split(","),
                                list(zip(arguments.sources, arguments.targets, strict=True)),
                                options_from(arguments, arguments.project), ctx, repo=arguments.repo)
            print(f"[fetch] recorded {name} job {','.join(record.jobs)}: {len(record.paths)} folder(s) when it ends")
            return 0
        if arguments.action == "status":
            for name in names:
                if arguments.prune is not None:
                    print(f"[fetch] {name}: pruned {prune(name, arguments.prune, ctx.env, ctx.clock())} record(s)")
                print("\n".join(status_lines(name, ctx.env)) or f"[fetch] {name}: no records")
            return 0
        if arguments.action == "run":
            for name in names:
                lines = run_site(name, Settings.from_site(site, name), ctx, force=frozenset(arguments.force),
                                 then_again=arguments.then_again)
                print("\n".join(lines)) if lines else None
            return 0
        owner = next((n for n in names if fr.find(fr.store_dir(n, ctx.env), arguments.job)), None)
        if owner is None:
            raise FetchRefused(f"no fetch record lists job {arguments.job} (fetch_back.py status)")
        while True:  # wait
            found = fr.find(fr.store_dir(owner, ctx.env), arguments.job)
            if not fr.is_final(found):
                for line in run_site(owner, Settings.from_site(site, owner), ctx):
                    print(line)
                found = fr.find(fr.store_dir(owner, ctx.env), arguments.job)
            if fr.is_final(found):
                print("\n".join(status_lines(owner, ctx.env)))
                return 0 if found.state == fr.RecordState.DONE else 1
            sleep(site.tuning.fetch_every_s)
    except (FetchRefused, remote.RemoteRefused) as error:
        print(f"[fetch] REFUSED: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
