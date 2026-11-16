"""Fetch-back records: what to pull home when a remote job ends, and the rule that moves a record on (D52).

A record is private to its submitter (mode 0600 under the user's own state folder, never under the tools' shared
state folder: D24, D52). `advance` is pure: one `sacct` answer in, the record's next state out; `fetch_back.py` does the
I/O. Design: section 2 (record), 3 (job end), 4 (retries), 6 (failed jobs).
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from transfer import EndState, end_state

#: The on-disk format; a file of another schema is skipped, not guessed at.
SCHEMA = 1
#: Below the user's state folder; one subfolder per remote site.
STORE = Path("gpujobs") / "fetch"
PRIVATE_FILE = 0o600
PRIVATE_DIR = 0o700


class RecordState(StrEnum):
    WAITING = "waiting"  # the jobs have not all ended, or the remote did not answer yet
    FETCHING = "fetching"  # ended as the record wants: its folders are being pulled
    BLOCKED = "blocked"  # the remote refused our login (an expired certificate): retried after the user renews it
    DONE = "done"  # every folder pulled
    SKIPPED = "skipped"  # ended in a state the record does not fetch (`--fetch-on completed` and a job FAILED)
    FAILED = "failed"  # a folder could not be pulled for a reason a retry will not fix
    EXPIRED = "expired"  # still not ended at the deadline
    LOST = "lost"  # the remote's accounting does not know a job id


FINAL = frozenset({RecordState.DONE, RecordState.SKIPPED, RecordState.FAILED, RecordState.EXPIRED, RecordState.LOST})


class PathState(StrEnum):
    PENDING = "pending"
    DONE = "done"
    BLOCKED = "blocked"  # not enough free space here: tried again next time
    FAILED = "failed"


class FetchOn(StrEnum):
    COMPLETED = "completed"  # only when every job ended COMPLETED (a folder that arrives still means success)
    FINISHED = "finished"  # whatever state they ended in


class Existing(StrEnum):
    UPDATE = "update"  # rsync -u: a file newer here is kept
    KEEP = "keep"  # --ignore-existing: nothing already here is touched
    REPLACE = "replace"  # rsync's own quick check


@dataclass(frozen=True)
class FetchPath:
    remote: str  # absolute, or relative to the remote login home
    local: str  # absolute
    filters: tuple[tuple[str, str], ...] = ()  # ("+" | "-", rsync pattern), in rsync's order: the first match wins
    max_size: str | None = None
    state: PathState = PathState.PENDING
    bytes: int = 0
    attempts: int = 0
    last_error: str = ""


@dataclass(frozen=True)
class Record:
    site: str
    jobs: tuple[str, ...]
    created: float
    deadline: float
    fetch_on: FetchOn
    existing: Existing
    keep_free_gb: float
    repo: str  # every local folder must stay inside this one (design 8.5)
    paths: tuple[FetchPath, ...]
    then: str | None = None  # run once from `repo` after a successful fetch (design 17.2)
    then_result: dict | None = None  # {"exit": int, "at": float, "log": str}
    owner: str = ""  # sacct User and WorkDir from the first answer (design 8)
    workdir: str = ""
    job_states: dict[str, str] = field(default_factory=dict)
    state: RecordState = RecordState.WAITING
    next_try: float = 0.0
    failures: int = 0
    empty_since: float | None = None
    finished_at: float | None = None
    force: bool = False  # `run --force`: fetch a skipped record once


@dataclass(frozen=True)
class Answer:
    """One `sacct` reading for a record's jobs."""

    reached: bool  # False: ssh failed or timed out, so there is no answer (not the same as an unknown id)
    rows: Mapping[str, tuple[str, str, str]] = field(default_factory=dict)  # id -> (State, User, WorkDir)


@dataclass(frozen=True)
class Policy:
    """The remote's settings that drive `advance` (site file `[remote.NAME]`, passed in: this module reads no config)."""

    retry_s: tuple[int, ...]  # the pause after the 1st, 2nd, ... consecutive failure; the last repeats
    lost_after_s: int  # an id the accounting has not known for this long is lost (a fresh submission can lag)


def store_dir(site: str, env: Mapping[str, str] = os.environ) -> Path:
    """`${XDG_STATE_HOME:-~/.local/state}/gpujobs/fetch/<site>`."""
    base = Path(env["XDG_STATE_HOME"]) if env.get("XDG_STATE_HOME") else Path(env.get("HOME") or Path.home()) / ".local" / "state"
    return base / STORE / site


def _to_json(record: Record) -> dict:
    return {"schema": SCHEMA, **dataclasses.asdict(record)}


def _from_json(raw: dict) -> Record:
    if raw.get("schema") != SCHEMA:
        msg = f"schema {raw.get('schema')!r}, expected {SCHEMA}"
        raise ValueError(msg)
    fields = {k: v for k, v in raw.items() if k != "schema"}
    paths = tuple(
        FetchPath(**{**p, "filters": tuple((sign, pattern) for sign, pattern in p["filters"]), "state": PathState(p["state"])})
        for p in fields.pop("paths")
    )
    return Record(
        **{
            **fields,
            "jobs": tuple(fields["jobs"]),
            "fetch_on": FetchOn(fields["fetch_on"]),
            "existing": Existing(fields["existing"]),
            "state": RecordState(fields["state"]),
            "paths": paths,
        }
    )


def path_of(record: Record, folder: Path) -> Path:
    return folder / f"{record.jobs[0]}.json"


def save(record: Record, folder: Path) -> Path:
    """Write the record atomically (a temporary file, then a rename), readable by its owner only."""
    folder.mkdir(parents=True, exist_ok=True, mode=PRIVATE_DIR)
    target = path_of(record, folder)
    fd, temp = tempfile.mkstemp(dir=folder, prefix=f".{target.name}.")  # mkstemp creates it 0600
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            json.dump(_to_json(record), out, indent=1, sort_keys=True)
        os.chmod(temp, PRIVATE_FILE)
        os.replace(temp, target)
    except BaseException:
        Path(temp).unlink(missing_ok=True)
        raise
    return target


def load_all(folder: Path) -> list[Record]:
    """Every record in the folder, oldest job first; a file that cannot be read is named on stderr and skipped."""
    out = []
    for path in sorted(folder.glob("*.json")) if folder.is_dir() else []:
        try:
            out.append(_from_json(json.loads(path.read_text(encoding="utf-8"))))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(f"[fetch] skipping {path}: {exc}", file=sys.stderr)
    return out


def find(folder: Path, job: str) -> Record | None:
    """The record that lists this job id."""
    return next((record for record in load_all(folder) if job in record.jobs), None)


def is_final(record: Record) -> bool:
    return record.state in FINAL


def retry_after(record: Record, now: float, policy: Policy) -> Record:
    """One more consecutive failure: wait the next step of `retry_s` (the last step repeats)."""
    failures = record.failures + 1
    pause = policy.retry_s[min(failures, len(policy.retry_s)) - 1]
    return dataclasses.replace(record, failures=failures, next_try=now + pause)


def _waiting(record: Record, now: float) -> Record:
    if now > record.deadline:
        return dataclasses.replace(record, state=RecordState.EXPIRED, finished_at=now)
    return dataclasses.replace(record, state=RecordState.WAITING)


def advance(record: Record, answer: Answer, now: float, policy: Policy) -> Record:
    """The record after one `sacct` answer (design 3, 4, 6). A final record stays as it is, except a skipped one that
    `run --force` re-opened."""
    if is_final(record) and not (record.force and record.state == RecordState.SKIPPED):
        return record
    if not answer.reached:
        moved = retry_after(record, now, policy)
        return _waiting(moved, now) if moved.state == RecordState.WAITING else moved
    rows = {job: row for job, row in answer.rows.items() if job in record.jobs}
    first = next(iter(rows.values()), None)
    record = dataclasses.replace(
        record,
        failures=0,
        next_try=0.0,
        job_states={**record.job_states, **{job: row[0] for job, row in rows.items()}},
        owner=record.owner or (first[1] if first else ""),
        workdir=record.workdir or (first[2] if first else ""),
    )
    if any(job not in rows for job in record.jobs):
        since = record.empty_since if record.empty_since is not None else now
        record = dataclasses.replace(record, empty_since=since)
        if now - since >= policy.lost_after_s:
            return dataclasses.replace(record, state=RecordState.LOST, finished_at=now)
        return _waiting(record, now)
    record = dataclasses.replace(record, empty_since=None)
    ends = [end_state(rows[job][0]) for job in record.jobs]
    if any(end is None for end in ends):
        return _waiting(record, now)
    if record.force or record.fetch_on == FetchOn.FINISHED or all(end == EndState.COMPLETED for end in ends):
        return dataclasses.replace(record, state=RecordState.FETCHING, force=False, finished_at=None)
    return dataclasses.replace(record, state=RecordState.SKIPPED, finished_at=now)
