#!/usr/bin/env python3
"""Weekly dependency upgrade of every uv project on this machine (owner's ruling 2026-10-29, automatic).

Policy: ~/.claude/software-upgrade-policy.md ("RULING 2026-10-29", "A project upgrade").

    weekly.py run [--project NAME] [--dry-run]
    weekly.py apply-held NAME          # apply a held upgrade's kept lock, its check skipped (owner 2026-10-29)

Per project in `projects.toml` (see README.md): skip when a job of the project is queued or running, or when its
uv.lock does not describe its venv; otherwise `uv lock --upgrade` in a scratch copy; build a side venv from the new lock;
check it (numbers: the project's probe under the old and the new venv in ONE slurm job, applied only when every value is
identical; tests: the project's fast test command in the side venv); then APPLY (commit uv.lock alone, `uv sync --frozen`,
hostguard, restart the project's services) or HOLD (announced by `session_notice.py`; applied at the study boundary
with `apply-held`, or cleared by the next run once a failing-tests hold is fixed in the project's code). Records:
~/.local/state/upgrades/<name>.json and weekly.log (a dry run writes dry-run/<name>.json instead).

A project that checks by `tests` keeps its PyTorch stack at the locked version: tests do not measure numbers or speed, so
PyTorch moves only through a `numbers` check or `torch_upgrade.py`.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path

import upgrade_common as common

REGISTRY = Path(__file__).resolve().parent / "projects.toml"
STATE = Path.home() / ".local/state/upgrades"
CACHE = Path.home() / "cache/upgrades"
COMMIT_SUBJECT = "chore(deps): Weekly dependency upgrade"
# The PyTorch stack: torch pins its own CUDA libraries and Triton.
STACK = re.compile(r"^(torch|torchvision|torchaudio|triton|nvidia-|cuda-)")


class Check(StrEnum):
    NUMBERS = "numbers"
    TESTS = "tests"


class Status(StrEnum):
    SKIPPED = "skipped"
    UP_TO_DATE = "up_to_date"
    APPLIED = "applied"
    HELD = "held"
    WOULD_APPLY = "would_apply"  # a dry run whose check passed
    ERROR = "error"


class LockState(StrEnum):
    CLEAN = "clean"  # tracked, no uncommitted change: the upgrade is committed
    DIRTY = "dirty"  # tracked with uncommitted changes, or not tracked and not ignored: never touched
    IGNORED = "ignored"  # git-ignored (a library's lock): applied without a commit, the old one kept in the scratch


@dataclass(frozen=True)
class Project:
    name: str
    path: Path
    check: Check
    command: str
    gpu: bool = False
    services: tuple[str, ...] = ()
    sync: tuple[
        str, ...
    ] = ()  # the uv flags that rebuild this project's venv: --extra X, --all-extras, ...
    mem: str = "8G"
    cpus: int | None = (
        None  # slurm cores for the check; unset = the cluster default (4 without a GPU)
    )


@dataclass(frozen=True)
class Change:
    name: str
    old: str | None
    new: str | None

    def __str__(self) -> str:
        return f"{self.name} {self.old or '-'} -> {self.new or '-'}"


@dataclass
class CheckResult:
    kind: Check
    passed: bool
    seconds: float
    detail: str
    numbers: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Outcome:
    status: Status
    reason: str


def load_registry(path: Path = REGISTRY) -> list[Project]:
    entries = tomllib.loads(path.read_text()).get("project", [])
    known = {
        "name",
        "path",
        "check",
        "command",
        "gpu",
        "services",
        "sync",
        "mem",
        "cpus",
    }
    projects: list[Project] = []
    for entry in entries:
        extra = set(entry) - known
        if extra:
            msg = f"{path}: project {entry.get('name')!r} has unknown keys {sorted(extra)}"
            raise ValueError(msg)
        project = Project(
            name=entry["name"],
            path=Path(entry["path"]),
            check=Check(entry["check"]),
            command=entry["command"],
            gpu=bool(entry.get("gpu", False)),
            services=tuple(entry.get("services", ())),
            sync=tuple(entry.get("sync", ())),
            mem=entry.get("mem", "8G"),
            cpus=entry.get("cpus"),
        )
        if any(p.name == project.name for p in projects):
            msg = f"{path}: duplicate project {project.name!r}"
            raise ValueError(msg)
        projects.append(project)
    return projects


# ---- projects found without a registry entry (owner 2026-10-29: "I don't have to tell the projects") ----
# A uv project under these roots that projects.toml does not name joins by itself, with the cheapest honest check:
# its own tests (first failure stops them, five-minute cap) or, with no tests folder, an import of its package.
# projects.toml stays the place to give one a faster test selection, a probe, services or a GPU.
DISCOVER_ROOTS = (Path.home() / "dev", Path.home() / "dev/turing")
DISCOVERED_TEST_CAP_S = 300
# Tried in order; the first whose `uv export` matches the venv is how that venv was built.
SYNC_CANDIDATES: tuple[tuple[str, ...], ...] = (
    (),
    ("--all-extras",),
    ("--extra", "dev"),
)


def is_candidate(path: Path) -> bool:
    """Its own git repository (a worktree's `.git` is a file), a pyproject, a uv.lock and a venv."""
    return (
        not path.name.startswith(("_", "."))
        and (path / ".git").is_dir()
        and (path / "pyproject.toml").is_file()
        and (path / "uv.lock").is_file()
        and (path / ".venv/bin/python").exists()
    )


def pick_sync(path: Path) -> tuple[str, ...]:
    for flags in SYNC_CANDIDATES:
        try:
            if not describes_venv(
                Project(path.name, path, Check.TESTS, "", sync=flags)
            ):
                return flags
        except subprocess.CalledProcessError:
            continue
    return ()  # none matches: the weekly run then skips it, saying the lock does not describe the venv


def default_project(path: Path) -> Project:
    if (path / "tests").is_dir():
        command = f"timeout -k 5 {DISCOVERED_TEST_CAP_S} python -m pytest -q -x -p no:cacheprovider tests"
    else:
        name = (
            tomllib.loads((path / "pyproject.toml").read_text())
            .get("project", {})
            .get("name", path.name)
        )
        command = f'python -c "import {normalise(name).replace("-", "_")}"'
    return Project(path.name, path, Check.TESTS, command, sync=pick_sync(path))


def discover(
    registered: list[Project], roots: tuple[Path, ...] = DISCOVER_ROOTS
) -> list[Project]:
    known_paths = {p.path.resolve() for p in registered}
    known_names = {p.name for p in registered}
    found: list[Project] = []
    for root in roots:
        for path in sorted(root.iterdir()) if root.is_dir() else []:
            if (
                path.resolve() in known_paths
                or path.name in known_names
                or not is_candidate(path)
            ):
                continue
            found.append(default_project(path))
            known_paths.add(path.resolve())
            known_names.add(path.name)
    return found


# ---- does the lock describe the venv? ------------------------------------------------------------


def normalise(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_export(text: str) -> dict[str, set[str]]:
    """`uv export --no-hashes` -> name -> {versions or '@ url'}; a name may be locked at several versions."""
    locked: dict[str, set[str]] = {}
    for raw in text.splitlines():
        line = raw.split(" ;")[0].split(" #")[0].strip()
        if not line or line.startswith(("#", "-")):
            continue
        match = re.match(r"([A-Za-z0-9._-]+)(?:\[[^\]]*\])?\s*(==|@)\s*(\S+)", line)
        if match:
            name, op, value = match.groups()
            locked.setdefault(normalise(name), set()).add(
                value if op == "==" else f"@ {value.replace('.git@', '@')}"
            )
    return locked


def parse_freeze(text: str) -> dict[str, str | None]:
    """`uv pip freeze` -> name -> version, '@ url' for a direct URL, None for an editable install."""
    installed: dict[str, str | None] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-e"):
            match = re.search(r"#egg=([A-Za-z0-9._-]+)", line)
            if match:
                installed[normalise(match.group(1))] = None
            continue
        match = re.match(r"([A-Za-z0-9._-]+)(?:\[[^\]]*\])?\s*(==|@)\s*(\S+)", line)
        if match:
            name, op, value = match.groups()
            installed[normalise(name)] = (
                value if op == "==" else f"@ {value.replace('.git@', '@')}"
            )
    return installed


def lock_mismatches(
    locked: dict[str, set[str]], installed: dict[str, str | None], own: str
) -> list[str]:
    """Where the venv differs from the lock. Lock-only packages (extras and platforms not installed) are not a
    difference; editable installs and the project itself are not compared; local versions (+cu130) compare whole."""
    problems: list[str] = []
    for name, version in sorted(installed.items()):
        if version is None or name == normalise(own):
            continue
        wanted = locked.get(name)
        if wanted is None:
            problems.append(f"{name} {version} is installed but not in the lock")
        elif version not in wanted:
            problems.append(f"{name}: venv {version}, lock {', '.join(sorted(wanted))}")
    return problems


def describes_venv(project: Project) -> list[str]:
    export = common.run(
        ["uv", "export", "--frozen", "--no-hashes", "--no-emit-project", *project.sync],
        cwd=project.path,
    )
    freeze = common.run(
        ["uv", "pip", "freeze", "--python", str(project.path / ".venv/bin/python")]
    )
    return lock_mismatches(parse_export(export), parse_freeze(freeze), project.name)


# ---- the lock files ------------------------------------------------------------------------------


def lock_versions(lock: Path) -> dict[str, str]:
    """name -> version for every registry package of a uv.lock (several versions joined by ' / ')."""
    seen: dict[str, set[str]] = {}
    for package in tomllib.loads(lock.read_text()).get("package", []):
        source = package.get("source", {})
        if "version" in package and ("registry" in source or "git" in source):
            seen.setdefault(package["name"], set()).add(package["version"])
    return {name: " / ".join(sorted(v)) for name, v in seen.items()}


def changed_packages(old: Path, new: Path) -> list[Change]:
    before, after = lock_versions(old), lock_versions(new)
    return [
        Change(name, before.get(name), after.get(name))
        for name in sorted(before.keys() | after.keys())
        if before.get(name) != after.get(name)
    ]


def upgrade_arguments(lock: Path, kind: Check) -> list[str]:
    """`--upgrade` for a numbers project; for a tests project every registry package but the PyTorch stack."""
    if kind is Check.NUMBERS:
        return ["--upgrade"]
    names = sorted(name for name in lock_versions(lock) if not STACK.match(name))
    return [arg for name in names for arg in ("--upgrade-package", name)]


# ---- the decision --------------------------------------------------------------------------------


def verdict(result: CheckResult) -> str | None:
    """None when the check passes; otherwise why not. Numbers pass only when every value is identical."""
    if result.kind is Check.NUMBERS:
        if not result.numbers.get("values_compared"):
            return "the probe returned no values to compare"
        if result.numbers["largest_relative_difference"] != 0:
            return (
                "the numbers moved: largest relative difference "
                f"{result.numbers['largest_relative_difference']:.2e}"
            )
        return None
    return None if result.passed else f"the tests failed ({result.detail})"


def decide(
    *,
    running_jobs: list[str],
    lock_problems: list[str],
    changes: list[Change] | None,
    lock: LockState,
    stack_moved: Change | None = None,
    result: CheckResult | None,
    dry_run: bool,
) -> Outcome:
    """The whole policy, in the order it is applied; every input is a fact already measured."""
    if running_jobs:
        return Outcome(
            Status.SKIPPED,
            f"a job of this project is queued or running: {running_jobs[0]}",
        )
    if lock_problems:
        return Outcome(
            Status.SKIPPED,
            f"uv.lock does not describe the venv ({len(lock_problems)} differences, e.g. {lock_problems[0]})",
        )
    if not changes:
        return Outcome(Status.UP_TO_DATE, "uv lock --upgrade changed nothing")
    if lock is LockState.DIRTY:
        return Outcome(
            Status.HELD, "uv.lock has uncommitted changes or is untracked; not touched"
        )
    if stack_moved is not None:
        return Outcome(
            Status.HELD,
            f"the PyTorch stack moved ({stack_moved}) and this project's tests do not measure it",
        )
    if result is None:
        return Outcome(Status.ERROR, "no check result")
    reason = verdict(result)
    if reason:
        return Outcome(Status.HELD, reason)
    how = "identical numbers" if result.kind is Check.NUMBERS else "tests pass"
    return Outcome(Status.WOULD_APPLY if dry_run else Status.APPLIED, how)


# ---- doing it ------------------------------------------------------------------------------------


def running_jobs(name: str) -> list[str]:
    names = subprocess.run(
        ["squeue", "-h", "-o", "%j"], check=True, text=True, capture_output=True
    ).stdout.split()
    return [job for job in names if job.startswith(f"{name}:")]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lock_state(project: Project) -> LockState:
    git = ["git", "-C", str(project.path)]
    if (
        subprocess.run([*git, "check-ignore", "-q", "uv.lock"], check=False).returncode
        == 0
    ):
        return LockState.IGNORED
    tracked = subprocess.run(
        [*git, "ls-files", "--error-unmatch", "uv.lock"],
        capture_output=True,
        check=False,
    )
    changed = common.run([*git, "status", "--porcelain", "--", "uv.lock"]).strip()
    return LockState.DIRTY if tracked.returncode != 0 or changed else LockState.CLEAN


def scratch_copy(project: Project, scratch: Path) -> Path:
    """pyproject.toml, uv.lock and the pyproject of every workspace member, in a directory of their own."""
    work = scratch / "project"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    pyproject = tomllib.loads((project.path / "pyproject.toml").read_text())
    uv = pyproject.get("tool", {}).get("uv", {})
    for source in uv.get("sources", {}).values():
        if isinstance(source, dict) and "path" in source:
            msg = "a path source needs its directory beside the copy; not supported"
            raise ValueError(msg)
    for member in uv.get("workspace", {}).get("members", []):
        for found in project.path.glob(member):
            if (found / "pyproject.toml").exists():
                target = work / found.relative_to(project.path)
                target.mkdir(parents=True, exist_ok=True)
                shutil.copy2(found / "pyproject.toml", target)
    for name in ("pyproject.toml", "uv.lock"):
        shutil.copy2(project.path / name, work / name)
    return work


def build_side_venv(project: Project, work: Path, venv: Path) -> None:
    """The new lock, installed exactly (uv sync --frozen: one resolve done by the lock), then the project itself."""
    old_python = project.path / ".venv/bin/python"
    common.run(
        [
            "uv",
            "sync",
            "--frozen",
            "--no-install-project",
            "--python",
            str(old_python),
            *project.sync,
        ],
        cwd=work,
        env={
            **os.environ,
            "UV_PROJECT_ENVIRONMENT": str(venv),
            "UV_COMPILE_BYTECODE": "1",
        },
    )
    common.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(venv / "bin/python"),
            "--no-deps",
            "-e",
            str(project.path),
        ]
    )
    if common.has_hostguard(project.path / ".venv"):
        common.install_hostguard(venv)


def check_numbers(project: Project, venv: Path, scratch: Path) -> CheckResult:
    out = scratch / "check"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    common.submit_and_wait(
        project.name,
        f"weekly upgrade check (numbers): {project.name}",
        "weekly-upgrade-check",
        common.probe_legs(project.path / ".venv", venv, project.command, out),
        project.path,
        out / "job.log",
        gpu=project.gpu,
        mem=project.mem,
        cpus=project.cpus,
    )
    probes = common.read_probes(out)
    if probes is None:
        return CheckResult(
            Check.NUMBERS,
            False,
            0.0,
            f"a probe leg wrote nothing; see {out / 'job.log'}",
        )
    old, new = probes
    try:
        diff = common.largest_relative_difference(old["values"], new["values"])
    except (KeyError, ValueError) as error:
        return CheckResult(
            Check.NUMBERS, False, 0.0, f"probe output not comparable: {error!r}"
        )
    numbers = {
        "values_compared": len(old["values"]),
        "largest_relative_difference": diff,
        "seconds_per_step_old": old.get("seconds_per_step"),
        "seconds_per_step_new": new.get("seconds_per_step"),
    }
    return CheckResult(
        Check.NUMBERS,
        diff == 0,
        float(new.get("seconds_per_step") or 0.0),
        "identical values" if diff == 0 else f"values differ by {diff:.2e}",
        numbers,
    )


def check_tests(project: Project, venv: Path, scratch: Path) -> CheckResult:
    out = scratch / "check"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    env = f"VIRTUAL_ENV={venv} PATH={venv}/bin:$PATH" + (
        "" if project.gpu else " CUDA_VISIBLE_DEVICES="
    )
    runner = f"{sys.executable} {Path(common.__file__)} timed {out} {project.path}"
    common.submit_and_wait(
        project.name,
        f"weekly upgrade check (tests): {project.name}",
        "weekly-upgrade-check",
        f"export {env}; {runner} -- {project.command}",
        project.path,
        out / "job.log",
        gpu=project.gpu,
        mem=project.mem,
        cpus=project.cpus,
    )
    result = out / "result.json"
    if not result.exists():
        return CheckResult(
            Check.TESTS, False, 0.0, f"the job wrote no result; see {out / 'job.log'}"
        )
    data = json.loads(result.read_text())
    lines = (
        (out / "output.log").read_text().strip().splitlines()
        if (out / "output.log").exists()
        else []
    )
    failed = [
        line.split(" - ")[0] for line in lines if line.startswith(("FAILED ", "ERROR "))
    ]
    return CheckResult(
        Check.TESTS,
        data["returncode"] == 0,
        data["seconds"],
        f"exit {data['returncode']} in {data['seconds']:.1f} s: {lines[-1] if lines else ''}"
        + (f"; {', '.join(failed[:3])}" if failed else ""),
    )


def apply(
    project: Project,
    new_lock: Path,
    changes: list[Change],
    before: str,
    state: LockState,
    note: str = "",
) -> str:
    """Put the new lock in, sync, commit uv.lock alone (unless git-ignored), restart services; returns the commit sha."""
    lock = project.path / "uv.lock"
    if sha(lock) != before:
        raise RuntimeError("uv.lock changed while the check ran; not applied")
    if running_jobs(project.name):
        raise RuntimeError(
            "a job of this project started while the check ran; not applied"
        )
    original = lock.read_bytes()
    shutil.copy2(new_lock, lock)
    try:
        common.run(["uv", "sync", "--frozen", *project.sync], cwd=project.path)
    except subprocess.CalledProcessError:
        lock.write_bytes(original)
        raise
    if common.has_hostguard(project.path / ".venv"):
        common.install_hostguard(project.path / ".venv")
    commit = ""
    if state is LockState.CLEAN:
        body = "\n".join(str(change) for change in changes) + (
            f"\n\n{note}" if note else ""
        )
        common.run(
            [
                "git",
                "-C",
                str(project.path),
                "commit",
                "-m",
                COMMIT_SUBJECT,
                "-m",
                body,
                "--",
                "uv.lock",
            ]
        )
        commit = common.run(
            ["git", "-C", str(project.path), "rev-parse", "--short", "HEAD"]
        ).strip()
    for unit in project.services:
        active = (
            subprocess.run(
                ["systemctl", "--user", "is-active", "--quiet", unit]
            ).returncode
            == 0
        )
        if active:
            common.run(["systemctl", "--user", "restart", unit])
    return commit


def process(
    project: Project, *, dry_run: bool
) -> tuple[Outcome, list[Change], CheckResult | None, str]:
    """One project end to end; returns what to record."""
    facts = {
        "running_jobs": [],
        "lock_problems": [],
        "changes": None,
        "lock": LockState.CLEAN,
        "result": None,
        "dry_run": dry_run,
    }
    jobs = running_jobs(project.name)
    problems = [] if jobs else describes_venv(project)
    if jobs or problems:
        return (
            decide(**{**facts, "running_jobs": jobs, "lock_problems": problems}),
            [],
            None,
            "",
        )
    scratch = CACHE / project.name
    scratch.mkdir(parents=True, exist_ok=True)
    work = scratch_copy(project, scratch)
    before = sha(project.path / "uv.lock")
    old_lock = scratch / "uv.lock.old"
    shutil.copy2(work / "uv.lock", old_lock)
    common.run(
        [
            "uv",
            "lock",
            *upgrade_arguments(old_lock, project.check),
            "--python",
            str(project.path / ".venv/bin/python"),
        ],
        cwd=work,
    )
    changes = changed_packages(old_lock, work / "uv.lock")
    shutil.copy2(work / "uv.lock", scratch / "uv.lock.new")
    facts |= {"changes": changes, "lock": lock_state(project)}
    stack = next((change for change in changes if STACK.match(change.name)), None)
    if (
        not changes
        or facts["lock"] is LockState.DIRTY
        or (stack and project.check is Check.TESTS)
    ):
        return (
            decide(
                **facts, stack_moved=stack if project.check is Check.TESTS else None
            ),
            changes,
            None,
            "",
        )
    venv = scratch / "venv"
    shutil.rmtree(venv, ignore_errors=True)
    try:
        build_side_venv(project, work, venv)
        check = check_numbers if project.check is Check.NUMBERS else check_tests
        result = check(project, venv, scratch)
    finally:
        shutil.rmtree(venv, ignore_errors=True)
    outcome = decide(**{**facts, "result": result})
    commit = ""
    if outcome.status is Status.APPLIED:
        commit = apply(project, scratch / "uv.lock.new", changes, before, facts["lock"])
    return outcome, changes, result, commit


def keeps_record(existing: dict | None, new: Status, now: dt.datetime) -> bool:
    """A quiet outcome (nothing to upgrade, or skipped for now) must not erase what `session_notice.py` still announces:
    a hold, or an upgrade applied within the last week."""
    if existing is None or new not in (Status.UP_TO_DATE, Status.SKIPPED):
        return False
    if existing.get("status") == Status.HELD:
        return (
            new is Status.SKIPPED
        )  # up to date means someone applied it by hand: the hold is over
    if existing.get("status") == Status.APPLIED:
        return now - dt.datetime.fromisoformat(existing["date"]) <= dt.timedelta(days=7)
    return False


def record(
    project: Project,
    outcome: Outcome,
    changes: list[Change],
    result: CheckResult | None,
    commit: str,
    *,
    dry_run: bool,
) -> None:
    now = dt.datetime.now(dt.UTC).replace(microsecond=0)
    folder = STATE / "dry-run" if dry_run else STATE
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{project.name}.json"
    existing = json.loads(target.read_text()) if target.exists() else None
    data = {
        "name": project.name,
        "date": now.isoformat(),
        "status": str(outcome.status),
        "reason": outcome.reason,
        "changed": [asdict(change) for change in changes],
        "check": None
        if result is None
        else {**asdict(result), "kind": str(result.kind)},
        "commit": commit,
        "dry_run": dry_run,
    }
    if not keeps_record(existing, outcome.status, now):
        target.write_text(json.dumps(data, indent=2) + "\n")
    line = (
        f"{now.isoformat()} {'dry-run ' if dry_run else ''}{project.name} {outcome.status} "
        f"({len(changes)} packages) {outcome.reason}"
        + (f" commit {commit}" if commit else "")
    )
    with (STATE / "weekly.log").open("a") as log:
        log.write(line + "\n")
    print(line, flush=True)


def held_refusal(
    name: str,
    record: dict | None,
    *,
    old_lock: Path,
    new_lock: Path,
    current_lock_sha: str,
    lock: LockState,
    lock_problems: list[str],
) -> str | None:
    """None when a held upgrade may be applied without its check; otherwise why not, naming what to do instead."""
    rerun = f"run `weekly.py run --project {name}`"
    if record is None or record.get("status") != Status.HELD:
        status = record.get("status") if record else "no record"
        return f"nothing is held (last weekly outcome: {status}); {rerun} to check for an upgrade"
    check = record.get("check") or {}
    if check.get("kind") == Check.TESTS and not check.get("passed"):
        return (
            f"held because its tests failed ({record['reason']}); fix the project's code against the new versions, "
            f"then {rerun}: it applies once the tests pass"
        )
    if lock is LockState.DIRTY:
        return "uv.lock has uncommitted changes or is untracked; commit or discard them first"
    if not (old_lock.exists() and new_lock.exists()):
        return (
            f"the kept locks are gone ({old_lock.parent}); {rerun} to make them again"
        )
    if sha(old_lock) != current_lock_sha:
        return f"uv.lock changed since the hold, so the kept upgrade is stale; {rerun}"
    if lock_problems:
        return (
            f"uv.lock does not describe the venv ({len(lock_problems)} differences, e.g. {lock_problems[0]}); "
            "bring them back in line first"
        )
    return None


def apply_held(name: str) -> int:
    """Apply the lock a hold kept, without its check: the owner's call at a study boundary (owner 2026-10-29)."""
    projects = {p.name: p for p in all_projects()}
    if name not in projects:
        sys.exit(f"unknown project: {name}")
    project = projects[name]
    scratch = CACHE / name
    target = STATE / f"{name}.json"
    held = json.loads(target.read_text()) if target.exists() else None
    STATE.mkdir(parents=True, exist_ok=True)
    with (STATE / "weekly.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            sys.exit("another weekly upgrade is running")
        state = lock_state(project)
        current = sha(project.path / "uv.lock")
        refusal = held_refusal(
            name,
            held,
            old_lock=scratch / "uv.lock.old",
            new_lock=scratch / "uv.lock.new",
            current_lock_sha=current,
            lock=state,
            lock_problems=describes_venv(project),
        )
        if refusal:
            print(f"[apply-held] {name}: not applied: {refusal}", flush=True)
            return 1
        changes = changed_packages(scratch / "uv.lock.old", scratch / "uv.lock.new")
        note = f"Applied by hand from a hold; the check was skipped. Held because: {held['reason']}"
        print(f"[apply-held] {name}: {note}", flush=True)
        commit = apply(project, scratch / "uv.lock.new", changes, current, state, note)
        record(
            project,
            Outcome(
                Status.APPLIED,
                f"by hand from the hold, check skipped (held: {held['reason']})",
            ),
            changes,
            None,
            commit,
            dry_run=False,
        )
    return 0


def all_projects() -> list[Project]:
    projects = load_registry()
    found = discover(projects)
    for project in found:
        print(
            f"[weekly] found {project.name} ({project.path}), not in projects.toml: check `{project.command}`"
        )
    return projects + found


def run_all(names: list[str], *, dry_run: bool) -> int:
    projects = all_projects()
    if names:
        unknown = set(names) - {p.name for p in projects}
        if unknown:
            sys.exit(f"unknown project(s): {sorted(unknown)}")
        projects = [p for p in projects if p.name in names]
    STATE.mkdir(parents=True, exist_ok=True)
    with (STATE / "weekly.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            sys.exit("another weekly upgrade is running")
        for project in projects:
            try:
                outcome, changes, result, commit = process(project, dry_run=dry_run)
            except subprocess.CalledProcessError as error:
                tail = (error.stderr or "").strip().splitlines()[-3:]
                outcome, changes, result, commit = (
                    Outcome(
                        Status.ERROR,
                        f"{' '.join(map(str, error.cmd))[:120]} failed: {' | '.join(tail)}",
                    ),
                    [],
                    None,
                    "",
                )
            except Exception as error:  # noqa: BLE001 - one project's failure must not stop the others
                outcome, changes, result, commit = (
                    Outcome(Status.ERROR, repr(error)),
                    [],
                    None,
                    "",
                )
            record(project, outcome, changes, result, commit, dry_run=dry_run)
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument(
        "--project",
        action="append",
        default=[],
        help="repeatable; default every project",
    )
    r.add_argument(
        "--dry-run",
        action="store_true",
        help="check, record under dry-run/, apply nothing",
    )
    h = sub.add_parser(
        "apply-held", help="apply a held upgrade's kept lock without its check"
    )
    h.add_argument("name")
    args = ap.parse_args()
    if args.cmd == "apply-held":
        sys.exit(apply_held(args.name))
    sys.exit(run_all(args.project, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
