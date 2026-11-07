"""The weekly upgrade's decisions: no network, no GPU, no slurm."""

from __future__ import annotations

import datetime as dt
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import session_notice
import upgrade_common
import weekly
from weekly import Change, Check, CheckResult, Status

CHANGES = [Change("numpy", "2.4.0", "2.4.1")]


def result(
    kind: Check, *, passed: bool = True, diff: float = 0.0, compared: int = 4
) -> CheckResult:
    return CheckResult(
        kind,
        passed,
        1.0,
        "d",
        {"values_compared": compared, "largest_relative_difference": diff}
        if kind is Check.NUMBERS
        else {},
    )


def decide(**given):
    facts = {
        "running_jobs": [],
        "lock_problems": [],
        "changes": CHANGES,
        "lock": weekly.LockState.CLEAN,
        "result": result(Check.TESTS),
        "dry_run": False,
    }
    return weekly.decide(**{**facts, **given}).status


def test_registry_parses_validates_and_the_example_and_the_local_one_load(tmp_path: Path) -> None:
    good = tmp_path / "p.toml"
    good.write_text(
        '[[project]]\nname="a"\npath="/x"\ncheck="tests"\ncommand="python -m pytest"\n'
        'services=["u.service"]\nsync=["--extra","dev"]\n'
    )
    (project,) = weekly.load_registry(good)
    assert (project.check, project.gpu, project.services, project.sync) == (
        Check.TESTS,
        False,
        ("u.service",),
        ("--extra", "dev"),
    )
    for bad in ('check="sometimes"', 'colour="red"'):  # an unknown kind; an unknown key
        good.write_text(
            f'[[project]]\nname="a"\npath="/x"\ncommand="c"\n{bad}\n'
            + ('check="tests"\n' if "colour" in bad else "")
        )
        with pytest.raises(ValueError):
            weekly.load_registry(good)
    good.write_text(
        '[[project]]\nname="a"\npath="/x"\ncheck="tests"\ncommand="c"\n' * 2
    )
    with pytest.raises(ValueError, match="duplicate"):
        weekly.load_registry(good)
    assert weekly.load_registry(weekly.REGISTRY.with_name("projects.example.toml"))
    if weekly.REGISTRY.exists():  # this machine's own list (untracked): every project in it must be upgradable
        real = weekly.load_registry()
        assert real and all(
            (p.path / "uv.lock").exists() and (p.path / ".venv/bin/python").exists()
            for p in real
        )


def test_the_lock_describes_the_venv_only_when_every_installed_package_matches() -> (
    None
):
    locked = weekly.parse_export(
        "torch==2.10.0+cu130 ; sys_platform == 'linux'\n"
        "uvicorn[standard]==0.30.0\n"
        "numpy==2.4.0 ; python_full_version < '3.13'\nnumpy==2.5.0 ; python_full_version >= '3.13'\n"
        "agentcall @ git+https://example.com/agentcall@abc\n"
        "colorama==0.4.6 ; sys_platform == 'win32'\n"
        "-e .\n"
    )
    freeze = (
        "torch==2.10.0+cu130\nUvicorn==0.30.0\nnumpy==2.5.0\nagentcall @ git+https://example.com/agentcall.git@abc\n"
        "-e file:///home/x/proj#egg=proj\n-e file:///home/x/other#egg=other\n-e file:///home/x/plain\n"
    )
    assert (
        weekly.lock_mismatches(locked, weekly.parse_freeze(freeze), "proj") == []
    )  # lock-only colorama is fine
    wrong = weekly.parse_freeze(
        "torch==2.10.0+cu126\nnumpy==2.4.1\nextra==1.0\nagentcall @ git+https://example.com/agentcall@def\n"
    )
    problems = weekly.lock_mismatches(locked, wrong, "proj")
    assert len(problems) == 4
    assert any("torch" in p and "2.10.0+cu126" in p for p in problems)


def test_the_decision_in_the_order_the_policy_applies_it() -> None:
    assert decide(running_jobs=["p:restart:x"]) is Status.SKIPPED
    assert decide(lock_problems=["torch differs"]) is Status.SKIPPED
    assert decide(changes=[], result=None) is Status.UP_TO_DATE
    assert decide(lock=weekly.LockState.DIRTY) is Status.HELD
    assert decide(lock=weekly.LockState.IGNORED) is Status.APPLIED
    assert decide(result=result(Check.TESTS, passed=False)) is Status.HELD
    assert decide(result=result(Check.TESTS)) is Status.APPLIED
    assert decide(result=result(Check.TESTS), dry_run=True) is Status.WOULD_APPLY
    assert decide(result=result(Check.NUMBERS)) is Status.APPLIED
    assert (
        decide(result=result(Check.NUMBERS, diff=1e-9)) is Status.HELD
    )  # any difference holds
    assert (
        decide(result=result(Check.NUMBERS, compared=0)) is Status.HELD
    )  # nothing compared is not identical
    assert decide(stack_moved=Change("torch", "2.10.0", "2.15.0")) is Status.HELD
    assert decide(result=None) is Status.ERROR


def test_a_running_job_is_matched_by_project_prefix_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(
            stdout="guard:restart:x\nguardian:resume:y\nlocalmodels:resume:z\n"
        ),
    )
    assert weekly.running_jobs("guard") == ["guard:restart:x"]
    assert weekly.running_jobs("service-c") == []


def test_lock_changes_and_the_frozen_pytorch_stack(tmp_path: Path) -> None:
    def lock(**versions: str) -> Path:
        path = tmp_path / f"{len(list(tmp_path.iterdir()))}.lock"
        path.write_text(
            "".join(
                f'[[package]]\nname = "{n}"\nversion = "{v}"\nsource = {{ registry = "https://pypi.org/simple" }}\n'
                for n, v in versions.items()
            )
            + '[[package]]\nname = "me"\nversion = "0.1"\nsource = { editable = "." }\n'
        )
        return path

    old, new = (
        lock(numpy="1", torch="2", nvidia_cublas="3"),
        lock(numpy="2", torch="2", extra="9"),
    )
    assert [str(c) for c in weekly.changed_packages(old, new)] == [
        "extra - -> 9",
        "numpy 1 -> 2",
        "nvidia_cublas 3 -> -",
    ]
    keep = weekly.upgrade_arguments(
        lock(numpy="1", torch="2", **{"nvidia-cublas": "3"}), Check.TESTS
    )
    assert keep == ["--upgrade-package", "numpy"]
    assert weekly.upgrade_arguments(old, Check.NUMBERS) == ["--upgrade"]


def test_a_path_source_cannot_be_copied_and_is_refused(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[tool.uv.sources]\nhelper = { path = "../helper" }\n'
    )
    (tmp_path / "uv.lock").write_text("")
    with pytest.raises(ValueError, match="path source"):
        weekly.scratch_copy(
            weekly.Project("p", tmp_path, Check.TESTS, "c"), tmp_path / "scratch"
        )


def test_the_state_of_the_lock_file_in_git(tmp_path: Path) -> None:
    def git(*args: str) -> None:
        subprocess.run(
            ["git", "-C", str(tmp_path), *args], check=True, capture_output=True
        )

    git("init", "-q")
    project = weekly.Project("p", tmp_path, Check.TESTS, "c")
    assert weekly.lock_state(project) is weekly.LockState.DIRTY  # no uv.lock tracked
    (tmp_path / "uv.lock").write_text("a")
    git("add", "uv.lock")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "x")
    assert weekly.lock_state(project) is weekly.LockState.CLEAN
    (tmp_path / "uv.lock").write_text("b")
    assert weekly.lock_state(project) is weekly.LockState.DIRTY
    git("rm", "-q", "--cached", "uv.lock")
    (tmp_path / ".gitignore").write_text("uv.lock\n")
    assert weekly.lock_state(project) is weekly.LockState.IGNORED


def test_the_timed_runner_records_exit_code_and_seconds(tmp_path: Path) -> None:
    code = upgrade_common.timed(
        tmp_path / "o", tmp_path, ["python3", "-c", "raise SystemExit(3)"]
    )
    data = json.loads((tmp_path / "o/result.json").read_text())
    assert code == data["returncode"] == 3 and data["seconds"] >= 0


def test_the_session_notice_names_held_and_recent_applied_only() -> None:
    now = dt.datetime(2026, 11, 1, tzinfo=dt.UTC)
    base = {
        "name": "p",
        "changed": [{"name": "numpy", "old": "1", "new": "2"}],
        "reason": "the numbers moved",
        "commit": "abc",
    }
    held = session_notice.notice(
        {**base, "status": "held", "date": "2026-10-01T00:00:00"}, now
    )
    assert (
        held and "HELD" in held and "numpy 1 -> 2" in held and "numbers moved" in held
    )
    assert "abc" in session_notice.notice(
        {**base, "status": "applied", "date": "2026-10-30T00:00:00+00:00"}, now
    )
    assert (
        session_notice.notice(
            {**base, "status": "applied", "date": "2026-10-01T00:00:00+00:00"}, now
        )
        is None
    )
    assert (
        session_notice.notice(
            {**base, "status": "up_to_date", "date": "2026-10-31T00:00:00"}, now
        )
        is None
    )
    drift = session_notice.notice(
        {
            **base,
            "status": "skipped",
            "date": "2026-10-31T00:00:00",
            "reason": "uv.lock does not describe the venv (6 differences, e.g. anyio)",
        },
        now,
    )
    assert drift and "SKIP" in drift and "uv sync --frozen" in drift
    assert (
        session_notice.notice(
            {
                **base,
                "status": "skipped",
                "date": "2026-10-31T00:00:00",
                "reason": "a job of this project is queued or running: x",
            },
            now,
        )
        is None
    )
    projects = [
        weekly.Project("a", Path("/x/a"), Check.TESTS, "c"),
        weekly.Project("b", Path("/x/a/b"), Check.TESTS, "c"),
    ]
    assert session_notice.project_for(Path("/x/a/b/src"), projects).name == "b"
    assert session_notice.project_for(Path("/x/ab"), projects) is None


def test_apply_commits_the_lock_alone_unless_it_is_ignored_and_refuses_a_moved_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def git(*args: str) -> str:
        return subprocess.run(
            [
                "git",
                "-C",
                str(tmp_path),
                "-c",
                "user.name=t",
                "-c",
                "user.email=t@t",
                *args,
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout

    git("init", "-q")
    (tmp_path / "uv.lock").write_text("old")
    (tmp_path / "other.txt").write_text("staged by someone else")
    git("add", "uv.lock", "other.txt")
    git("commit", "-qm", "base")
    (tmp_path / "other.txt").write_text("edited and staged")
    git("add", "other.txt")
    new = tmp_path / "new.lock"
    new.write_text("new")
    real_run = upgrade_common.run
    monkeypatch.setattr(
        upgrade_common,
        "run",
        lambda cmd, **k: "" if cmd[0] == "uv" else real_run(cmd, **k),
    )
    monkeypatch.setattr(weekly, "running_jobs", lambda name: [])
    project = weekly.Project("p", tmp_path, Check.TESTS, "c")
    before = weekly.sha(tmp_path / "uv.lock")

    with pytest.raises(RuntimeError, match="changed while"):
        weekly.apply(project, new, CHANGES, "not-the-digest", weekly.LockState.CLEAN)
    assert (tmp_path / "uv.lock").read_text() == "old"
    commit = weekly.apply(project, new, CHANGES, before, weekly.LockState.CLEAN)
    assert (tmp_path / "uv.lock").read_text() == "new" and commit == git(
        "rev-parse", "--short", "HEAD"
    ).strip()
    assert git("show", "--name-only", "--format=%s%n%b", "HEAD").split() == [
        "chore(deps):",
        "Weekly",
        "dependency",
        "upgrade",
        "numpy",
        "2.4.0",
        "->",
        "2.4.1",
        "uv.lock",
    ]
    assert git("diff", "--cached", "--name-only").split() == [
        "other.txt"
    ]  # someone else's staged work is untouched
    new.write_text("newer")
    assert (
        weekly.apply(
            project,
            new,
            CHANGES,
            weekly.sha(tmp_path / "uv.lock"),
            weekly.LockState.IGNORED,
        )
        == ""
    )
    assert (
        git("rev-list", "--count", "HEAD").strip() == "2"
    )  # ignored lock: written, not committed
    monkeypatch.setattr(weekly, "running_jobs", lambda name: ["p:restart:x"])
    with pytest.raises(RuntimeError, match="started while"):
        weekly.apply(
            project,
            new,
            CHANGES,
            weekly.sha(tmp_path / "uv.lock"),
            weekly.LockState.CLEAN,
        )


def test_a_quiet_week_does_not_erase_a_hold_or_a_recent_apply() -> None:
    now = dt.datetime(2026, 11, 1, tzinfo=dt.UTC)
    recent, old = "2026-10-30T00:00:00+00:00", "2026-10-01T00:00:00+00:00"
    keeps = weekly.keeps_record
    assert keeps({"status": "applied", "date": recent}, Status.UP_TO_DATE, now)
    assert not keeps({"status": "applied", "date": old}, Status.UP_TO_DATE, now)
    assert keeps(
        {"status": "held", "date": old}, Status.SKIPPED, now
    )  # skipped only because a job was running
    assert not keeps(
        {"status": "held", "date": old}, Status.UP_TO_DATE, now
    )  # applied by hand: the hold is over
    assert not keeps({"status": "applied", "date": recent}, Status.HELD, now)
    assert not keeps(None, Status.UP_TO_DATE, now)


def test_record_writes_the_state_file_and_a_quiet_run_leaves_an_apply_standing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(weekly, "STATE", tmp_path)
    project = weekly.Project("p", tmp_path, Check.TESTS, "c")
    weekly.record(
        project,
        weekly.Outcome(Status.APPLIED, "tests pass"),
        CHANGES,
        None,
        "abc1234",
        dry_run=False,
    )
    weekly.record(
        project,
        weekly.Outcome(Status.UP_TO_DATE, "nothing"),
        [],
        None,
        "",
        dry_run=False,
    )
    kept = json.loads((tmp_path / "p.json").read_text())
    assert (kept["status"], kept["commit"], kept["changed"][0]["name"]) == (
        "applied",
        "abc1234",
        "numpy",
    )
    assert (
        len((tmp_path / "weekly.log").read_text().splitlines()) == 2
    )  # the log still has both runs
    weekly.record(
        project,
        weekly.Outcome(Status.WOULD_APPLY, "tests pass"),
        CHANGES,
        None,
        "",
        dry_run=True,
    )
    assert (
        json.loads((tmp_path / "dry-run/p.json").read_text())["status"] == "would_apply"
    )
    assert (
        json.loads((tmp_path / "p.json").read_text())["status"] == "applied"
    )  # a dry run never touches the real record


def test_a_new_uv_project_joins_by_itself_and_a_worktree_or_registered_one_does_not(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def make(name: str, *, git_dir: bool = True, tests: bool = False) -> Path:
        path = tmp_path / name
        (path / ".venv/bin").mkdir(parents=True)
        (path / ".venv/bin/python").touch()
        (path / "uv.lock").touch()
        (path / "pyproject.toml").write_text(f'[project]\nname = "{name}"\n')
        if git_dir:
            (path / ".git").mkdir()
        else:
            (path / ".git").write_text("gitdir: elsewhere")
        if tests:
            (path / "tests").mkdir()
        return path

    make("new-lib")
    make("app", tests=True)
    make("a-worktree", git_dir=False)
    registered = make("listed")
    make("_archive")
    make(".hidden-tool")
    monkeypatch.setattr(weekly, "pick_sync", lambda path: ("--all-extras",))
    listed = weekly.Project("listed", registered, Check.TESTS, "true")

    found = {p.name: p for p in weekly.discover([listed], roots=(tmp_path,))}

    assert sorted(found) == ["app", "new-lib"]
    assert found["new-lib"].command == 'python -c "import new_lib"'
    assert "pytest -q -x" in found["app"].command
    assert "timeout -k 5 300" in found["app"].command
    assert found["app"].sync == ("--all-extras",)


def test_apply_held_refuses_what_a_check_must_settle_and_applies_a_kept_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old, new = tmp_path / "uv.lock.old", tmp_path / "uv.lock.new"
    old.write_text("old")
    new.write_text("new")
    held = {
        "name": "p",
        "status": "held",
        "reason": "the numbers moved",
        "check": {"kind": "numbers", "passed": False},
    }

    def refusal(record=held, **given):
        facts = {
            "old_lock": old,
            "new_lock": new,
            "current_lock_sha": weekly.sha(old),
            "lock": weekly.LockState.CLEAN,
            "lock_problems": [],
        }
        return weekly.held_refusal("p", record, **{**facts, **given})

    assert refusal() is None
    assert "nothing is held" in refusal(None)
    assert "nothing is held" in refusal({**held, "status": "applied"})
    tests_failed = {**held, "check": {"kind": "tests", "passed": False}}
    assert "fix the project's code" in refusal(tests_failed)
    assert (
        refusal({**held, "check": None}) is None
    )  # e.g. held because the PyTorch stack moved
    assert "uncommitted" in refusal(lock=weekly.LockState.DIRTY)
    assert "kept locks are gone" in refusal(new_lock=tmp_path / "missing")
    assert "stale" in refusal(current_lock_sha="moved")
    assert "does not describe" in refusal(lock_problems=["x: venv 1, lock 2"])

    project = weekly.Project("p", tmp_path / "repo", Check.NUMBERS, "c")
    (tmp_path / "repo").mkdir()
    (tmp_path / "repo/uv.lock").write_text("old")
    state, cache = tmp_path / "state", tmp_path / "cache"
    (cache / "p").mkdir(parents=True)
    old.rename(cache / "p/uv.lock.old")
    new.rename(cache / "p/uv.lock.new")
    state.mkdir()
    (state / "p.json").write_text(json.dumps(held))
    applied = []
    monkeypatch.setattr(weekly, "STATE", state)
    monkeypatch.setattr(weekly, "CACHE", cache)
    monkeypatch.setattr(weekly, "all_projects", lambda: [project])
    monkeypatch.setattr(weekly, "lock_state", lambda p: weekly.LockState.CLEAN)
    monkeypatch.setattr(weekly, "describes_venv", lambda p: [])
    monkeypatch.setattr(weekly, "changed_packages", lambda a, b: CHANGES)
    monkeypatch.setattr(weekly, "apply", lambda *a: applied.append(a) or "abc1234")
    assert weekly.apply_held("p") == 0
    assert applied[0][3] == weekly.sha(
        tmp_path / "repo/uv.lock"
    )  # the lock it found is the one apply guards
    assert "check was skipped" in applied[0][5]
    kept = json.loads((state / "p.json").read_text())
    assert (kept["status"], kept["commit"]) == (
        "applied",
        "abc1234",
    ) and "check skipped" in kept["reason"]
    assert weekly.apply_held("p") == 1  # applied: nothing is held any more
