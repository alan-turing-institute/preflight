"""remote.py: what `submit.py --site` ships, checks and runs, with ssh and rsync replaced by a recording fake."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import remote
import site_config

REMOTE = site_config.Remote("box", "sync", data_root="/data/ds")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "lib.py").write_text("X = 1\n")
    (tmp_path / "run.py").write_text("import lib\n")
    (tmp_path / "data" / "set").mkdir(parents=True)
    return tmp_path


def arguments(repo: Path, **given: object) -> argparse.Namespace:
    base = {
        "project": "p", "what": "w", "job_class": "restart", "name": "n", "wait_limit": "1h", "level": "normal",
        "max_jobs": None, "mem": "8G", "gpus": 1, "cpus": None, "after": [], "resume_ok": False, "entry": "run.py",
        "repo": repo, "search": ["src"], "data": ["data/set"], "workdir": repo, "command": ["python", "run.py", "a b"],
        "dry_run": False, "site": "turing", "service": False, "anywhere": False, "suspend": None, "no_suspend": False,
        "log": None, "data_base": None, "data_env": None,
    }
    return argparse.Namespace(**{**base, **given})


class Box:
    """Stands in for subprocess.run: records calls; answers md5sum with the real digests unless told to differ."""

    def __init__(self, repo: Path, *, stale: bool = False, stray: str = "", old: bool = False) -> None:
        self.calls: list[list[str]] = []
        self.sent: list[str] = []  # what was piped to the other site's Python
        self.repo, self.stale, self.stray, self.old = repo, stale, stray, old

    def __call__(self, argv: list[str], **kw: object) -> SimpleNamespace:
        self.calls.append(argv)
        out = ""
        if kw.get("input"):  # the import check: refused when `stray` says Python there loads a name from elsewhere
            self.sent.append(str(kw["input"]))
            return SimpleNamespace(returncode=1 if self.stray else 0, stdout="", stderr=self.stray)
        if argv[-1].endswith("gpujobs-submit --help"):
            return SimpleNamespace(returncode=0, stdout="usage: submit.py" + ("" if self.old else " [--check CMD]"), stderr="")
        if argv[0] == "rsync":
            out = ">f.st...... run.py\n"
        elif argv[0] == "ssh" and "md5sum" in argv[-1]:
            names = argv[-1].split("md5sum ")[1].replace("'", "").split()
            out = "".join(
                f"{'0' * 32 if self.stale else hashlib.md5((self.repo / n).read_bytes()).hexdigest()}  {n}\n" for n in names
            )
        elif argv[0] == "ssh" and "gpujobs-submit" in argv[-1]:
            out = "[submit] job 7 (normal); log /var/lib/gpujobs/logs/n.log\n"
        return SimpleNamespace(returncode=0, stdout=out, stderr="")


def test_sync_then_submit_in_the_same_place_with_the_same_arguments(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    box = Box(repo)
    monkeypatch.setattr(remote.subprocess, "run", box)
    result = remote.submit(arguments(repo), SimpleNamespace(remotes={"turing": REMOTE}))
    assert result.startswith("[submit] job 7")
    rsync = next(c for c in box.calls if c[0] == "rsync")
    assert f"{repo}/./run.py" in rsync and f"{repo}/./src/lib.py" in rsync  # the import closure, deferred module included
    assert f"{repo}/./data/set/" in rsync and "--checksum" in rsync and rsync[-1] == "box:sync/p/"
    order = [c[0] if c[0] == "rsync" else c[-1].split()[0] for c in box.calls]
    assert order.index("rsync") < order.index("cd") and "md5sum" in " ".join(c[-1] for c in box.calls)  # verified before submitting
    submitted = box.calls[-1][-1]
    assert submitted.startswith("cd sync/p/. && gpujobs-submit --project p --what w --class restart --name n")
    assert "--gpus 1" in submitted and "--mem 8G" in submitted and submitted.endswith("-- python run.py 'a b'")
    assert "--max-jobs" not in submitted  # none asked: the other site decides, never the text "None"


def test_data_outside_the_repo_lands_under_data_root_and_the_job_is_told_where(
    repo: Path, tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = tmp_path_factory.mktemp("store")
    (store / "gpe" / "laser").mkdir(parents=True)
    box = Box(repo)
    monkeypatch.setattr(remote.subprocess, "run", box)
    remote.submit(
        arguments(repo, data=[str(store / "gpe" / "laser")], data_base=store, data_env="DATASETS"),
        SimpleNamespace(remotes={"turing": REMOTE}),
    )
    data_sync = next(c for c in box.calls if c[0] == "rsync" and c[-2].endswith("/gpe/laser/"))
    assert data_sync[-2] == f"{store}/./gpe/laser/" and data_sync[-1] == "box:/data/ds/"  # nested name kept, fixed place
    assert "--no-implied-dirs" in data_sync and "--checksum" in data_sync  # shared parent folders are left alone
    assert box.calls[-1][-1].endswith("-- env DATASETS=/data/ds python run.py 'a b'")


def test_a_mismatch_after_the_copy_queues_nothing(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    box = Box(repo, stale=True)
    monkeypatch.setattr(remote.subprocess, "run", box)
    with pytest.raises(remote.RemoteRefused, match="still differs"):
        remote.submit(arguments(repo), SimpleNamespace(remotes={"turing": REMOTE}))
    assert not any("gpujobs-submit" in c[-1] for c in box.calls if c[0] == "ssh")


def test_dry_run_copies_nothing_and_what_cannot_be_shipped_is_refused(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    box = Box(repo)
    monkeypatch.setattr(remote.subprocess, "run", box)
    site = SimpleNamespace(remotes={"turing": REMOTE})
    bare = SimpleNamespace(remotes={"turing": site_config.Remote("box", "sync")})
    with pytest.raises(remote.RemoteRefused, match="needs data_root"):  # outside data with nowhere to put it
        remote.submit(arguments(repo, data=[str(repo.parent)]), bare)
    assert "nothing copied" in remote.submit(arguments(repo, dry_run=True), site) and box.calls == []
    for bad, message in (
        ({"data": ["nope"]}, "not a folder"),
        ({"data": [str(repo.parent)], "data_base": repo / "data"}, "outside --data-base"),
        ({"service": True}, "belongs to the site that owns the queue"),
        ({"entry": None}, "needs --repo and --entry"),
        ({"site": "nowhere"}, "names no such remote"),
    ):
        with pytest.raises(remote.RemoteRefused, match=message):
            remote.submit(arguments(repo, **bad), site)
    (repo / "run.py").write_text("import lib\nimport ghost_in_repo\n")
    (repo / "elsewhere").mkdir()
    (repo / "elsewhere" / "ghost_in_repo.py").write_text("")
    with pytest.raises(remote.RemoteRefused, match="closure could not be built"):  # a repo import on no search root
        remote.submit(arguments(repo), site)


def test_the_check_runs_there_after_the_copy_with_the_jobs_data_path_and_an_old_site_is_refused(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D47 with --site: the check travels to the other site's submit.py (it runs after the copy, where the files and
    data are), under the job's own `env NAME=path`; a site whose submit.py has no --check is refused before any copy."""
    site = SimpleNamespace(remotes={"turing": REMOTE})
    box = Box(repo)
    monkeypatch.setattr(remote.subprocess, "run", box)
    remote.submit(arguments(repo, check="python -m dry_run", data_env="DATASETS"), site)
    assert "--check 'env DATASETS=/data/ds python -m dry_run'" in box.calls[-1][-1]  # one word, the data path first
    assert any(c[0] == "rsync" for c in box.calls)
    old = Box(repo, old=True)
    monkeypatch.setattr(remote.subprocess, "run", old)
    with pytest.raises(remote.RemoteRefused, match="predates submit-time checks"):
        remote.submit(arguments(repo, no_check="no inputs"), site)
    assert not any(c[0] == "rsync" for c in old.calls)  # nothing was copied
    plain = Box(repo, old=True)  # no check asked: an old site is fine, and is not asked
    monkeypatch.setattr(remote.subprocess, "run", plain)
    remote.submit(arguments(repo), site)
    assert not any(c[-1].endswith("--help") for c in plain.calls)


def test_the_synced_names_must_load_from_the_synced_folder_before_anything_is_queued(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Python there is the job's own (named, or the command's first word), run from the job's folder; a name it loads
    from elsewhere queues nothing; a command that is not a Python needs --remote-python."""
    site = SimpleNamespace(remotes={"turing": REMOTE})
    box = Box(repo, stray="lib -> /elsewhere/src/lib.py")
    monkeypatch.setattr(remote.subprocess, "run", box)
    with pytest.raises(remote.RemoteRefused, match="/elsewhere/src/lib.py"):
        remote.submit(arguments(repo, command=[".venv/bin/python", "run.py"]), site)
    ran = next(c[-1] for c in box.calls if " - '" in c[-1])
    assert ran.startswith("cd sync/p/. && .venv/bin/python - '") and '"names": ["lib", "run"]' in ran
    assert '"roots": ["src", "."]' in ran and '"project": "."' in ran  # the roots the closure was walked with
    assert "find_spec" in box.sent[0]
    assert not any("gpujobs-submit --project" in c[-1] for c in box.calls)  # nothing was queued
    clean = Box(repo)
    monkeypatch.setattr(remote.subprocess, "run", clean)
    with pytest.raises(remote.RemoteRefused, match="--remote-python"):
        remote.submit(arguments(repo, command=["uv", "run", "python", "run.py"]), site)
    assert clean.calls == []  # refused before any copy
    remote.submit(arguments(repo, command=["uv", "run", "python", "run.py"], remote_python=".venv/bin/python"), site)
    assert any(c[-1].startswith("cd sync/p/. && .venv/bin/python - ") for c in clean.calls)
    clean.calls.clear()  # a command led by `env NAME=value`: the check runs under the same variables (PYTHONPATH)
    remote.submit(arguments(repo, command=["env", "PYTHONPATH=/g", "py/bin/python3", "-m", "run"]), site)
    assert any(c[-1].startswith("cd sync/p/. && env PYTHONPATH=/g py/bin/python3 - ") for c in clean.calls)


def test_python_that_would_load_a_package_from_elsewhere_is_caught_by_the_real_check(tmp_path: Path) -> None:
    """2026-11-16: `src/pkg` shipped without its `__init__.py` is a namespace portion, and a regular `pkg` later on the
    path (the venv's editable install of another checkout) wins. Run the real script under a real interpreter."""
    project, other = tmp_path / "project", tmp_path / "editable"
    (project / "src" / "pkg").mkdir(parents=True)
    (project / "src" / "pkg" / "mod.py").write_text("")
    (other / "pkg").mkdir(parents=True)
    (other / "pkg" / "__init__.py").write_text("")
    spec = json.dumps({"project": str(project), "roots": ["src"], "names": ["pkg"]})

    def run() -> subprocess.CompletedProcess[str]:
        return subprocess.run([sys.executable, "-", spec], input=remote.IMPORT_CHECK.read_text(), capture_output=True,
                              text=True, check=False, cwd=project, env={"PYTHONPATH": str(other)})

    refused = run()
    assert refused.returncode == 1 and f"pkg -> {other / 'pkg' / '__init__.py'}" in refused.stderr
    assert "add the missing __init__.py" in refused.stderr  # names the remedy
    (project / "src" / "pkg" / "__init__.py").write_text("")
    assert run().returncode == 0


def test_a_symlink_that_would_dangle_there_refuses_the_data_folder_naming_it(repo: Path) -> None:
    """2026-11-15: a link synced without its 487 MB target died in seconds. A link inside its folder travels fine."""
    outside = repo.parent / "big.bin"
    outside.write_text("x")
    folder = repo / "data" / "set"
    (folder / "real.bin").write_text("y")
    (folder / "ok.bin").symlink_to("real.bin")  # relative, stays inside
    assert remote.plan(arguments(repo), REMOTE).data == ["data/set"]
    (folder / "weights.bin").symlink_to(outside)  # absolute
    with pytest.raises(remote.RemoteRefused, match="weights.bin -> .*big.bin.*Name the folder it points at") as refused:
        remote.plan(arguments(repo), REMOTE)
    assert "ok.bin" not in str(refused.value)
    (folder / "weights.bin").unlink()
    (folder / "up.bin").symlink_to(Path("..") / ".." / ".." / "big.bin")  # relative, leaves the folder
    with pytest.raises(remote.RemoteRefused, match="up.bin"):
        remote.plan(arguments(repo), REMOTE)


def test_fetch_back_folders_must_be_inside_the_repo_and_land_where_asked(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (repo / "out").mkdir()
    (repo / "notes.txt").write_text("x")
    pairs = remote.fetch_pairs(arguments(repo, fetch_back=["out", "runs/new"]), REMOTE)
    assert pairs == [("sync/p/out", str(repo / "out")), ("sync/p/runs/new", str(repo / "runs" / "new"))]  # may not exist yet
    moved = remote.fetch_pairs(arguments(repo, fetch_back=["out"], fetch_to=["/cache/p/out"]), REMOTE)
    assert moved == [("sync/p/out", "/cache/p/out")]
    for given, extra, fragment in [
        ("../elsewhere", {}, "inside --repo"), (".", {}, "inside --repo"), ("out/../..", {}, "inside --repo"),
        ("notes.txt", {}, "not a folder"), ("out", {"fetch_to": ["/a", "/b"]}, "one --fetch-to for each"),
        ("out", {"fetch_to": ["cache/out"]}, "absolute path"),
    ]:
        with pytest.raises(remote.RemoteRefused, match=re.escape(fragment)):
            remote.fetch_pairs(arguments(repo, fetch_back=[given], **extra), REMOTE)
    box = Box(repo)
    monkeypatch.setattr(remote.subprocess, "run", box)
    fetch_only = site_config.Remote("box", "sync", submit_command="")
    with pytest.raises(remote.RemoteRefused, match="fetch_back.py add --site turing"):
        remote.submit(arguments(repo), SimpleNamespace(remotes={"turing": fetch_only}))
    assert box.calls == []
